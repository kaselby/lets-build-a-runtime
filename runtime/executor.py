"""Executor: walks an execution plan and dispatches to backend kernels.

The executor is the "runtime" half of the system — it allocates memory,
binds buffers, and dispatches each node to the appropriate kernel. The
ExecutionPlan is the "compiled" half — static, reusable, serializable.

Two execution modes:
  1. Per-op dispatch (Python loop + backend kernels via ctypes)
  2. Compiled C dispatch (one ctypes call for the whole plan)

Backends register kernel implementations for op types. The executor
tries backends in priority order (first match wins), so you get
automatic fallback: [c_backend, numpy_backend] means C handles what
it can, numpy catches the rest.
"""

import ctypes
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from .ir import Graph, Node, OpType
from .planner import ExecutionPlan


# Kernel contract: given input buffers and a pre-allocated output buffer,
# compute the result and write it into output. No allocations, no returns.
KernelFn = Callable[[list[np.ndarray], np.ndarray, dict[str, Any]], None]


class Backend(Protocol):
    """Interface for an execution backend (CPU, CUDA, etc.)."""
    name: str

    def get_kernel(self, op: OpType) -> KernelFn | None:
        """Return a kernel for this op, or None if unsupported."""
        ...


# ---------------------------------------------------------------------------
# C executor types and loading
# ---------------------------------------------------------------------------

MAX_INPUTS = 8
MAX_DIMS = 16
FLOAT_PTR = ctypes.POINTER(ctypes.c_float)


class COpNode(ctypes.Structure):
    """Mirrors the OpNode struct in executor.c."""
    _fields_ = [
        ("op", ctypes.c_int),
        ("n_inputs", ctypes.c_int),
        ("inputs", FLOAT_PTR * MAX_INPUTS),
        ("output", FLOAT_PTR),
        ("out_shape", ctypes.c_int * MAX_DIMS),
        ("n_dims", ctypes.c_int),
        ("extra", ctypes.c_int * MAX_DIMS),
    ]


def _load_c_executor() -> ctypes.CDLL | None:
    """Try to load the compiled C executor library."""
    csrc_dir = Path(__file__).parent.parent / "csrc"
    if sys.platform == "darwin":
        lib_path = csrc_dir / "libexecutor.dylib"
    else:
        lib_path = csrc_dir / "libexecutor.so"
    if not lib_path.exists():
        return None
    lib = ctypes.CDLL(str(lib_path))
    lib.execute.argtypes = [ctypes.POINTER(COpNode), ctypes.c_int]
    lib.execute.restype = None
    return lib


_c_executor_lib = _load_c_executor()


@dataclass
class CompiledPlan:
    """A plan compiled into C-friendly OpNode array, ready for fast execution.

    Built once per plan, reused across inference calls. Only the graph
    input pointers need patching before each call.
    """
    nodes: ctypes.Array          # COpNode array
    n_nodes: int
    arena: np.ndarray            # owned arena buffer
    graph: Graph                 # reference to the original graph
    # Map: graph input tensor name -> list of (node_index, input_slot) to patch
    input_slots: dict[str, list[tuple[int, int]]]
    # Keep references to scratch views so they aren't garbage collected
    _scratch_views: list[np.ndarray] = field(default_factory=list, repr=False)


@dataclass
class Executor:
    """Dispatches execution plan nodes to backend kernels.

    Holds an ordered list of backends tried in priority order.
    Allocates and reuses an arena buffer across inference calls.
    """
    backends: list[Backend]
    _arena: np.ndarray | None = field(default=None, repr=False)

    def execute(self, plan: ExecutionPlan,
                inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference on an execution plan.

        Args:
            plan: The compiled execution plan (graph + memory layout).
            inputs: Map of input tensor names to numpy arrays.

        Returns:
            Map of output tensor names to numpy arrays (copies, not arena views).
        """
        graph = plan.graph

        # Allocate or reuse arena
        arena = self._get_arena(plan.arena_size)

        # Bind all buffers
        self._bind_inputs(graph, inputs)
        self._bind_intermediates(graph, plan.offsets, arena)

        # Execute
        for node in plan.order:
            if node.op == OpType.RESHAPE:
                # Zero-copy: output is a reshaped view of input's buffer
                input_buf = graph.tensors[node.inputs[0]].buffer
                graph.tensors[node.output].buffer = input_buf.reshape(
                    graph.tensors[node.output].shape
                )
                continue
            kernel = self._resolve_kernel(node.op)
            input_buffers = [graph.tensors[inp].buffer for inp in node.inputs]
            # Append scratch buffer if planner allocated one for this node
            if node.id in plan.scratch:
                offset, size_bytes = plan.scratch[node.id]
                scratch_buf = arena[offset:offset + size_bytes].view(np.float32)
                input_buffers.append(scratch_buf)
            output_buffer = graph.tensors[node.output].buffer
            kernel(input_buffers, output_buffer, node.attrs)

        # Collect outputs (copy so caller isn't holding arena views)
        return {
            name: graph.tensors[name].buffer.copy()
            for name in graph.outputs
        }

    def _resolve_kernel(self, op: OpType) -> KernelFn:
        """Find the first backend that supports this op."""
        for backend in self.backends:
            kernel = backend.get_kernel(op)
            if kernel is not None:
                return kernel
        raise RuntimeError(f"No backend supports {op.name}")

    def _get_arena(self, size: int) -> np.ndarray:
        """Allocate or reuse the arena buffer."""
        if self._arena is None or len(self._arena) < size:
            self._arena = np.zeros(size, dtype=np.uint8)
        return self._arena

    def _bind_inputs(self, graph: Graph, inputs: dict[str, np.ndarray]) -> None:
        """Point input tensors at user-provided arrays."""
        for name in graph.inputs:
            if name not in inputs:
                raise ValueError(f"Missing input tensor: '{name}'")
            graph.tensors[name].buffer = inputs[name]

    def _bind_intermediates(self, graph: Graph, offsets: dict[str, int],
                            arena: np.ndarray) -> None:
        """Point intermediate tensors at their arena views."""
        for name, offset in offsets.items():
            tensor = graph.tensors[name]
            dtype = np.dtype(tensor.dtype)
            size_bytes = int(np.prod(tensor.shape)) * dtype.itemsize
            flat_view = arena[offset:offset + size_bytes]
            tensor.buffer = flat_view.view(dtype).reshape(tensor.shape)

    # ------------------------------------------------------------------
    # Compiled C execution (one ctypes call for the whole plan)
    # ------------------------------------------------------------------

    def compile_plan(self, plan: ExecutionPlan) -> CompiledPlan:
        """Compile an execution plan into a C-friendly OpNode array.

        Allocates the arena, binds intermediate and constant buffers,
        and builds the struct array with all pointers pre-resolved.
        Only graph input pointers are left as NULL — patched per call.
        """
        if _c_executor_lib is None:
            raise RuntimeError("C executor library not found — build csrc/ first")

        graph = plan.graph
        arena = self._get_arena(plan.arena_size)

        # Bind intermediates so we can grab their pointers
        self._bind_intermediates(graph, plan.offsets, arena)

        # Bind RESHAPE aliases (zero-copy views of their input's buffer)
        for node in plan.order:
            if node.op == OpType.RESHAPE:
                inp_buf = graph.tensors[node.inputs[0]].buffer
                out_tensor = graph.tensors[node.output]
                out_tensor.buffer = inp_buf.reshape(out_tensor.shape)

        # Build the COpNode array — skip RESHAPE nodes since they're
        # zero-copy aliases already bound above (no C work to do)
        exec_order = [node for node in plan.order if node.op != OpType.RESHAPE]
        n = len(exec_order)
        NodeArray = COpNode * n
        nodes = NodeArray()

        # Track which slots need input patching per call
        input_names = set(graph.inputs)
        input_slots: dict[str, list[tuple[int, int]]] = {
            name: [] for name in graph.inputs
        }

        # Bind scratch buffers as arena views (keep references to prevent GC)
        scratch_views: list[np.ndarray] = []

        for i, node in enumerate(exec_order):
            c_node = nodes[i]
            c_node.op = node.op.value
            c_node.n_inputs = len(node.inputs)

            # Resolve input pointers
            for j, inp_name in enumerate(node.inputs):
                if inp_name in input_names:
                    # Will be patched per call — record the slot
                    input_slots[inp_name].append((i, j))
                    c_node.inputs[j] = ctypes.cast(0, FLOAT_PTR)  # NULL for now
                else:
                    tensor = graph.tensors[inp_name]
                    c_node.inputs[j] = tensor.buffer.ctypes.data_as(FLOAT_PTR)

            # Append scratch pointer as extra input if planner allocated one
            if node.id in plan.scratch:
                offset, size_bytes = plan.scratch[node.id]
                scratch_view = arena[offset:offset + size_bytes].view(np.float32)
                scratch_views.append(scratch_view)  # prevent GC
                scratch_idx = c_node.n_inputs
                c_node.inputs[scratch_idx] = scratch_view.ctypes.data_as(FLOAT_PTR)
                c_node.n_inputs += 1

            # Resolve output pointer
            out_tensor = graph.tensors[node.output]
            c_node.output = out_tensor.buffer.ctypes.data_as(FLOAT_PTR)

            # Fill shape info
            shape = out_tensor.shape
            c_node.n_dims = len(shape)
            for d in range(len(shape)):
                c_node.out_shape[d] = shape[d]

            # Fill op-specific extras
            self._fill_extras(c_node, node, graph)

        return CompiledPlan(
            nodes=nodes,
            n_nodes=n,
            arena=arena,
            graph=graph,
            input_slots=input_slots,
            _scratch_views=scratch_views,
        )

    def execute_compiled(self, compiled: CompiledPlan,
                         inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Execute a compiled plan. One ctypes call."""
        # Patch input pointers
        for name, slots in compiled.input_slots.items():
            if name not in inputs:
                raise ValueError(f"Missing input tensor: '{name}'")
            ptr = inputs[name].ctypes.data_as(FLOAT_PTR)
            for node_idx, slot_idx in slots:
                compiled.nodes[node_idx].inputs[slot_idx] = ptr

        # One C call for the whole plan
        _c_executor_lib.execute(compiled.nodes, compiled.n_nodes)

        # Collect outputs
        return {
            name: compiled.graph.tensors[name].buffer.copy()
            for name in compiled.graph.outputs
        }

    @staticmethod
    def _fill_extras(c_node: COpNode, node: Node, graph: Graph) -> None:
        """Fill op-specific extra fields on a COpNode."""
        if node.op in (OpType.MATMUL, OpType.MATMUL_ADD):
            # extra[0] = K, extra[1] = trans_b, extra[2] = b_is_2d,
            # extra[3] = alpha as float bits (0 means 1.0)
            a_shape = graph.tensors[node.inputs[0]].shape
            b_shape = graph.tensors[node.inputs[1]].shape
            c_node.extra[0] = a_shape[-1]
            c_node.extra[1] = 1 if node.attrs.get("transpose_b") else 0
            c_node.extra[2] = 1 if len(b_shape) == 2 and len(a_shape) > 2 else 0
            alpha = node.attrs.get("alpha", 1.0)
            if alpha != 1.0:
                c_node.extra[3] = struct.unpack('i', struct.pack('f', alpha))[0]

        elif node.op == OpType.TRANSPOSE:
            in_shape = graph.tensors[node.inputs[0]].shape
            if len(in_shape) == 2:
                # 2D fast path: extra = [rows, cols]
                c_node.extra[0] = in_shape[0]
                c_node.extra[1] = in_shape[1]
            else:
                # N-dim swapaxes: decompose into [outer, A, middle, B, inner]
                dim0 = node.attrs.get("dim0", 0)
                dim1 = node.attrs.get("dim1", 1)
                if dim0 > dim1:
                    dim0, dim1 = dim1, dim0
                outer = 1
                for d in range(dim0):
                    outer *= in_shape[d]
                A = in_shape[dim0]
                middle = 1
                for d in range(dim0 + 1, dim1):
                    middle *= in_shape[d]
                B = in_shape[dim1]
                inner = 1
                for d in range(dim1 + 1, len(in_shape)):
                    inner *= in_shape[d]
                c_node.extra[0] = outer
                c_node.extra[1] = A
                c_node.extra[2] = middle
                c_node.extra[3] = B
                c_node.extra[4] = inner

        elif node.op == OpType.ADD:
            if "scalar" in node.attrs:
                # extra[0] = 2 (scalar mode), extra[1] = scalar as float bits
                c_node.extra[0] = 2
                c_node.extra[1] = struct.unpack('i', struct.pack('f', node.attrs["scalar"]))[0]
            else:
                # extra[0] = 1 if element-wise (same shape), 0 if bias broadcast
                a_shape = graph.tensors[node.inputs[0]].shape
                b_shape = graph.tensors[node.inputs[1]].shape
                c_node.extra[0] = 1 if a_shape == b_shape else 0

        elif node.op in (OpType.DIV, OpType.SUB, OpType.MUL):
            if "scalar" in node.attrs:
                # extra[0] = 1 (scalar mode), extra[1] = scalar as float bits
                c_node.extra[0] = 1
                c_node.extra[1] = struct.unpack('i', struct.pack('f', node.attrs["scalar"]))[0]

        elif node.op in (OpType.MAX, OpType.SUM):
            # extra[0] = axis, extra[1] = axis_size (from input shape)
            axis = node.attrs.get("axis", -1)
            in_shape = graph.tensors[node.inputs[0]].shape
            c_node.extra[0] = axis
            c_node.extra[1] = in_shape[axis]

        elif node.op == OpType.LAYERNORM:
            # extra[0] = eps as float bits packed into int
            eps = node.attrs.get("eps", 1e-5)
            c_node.extra[0] = struct.unpack('i', struct.pack('f', eps))[0]

        elif node.op == OpType.ATTENTION:
            # extra[0] = seq_len, extra[1] = head_dim
            # Q shape is [..., seq_len, head_dim] (3D or 4D)
            q_shape = graph.tensors[node.inputs[0]].shape
            c_node.extra[0] = q_shape[-2]  # seq_len
            c_node.extra[1] = q_shape[-1]  # head_dim

