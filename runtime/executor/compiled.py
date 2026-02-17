"""Compiled executor: single ctypes call for the whole graph.

Builds an array of COpNode structs with all pointers pre-resolved at
compile time. Per-call cost is just patching graph input pointers and
one C function call. No Python-level dispatch, no kernel resolution.
"""

import ctypes
import struct
import time

import numpy as np

from ..ir import FOLD_ONLY_BASE, Graph, Node, OpType
from ..ops import OP_REGISTRY
from ..planner import MemoryPlan
from .common import COpNode, Executor, RunProfile, MAX_INPUTS, MAX_DIMS, _c_executor_lib


class CompiledExecutor(Executor):
    """Compiles an execution plan into a C struct array for fast dispatch."""

    def __init__(self, profile: bool = False) -> None:
        super().__init__(profile=profile)
        # Set during compile()
        self._nodes: ctypes.Array | None = None
        self._n_nodes: int = 0
        self._graph: Graph | None = None
        # graph input name -> list of (node_index, input_slot) to patch per call
        self._input_slots: dict[str, list[tuple[int, int]]] = {}
        # Keep references to arena/scratch views to prevent GC
        self._refs: list[np.ndarray] = []

    def compile(self, plan: MemoryPlan) -> None:
        """Build COpNode struct array with all pointers pre-resolved.

        After this, only graph input pointers need patching per call.
        """
        if _c_executor_lib is None:
            raise RuntimeError("C executor library not found — build csrc/ first")

        graph = plan.graph
        self._graph = graph
        arena = self._get_arena(plan.arena_size)

        # Bind all non-input buffers so we can grab their pointers
        self._bind_intermediates(graph, plan.offsets, arena)
        external = set(graph.inputs) | set(graph.constants)

        # Filter to nodes that need C dispatch (skip aliases, but not aliases of externals)
        exec_order = []
        for node in plan.order:
            if self._is_alias(node) and node.inputs[0] not in external:
                continue
            if node.op.value >= FOLD_ONLY_BASE:
                raise RuntimeError(
                    f"Fold-only op {node.op.name} reached compiled executor — "
                    f"should have been eliminated by constant folding"
                )
            exec_order.append(node)

        # Build struct array
        n = len(exec_order)
        self._nodes = (COpNode * n)()
        self._n_nodes = n
        self._input_slots = {name: [] for name in graph.inputs}
        self._refs = []

        for i, node in enumerate(exec_order):
            self._build_node(i, node, graph, plan, arena)

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Patch input pointers and execute the whole graph in one C call."""
        graph = self._graph
 
        # Patch input pointers
        for name, slots in self._input_slots.items():
            if name not in inputs:
                raise ValueError(f"Missing input tensor: '{name}'")
            ptr = inputs[name].ctypes.data
            for node_idx, slot_idx in slots:
                self._nodes[node_idx].inputs[slot_idx] = ptr

        # One call
        if self._profile:
            t0 = time.perf_counter_ns()
            _c_executor_lib.execute(self._nodes, self._n_nodes)
            total_ns = time.perf_counter_ns() - t0
            self._last_profile = RunProfile(total_ns=total_ns)
        else:
            _c_executor_lib.execute(self._nodes, self._n_nodes)

        # Copy outputs (caller shouldn't hold arena views)
        return {
            name: graph.tensors[name].buffer.copy()
            for name in graph.outputs
        }

    # ------------------------------------------------------------------
    # Compile helpers
    # ------------------------------------------------------------------

    def _is_alias(self, node: Node) -> bool:
        """Check if a node is an alias op that doesn't need C dispatch."""
        op_def = OP_REGISTRY.get(node.op)
        return op_def is not None and op_def.is_alias(node)

    def _build_node(self, i: int, node: Node, graph: Graph,
                    plan: MemoryPlan, arena: np.ndarray) -> None:
        """Populate one COpNode struct."""
        c_node = self._nodes[i]
        c_node.op = node.op.value
        c_node.n_inputs = len(node.inputs)

        # Resolve input pointers
        input_names = set(graph.inputs)
        for j, inp_name in enumerate(node.inputs):
            if inp_name in input_names:
                self._input_slots[inp_name].append((i, j))
                c_node.inputs[j] = 0  # patched per call
            else:
                c_node.inputs[j] = graph.tensors[inp_name].buffer.ctypes.data

        # Scratch buffer as extra input
        if node.id in plan.scratch:
            offset, size_bytes = plan.scratch[node.id]
            scratch_view = arena[offset:offset + size_bytes].view(np.float32)
            self._refs.append(scratch_view)
            scratch_idx = c_node.n_inputs
            c_node.inputs[scratch_idx] = scratch_view.ctypes.data
            c_node.n_inputs += 1

        # Output pointer
        c_node.output = graph.tensors[node.output].buffer.ctypes.data

        # Output shape and element size
        out_tensor = graph.tensors[node.output]
        shape = out_tensor.shape
        c_node.n_dims = len(shape)
        for d in range(len(shape)):
            c_node.out_shape[d] = shape[d]
        c_node.elem_size = np.dtype(out_tensor.dtype).itemsize

        # Op-specific extras
        _fill_extras(c_node, node, graph)


# ---------------------------------------------------------------------------
# Op-specific extra packing
# ---------------------------------------------------------------------------

def _fill_extras(c_node: COpNode, node: Node, graph: Graph) -> None:
    """Pack op-specific parameters into COpNode.extra[]."""
    op_def = OP_REGISTRY.get(node.op)
    if op_def is None or op_def.extras is None:
        return
    for i, val in enumerate(op_def.extras(node, graph)):
        c_node.extra[i] = val
