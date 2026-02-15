"""Graph optimization passes.

A pass is any callable with signature (Graph) -> bool, where the return
value indicates whether the pass modified the graph. Passes mutate the
graph in-place.

The optimization pipeline is a configurable list of passes. Passes can
be reordered, repeated, or swapped out. A convenience function runs
passes until no pass reports changes (fixed-point iteration).
"""

from dataclasses import dataclass
from typing import Callable
import numpy as np

from ..ir import Graph, Node, OpType
from ..ops import OP_REGISTRY

# A pass takes a Graph, mutates it, and returns True if it made changes.
Pass = Callable[[Graph], bool]


# Default pass ordering. Users can build their own pipeline instead.
DEFAULT_PIPELINE: list[Pass] = []  # Populated after passes are defined


@dataclass
class PassResult:
    """Record of a single optimization pass execution."""
    name: str
    changed: bool
    nodes_before: int
    nodes_after: int

    def __str__(self) -> str:
        if self.changed:
            delta = self.nodes_after - self.nodes_before
            sign = "+" if delta >= 0 else ""
            return (f"[pass] {self.name}: {self.nodes_before} -> "
                    f"{self.nodes_after} nodes ({sign}{delta})")
        return f"[pass] {self.name}: no changes"


def run_pipeline(graph: Graph, pipeline: list[Pass] | None = None,
                 log: list[PassResult] | None = None) -> None:
    """Run a list of passes on the graph, once each in order.

    If log is provided, appends a PassResult for each pass.
    """
    for p in (pipeline or DEFAULT_PIPELINE):
        n_before = len(graph.nodes)
        changed = p(graph)
        if log is not None:
            log.append(PassResult(
                name=getattr(p, '__name__', str(p)),
                changed=changed,
                nodes_before=n_before,
                nodes_after=len(graph.nodes),
            ))


def run_until_stable(graph: Graph, pipeline: list[Pass] | None = None,
                     max_iterations: int = 10,
                     log: list[PassResult] | None = None) -> int:
    """Run passes repeatedly until none of them make changes.

    Returns the number of iterations performed. Useful when passes
    create new opportunities for each other (e.g., constant folding
    after fusion may enable more DCE).
    """
    passes = pipeline or DEFAULT_PIPELINE
    for i in range(max_iterations):
        changed = False
        for p in passes:
            n_before = len(graph.nodes)
            result = p(graph)
            changed |= result
            if log is not None:
                log.append(PassResult(
                    name=getattr(p, '__name__', str(p)),
                    changed=result,
                    nodes_before=n_before,
                    nodes_after=len(graph.nodes),
                ))
        if not changed:
            return i + 1
    return max_iterations


# ---------------------------------------------------------------------------
# MATMUL absorption
# ---------------------------------------------------------------------------

def _absorb_transpose_b(node: Node, graph: Graph) -> bool:
    """Absorb a TRANSPOSE on the B input into transpose_b=True.

    Only absorbs last-two-dim swaps (that's what CblasTrans means).
    The TRANSPOSE must be the sole consumer path — if other nodes also
    read the transposed tensor, we can't remove it.
    """
    if node.attrs.get("transpose_b"):
        return False

    b_name = node.inputs[1]
    b_producer = graph.producer(b_name)
    if (b_producer is None
            or b_producer.op != OpType.TRANSPOSE
            or len(graph.consumers(b_name)) != 1):
        return False

    # Only absorb if the transpose swaps the last two dims
    ndim = len(graph.tensors[b_producer.inputs[0]].shape)
    dim0 = b_producer.attrs.get("dim0", 0)
    dim1 = b_producer.attrs.get("dim1", 1)
    if {dim0, dim1} != {ndim - 2, ndim - 1}:
        return False

    graph.rewire_input(node.id, b_name, b_producer.inputs[0])
    node.attrs["transpose_b"] = True
    graph.remove_node(b_producer.id)
    graph.remove_tensor(b_name)
    return True


def _pretranspose_constant_b(node: Node, graph: Graph) -> bool:
    """Pre-transpose a constant B input and set transpose_b=True.

    When B is a constant (e.g., Conv1D weight in [K, N] layout) and
    transpose_b isn't already set, physically transpose the buffer to
    [N, K] so CBLAS can use CblasTrans for better packing performance.

    Only applies to 2D constants — batched matmuls with constant B
    are rare and the benefit is less clear.
    """
    if node.attrs.get("transpose_b"):
        return False

    b_name = node.inputs[1]
    if b_name not in graph.constants:
        return False

    b_info = graph.tensors[b_name]
    if b_info.buffer is None or len(b_info.shape) != 2:
        return False

    
    b_info.buffer = np.ascontiguousarray(b_info.buffer.T)
    b_info.shape = b_info.buffer.shape
    node.attrs["transpose_b"] = True
    return True


def _fold_scalar_into_alpha(node: Node, scalar: float, op: OpType) -> None:
    """Fold a scalar MUL or DIV into a MATMUL's alpha parameter.

    Matmul is bilinear, so s*(A@B) = (s*A)@B = A@(s*B).
    """
    alpha = node.attrs.get("alpha", 1.0)
    if op == OpType.MUL:
        node.attrs["alpha"] = alpha * scalar
    else:
        node.attrs["alpha"] = alpha / scalar


def _absorb_input_scalars(node: Node, graph: Graph) -> bool:
    """Absorb scalar MUL/DIV on either MATMUL input into alpha."""
    changed = False
    for inp_name in list(node.inputs):
        producer = graph.producer(inp_name)
        if (producer is not None
                and producer.op in (OpType.MUL, OpType.DIV)
                and "scalar" in producer.attrs
                and len(graph.consumers(inp_name)) == 1):
            _fold_scalar_into_alpha(node, producer.attrs["scalar"], producer.op)
            graph.rewire_input(node.id, inp_name, producer.inputs[0])
            graph.remove_node(producer.id)
            graph.remove_tensor(inp_name)
            changed = True
    return changed


def _absorb_output_scalar(node: Node, graph: Graph) -> bool:
    """Absorb a scalar MUL/DIV on the MATMUL output into alpha."""
    consumers = graph.consumers(node.output)
    if (len(consumers) != 1
            or consumers[0].op not in (OpType.MUL, OpType.DIV)
            or "scalar" not in consumers[0].attrs):
        return False

    consumer = consumers[0]
    _fold_scalar_into_alpha(node, consumer.attrs["scalar"], consumer.op)

    # Rewire everything that read the MUL/DIV output to read the MATMUL output
    for downstream in graph.consumers(consumer.output):
        graph.rewire_input(downstream.id, consumer.output, node.output)
    if consumer.output in graph.outputs:
        idx = graph.outputs.index(consumer.output)
        graph.outputs[idx] = node.output

    graph.remove_node(consumer.id)
    graph.remove_tensor(consumer.output)
    return True


def absorb_into_matmul(graph: Graph) -> bool:
    """Absorb adjacent ops into MATMUL nodes via sgemm parameters.

    For each MATMUL, tries four absorptions:
      1. TRANSPOSE on B input -> transpose_b=True (CblasTrans)
      2. Constant B without transpose_b -> pre-transpose buffer, set transpose_b
      3. Scalar MUL/DIV on inputs -> fold into alpha
      4. Scalar MUL/DIV on output -> fold into alpha

    Must run before constant folding — otherwise folding eagerly
    materializes transposes and scalars, destroying the patterns.
    """
    changed = False
    for node in list(graph):
        if node.op != OpType.MATMUL:
            continue
        changed |= _absorb_transpose_b(node, graph)
        changed |= _pretranspose_constant_b(node, graph)
        changed |= _absorb_input_scalars(node, graph)
        changed |= _absorb_output_scalar(node, graph)
    return changed


# ---------------------------------------------------------------------------
# Mask absorption into attention
# ---------------------------------------------------------------------------

def _is_causal_mask(buf: np.ndarray) -> bool:
    """Check if a constant buffer is a causal attention mask.

    Recognizes two formats:
      - Bool: lower-triangular True (HuggingFace-style)
      - Float: upper-triangular -inf, rest ~0 (additive mask style)
    """
    if buf.ndim < 2 or buf.shape[-1] != buf.shape[-2]:
        return False
    S = buf.shape[-1]
    if S < 2:
        return True  # Trivially causal for seq_len=1
    mat = buf.reshape(-1, S, S)[0]

    if buf.dtype == np.bool_:
        return np.array_equal(mat, np.tril(np.ones((S, S), dtype=np.bool_)))

    if np.issubdtype(buf.dtype, np.floating):
        lower = np.tril(mat)
        upper_strict = mat[np.triu_indices(S, k=1)]
        return np.allclose(lower, 0, atol=1e-6) and np.all(upper_strict < -1e4)

    return False


def absorb_mask_into_attention(graph: Graph) -> bool:
    """Replace ATTENTION(Q, K, V, constant_mask) with ATTENTION(Q, K, V, causal=True).

    After constant folding, attention mask tensors become constants. If the
    mask is a standard causal pattern, drop it and set causal=True so the
    kernel computes the mask on the fly.
    """
    changed = False
    constant_set = set(graph.constants)

    for node in list(graph):
        if node.op != OpType.ATTENTION or len(node.inputs) != 4:
            continue
        if node.attrs.get("causal"):
            continue

        mask_name = node.inputs[3]
        if mask_name not in constant_set:
            continue

        mask_info = graph.tensors[mask_name]
        if mask_info.buffer is None or not _is_causal_mask(mask_info.buffer):
            continue

        node.attrs["causal"] = True
        graph._consumers[mask_name].remove(node.id)
        node.inputs = node.inputs[:3]
        graph._order = None
        changed = True

    return changed


# ---------------------------------------------------------------------------
# Constant folding
# ---------------------------------------------------------------------------

def constant_fold(graph: Graph) -> bool:
    """Evaluate nodes whose inputs are all constants.

    The result becomes a new constant tensor (buffer set to the computed
    numpy array) and the node is removed. Uses evaluators from the
    centralized op registry. If input constants become dead after folding,
    DCE will clean them up.
    """
    changed = False
    constant_set = set(graph.constants)

    for node in list(graph):  # topological order — ensures inputs are folded first
        if not all(inp in constant_set for inp in node.inputs):
            continue

        op_def = OP_REGISTRY.get(node.op)
        if op_def is None or op_def.evaluator is None:
            continue

        input_buffers = [graph.tensors[inp].buffer for inp in node.inputs]
        result = op_def.evaluator(input_buffers, node.attrs)

        # Ensure contiguous layout — evaluators may return views (e.g. swapaxes)
        # and downstream C kernels need contiguous data.
        # Also enforce declared dtype — numpy may promote (e.g. int64 + 0.0 → float64)
        declared_dtype = np.dtype(graph.tensors[node.output].dtype)
        if result.dtype != declared_dtype:
            result = result.astype(declared_dtype)
        graph.tensors[node.output].buffer = np.ascontiguousarray(result)
        graph.constants.append(node.output)
        constant_set.add(node.output)

        graph.remove_node(node.id)
        changed = True

    return changed


# ---------------------------------------------------------------------------
# Dead code elimination
# ---------------------------------------------------------------------------

def eliminate_dead_code(graph: Graph) -> bool:
    """Remove nodes whose outputs have no consumers and aren't graph outputs.

    Repeats until stable — removing a node may make its predecessors dead.
    Also cleans up unused constant tensors.
    """
    changed = False
    output_set = set(graph.outputs)

    # Remove dead compute nodes
    while True:
        dead = [
            node for node in graph.nodes.values()
            if not graph.consumers(node.output) and node.output not in output_set
        ]
        if not dead:
            break
        for node in dead:
            graph.remove_node(node.id)
            graph.remove_tensor(node.output)
        changed = True

    # Remove unused constants (no consumers and not a graph output)
    dead_constants = [
        name for name in graph.constants
        if not graph.consumers(name) and name not in output_set
    ]
    for name in dead_constants:
        graph.constants.remove(name)
        graph.remove_tensor(name)
        changed = True

    return changed



# ---------------------------------------------------------------------------
# Default pipeline
# ---------------------------------------------------------------------------

from .fusion import fuse, fuse_dags

DEFAULT_PIPELINE.extend([
    absorb_into_matmul,
    constant_fold,
    absorb_mask_into_attention,
    fuse_dags,
    fuse,
    eliminate_dead_code,
])