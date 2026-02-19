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

def _has_unresolved_symbols(attrs: dict, symbols: set[str]) -> bool:
    """Check if any attr value contains an unresolved symbol string."""
    for v in attrs.values():
        if isinstance(v, str) and v in symbols:
            return True
        if isinstance(v, (tuple, list)):
            if any(isinstance(x, str) and x in symbols for x in v):
                return True
    return False


def constant_fold(graph: Graph) -> bool:
    """Evaluate nodes whose inputs are all constants.

    The result becomes a new constant tensor (buffer set to the computed
    numpy array) and the node is removed. Uses evaluators from the
    centralized op registry. If input constants become dead after folding,
    DCE will clean them up.

    Nodes whose attrs contain unresolved symbol strings (from dynamic
    shapes) are skipped — they'll be folded post-resolution when
    symbols have concrete values.
    """
    changed = False
    constant_set = set(graph.constants)
    symbols = set(graph.dynamic_dims.keys())

    for node in list(graph):  # topological order — ensures inputs are folded first
        if not all(inp in constant_set for inp in node.inputs):
            continue

        op_def = OP_REGISTRY.get(node.op)
        if op_def is None or op_def.evaluator is None:
            continue

        if symbols and _has_unresolved_symbols(node.attrs, symbols):
            continue

        input_buffers = [graph.tensors[inp].buffer for inp in node.inputs]
        result = op_def.evaluator(input_buffers, node.attrs)

        # Ensure contiguous layout — evaluators may return views (e.g. swapaxes)
        # and downstream C kernels need contiguous data.
        # Also enforce declared dtype — numpy may promote (e.g. int64 + 0.0 → float64)
        declared_dtype = np.dtype(graph.tensors[node.output].dtype)
        if result.dtype != declared_dtype:
            result = result.astype(declared_dtype)
        out_tensor = graph.tensors[node.output]
        out_tensor.buffer = np.ascontiguousarray(result)
        out_tensor.shape = result.shape
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
# Parallel matmul merging
# ---------------------------------------------------------------------------

def merge_parallel_matmuls(graph: Graph) -> bool:
    """Merge multiple MATMULs sharing the same input into one larger MATMUL.

    Detects groups of MATMUL nodes that read the same activation tensor and
    have constant weight matrices with compatible shapes (same K, same
    trans_b). Replaces N matmuls with one merged matmul + N SLICE nodes.

    Typical targets: QKV projections (3→1), gate/up projections (2→1).
    """
    changed = False
    constant_set = set(graph.constants)

    # Find tensors consumed by 2+ MATMULs
    from collections import defaultdict
    matmul_groups: dict[str, list[Node]] = defaultdict(list)
    for node in graph:
        if node.op == OpType.MATMUL and node.inputs[1] in constant_set:
            matmul_groups[node.inputs[0]].append(node)

    for shared_input, group in matmul_groups.items():
        if len(group) < 2:
            continue

        # Validate: same K, same trans_b, all weights are 2D constants
        first = group[0]
        trans_b = first.attrs.get("transpose_b", False)
        alpha = first.attrs.get("alpha", 1.0)
        w0_shape = graph.tensors[first.inputs[1]].shape
        if len(w0_shape) != 2:
            continue

        K = w0_shape[1] if trans_b else w0_shape[0]
        compatible = True
        for node in group[1:]:
            if node.attrs.get("transpose_b", False) != trans_b:
                compatible = False
                break
            if node.attrs.get("alpha", 1.0) != alpha:
                compatible = False
                break
            ws = graph.tensors[node.inputs[1]].shape
            if len(ws) != 2:
                compatible = False
                break
            node_K = ws[1] if trans_b else ws[0]
            if node_K != K:
                compatible = False
                break
        if not compatible:
            continue

        # Compute merged weight shape and output slices
        # With trans_b: weight is (N, K), output dim is N (axis 0)
        # Without trans_b: weight is (K, N), output dim is N (axis 1)
        cat_axis = 0 if trans_b else 1
        weights = [graph.tensors[n.inputs[1]].buffer for n in group]
        merged_weight = np.concatenate(weights, axis=cat_axis)

        # Register merged weight as a constant
        merged_w_name = f"_merged_weight_{group[0].output}"
        w_info = graph.add_tensor(merged_w_name, merged_weight.shape, "float32")
        w_info.buffer = merged_weight
        graph.constants.append(merged_w_name)
        constant_set.add(merged_w_name)

        # Compute merged output shape
        in_shape = graph.tensors[shared_input].shape
        N_total = merged_weight.shape[0] if trans_b else merged_weight.shape[1]
        merged_out_shape = (*in_shape[:-1], N_total)

        # Register merged output tensor
        merged_out_name = f"_merged_matmul_{group[0].output}"
        graph.add_tensor(merged_out_name, merged_out_shape, "float32")

        # Add merged MATMUL node
        merged_attrs = {"transpose_b": trans_b}
        if alpha != 1.0:
            merged_attrs["alpha"] = alpha
        graph.add_node(OpType.MATMUL, [shared_input, merged_w_name],
                        merged_out_name, merged_attrs)

        # Replace each original MATMUL with a SLICE from the merged output
        last_dim = len(merged_out_shape) - 1
        offset = 0
        for node in group:
            w_shape = graph.tensors[node.inputs[1]].shape
            N_i = w_shape[0] if trans_b else w_shape[1]

            # Rewire: remove original MATMUL, add SLICE keeping same output name
            old_output = node.output
            old_weight = node.inputs[1]
            graph.remove_node(node.id)

            graph.add_node(OpType.SLICE, [merged_out_name], old_output,
                           {"dim": last_dim, "start": offset, "end": offset + N_i})
            offset += N_i

            # Clean up now-unused original weight constant
            if not graph.consumers(old_weight):
                if old_weight in graph.constants:
                    graph.constants.remove(old_weight)
                graph.remove_tensor(old_weight)

        changed = True

    return changed


# ---------------------------------------------------------------------------
# Pass pipelines
# ---------------------------------------------------------------------------

from .fusion import fuse, fuse_dags

# Pre-resolution: structural passes on the (possibly symbolic) graph.
# Constant folding folds what it can (weight-only subgraphs); ops with
# unresolved symbolic attrs are skipped and left for post-resolution.
PRE_RESOLUTION_PIPELINE: list[Pass] = [
    absorb_into_matmul,
    constant_fold,
    absorb_mask_into_attention,
    fuse_dags,
    merge_parallel_matmuls,
    fuse,
    eliminate_dead_code,
]

# Post-resolution: runs on the concrete resolved graph.  Folds ops that
# depended on dynamic dims (e.g., ARANGE with end=seq_len) and cleans up
# any dead subgraphs that folding exposes.
POST_RESOLUTION_PIPELINE: list[Pass] = [
    constant_fold,
    absorb_mask_into_attention,
    eliminate_dead_code,
]

# Backwards compat — existing tests and scripts use DEFAULT_PIPELINE.
DEFAULT_PIPELINE.extend(PRE_RESOLUTION_PIPELINE)