"""Graph optimization passes.

A pass is any callable with signature (Graph) -> bool, where the return
value indicates whether the pass modified the graph. Passes mutate the
graph in-place.

The optimization pipeline is a configurable list of passes. Passes can
be reordered, repeated, or swapped out. A convenience function runs
passes until no pass reports changes (fixed-point iteration).

The fusion pass uses a registry of FusionPattern objects that describe
what to match and what to replace it with. Patterns have priority levels
— lower priority number = matched first. Within a priority level,
longer patterns are tried first (greedy matching).
"""

from dataclasses import dataclass
from itertools import groupby
from typing import Any, Callable

import numpy as np

from .ir import Graph, Node, OpType


# ---------------------------------------------------------------------------
# Pass interface and pipeline
# ---------------------------------------------------------------------------

# A pass takes a Graph, mutates it, and returns True if it made changes.
Pass = Callable[[Graph], bool]


# Default pass ordering. Users can build their own pipeline instead.
DEFAULT_PIPELINE: list[Pass] = []  # Populated at bottom of file after passes are defined


def run_pipeline(graph: Graph, pipeline: list[Pass] | None = None) -> None:
    """Run a list of passes on the graph, once each in order."""
    for p in (pipeline or DEFAULT_PIPELINE):
        p(graph)


def run_until_stable(graph: Graph, pipeline: list[Pass] | None = None,
                     max_iterations: int = 10) -> int:
    """Run passes repeatedly until none of them make changes.

    Returns the number of iterations performed. Useful when passes
    create new opportunities for each other (e.g., constant folding
    after fusion may enable more DCE).
    """
    passes = pipeline or DEFAULT_PIPELINE
    for i in range(max_iterations):
        changed = False
        for p in passes:
            changed |= p(graph)
        if not changed:
            return i + 1
    return max_iterations


# ---------------------------------------------------------------------------
# MATMUL absorption
# ---------------------------------------------------------------------------

def absorb_into_matmul(graph: Graph) -> bool:
    """Absorb adjacent ops into MATMUL nodes via sgemm parameters.

    For each MATMUL, checks three things:

    1. **Transpose absorption (backward, B input):** If the B input comes
       from a TRANSPOSE that swaps the last two dims, absorb it as
       transpose_b=True. CblasTrans has better memory access than
       materializing the transposed matrix.

    2. **Scalar absorption (backward, either input):** If either input
       comes from a scalar MUL or DIV, fold the scalar into alpha.
       Matmul is bilinear: s*(A@B) = (s*A)@B = A@(s*B).

    3. **Scalar absorption (forward):** If the MATMUL's sole consumer
       is a scalar MUL or DIV, fold it into alpha the same way.

    Must run before constant folding — otherwise folding eagerly
    materializes transposes and scalars, destroying the patterns.
    """
    changed = False

    for node in list(graph):
        if node.op != OpType.MATMUL:
            continue

        # --- Transpose absorption on B input ---
        if not node.attrs.get("transpose_b"):
            b_name = node.inputs[1]
            b_producer = graph.producer(b_name)
            if (b_producer is not None
                    and b_producer.op == OpType.TRANSPOSE
                    and len(graph.consumers(b_name)) == 1):
                # Only absorb last-two-dim swaps (that's what transpose_b means)
                b_input_shape = graph.tensors[b_producer.inputs[0]].shape
                ndim = len(b_input_shape)
                dim0 = b_producer.attrs.get("dim0", 0)
                dim1 = b_producer.attrs.get("dim1", 1)
                if {dim0, dim1} == {ndim - 2, ndim - 1}:
                    original_b = b_producer.inputs[0]
                    graph.rewire_input(node.id, b_name, original_b)
                    node.attrs["transpose_b"] = True
                    graph.remove_node(b_producer.id)
                    graph.remove_tensor(b_name)
                    changed = True

        # --- Scalar absorption on inputs (backward) ---
        for i, inp_name in enumerate(list(node.inputs)):
            producer = graph.producer(inp_name)
            if (producer is not None
                    and producer.op in (OpType.MUL, OpType.DIV)
                    and "scalar" in producer.attrs
                    and len(graph.consumers(inp_name)) == 1):
                scalar = producer.attrs["scalar"]
                alpha = node.attrs.get("alpha", 1.0)
                if producer.op == OpType.MUL:
                    node.attrs["alpha"] = alpha * scalar
                else:
                    node.attrs["alpha"] = alpha / scalar
                # Rewire MATMUL to consume the MUL/DIV's input directly
                original = producer.inputs[0]
                graph.rewire_input(node.id, inp_name, original)
                graph.remove_node(producer.id)
                graph.remove_tensor(inp_name)
                changed = True

        # --- Scalar absorption on output (forward) ---
        consumers = graph.consumers(node.output)
        if (len(consumers) == 1
                and consumers[0].op in (OpType.MUL, OpType.DIV)
                and "scalar" in consumers[0].attrs):
            consumer = consumers[0]
            scalar = consumer.attrs["scalar"]
            alpha = node.attrs.get("alpha", 1.0)
            if consumer.op == OpType.MUL:
                node.attrs["alpha"] = alpha * scalar
            else:
                node.attrs["alpha"] = alpha / scalar
            # Rewire: consumers of the MUL/DIV now consume the MATMUL output
            for downstream in graph.consumers(consumer.output):
                graph.rewire_input(downstream.id, consumer.output, node.output)
            if consumer.output in graph.outputs:
                idx = graph.outputs.index(consumer.output)
                graph.outputs[idx] = node.output
            graph.remove_node(consumer.id)
            graph.remove_tensor(consumer.output)
            changed = True

    return changed


# ---------------------------------------------------------------------------
# Constant folding
# ---------------------------------------------------------------------------

# Numpy evaluators for constant folding. Maps OpType to a callable
# that takes (list[np.ndarray], dict[str, Any]) -> np.ndarray.
# Also reusable by the executor for a pure-Python fallback path.
NumpyEvaluator = Callable[[list[np.ndarray], dict[str, Any]], np.ndarray]
NUMPY_EVALUATORS: dict[OpType, NumpyEvaluator] = {}


def register_evaluator(op: OpType, fn: NumpyEvaluator) -> None:
    """Register a numpy evaluator for an op type."""
    NUMPY_EVALUATORS[op] = fn


register_evaluator(OpType.ADD, lambda ins, attrs: ins[0] + (attrs["scalar"] if "scalar" in attrs else ins[1]))
register_evaluator(OpType.RELU, lambda ins, attrs: np.maximum(ins[0], 0))
register_evaluator(OpType.MATMUL, lambda ins, attrs: (
    ins[0] @ (np.swapaxes(ins[1], -2, -1) if attrs.get("transpose_b") else ins[1]) * attrs.get("alpha", 1.0)
))
register_evaluator(OpType.TRANSPOSE, lambda ins, attrs: np.swapaxes(ins[0], attrs["dim0"], attrs["dim1"]))
register_evaluator(OpType.PERMUTE, lambda ins, attrs: np.transpose(ins[0], attrs["axes"]))
register_evaluator(OpType.DIV, lambda ins, attrs: ins[0] / (attrs["scalar"] if "scalar" in attrs else ins[1]))
register_evaluator(OpType.SUB, lambda ins, attrs: ins[0] - (attrs["scalar"] if "scalar" in attrs else ins[1]))
register_evaluator(OpType.MUL, lambda ins, attrs: ins[0] * (attrs["scalar"] if "scalar" in attrs else ins[1]))
register_evaluator(OpType.EXP, lambda ins, attrs: np.exp(ins[0]))
register_evaluator(OpType.MAX, lambda ins, attrs: np.max(ins[0], axis=attrs["axis"], keepdims=attrs.get("keepdim", False)))
register_evaluator(OpType.SUM, lambda ins, attrs: np.sum(ins[0], axis=attrs["axis"], keepdims=attrs.get("keepdim", False)))


def _eval_softmax(ins: list[np.ndarray], attrs: dict[str, Any]) -> np.ndarray:
    x = ins[0]
    axis = attrs["axis"]
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)

register_evaluator(OpType.SOFTMAX, _eval_softmax)
register_evaluator(OpType.RESHAPE, lambda ins, attrs: ins[0].reshape(attrs["shape"]))


def _eval_layernorm(ins: list[np.ndarray], attrs: dict[str, Any]) -> np.ndarray:
    x, gamma, beta = ins[0], ins[1], ins[2]
    eps = attrs.get("eps", 1e-5)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta

register_evaluator(OpType.LAYERNORM, _eval_layernorm)

register_evaluator(OpType.POW, lambda ins, attrs: np.power(ins[0], attrs["scalar"]))
register_evaluator(OpType.TANH, lambda ins, attrs: np.tanh(ins[0]))
register_evaluator(OpType.GELU, lambda ins, attrs: (
    0.5 * ins[0] * (1 + np.tanh(0.7978845608 * (ins[0] + 0.044715 * ins[0]**3)))
))

register_evaluator(OpType.MATMUL_ADD, lambda ins, attrs: (
    ins[0] @ (ins[1].T if attrs.get("transpose_b") else ins[1]) + ins[2]
))

register_evaluator(OpType.FUSED_BIAS_RELU, lambda ins, attrs: np.maximum(ins[0] + ins[1], 0))


def _eval_attention(ins: list[np.ndarray], attrs: dict[str, Any]) -> np.ndarray:
    Q, K, V = ins[0], ins[1], ins[2]
    scale = 1.0 / np.sqrt(Q.shape[-1])
    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) * scale
    if attrs.get("causal"):
        seq_len = Q.shape[-2]
        mask = np.triu(np.full((seq_len, seq_len), -np.inf, dtype=np.float32), k=1)
        scores = scores + mask
    scores -= np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores)
    weights = e / np.sum(e, axis=-1, keepdims=True)
    return np.matmul(weights, V)

register_evaluator(OpType.ATTENTION, _eval_attention)

# --- Mask/infrastructure op evaluators (constant-folded, never executed) ---

# CAST: convert dtype
register_evaluator(OpType.CAST, lambda ins, attrs: ins[0].astype(attrs["target_dtype"]))

# EXPAND: broadcast to target shape
register_evaluator(OpType.EXPAND, lambda ins, attrs: np.broadcast_to(ins[0], attrs["shape"]))

# SLICE_TENSOR: slice along a dimension
def _eval_slice_tensor(ins: list[np.ndarray], attrs: dict[str, Any]) -> np.ndarray:
    x = ins[0]
    dim = attrs["dim"]
    start = attrs.get("start", 0)
    end = attrs.get("end", x.shape[dim])
    slices = tuple(slice(start, end) if d == dim else slice(None) for d in range(x.ndim))
    return x[slices]
register_evaluator(OpType.SLICE_TENSOR, _eval_slice_tensor)

# DIFF: np.diff with optional prepend
def _eval_diff(ins: list[np.ndarray], attrs: dict[str, Any]) -> np.ndarray:
    x = ins[0]
    n = attrs.get("n", 1)
    dim = attrs.get("dim", -1)
    prepend = ins[1] if len(ins) > 1 else None
    return np.diff(x, n=n, axis=dim, prepend=prepend)
register_evaluator(OpType.DIFF, _eval_diff)

# CMP_NE: not-equal (scalar or tensor)
register_evaluator(OpType.CMP_NE, lambda ins, attrs: ins[0] != (attrs["scalar"] if "scalar" in attrs else ins[1]))

# CMP_LE: less-than-or-equal
register_evaluator(OpType.CMP_LE, lambda ins, attrs: ins[0] <= ins[1])

# CMP_EQ: equality
register_evaluator(OpType.CMP_EQ, lambda ins, attrs: ins[0] == ins[1])

# CUMSUM: cumulative sum along axis
register_evaluator(OpType.CUMSUM, lambda ins, attrs: np.cumsum(ins[0], axis=attrs["dim"]))

# BITWISE_AND: element-wise AND
register_evaluator(OpType.BITWISE_AND, lambda ins, attrs: ins[0] & ins[1])

# INDEX: advanced (fancy) indexing
def _eval_index(ins: list[np.ndarray], attrs: dict[str, Any]) -> np.ndarray:
    tensor = ins[0]
    n_indices = attrs.get("n_indices", len(ins) - 1)
    none_positions = set(attrs.get("none_positions", []))
    indices = []
    tensor_idx = 1
    for i in range(n_indices):
        if i in none_positions:
            indices.append(slice(None))
        else:
            indices.append(ins[tensor_idx])
            tensor_idx += 1
    return tensor[tuple(indices)]
register_evaluator(OpType.INDEX, _eval_index)


def constant_fold(graph: Graph) -> bool:
    """Evaluate nodes whose inputs are all constants.

    The result becomes a new constant tensor (buffer set to the computed
    numpy array) and the node is removed. If input constants become dead
    after folding, DCE will clean them up.
    """
    changed = False
    constant_set = set(graph.constants)

    for node in list(graph):  # topological order — ensures inputs are folded first
        # All inputs must be constants with loaded buffers
        if not all(inp in constant_set for inp in node.inputs):
            continue

        evaluator = NUMPY_EVALUATORS.get(node.op)
        if evaluator is None:
            continue

        input_buffers = [graph.tensors[inp].buffer for inp in node.inputs]
        result = evaluator(input_buffers, node.attrs)

        # Promote the output tensor to a constant.
        # Ensure contiguous layout — evaluators may return views (e.g. swapaxes)
        # and downstream C kernels need contiguous data.
        # Also enforce declared dtype — numpy may promote (e.g. int64 + 0.0 → float64)
        # which would break downstream ops expecting the original type.
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
# Mask absorption into attention
# ---------------------------------------------------------------------------

def absorb_mask_into_attention(graph: Graph) -> bool:
    """Replace ATTENTION(Q, K, V, constant_mask) with ATTENTION(Q, K, V, causal=True).

    After constant folding, attention mask tensors from HuggingFace-style models
    become constants. If the mask is a standard causal pattern (lower-triangular True
    for bool masks, or upper-triangular -inf for float masks), we can replace it with
    the causal flag and let the kernel compute the mask on the fly.

    This is a general optimization — it recognizes the *result* (a causal mask constant),
    not the specific ops that generated it.
    """
    changed = False
    constant_set = set(graph.constants)

    for node in list(graph):
        if node.op != OpType.ATTENTION:
            continue
        if len(node.inputs) != 4:
            continue  # Already has only Q, K, V — no mask to absorb
        if node.attrs.get("causal"):
            continue  # Already marked causal

        mask_name = node.inputs[3]
        if mask_name not in constant_set:
            continue  # Non-constant mask — can’t fold

        mask_info = graph.tensors[mask_name]
        if mask_info.buffer is None:
            continue

        buf = mask_info.buffer

        # Check if this is a causal mask
        if _is_bool_causal_mask(buf) or _is_float_causal_mask(buf):
            node.attrs["causal"] = True
            # Remove mask from inputs
            graph._consumers[mask_name].remove(node.id)
            node.inputs = node.inputs[:3]  # Keep only Q, K, V
            graph._order = None  # Invalidate cached order
            changed = True

    return changed


def _is_bool_causal_mask(buf: np.ndarray) -> bool:
    """Check if a bool array is a causal mask (lower-triangular True)."""
    if buf.dtype != np.bool_:
        return False
    if buf.ndim < 2 or buf.shape[-1] != buf.shape[-2]:
        return False
    S = buf.shape[-1]
    if S < 2:
        return True  # Trivially causal for seq_len=1
    # Check first 2D slice (broadcast means all slices are the same)
    mat = buf.reshape(-1, S, S)[0]
    expected = np.tril(np.ones((S, S), dtype=np.bool_))
    return np.array_equal(mat, expected)


def _is_float_causal_mask(buf: np.ndarray) -> bool:
    """Check if a float array is a causal mask (upper-triangular -inf, rest ~0)."""
    if not np.issubdtype(buf.dtype, np.floating):
        return False
    if buf.ndim < 2 or buf.shape[-1] != buf.shape[-2]:
        return False
    S = buf.shape[-1]
    if S < 2:
        return True
    mat = buf.reshape(-1, S, S)[0]
    lower = np.tril(mat)
    upper_strict = mat[np.triu_indices(S, k=1)]
    return np.allclose(lower, 0, atol=1e-6) and np.all(upper_strict < -1e4)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

@dataclass
class FusionPattern:
    """A pattern of ops that can be fused into a single op.

    Describes a linear chain of ops to match in the graph and what to
    replace them with. The validator callback allows additional checks
    beyond op type matching (e.g., verifying that an ADD is a bias add
    rather than an arbitrary element-wise add).

    Patterns are grouped by priority level — lower priority number means
    higher precedence. Within a level, longer patterns are tried first
    (greedy matching). This lets you control which fusions win when
    patterns compete for the same node (e.g., ADD+RELU vs MATMUL+ADD).
    """
    name: str
    # Op types to match, in order from first to last in the chain
    pattern: list[OpType]
    # The fused op type to replace the chain with
    fused_op: OpType
    # Lower = higher precedence. Patterns at priority 0 get first pick of nodes.
    priority: int = 0
    # Optional validation: given the matched chain of nodes, return True
    # if this fusion is valid. Use for structural checks beyond op types.
    validator: Callable[[list[Node], Graph], bool] | None = None
    # Optional attr builder: given matched nodes, produce attrs for the
    # fused node. Use for carrying forward flags like transpose dims.
    build_attrs: Callable[[list[Node], Graph], dict] | None = None
    # Optional input builder: given matched nodes, graph, and the default
    # external inputs, return the actual inputs for the fused node. Use
    # when fusion needs to drop inputs (e.g., removing a mask constant
    # when replacing explicit masking with a kernel flag).
    build_inputs: Callable[[list[Node], Graph, list[str]], list[str]] | None = None


# Registry of fusion patterns
FUSION_PATTERNS: list[FusionPattern] = []


def register_fusion(pattern: FusionPattern) -> None:
    """Add a fusion pattern to the registry."""
    FUSION_PATTERNS.append(pattern)


def fuse(graph: Graph, patterns: list[FusionPattern] | None = None) -> bool:
    """Apply fusion patterns from the registry to the graph.

    Patterns are grouped by priority level (lower = first). Within each
    level, the graph is walked in topological order and patterns are
    tried longest-first (greedy). Each priority level gets a full sweep
    before the next level runs, so high-priority patterns claim nodes
    before lower-priority ones see them.

    Accepts an optional patterns list for testing alternative orderings.
    """
    all_patterns = patterns if patterns is not None else FUSION_PATTERNS
    if not all_patterns:
        return False

    changed = False
    sorted_patterns = sorted(all_patterns, key=lambda p: p.priority)

    for _priority, group in groupby(sorted_patterns, key=lambda p: p.priority):
        # Within this priority level: sort by pattern length descending (greedy)
        level_patterns = sorted(group, key=lambda p: len(p.pattern), reverse=True)

        fused_ids: set[int] = set()
        for node in list(graph):
            if node.id in fused_ids:
                continue
            for pattern in level_patterns:
                chain = _try_match(node, pattern, graph, fused_ids)
                if chain is not None:
                    _apply_fusion(chain, pattern, graph)
                    fused_ids.update(n.id for n in chain)
                    changed = True
                    break

    return changed


def _try_match(
    start: Node, pattern: FusionPattern, graph: Graph, fused_ids: set[int]
) -> list[Node] | None:
    """Try to match a fusion pattern as a linear chain starting at `start`."""
    if start.op != pattern.pattern[0]:
        return None

    chain = [start]
    current = start

    for next_op in pattern.pattern[1:]:
        consumers = graph.consumers(current.output)
        # Sole consumer constraint: can't fuse if the intermediate is used elsewhere
        if len(consumers) != 1:
            return None
        next_node = consumers[0]
        if next_node.id in fused_ids or next_node.op != next_op:
            return None
        chain.append(next_node)
        current = next_node

    if pattern.validator and not pattern.validator(chain, graph):
        return None

    return chain


def _apply_fusion(chain: list[Node], pattern: FusionPattern, graph: Graph) -> None:
    """Replace a matched chain with a single fused node."""
    last = chain[-1]

    # Collect inputs from outside the chain
    chain_outputs = {n.output for n in chain}
    external_inputs = []
    for node in chain:
        for inp in node.inputs:
            if inp not in chain_outputs and inp not in external_inputs:
                external_inputs.append(inp)

    attrs = pattern.build_attrs(chain, graph) if pattern.build_attrs else {}

    # Let the pattern customize which inputs the fused node receives.
    # Default: all external inputs. Patterns can drop inputs (e.g., a mask
    # constant that's replaced by a kernel flag).
    if pattern.build_inputs:
        external_inputs = pattern.build_inputs(chain, graph, external_inputs)

    # Remove original nodes and intermediate tensors FIRST —
    # remove_node pops _producer[output], so if we added the fused node
    # first, removing the last chain node would clobber its producer entry.
    for node in chain:
        graph.remove_node(node.id)
        if node.output != last.output:
            graph.remove_tensor(node.output)

    # Now create the fused node (safe — old producer entries are cleared)
    graph.add_node(pattern.fused_op, external_inputs, last.output, attrs)


# ---------------------------------------------------------------------------
# Registered fusion patterns
# ---------------------------------------------------------------------------

# Priority 0: activation fusions (claim ADD+RELU before MATMUL+ADD can)

def _validate_bias_relu(chain: list[Node], graph: Graph) -> bool:
    """ADD must be a bias broadcast: second input is 1D matching last dim."""
    add_node = chain[0]
    a_shape = graph.tensors[add_node.inputs[0]].shape
    b_shape = graph.tensors[add_node.inputs[1]].shape
    return len(b_shape) == 1 and b_shape[0] == a_shape[-1]

register_fusion(FusionPattern(
    name="bias_relu",
    pattern=[OpType.ADD, OpType.RELU],
    fused_op=OpType.FUSED_BIAS_RELU,
    priority=0,
    validator=_validate_bias_relu,
))

# Priority 1: MATMUL+ADD (catches standalone bias adds not claimed by bias_relu)

def _validate_matmul_add(chain: list[Node], graph: Graph) -> bool:
    """The ADD's non-chain input must be a 1D bias vector."""
    matmul_node, add_node = chain
    for inp in add_node.inputs:
        if inp != matmul_node.output:
            return len(graph.tensors[inp].shape) == 1
    return False

def _build_matmul_add_attrs(chain: list[Node], graph: Graph) -> dict:
    """Carry forward MATMUL attrs (transpose_b, etc.)."""
    return dict(chain[0].attrs)

register_fusion(FusionPattern(
    name="matmul_add",
    pattern=[OpType.MATMUL, OpType.ADD],
    fused_op=OpType.MATMUL_ADD,
    priority=1,
    validator=_validate_matmul_add,
    build_attrs=_build_matmul_add_attrs,
))


# Attention fusion: MATMUL(Q @ K^T) → SOFTMAX → MATMUL(weights @ V)
# The scalar scale (1/sqrt(d_k)) is already absorbed into the first MATMUL's alpha
# by absorb_into_matmul, so only 3 nodes remain. External inputs collected by
# _apply_fusion are [Q, K, V] — exactly what the ATTENTION kernel expects.

def _validate_attention(chain: list[Node], graph: Graph) -> bool:
    """Verify the MATMUL→SOFTMAX→MATMUL chain is really attention."""
    qk_matmul, softmax, wv_matmul = chain
    # First matmul should be Q @ K^T (transpose_b absorbed from earlier pass)
    if not qk_matmul.attrs.get("transpose_b"):
        return False
    # Softmax should be along last axis
    if softmax.attrs.get("axis") != len(graph.tensors[softmax.output].shape) - 1:
        return False
    return True

register_fusion(FusionPattern(
    name="attention",
    pattern=[OpType.MATMUL, OpType.SOFTMAX, OpType.MATMUL],
    fused_op=OpType.ATTENTION,
    priority=0,
    validator=_validate_attention,
))


# Causal attention fusion: MATMUL(Q @ K^T) → ADD(mask) → SOFTMAX → MATMUL(weights @ V)
# Models that apply an explicit causal mask before softmax (rather than using SDPA's
# is_causal flag) produce this 4-node pattern. The ADD's second input is a constant
# upper-triangular -inf mask. We validate the mask shape and values, drop it from the
# fused node's inputs via build_inputs, and set causal=True so the C kernel applies
# the mask computationally (a branch is cheaper than reading an S×S tensor).

def _is_causal_mask(tensor_name: str, graph: Graph) -> bool:
    """Check if a tensor is a causal (upper-triangular -inf) attention mask."""
    info = graph.tensors[tensor_name]
    if tensor_name not in graph.constants or info.buffer is None:
        return False
    buf = info.buffer
    # Shape should be broadcastable to [B, H, S, S] — typically [1, 1, S, S] or [S, S]
    if buf.ndim < 2 or buf.shape[-1] != buf.shape[-2]:
        return False
    # Check the last two dims: diagonal and below should be ~0, above should be -inf
    mat = buf.reshape(-1, buf.shape[-2], buf.shape[-1])[0]  # take first slice
    S = mat.shape[0]
    if S < 2:
        return True  # trivially causal for seq_len=1
    lower = np.tril(mat)
    upper_strict = mat[np.triu_indices(S, k=1)]
    return np.allclose(lower, 0, atol=1e-6) and np.all(upper_strict < -1e4)


def _validate_causal_attention(chain: list[Node], graph: Graph) -> bool:
    """Verify the MATMUL→ADD(mask)→SOFTMAX→MATMUL chain is causal attention."""
    qk_matmul, add_mask, softmax, wv_matmul = chain
    if not qk_matmul.attrs.get("transpose_b"):
        return False
    if softmax.attrs.get("axis") != len(graph.tensors[softmax.output].shape) - 1:
        return False
    # The ADD's non-chain input should be a causal mask constant
    mask_input = None
    for inp in add_mask.inputs:
        if inp != qk_matmul.output:
            mask_input = inp
            break
    if mask_input is None:
        return False
    return _is_causal_mask(mask_input, graph)


def _build_causal_attention_inputs(
    chain: list[Node], graph: Graph, external_inputs: list[str]
) -> list[str]:
    """Drop the mask tensor — the kernel computes causal masking internally."""
    qk_matmul, add_mask, softmax, wv_matmul = chain
    # Find the mask input (the ADD's non-chain input)
    mask_input = None
    for inp in add_mask.inputs:
        if inp != qk_matmul.output:
            mask_input = inp
            break
    return [inp for inp in external_inputs if inp != mask_input]


register_fusion(FusionPattern(
    name="causal_attention",
    pattern=[OpType.MATMUL, OpType.ADD, OpType.SOFTMAX, OpType.MATMUL],
    fused_op=OpType.ATTENTION,
    priority=0,
    validator=_validate_causal_attention,
    build_attrs=lambda chain, graph: {"causal": True},
    build_inputs=_build_causal_attention_inputs,
))


# ---------------------------------------------------------------------------
# GELU recognition
# ---------------------------------------------------------------------------

def _approx_eq(val: float, target: float, tol: float = 1e-4) -> bool:
    """Check if a scalar value approximately equals the target."""
    return abs(val - target) < tol


def recognize_gelu(graph: Graph) -> bool:
    """Recognize the GELU tanh approximation pattern and replace with a single GELU node.

    The pattern (from GPT-2):
      x → pow(3) → mul(0.044715) → add(x) → mul(sqrt(2/pi)) → tanh → add(1) → mul(half_x)
    where half_x = mul(x, 0.5)

    Finds TANH nodes and walks backward/forward to verify the full pattern,
    then replaces the 8-node subgraph with a single GELU(x) → final_output.
    """
    changed = False

    for node in list(graph):
        if node.op != OpType.TANH:
            continue

        # Walk backward from TANH to find the pattern
        # TANH input should be MUL with scalar ≈ sqrt(2/pi) = 0.7978845608
        tanh_input_producer = graph.producer(node.inputs[0])
        if (tanh_input_producer is None
                or tanh_input_producer.op != OpType.MUL
                or "scalar" not in tanh_input_producer.attrs
                or not _approx_eq(tanh_input_producer.attrs["scalar"], 0.7978845608)):
            continue
        mul_sqrt2pi = tanh_input_producer

        # MUL(sqrt2pi) input should be ADD(x + 0.044715*x^3)
        add_inner = graph.producer(mul_sqrt2pi.inputs[0])
        if add_inner is None or add_inner.op != OpType.ADD:
            continue
        # ADD should have 2 tensor inputs (x and 0.044715*x^3)
        if len(add_inner.inputs) != 2 or "scalar" in add_inner.attrs:
            continue

        # One input to ADD is x, the other is MUL(0.044715) → POW(3)
        # Try both orderings
        root_x = None
        mul_coeff_node = None
        for i in range(2):
            candidate_mul = graph.producer(add_inner.inputs[i])
            candidate_x = add_inner.inputs[1 - i]
            if (candidate_mul is not None
                    and candidate_mul.op == OpType.MUL
                    and "scalar" in candidate_mul.attrs
                    and _approx_eq(candidate_mul.attrs["scalar"], 0.044715)):
                # Verify MUL(0.044715) input is POW(3) of x
                pow_node = graph.producer(candidate_mul.inputs[0])
                if (pow_node is not None
                        and pow_node.op == OpType.POW
                        and _approx_eq(pow_node.attrs.get("scalar", 0), 3.0)
                        and pow_node.inputs[0] == candidate_x):
                    root_x = candidate_x
                    mul_coeff_node = candidate_mul
                    break

        if root_x is None:
            continue
        pow_node = graph.producer(mul_coeff_node.inputs[0])

        # Walk forward from TANH: TANH → ADD(1.0) → MUL(half_x)
        tanh_consumers = graph.consumers(node.output)
        if len(tanh_consumers) != 1:
            continue
        add_one = tanh_consumers[0]
        if (add_one.op != OpType.ADD
                or "scalar" not in add_one.attrs
                or not _approx_eq(add_one.attrs["scalar"], 1.0)):
            continue

        add_one_consumers = graph.consumers(add_one.output)
        if len(add_one_consumers) != 1:
            continue
        final_mul = add_one_consumers[0]
        if final_mul.op != OpType.MUL:
            continue

        # final_mul should multiply (tanh+1) by half_x = x * 0.5
        # Find which input is the add_one output and which is half_x
        other_input = None
        for inp in final_mul.inputs:
            if inp != add_one.output:
                other_input = inp
                break
        if other_input is None:
            continue

        half_x_producer = graph.producer(other_input)
        if (half_x_producer is None
                or half_x_producer.op != OpType.MUL
                or "scalar" not in half_x_producer.attrs
                or not _approx_eq(half_x_producer.attrs["scalar"], 0.5)
                or half_x_producer.inputs[0] != root_x):
            continue

        # Pattern matched! Replace all 8 nodes with GELU(x) → final_output
        subgraph_nodes = [
            pow_node, mul_coeff_node, add_inner, mul_sqrt2pi,
            node, add_one, half_x_producer, final_mul,
        ]
        final_output = final_mul.output

        # Collect intermediate tensor names to remove (not root_x, not final_output)
        chain_outputs = {n.output for n in subgraph_nodes}

        # Remove all subgraph nodes
        for n in subgraph_nodes:
            graph.remove_node(n.id)
            if n.output != final_output:
                graph.remove_tensor(n.output)

        # Add GELU node: input is root_x, output is final_output
        graph.add_node(OpType.GELU, [root_x], final_output)
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

DEFAULT_PIPELINE.extend([
    absorb_into_matmul,
    constant_fold,
    absorb_mask_into_attention,
    recognize_gelu,
    fuse,
    eliminate_dead_code,
])
