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

register_evaluator(OpType.MATMUL_ADD, lambda ins, attrs: (
    ins[0] @ (ins[1].T if attrs.get("transpose_b") else ins[1]) + ins[2]
))

register_evaluator(OpType.FUSED_BIAS_RELU, lambda ins, attrs: np.maximum(ins[0] + ins[1], 0))


def _eval_attention(ins: list[np.ndarray], attrs: dict[str, Any]) -> np.ndarray:
    Q, K, V = ins[0], ins[1], ins[2]
    scale = 1.0 / np.sqrt(Q.shape[-1])
    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) * scale
    scores -= np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores)
    weights = e / np.sum(e, axis=-1, keepdims=True)
    return np.matmul(weights, V)

register_evaluator(OpType.ATTENTION, _eval_attention)


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
        graph.tensors[node.output].buffer = np.ascontiguousarray(result)
        graph.constants.append(node.output)
        constant_set.add(node.output)

        graph.remove_node(node.id)
        changed = True

    return changed


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
    fuse,
    eliminate_dead_code,
])
