"""Fusion pass: pattern-based graph rewriting.

The fusion pass uses a registry of FusionPattern objects that describe
what to match and what to replace it with. Patterns have priority levels
— lower priority number = matched first. Within a priority level,
longer patterns are tried first (greedy matching).
"""

from dataclasses import dataclass
from itertools import groupby
from typing import Any, Callable

from ..ir import Graph, Node, OpType


# ---------------------------------------------------------------------------
# Fusion pattern definition
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


# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------

FUSION_PATTERNS: list[FusionPattern] = []


def register_fusion(pattern: FusionPattern) -> None:
    """Add a fusion pattern to the registry."""
    FUSION_PATTERNS.append(pattern)


# ---------------------------------------------------------------------------
# Matching and rewriting engine
# ---------------------------------------------------------------------------

def fuse(graph: Graph, patterns: list[FusionPattern] | None = None) -> bool:
    """Apply fusion patterns from the registry to the graph.

    Patterns are grouped by priority level (lower = first). Within each
    level, the graph is walked in topological order and patterns are
    tried longest-first (greedy). Each priority level gets a full sweep
    before the next level runs, so high-priority patterns claim nodes
    before lower-priority ones see them.
    """
    all_patterns = patterns if patterns is not None else FUSION_PATTERNS
    if not all_patterns:
        return False

    changed = False
    sorted_patterns = sorted(all_patterns, key=lambda p: p.priority)

    for _priority, group in groupby(sorted_patterns, key=lambda p: p.priority):
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

    if pattern.build_inputs:
        external_inputs = pattern.build_inputs(chain, graph, external_inputs)

    # Remove original nodes and intermediate tensors FIRST —
    # remove_node pops _producer[output], so if we added the fused node
    # first, removing the last chain node would clobber its producer entry.
    for node in chain:
        graph.remove_node(node.id)
        if node.output != last.output:
            graph.remove_tensor(node.output)

    graph.add_node(pattern.fused_op, external_inputs, last.output, attrs)


# ---------------------------------------------------------------------------
# Registered fusion patterns
# ---------------------------------------------------------------------------

import numpy as np
from .passes import _is_causal_mask

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

# Priority 0: gated activation fusions (claim ADD before MATMUL+ADD can)

def _validate_bias_gated(chain: list[Node], graph: Graph) -> bool:
    """ADD+act+MUL: ADD must be a bias broadcast, MUL must be element-wise."""
    add_node = chain[0]
    a_shape = graph.tensors[add_node.inputs[0]].shape
    b_shape = graph.tensors[add_node.inputs[1]].shape
    return len(b_shape) == 1 and b_shape[0] == a_shape[-1]

def _build_bias_gated_attrs(act: str):
    def builder(chain: list[Node], graph: Graph) -> dict:
        return {"act": act, "has_bias": True}
    return builder

def _build_bias_gated_inputs(chain: list[Node], graph: Graph, external_inputs: list[str]) -> list[str]:
    """Reorder inputs to [x, bias, up] — x is ADD's first input, bias is second, up is MUL's other input."""
    add_node, _act_node, mul_node = chain
    x = add_node.inputs[0]
    bias = add_node.inputs[1]
    up = [inp for inp in mul_node.inputs if inp not in {_act_node.output}][0]
    return [x, bias, up]

register_fusion(FusionPattern(
    name="bias_silu_mul",
    pattern=[OpType.ADD, OpType.SILU, OpType.MUL],
    fused_op=OpType.GATED_ACT,
    priority=0,
    validator=_validate_bias_gated,
    build_attrs=_build_bias_gated_attrs("silu"),
    build_inputs=_build_bias_gated_inputs,
))

register_fusion(FusionPattern(
    name="bias_gelu_mul",
    pattern=[OpType.ADD, OpType.GELU, OpType.MUL],
    fused_op=OpType.GATED_ACT,
    priority=0,
    validator=_validate_bias_gated,
    build_attrs=_build_bias_gated_attrs("gelu"),
    build_inputs=_build_bias_gated_inputs,
))

# Priority 1: bias-free gated activations (after priority 0 claims biased variants)

def _build_gated_attrs(act: str):
    def builder(chain: list[Node], graph: Graph) -> dict:
        return {"act": act, "has_bias": False}
    return builder

def _build_gated_inputs(chain: list[Node], graph: Graph, external_inputs: list[str]) -> list[str]:
    """Reorder inputs to [x, up] — x is the activation's input, up is MUL's other input."""
    act_node, mul_node = chain
    x = act_node.inputs[0]
    up = [inp for inp in mul_node.inputs if inp not in {act_node.output}][0]
    return [x, up]

register_fusion(FusionPattern(
    name="silu_mul",
    pattern=[OpType.SILU, OpType.MUL],
    fused_op=OpType.GATED_ACT,
    priority=1,
    build_attrs=_build_gated_attrs("silu"),
    build_inputs=_build_gated_inputs,
))

register_fusion(FusionPattern(
    name="gelu_mul",
    pattern=[OpType.GELU, OpType.MUL],
    fused_op=OpType.GATED_ACT,
    priority=1,
    build_attrs=_build_gated_attrs("gelu"),
    build_inputs=_build_gated_inputs,
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

def _validate_attention(chain: list[Node], graph: Graph) -> bool:
    """Verify the MATMUL→SOFTMAX→MATMUL chain is really attention."""
    qk_matmul, softmax, wv_matmul = chain
    if not qk_matmul.attrs.get("transpose_b"):
        return False
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
    info = graph.tensors[mask_input]
    if mask_input not in graph.constants or info.buffer is None:
        return False
    return _is_causal_mask(info.buffer)


def _build_causal_attention_inputs(
    chain: list[Node], graph: Graph, external_inputs: list[str]
) -> list[str]:
    """Drop the mask tensor — the kernel computes causal masking internally."""
    qk_matmul, add_mask, softmax, wv_matmul = chain
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


# GQA absorption: unsqueeze→expand→reshape feeding into ATTENTION's K or V
#
# GQA models replicate KV heads via RESHAPE(unsqueeze) → EXPAND → RESHAPE(flatten)
# before ATTENTION. This absorbs the chain into ATTENTION's group_size parameter,
# so the kernel strides K/V by bh/group_size instead of materializing the expansion.
#
# Fires twice (once for K, once for V) — the second pass picks up the new
# ATTENTION node created by the first.

def _validate_gqa(chain: list[Node], graph: Graph) -> bool:
    """Verify the RESHAPE→EXPAND→RESHAPE→ATTENTION chain is GQA head expansion."""
    reshape1, expand, reshape2, attn = chain
    expand_shape = expand.attrs.get("shape")
    original_shape = graph.tensors[reshape1.inputs[0]].shape
    if expand_shape is None or len(expand_shape) != len(original_shape) + 1:
        return False
    # The inserted dim (position 2) should be > 1 — that's the group_size
    group_size = expand_shape[2]
    if not isinstance(group_size, int) or group_size <= 1:
        return False
    # The chain output must feed into ATTENTION's K (input[1]) or V (input[2])
    chain_tensor = reshape2.output
    return chain_tensor in (attn.inputs[1], attn.inputs[2])


def _build_gqa_attrs(chain: list[Node], graph: Graph) -> dict:
    """Copy ATTENTION's attrs and add/update group_size from EXPAND's shape."""
    expand = chain[1]
    attn = chain[-1]
    attrs = dict(attn.attrs)
    attrs["group_size"] = expand.attrs["shape"][2]
    return attrs


def _build_gqa_inputs(
    chain: list[Node], graph: Graph, external_inputs: list[str]
) -> list[str]:
    """Replace the chain's connection to ATTENTION with the original tensor."""
    attn = chain[-1]
    chain_outputs = {n.output for n in chain[:-1]}
    original = chain[0].inputs[0]
    return [original if inp in chain_outputs else inp for inp in attn.inputs]


register_fusion(FusionPattern(
    name="gqa_attention",
    pattern=[OpType.RESHAPE, OpType.EXPAND, OpType.RESHAPE, OpType.ATTENTION],
    fused_op=OpType.ATTENTION,
    priority=0,
    validator=_validate_gqa,
    build_attrs=_build_gqa_attrs,
    build_inputs=_build_gqa_inputs,
))


# ---------------------------------------------------------------------------
# DAG fusion: branching/reconverging subgraphs (GELU, SiLU, etc.)
#
# Chain patterns match a linear sequence of ops. DAG patterns match a small
# directed acyclic subgraph where an input fans out to multiple paths that
# reconverge. The classic example is GELU's tanh approximation, where x
# feeds into three separate branches.
#
# General subgraph isomorphism is NP-complete, but these patterns are tiny
# (5-10 nodes) and anchored at a known root — matching is O(graph × pattern).
# ---------------------------------------------------------------------------

@dataclass
class DAGNode:
    """One node in a DAG fusion pattern.

    Inputs are string references: names of other pattern nodes for internal
    edges, or "x0", "x1", ... for external inputs from outside the pattern.
    Same external name = must be the same tensor in the graph.
    """
    op: OpType
    inputs: list[str]
    scalar: float | None = None     # if set, node.attrs["scalar"] must match
    commutative: bool = False       # try both input orderings when matching


@dataclass
class DAGFusionPattern:
    """A small DAG of ops that can be fused into a single op."""
    name: str
    nodes: dict[str, DAGNode]       # named pattern nodes
    root: str                       # which node's output becomes the fused output
    fused_op: OpType
    external_inputs: list[str]      # ordered external names → fused node's inputs
    priority: int = 0
    build_attrs: Callable[[dict[str, Node], Graph], dict] | None = None


DAG_FUSION_PATTERNS: list[DAGFusionPattern] = []


def register_dag_fusion(pattern: DAGFusionPattern) -> None:
    """Add a DAG fusion pattern to the registry."""
    DAG_FUSION_PATTERNS.append(pattern)


# ---------------------------------------------------------------------------
# DAG matching engine
# ---------------------------------------------------------------------------

_APPROX_TOL = 1e-4


def _approx_eq(a: float, b: float) -> bool:
    return abs(a - b) < _APPROX_TOL


def _match_dag_node(
    name: str, gnode: Node, pattern: DAGFusionPattern, graph: Graph,
    bindings: dict[str, Node], externals: dict[str, str],
    fused_ids: set[int],
) -> bool:
    """Recursively match pattern node `name` against graph node `gnode`.

    Binds pattern names to graph nodes in `bindings` and external refs to
    tensor names in `externals`. Uses snapshot/restore for backtracking
    when commutative ops need to try both input orderings.
    """
    if name in bindings:
        return bindings[name].id == gnode.id

    pnode = pattern.nodes[name]
    if gnode.op != pnode.op or gnode.id in fused_ids:
        return False
    if pnode.scalar is not None:
        actual = gnode.attrs.get("scalar")
        if actual is None or not _approx_eq(actual, pnode.scalar):
            return False
    if len(pnode.inputs) != len(gnode.inputs):
        return False

    # Try each valid input ordering (swap for commutative binary ops)
    orderings = [pnode.inputs]
    if pnode.commutative and len(pnode.inputs) == 2:
        orderings.append([pnode.inputs[1], pnode.inputs[0]])

    for ordering in orderings:
        snap_b, snap_e = dict(bindings), dict(externals)
        bindings[name] = gnode
        ok = True

        for ref, tensor in zip(ordering, gnode.inputs):
            if ref.startswith("x"):  # external input
                if ref in externals and externals[ref] != tensor:
                    ok = False
                    break
                externals[ref] = tensor
            else:  # internal — ref names another pattern node
                producer = graph.producer(tensor)
                if producer is None:
                    ok = False
                    break
                if not _match_dag_node(
                    ref, producer, pattern, graph, bindings, externals, fused_ids
                ):
                    ok = False
                    break

        if ok:
            return True

        # Restore state on failure
        bindings.clear()
        bindings.update(snap_b)
        externals.clear()
        externals.update(snap_e)

    return False


def _try_match_dag(
    root_node: Node, pattern: DAGFusionPattern, graph: Graph,
    fused_ids: set[int],
) -> tuple[dict[str, Node], dict[str, str]] | None:
    """Try matching a DAG pattern anchored at root_node."""
    bindings: dict[str, Node] = {}
    externals: dict[str, str] = {}

    if not _match_dag_node(
        pattern.root, root_node, pattern, graph, bindings, externals, fused_ids
    ):
        return None

    # Verify internal nodes have no consumers outside the pattern.
    # Count how many times each internal node is referenced as an input
    # within the pattern — the graph consumer count must match exactly.
    internal_refs: dict[str, int] = {}
    for pnode in pattern.nodes.values():
        for ref in pnode.inputs:
            if not ref.startswith("x"):
                internal_refs[ref] = internal_refs.get(ref, 0) + 1

    for ref_name, expected in internal_refs.items():
        if len(graph.consumers(bindings[ref_name].output)) != expected:
            return None

    return bindings, externals


def _apply_dag_fusion(
    bindings: dict[str, Node], externals: dict[str, str],
    pattern: DAGFusionPattern, graph: Graph,
) -> None:
    """Replace a matched DAG subgraph with a single fused node."""
    root_output = bindings[pattern.root].output
    inputs = [externals[name] for name in pattern.external_inputs]
    attrs = pattern.build_attrs(bindings, graph) if pattern.build_attrs else {}

    for gnode in bindings.values():
        graph.remove_node(gnode.id)
        if gnode.output != root_output:
            graph.remove_tensor(gnode.output)

    graph.add_node(pattern.fused_op, inputs, root_output, attrs)


def fuse_dags(graph: Graph, patterns: list[DAGFusionPattern] | None = None) -> bool:
    """Apply DAG fusion patterns to the graph.

    Same priority/greedy structure as chain fusion: patterns grouped by
    priority level, larger patterns tried first within each level.
    """
    all_patterns = patterns if patterns is not None else DAG_FUSION_PATTERNS
    if not all_patterns:
        return False

    changed = False
    sorted_patterns = sorted(all_patterns, key=lambda p: p.priority)

    for _priority, group in groupby(sorted_patterns, key=lambda p: p.priority):
        level_patterns = sorted(group, key=lambda p: len(p.nodes), reverse=True)
        fused_ids: set[int] = set()

        for node in list(graph):
            if node.id in fused_ids:
                continue
            for pattern in level_patterns:
                if node.op != pattern.nodes[pattern.root].op:
                    continue
                result = _try_match_dag(node, pattern, graph, fused_ids)
                if result is not None:
                    bindings, externals = result
                    _apply_dag_fusion(bindings, externals, pattern, graph)
                    fused_ids.update(n.id for n in bindings.values())
                    changed = True
                    break

    return changed


# ---------------------------------------------------------------------------
# Registered DAG patterns
# ---------------------------------------------------------------------------

# GELU tanh approximation (GPT-2):
#   x → pow(3) → mul(0.044715) → add(x,·) → mul(√(2/π)) → tanh
#                                                             ↓
#                         x → mul(0.5) → mul(·, add(tanh, 1))
#
# 8 nodes, x fans out to pow, add, and mul(0.5).

register_dag_fusion(DAGFusionPattern(
    name="gelu_tanh",
    nodes={
        "pow":    DAGNode(OpType.POW,  ["x0"],              scalar=3.0),
        "mul_c":  DAGNode(OpType.MUL,  ["pow"],             scalar=0.044715),
        "add_x":  DAGNode(OpType.ADD,  ["x0", "mul_c"],     commutative=True),
        "mul_s":  DAGNode(OpType.MUL,  ["add_x"],           scalar=0.7978845608),
        "tanh":   DAGNode(OpType.TANH, ["mul_s"]),
        "add_1":  DAGNode(OpType.ADD,  ["tanh"],            scalar=1.0),
        "half_x": DAGNode(OpType.MUL,  ["x0"],             scalar=0.5),
        "out":    DAGNode(OpType.MUL,  ["add_1", "half_x"], commutative=True),
    },
    root="out",
    fused_op=OpType.GELU,
    external_inputs=["x0"],
))


# SiLU (x * sigmoid(x) = x / (1 + exp(-x))):
#   x → NEG → EXP → ADD(scalar=1.0) → DIV
#   x ──────────────────────────────→ DIV → out
#
# 4 nodes, x0 fans out to NEG and DIV.

register_dag_fusion(DAGFusionPattern(
    name="silu",
    nodes={
        "neg":   DAGNode(OpType.NEG, ["x0"]),
        "exp":   DAGNode(OpType.EXP, ["neg"]),
        "denom": DAGNode(OpType.ADD, ["exp"], scalar=1.0),
        "out":   DAGNode(OpType.DIV, ["x0", "denom"]),
    },
    root="out",
    fused_op=OpType.SILU,
    external_inputs=["x0"],
))


# RMSNorm: x / sqrt(mean(x^2) + eps) * weight
#
# After exporter decomposes mean to SUM+DIV:
#   x → POW(2) → SUM(keepdim) → DIV(scalar=N) → ADD(scalar=eps) → RSQRT
#                                                                     ↓
#                                                  x → MUL(x, rsqrt) → MUL(·, weight)
#
# 7 nodes, x0 fans out to POW and normalize MUL.
# DIV and ADD have varying scalars (hidden_dim, eps) — matched by
# structure only (scalar=None). Scalar-vs-tensor is distinguished by
# input count: scalar variants have 1 input, tensor variants have 2.

def _rmsnorm_build_attrs(bindings: dict[str, 'Node'], graph: 'Graph') -> dict:
    """Extract eps from the ADD node."""
    return {"eps": bindings["add_eps"].attrs["scalar"]}

register_dag_fusion(DAGFusionPattern(
    name="rmsnorm",
    nodes={
        "pow":     DAGNode(OpType.POW,   ["x0"],              scalar=2.0),
        "sum":     DAGNode(OpType.SUM,   ["pow"]),
        "div":     DAGNode(OpType.DIV,   ["sum"]),             # scalar=hidden_dim
        "add_eps": DAGNode(OpType.ADD,   ["div"]),             # scalar=eps
        "rsqrt":   DAGNode(OpType.RSQRT, ["add_eps"]),
        "norm":    DAGNode(OpType.MUL,   ["x0", "rsqrt"],     commutative=True),
        "out":     DAGNode(OpType.MUL,   ["norm", "x1"],      commutative=True),
    },
    root="out",
    fused_op=OpType.RMSNORM,
    external_inputs=["x0", "x1"],  # x0=input, x1=weight
    build_attrs=_rmsnorm_build_attrs,
))
