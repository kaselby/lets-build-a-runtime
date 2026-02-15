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
# Registered fusion patterns will be added here as we review them
# ---------------------------------------------------------------------------
