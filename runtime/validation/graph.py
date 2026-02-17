"""Graph-level validators.

Covers POST_EXPORT, POST_OPTIMIZE, POST_RESOLVE, and
POST_RESOLVE_OPTIMIZE phases. All validators receive a Graph.
"""

from ..ir import FOLD_ONLY_BASE, Graph, OpType
from ..ops import OP_REGISTRY
from .core import Phase, Severity, ValidationResult, register_validator


# ---------------------------------------------------------------------------
# Structural integrity (shared by POST_EXPORT and POST_OPTIMIZE)
# ---------------------------------------------------------------------------

def _check_structure(graph: Graph, validator_name: str) -> list[ValidationResult]:
    """Verify internal consistency of graph topology and tensor registry.

    Checks that every reference resolves, every role assignment is valid,
    and the graph is acyclic. This is the migrated logic from the old
    Graph.validate() method.
    """
    results = []
    r = lambda sev, msg: results.append(ValidationResult(validator_name, sev, msg))

    # --- Tensor registry consistency ---
    for node in graph.nodes.values():
        for t in node.inputs:
            if t not in graph.tensors:
                r(Severity.ERROR,
                  f"Node {node.id} ({node.op.name}) references unknown input tensor '{t}'")
        if node.output not in graph.tensors:
            r(Severity.ERROR,
              f"Node {node.id} ({node.op.name}) output tensor '{node.output}' not in registry")

    # --- Role tensors exist ---
    for role, names in [("input", graph.inputs), ("constant", graph.constants),
                        ("output", graph.outputs)]:
        for name in names:
            if name not in graph.tensors:
                r(Severity.ERROR, f"Graph {role} '{name}' not in tensor registry")

    # --- Inputs and constants must not have producer nodes ---
    for name in graph.inputs:
        if graph.producer(name) is not None:
            r(Severity.ERROR,
              f"Graph input '{name}' has a producer node (should be external)")
    for name in graph.constants:
        if graph.producer(name) is not None:
            r(Severity.ERROR,
              f"Graph constant '{name}' has a producer node (should be external)")

    # --- Outputs must have producer nodes ---
    for name in graph.outputs:
        if graph.producer(name) is None:
            r(Severity.ERROR, f"Graph output '{name}' has no producer node")

    # --- All consumed non-external tensors must have a source ---
    external = set(graph.inputs) | set(graph.constants)
    for node in graph.nodes.values():
        for t in node.inputs:
            if t not in external and graph.producer(t) is None:
                r(Severity.ERROR,
                  f"Node {node.id} ({node.op.name}) consumes '{t}' "
                  f"which has no producer and is not an input/constant")

    # --- Cycle detection ---
    order = graph._toposort()
    if len(order) != len(graph.nodes):
        visited = {n.id for n in order}
        stuck = [nid for nid in graph.nodes if nid not in visited]
        r(Severity.ERROR, f"Graph has cycles involving nodes: {stuck}")

    # --- Dead nodes (warning — DCE should have caught these post-optimize) ---
    consumed = set()
    for node in graph.nodes.values():
        consumed.update(node.inputs)
    output_set = set(graph.outputs)
    for node in graph.nodes.values():
        if node.output not in consumed and node.output not in output_set:
            r(Severity.WARNING,
              f"Dead node {node.id} ({node.op.name}): "
              f"output '{node.output}' is never consumed")

    return results


@register_validator("structural_integrity", Phase.POST_EXPORT)
def validate_post_export(graph: Graph) -> list[ValidationResult]:
    """Verify structural integrity of the freshly-exported graph."""
    return _check_structure(graph, "structural_integrity")


@register_validator("structural_integrity", Phase.POST_OPTIMIZE)
def validate_post_optimize(graph: Graph) -> list[ValidationResult]:
    """Re-verify structural integrity after optimization passes."""
    return _check_structure(graph, "structural_integrity")


# ---------------------------------------------------------------------------
# Resolution completeness (POST_RESOLVE)
# ---------------------------------------------------------------------------

def _has_symbol(val, symbols: set[str]) -> bool:
    """Check if a value contains an unresolved symbol string."""
    if isinstance(val, str) and val in symbols:
        return True
    if isinstance(val, (tuple, list)):
        return any(_has_symbol(x, symbols) for x in val)
    return False


@register_validator("resolution_completeness", Phase.POST_RESOLVE)
def validate_post_resolve(graph: Graph) -> list[ValidationResult]:
    """Verify that all symbolic dimensions have been resolved to concrete values.

    After resolve_graph(), nothing symbolic should remain — all tensor
    shapes should be tuples of ints, and all node attrs should contain
    only concrete values.
    """
    NAME = "resolution_completeness"
    results = []

    # Check tensor shapes for non-integer dimensions
    for name, tensor in graph.tensors.items():
        for i, d in enumerate(tensor.shape):
            if not isinstance(d, int):
                results.append(ValidationResult(NAME, Severity.ERROR,
                    f"Tensor '{name}' has non-integer dimension at index {i}: {d!r}"))

    # Check attrs for unresolved symbol strings
    symbols = set(graph.dynamic_dims.keys()) if graph.dynamic_dims else set()
    if symbols:
        for node in graph.nodes.values():
            for key, val in node.attrs.items():
                if _has_symbol(val, symbols):
                    results.append(ValidationResult(NAME, Severity.ERROR,
                        f"Node {node.id} ({node.op.name}) attr '{key}' "
                        f"contains unresolved symbol: {val!r}"))

    return results


# ---------------------------------------------------------------------------
# Execution readiness (POST_RESOLVE_OPTIMIZE)
# ---------------------------------------------------------------------------

@register_validator("execution_readiness", Phase.POST_RESOLVE_OPTIMIZE)
def validate_execution_readiness(graph: Graph) -> list[ValidationResult]:
    """Verify the graph is ready for planning and execution.

    After all passes (pre- and post-resolution), the graph must contain
    only ops the runtime can dispatch. Fold-only ops should be gone,
    every op should have registry coverage, and alias chains should be
    well-formed.
    """
    NAME = "execution_readiness"
    results = []

    # --- No fold-only ops ---
    for node in graph.nodes.values():
        if node.op.value >= FOLD_ONLY_BASE:
            results.append(ValidationResult(NAME, Severity.ERROR,
                f"Fold-only op {node.op.name} (node {node.id}) "
                f"was not eliminated by constant folding"))

    # --- All ops have OP_REGISTRY coverage ---
    for node in graph.nodes.values():
        if node.op not in OP_REGISTRY:
            results.append(ValidationResult(NAME, Severity.ERROR,
                f"Op {node.op.name} (node {node.id}) has no OP_REGISTRY entry"))

    # --- Alias chain integrity ---
    for node in graph.nodes.values():
        op_def = OP_REGISTRY.get(node.op)
        if op_def is None or not op_def.is_alias(node):
            continue

        # Alias ops must have exactly one input
        if len(node.inputs) != 1:
            results.append(ValidationResult(NAME, Severity.ERROR,
                f"Alias op {node.op.name} (node {node.id}) "
                f"has {len(node.inputs)} inputs, expected 1"))
            continue

        # Walk alias chain — must terminate (no cycles)
        seen = {node.output}
        cursor = node.inputs[0]
        while cursor is not None:
            if cursor in seen:
                results.append(ValidationResult(NAME, Severity.ERROR,
                    f"Circular alias chain at node {node.id} ({node.op.name})"))
                break
            seen.add(cursor)
            prod = graph.producer(cursor)
            if prod is None:
                break
            prod_def = OP_REGISTRY.get(prod.op)
            if prod_def is None or not prod_def.is_alias(prod):
                break
            cursor = prod.inputs[0] if prod.inputs else None

        # SLICE byte offset must be non-negative
        if node.op == OpType.SLICE:
            byte_offset = node.attrs.get("byte_offset", 0)
            if isinstance(byte_offset, int) and byte_offset < 0:
                results.append(ValidationResult(NAME, Severity.ERROR,
                    f"SLICE node {node.id} has negative byte_offset: {byte_offset}"))

    return results
