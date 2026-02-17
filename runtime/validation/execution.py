"""Execution plan validators.

Covers PRE_EXECUTE phase. Separate validators for compiled and
interpreted execution paths, since they have different dispatch
requirements.
"""

from ..ir import OpType
from ..ops import OP_REGISTRY
from ..planner import ExecutionPlan
from .core import Phase, Severity, ValidationResult, register_validator


# Ops with entries in executor.c's dispatch_table.
# Must stay in sync with executor.c — update when adding C kernels.
_C_DISPATCH_OPS: frozenset[OpType] = frozenset({
    # Element-wise unary
    OpType.RELU, OpType.EXP, OpType.TANH, OpType.POW, OpType.GELU,
    # Element-wise binary
    OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV,
    # Reductions
    OpType.MAX, OpType.SUM, OpType.SOFTMAX,
    # MatMul
    OpType.MATMUL,
    # Shape / data movement
    OpType.RESHAPE, OpType.TRANSPOSE, OpType.SLICE, OpType.EMBEDDING,
    # Normalization
    OpType.LAYERNORM,
    # Fused ops
    OpType.MATMUL_ADD, OpType.FUSED_BIAS_RELU, OpType.ATTENTION,
})

# C backend kernel coverage (per-op dispatch via ctypes).
# Must stay in sync with c_backend.py CBackend.__init__.
_C_BACKEND_OPS: frozenset[OpType] = frozenset({
    OpType.MATMUL, OpType.MATMUL_ADD,
    OpType.ADD, OpType.RELU, OpType.TRANSPOSE,
    OpType.DIV, OpType.SUB, OpType.MUL, OpType.EXP,
    OpType.MAX, OpType.SUM, OpType.SOFTMAX,
    OpType.LAYERNORM, OpType.FUSED_BIAS_RELU, OpType.ATTENTION,
    OpType.POW, OpType.TANH, OpType.GELU,
    OpType.EMBEDDING, OpType.SLICE,
})


def _ops_needing_dispatch(plan: ExecutionPlan) -> list[tuple[int, OpType]]:
    """Collect (node_id, op) for nodes that will actually be dispatched.

    Filters out alias ops (skipped by both executors) and fold-only ops
    (which should have been caught earlier but are checked defensively).
    """
    graph = plan.graph
    external = set(graph.inputs) | set(graph.constants)
    ops = []
    for node in plan.memory.order:
        op_def = OP_REGISTRY.get(node.op)
        if op_def is not None and op_def.is_alias(node) and node.inputs[0] not in external:
            continue
        ops.append((node.id, node.op))
    return ops


@register_validator("compiled_dispatch", Phase.PRE_EXECUTE)
def validate_compiled_dispatch(plan: ExecutionPlan) -> list[ValidationResult]:
    """Verify every dispatched op has a C executor entry and extras packer.

    The compiled executor builds a COpNode struct array with pre-resolved
    pointers. If an op isn't in the C dispatch table, the executor aborts
    at runtime. This catches that at validation time.
    """
    if plan.executor_type != "compiled":
        return []

    NAME = "compiled_dispatch"
    results = []

    for node_id, op in _ops_needing_dispatch(plan):
        if op.value >= 100:
            results.append(ValidationResult(NAME, Severity.ERROR,
                f"Fold-only op {op.name} (node {node_id}) cannot be compiled"))
            continue

        if op not in _C_DISPATCH_OPS:
            results.append(ValidationResult(NAME, Severity.ERROR,
                f"Op {op.name} (node {node_id}) has no C dispatch entry"))

    return results


@register_validator("interpreted_dispatch", Phase.PRE_EXECUTE)
def validate_interpreted_dispatch(plan: ExecutionPlan) -> list[ValidationResult]:
    """Verify every dispatched op has a kernel in the backend chain.

    The interpreted executor walks the backend list and calls the first
    kernel it finds. If no backend supports an op, it raises at runtime.
    This catches that at validation time.
    """
    if plan.executor_type != "interpreted":
        return []

    NAME = "interpreted_dispatch"
    results = []
    available = _backend_op_set(plan.backend)

    for node_id, op in _ops_needing_dispatch(plan):
        if op.value >= 100:
            results.append(ValidationResult(NAME, Severity.ERROR,
                f"Fold-only op {op.name} (node {node_id}) cannot be dispatched"))
            continue

        if op not in available:
            results.append(ValidationResult(NAME, Severity.ERROR,
                f"Op {op.name} (node {node_id}) has no kernel "
                f"in backend '{plan.backend}'"))

    return results


def _backend_op_set(backend: str) -> frozenset[OpType]:
    """Determine the union of ops available for a backend configuration."""
    # Import numpy kernel registry for automatic sync
    from ..backends.numpy_backend import _KERNELS as numpy_kernels
    numpy_ops = frozenset(numpy_kernels.keys())

    if backend == "numpy":
        return numpy_ops
    if backend == "c":
        return _C_BACKEND_OPS
    if backend == "c+numpy":
        return _C_BACKEND_OPS | numpy_ops
    return frozenset()
