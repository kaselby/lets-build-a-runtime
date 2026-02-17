"""Validation framework for the inference runtime pipeline.

Validators are tagged checks that run at specific pipeline phases.
Each validator inspects a pipeline artifact (Graph, MemoryPlan, or
ExecutionPlan) and returns structured diagnostics.

The registry collects validators via decorator. The Session runs
them at the appropriate points, but they work standalone too:

    from runtime.validation import run_validators, Phase
    errors = run_validators(Phase.POST_OPTIMIZE, graph)

Validators are defined in submodules:
    graph.py      — graph-level checks (structural, semantic, shape)
    plan.py       — memory plan integrity (arena, offsets, lifetimes)
    execution.py  — execution plan checks (dispatch coverage, extras)

Core types live in core.py to avoid circular imports.
"""

from .core import (  # noqa: F401
    Phase,
    Severity,
    ValidationResult,
    ValidationError,
    Validator,
    VALIDATORS,
    register_validator,
    run_validators,
)

# Import submodules to trigger validator registration.
from . import graph, plan, execution  # noqa: F401
