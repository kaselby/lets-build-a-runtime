"""Core validation types, registry, and runner.

All types live here to avoid circular imports — validator submodules
import from core, and __init__ re-exports everything.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable


class Phase(Enum):
    """Pipeline checkpoints where validation runs.

    Each phase implies the type of artifact being validated:
        Graph phases:   POST_EXPORT, POST_OPTIMIZE, POST_RESOLVE,
                        POST_RESOLVE_OPTIMIZE
        MemoryPlan:     POST_PLAN
        ExecutionPlan:  PRE_EXECUTE
    """
    POST_EXPORT           = auto()   # raw graph from exporter
    POST_OPTIMIZE         = auto()   # after pre-resolution passes
    POST_RESOLVE          = auto()   # after symbol resolution + shape propagation
    POST_RESOLVE_OPTIMIZE = auto()   # after post-resolution passes (ready for planning)
    POST_PLAN             = auto()   # after memory planning
    PRE_EXECUTE           = auto()   # after full plan assembly (ready for compilation)


class Severity(Enum):
    """Diagnostic severity level.

    ERROR:   Execution will fail or produce incorrect results.
    WARNING: Suspicious but not necessarily fatal.
    INFO:    Diagnostic observation (dead code, suboptimal patterns).
    """
    ERROR   = auto()
    WARNING = auto()
    INFO    = auto()


@dataclass
class ValidationResult:
    """A single diagnostic from a validator."""
    validator: str
    severity: Severity
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.name}] {self.validator}: {self.message}"


class ValidationError(Exception):
    """Raised when validation produces fatal errors."""

    def __init__(self, phase: Phase, results: list[ValidationResult]) -> None:
        self.phase = phase
        self.results = results
        errors = [r for r in results if r.severity == Severity.ERROR]
        msg = f"Validation failed at {phase.name} ({len(errors)} error(s)):\n"
        msg += "\n".join(f"  {r}" for r in errors)
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass
class Validator:
    """A tagged validation check.

    Attributes:
        name: Human-readable identifier.
        phase: When this validator runs in the pipeline.
        check: Callable that inspects a pipeline artifact and returns
            diagnostics. Signature depends on phase — Graph for graph
            phases, MemoryPlan for POST_PLAN, ExecutionPlan for PRE_EXECUTE.
    """
    name: str
    phase: Phase
    check: Callable[..., list[ValidationResult]]


VALIDATORS: list[Validator] = []


def register_validator(name: str, phase: Phase):
    """Decorator to register a validation function.

    Usage:
        @register_validator("my_check", Phase.POST_OPTIMIZE)
        def check_something(graph: Graph) -> list[ValidationResult]:
            ...
    """
    def decorator(fn: Callable[..., list[ValidationResult]]):
        VALIDATORS.append(Validator(name=name, phase=phase, check=fn))
        return fn
    return decorator


def run_validators(
    phase: Phase,
    target: Any,
    *,
    fail_on: Severity = Severity.ERROR,
) -> list[ValidationResult]:
    """Run all validators registered for a phase.

    Args:
        phase: Which pipeline checkpoint to validate.
        target: The artifact to validate (Graph, MemoryPlan, or ExecutionPlan).
        fail_on: Raise ValidationError if any result meets or exceeds
            this severity. Set to None to collect without raising.

    Returns:
        All validation results (errors, warnings, and info).

    Raises:
        ValidationError: If any result's severity >= fail_on.
    """
    results: list[ValidationResult] = []
    for v in VALIDATORS:
        if v.phase == phase:
            results.extend(v.check(target))

    if fail_on is not None:
        fatal = [r for r in results if r.severity.value <= fail_on.value]
        if fatal:
            raise ValidationError(phase, results)

    return results
