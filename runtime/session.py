"""Session: the primary user-facing API for inference.

Wraps the full pipeline — optimize, plan, compile — behind a simple
create/run interface. Mirrors ONNX Runtime's InferenceSession pattern.

    graph = export_model(model, example_inputs)
    session = Session(graph)
    session.create()
    result = session.run({"x": input_data})

    # With observability:
    session = Session(graph)
    session.create(verbose=True)       # logs passes and graph summary
    print(session.plan_stats)          # memory plan statistics
    result = session.run(inputs, profile=True)
    print(session.last_profile)        # per-op timing breakdown
"""

from __future__ import annotations

import numpy as np

from .ir import Graph
from .ops import resolve_graph
from .passes.passes import (
    Pass, PassResult,
    PRE_RESOLUTION_PIPELINE, run_pipeline, run_until_stable,
)
from .planner import ExecutionPlan, PlanStats, plan
from .validation import Phase, Severity, run_validators
from .executor.common import Executor, RunProfile


class Session:
    """Inference session wrapping a graph and its executor.

    Holds the graph, runs optimization and planning on create(),
    and delegates execution to a pluggable Executor.

    Supports dynamic shapes via bindings: export once with symbolic dims,
    then rebind to different concrete values without re-exporting or
    re-optimizing.
    """

    def __init__(self, graph: Graph) -> None:
        self._graph = graph
        self._resolved: Graph | None = None  # concrete copy for planning/execution
        self._executor: Executor | None = None
        self._plan_stats: PlanStats | None = None
        self._pass_log: list[PassResult] | None = None
        self._optimized: bool = False
        # Saved execution settings for rebind()
        self._execution_mode: str = "compiled"
        self._backend: str = "c"
        self._profile: bool = False

    @property
    def graph(self) -> Graph:
        """The underlying graph (may be optimized after create())."""
        return self._graph

    @property
    def plan_stats(self) -> PlanStats | None:
        """Memory plan statistics (available after create())."""
        return self._plan_stats

    @property
    def pass_log(self) -> list[PassResult] | None:
        """Optimization pass results (available after create() with verbose=True)."""
        return self._pass_log

    @property
    def last_profile(self) -> RunProfile | None:
        """Profiling results from the most recent run() call."""
        if self._executor is None:
            return None
        return self._executor._last_profile

    def create(
        self,
        execution_mode: str = "compiled",
        backend: str = "c",
        optimization: str = "default",
        pipeline: list[Pass] | None = None,
        verbose: bool = False,
        profile: bool = False,
        bindings: dict[str, int] | None = None,
        validation: str = "normal",
    ) -> None:
        """Optimize, plan, and compile the graph for execution.

        Args:
            execution_mode: How to dispatch ops.
                "compiled"    — Single C function call (default, fastest).
                "interpreted" — Python loop dispatching to backend kernels.
                                Useful for debugging, ablation, or when the
                                C executor library isn't available.
            backend: Which kernels to use (interpreted mode only; compiled
                always uses the C dispatch table).
                "c"        — C kernels only (errors if a kernel is missing).
                "numpy"    — Pure numpy fallback.
                "c+numpy"  — C kernels with numpy fallback (default for
                             interpreted mode).
            optimization: Which passes to run on the graph.
                "none"       — Skip optimization entirely.
                "default"    — Run DEFAULT_PIPELINE once (absorption, folding,
                               fusion, DCE).
                "aggressive" — Iterate passes until no further changes.
            pipeline: Explicit pass list. Overrides `optimization` if set.
            verbose: Print graph summary and pass activity during creation.
            profile: Enable per-op profiling for subsequent run() calls.
                     Interpreted mode: per-op timing breakdown.
                     Compiled mode: total execution time only.
            bindings: Concrete values for dynamic symbols.
                e.g., {'L': 50} resolves all "L" symbols to 50.
                If the graph has dynamic_dims, this triggers shape
                resolution before planning.
            validation: How strictly to enforce validation checks.
                "strict"  — Fail on WARNING or ERROR.
                "normal"  — Fail on ERROR only (default).
                "none"    — Skip validation entirely.
        """
        # Save execution settings for rebind()
        self._execution_mode = execution_mode
        self._backend = backend
        self._profile = profile

        fail_on = _validation_severity(validation)

        # --- Optimize (only once, not on rebind) ---
        if not self._optimized:
            if verbose:
                print(self._graph.summary())
                print()

            log = [] if verbose else None

            if optimization == "default":
                run_pipeline(self._graph, pipeline, log=log)
            elif optimization == "aggressive":
                run_until_stable(self._graph, pipeline, log=log)

            if verbose and log:
                for entry in log:
                    print(entry)
                print()
                print(self._graph.summary())
                print()

            self._pass_log = log
            self._optimized = True

            self._validate(Phase.POST_OPTIMIZE, self._graph, fail_on, verbose)

        # --- Resolve dynamic shapes ---
        if bindings:
            self._resolved = resolve_graph(self._graph, bindings)
        else:
            self._resolved = self._graph

        self._validate(Phase.POST_RESOLVE_OPTIMIZE, self._resolved, fail_on, verbose)

        # --- Plan ---
        memory_plan = plan(self._resolved)
        self._plan_stats = memory_plan.stats

        if verbose and self._plan_stats:
            print(self._plan_stats)
            print()

        self._validate(Phase.POST_PLAN, memory_plan, fail_on, verbose)

        # --- Build execution plan and validate before compilation ---
        exec_plan = ExecutionPlan(
            graph=self._resolved,
            memory=memory_plan,
            executor_type=execution_mode,
            backend=backend,
        )
        self._validate(Phase.PRE_EXECUTE, exec_plan, fail_on, verbose)

        # --- Compile ---
        executor = self._make_executor(execution_mode, backend, profile)
        executor.compile(memory_plan)
        self._executor = executor

    def rebind(self, bindings: dict[str, int]) -> None:
        """Re-plan and re-compile the graph with new dynamic shape values.

        Skips optimization passes (they ran once during create()). Creates
        a fresh concrete copy from the original symbolic graph, then
        re-plans memory and re-compiles the executor.

        Args:
            bindings: Concrete values for dynamic symbols.
                e.g., {'L': 64} resolves all "L" symbols to 64.
        """
        self._resolved = resolve_graph(self._graph, bindings)

        memory_plan = plan(self._resolved)
        self._plan_stats = memory_plan.stats

        executor = self._make_executor(
            self._execution_mode, self._backend, self._profile
        )
        executor.compile(memory_plan)
        self._executor = executor

    def run(self, inputs: dict[str, np.ndarray],
            profile: bool = False) -> dict[str, np.ndarray]:
        """Run inference on the compiled graph.

        Args:
            inputs: Map of graph input names to numpy arrays.
            profile: Enable profiling for this call (overrides session default).

        Returns:
            Map of graph output names to numpy result arrays.
        """
        if self._executor is None:
            raise RuntimeError("Session not created — call create() first")

        if profile:
            self._executor._profile = True
        result = self._executor.run(inputs)
        if profile:
            self._executor._profile = False

        return result

    @staticmethod
    def _validate(phase: Phase, target, fail_on: Severity | None,
                  verbose: bool) -> None:
        """Run validators for a phase, optionally printing results."""
        if fail_on is None:
            return
        results = run_validators(phase, target, fail_on=fail_on)
        if verbose and results:
            for r in results:
                print(r)

    @staticmethod
    def _make_executor(execution_mode: str, backend: str,
                       profile: bool = False) -> Executor:
        """Construct an executor from high-level options."""
        if execution_mode == "compiled":
            from .executor.compiled import CompiledExecutor
            return CompiledExecutor(profile=profile)

        if execution_mode == "interpreted":
            from .executor.interpreted import InterpretedExecutor
            backends = _build_backend_chain(backend)
            return InterpretedExecutor(backends=backends, profile=profile)

        raise ValueError(
            f"Unknown execution_mode '{execution_mode}' "
            f"(expected 'compiled' or 'interpreted')"
        )


def _validation_severity(validation: str) -> Severity | None:
    """Map validation preference string to fail_on severity."""
    if validation == "strict":
        return Severity.WARNING
    if validation == "normal":
        return Severity.ERROR
    if validation == "none":
        return None
    raise ValueError(
        f"Unknown validation '{validation}' "
        f"(expected 'strict', 'normal', or 'none')"
    )


def _build_backend_chain(backend: str) -> list:
    """Construct a backend list from a preference string."""
    from .backends.c_backend import CBackend
    from .backends.numpy_backend import NumpyBackend

    if backend == "c":
        return [CBackend()]
    if backend == "numpy":
        return [NumpyBackend()]
    if backend == "c+numpy":
        return [CBackend(), NumpyBackend()]

    raise ValueError(
        f"Unknown backend '{backend}' "
        f"(expected 'c', 'numpy', or 'c+numpy')"
    )
