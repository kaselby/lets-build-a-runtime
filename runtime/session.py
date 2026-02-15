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
from .passes.passes import Pass, PassResult, DEFAULT_PIPELINE, run_pipeline, run_until_stable
from .planner import PlanStats, plan
from .executor.common import Executor, RunProfile


class Session:
    """Inference session wrapping a graph and its executor.

    Holds the graph, runs optimization and planning on create(),
    and delegates execution to a pluggable Executor.
    """

    def __init__(self, graph: Graph) -> None:
        self._graph = graph
        self._executor: Executor | None = None
        self._plan_stats: PlanStats | None = None
        self._pass_log: list[PassResult] | None = None

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
        """
        # --- Optimize ---
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

        # --- Plan ---
        execution_plan = plan(self._graph)
        self._plan_stats = execution_plan.stats

        if verbose and self._plan_stats:
            print(self._plan_stats)
            print()

        # --- Build executor ---
        executor = self._make_executor(execution_mode, backend, profile)
        executor.compile(execution_plan)
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
