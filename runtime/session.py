"""Session: the primary user-facing API for inference.

Wraps the full pipeline — optimize, plan, compile — behind a simple
create/run interface. Mirrors ONNX Runtime's InferenceSession pattern.

    graph = export_model(model, example_inputs)
    session = Session(graph)
    session.create()
    result = session.run({"x": input_data})

    # Or with options:
    session.create(execution_mode="interpreted", backend="numpy")
"""

from __future__ import annotations

import numpy as np

from .ir import Graph
from .passes.passes import Pass, DEFAULT_PIPELINE, run_pipeline, run_until_stable
from .planner import plan
from .executor.common import Executor


class Session:
    """Inference session wrapping a graph and its executor.

    Holds the graph, runs optimization and planning on create(),
    and delegates execution to a pluggable Executor.
    """

    def __init__(self, graph: Graph) -> None:
        self._graph = graph
        self._executor: Executor | None = None

    @property
    def graph(self) -> Graph:
        """The underlying graph (may be optimized after create())."""
        return self._graph

    def create(
        self,
        execution_mode: str = "compiled",
        backend: str = "c",
        optimization: str = "default",
        pipeline: list[Pass] | None = None,
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
        """
        # --- Optimize ---
        if optimization == "default":
            run_pipeline(self._graph, pipeline)
        elif optimization == "aggressive":
            run_until_stable(self._graph, pipeline)

        # --- Plan ---
        execution_plan = plan(self._graph)

        # --- Build executor ---
        executor = self._make_executor(execution_mode, backend)
        executor.compile(execution_plan)
        self._executor = executor

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference on the compiled graph.

        Args:
            inputs: Map of graph input names to numpy arrays.

        Returns:
            Map of graph output names to numpy result arrays.
        """
        if self._executor is None:
            raise RuntimeError("Session not created — call create() first")
        return self._executor.run(inputs)

    @staticmethod
    def _make_executor(execution_mode: str, backend: str) -> Executor:
        """Construct an executor from high-level options."""
        if execution_mode == "compiled":
            from .executor.compiled import CompiledExecutor
            return CompiledExecutor()

        if execution_mode == "interpreted":
            from .executor.interpreted import InterpretedExecutor
            backends = _build_backend_chain(backend)
            return InterpretedExecutor(backends=backends)

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
