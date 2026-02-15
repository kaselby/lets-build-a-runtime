"""Session: the primary user-facing API for inference.

Wraps the full pipeline — optimize, plan, compile — behind a simple
create/run interface. The compiled C executor is the default path.

    graph = export_model(model, example_inputs)
    session = Session(graph)
    session.create()
    result = session.run({"x": input_data})

For custom pipelines or the interpreted executor, use the underlying
components directly (passes, planner, executor/).
"""

from __future__ import annotations

import numpy as np

from .ir import Graph
from .passes.passes import Pass, DEFAULT_PIPELINE, run_pipeline
from .planner import ExecutionPlan, plan
from .executor.common import Executor


class Session:
    """Inference session wrapping a graph and its executor.

    Holds the graph, runs optimization and planning on create(),
    and delegates execution to a pluggable Executor.
    """

    def __init__(self, graph: Graph) -> None:
        self._graph = graph
        self._executor: Executor | None = None

    def create(
        self,
        pipeline: list[Pass] | None = None,
        executor: Executor | None = None,
    ) -> None:
        """Optimize, plan, and compile the graph for execution.

        Args:
            pipeline: Custom pass list. None = DEFAULT_PIPELINE.
            executor: Executor instance. None = CompiledExecutor.
        """
        run_pipeline(self._graph, pipeline or DEFAULT_PIPELINE)
        execution_plan = plan(self._graph)

        if executor is None:
            from .executor.compiled import CompiledExecutor
            executor = CompiledExecutor()

        executor.compile(execution_plan)
        self._executor = executor

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference with the compiled executor."""
        if self._executor is None:
            raise RuntimeError("Session not created — call create() first")
        return self._executor.run(inputs)
