"""Interpreted executor: Python loop dispatching each node to a backend kernel.

Walks the execution plan node-by-node, resolving kernels from a
priority-ordered backend chain. Useful for debugging, ablation
benchmarks, and as a reference implementation.

Not the primary execution path — see compiled.py for production use.
"""

from typing import Any, Callable, Protocol

import numpy as np

from ..ir import OpType
from ..ops import OP_REGISTRY
from ..planner import ExecutionPlan
from .common import Executor


# Kernel contract: given input buffers and a pre-allocated output buffer,
# compute the result and write it into output. No allocations, no returns.
KernelFn = Callable[[list[np.ndarray], np.ndarray, dict[str, Any]], None]


class Backend(Protocol):
    """Interface for an execution backend (CPU, CUDA, etc.)."""
    name: str

    def get_kernel(self, op: OpType) -> KernelFn | None:
        """Return a kernel for this op, or None if unsupported."""
        ...


class InterpretedExecutor(Executor):
    """Dispatches plan nodes to backend kernels one at a time."""

    def __init__(self, backends: list[Backend]) -> None:
        super().__init__()
        self.backends = backends
        self._plan: ExecutionPlan | None = None

    def compile(self, plan: ExecutionPlan) -> None:
        """Stash the plan. No compilation needed for interpreted dispatch."""
        self._plan = plan

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference by walking nodes in order and dispatching each one."""
        plan = self._plan
        graph = plan.graph
        arena = self._get_arena(plan.arena_size)

        self._bind_inputs(graph, inputs)
        self._bind_intermediates(graph, plan.offsets, arena)

        for node in plan.order:
            op_def = OP_REGISTRY.get(node.op)
            if op_def is not None and op_def.is_alias(node):
                continue
            if node.op.value >= 100:
                raise RuntimeError(
                    f"Fold-only op {node.op.name} reached interpreted executor — "
                    f"should have been eliminated by constant folding"
                )

            kernel = self._resolve_kernel(node.op)
            input_buffers = [graph.tensors[inp].buffer for inp in node.inputs]

            if node.id in plan.scratch:
                offset, size_bytes = plan.scratch[node.id]
                input_buffers.append(arena[offset:offset + size_bytes])

            kernel(input_buffers, graph.tensors[node.output].buffer, node.attrs)

        return {
            name: graph.tensors[name].buffer.copy()
            for name in graph.outputs
        }

    def _resolve_kernel(self, op: OpType) -> KernelFn:
        """Find the first backend that supports this op."""
        for backend in self.backends:
            kernel = backend.get_kernel(op)
            if kernel is not None:
                return kernel
        raise RuntimeError(f"No backend supports {op.name}")
