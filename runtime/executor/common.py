"""Shared types and utilities for both execution paths.

Contains the Executor base class, COpNode struct definition, C library
loading, arena/buffer management, and profiling types.
"""

import ctypes
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..ir import Graph
from ..planner import ExecutionPlan


# ---------------------------------------------------------------------------
# Profiling types
# ---------------------------------------------------------------------------

@dataclass
class OpTiming:
    """Timing for a single op execution."""
    node_id: int
    op: str              # OpType.name
    time_ns: int
    output_name: str
    output_shape: tuple[int, ...]


@dataclass
class RunProfile:
    """Profiling results from a single inference run."""
    op_timings: list[OpTiming] = field(default_factory=list)
    total_ns: int = 0

    def __str__(self) -> str:
        if not self.op_timings:
            total_ms = self.total_ns / 1e6
            return f"Execution Profile:\n  Total: {total_ms:.2f} ms (no per-op breakdown)"

        # Group by op type
        by_op: dict[str, list[int]] = defaultdict(list)
        for t in self.op_timings:
            by_op[t.op].append(t.time_ns)

        lines = [f"Execution Profile:"]
        total_ms = self.total_ns / 1e6
        lines.append(f"  Total: {total_ms:.2f} ms ({len(self.op_timings)} ops)")

        # Sort by total time descending
        op_totals = [(op, sum(times), len(times)) for op, times in by_op.items()]
        op_totals.sort(key=lambda x: x[1], reverse=True)

        for op, total_ns, count in op_totals:
            ms = total_ns / 1e6
            pct = total_ns / self.total_ns * 100 if self.total_ns > 0 else 0
            lines.append(f"  {op:<20} {ms:7.2f} ms ({pct:5.1f}%)  {count} ops")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Executor base class
# ---------------------------------------------------------------------------

class Executor(ABC):
    """Base class for execution backends.

    Subclasses implement compile() to do any one-time preparation,
    and run() to execute inference. The arena is managed here —
    allocated once, reused across inference calls, grown if needed.
    """

    def __init__(self, profile: bool = False) -> None:
        self._arena: np.ndarray | None = None
        self._profile = profile
        self._last_profile: RunProfile | None = None

    def _get_arena(self, size: int) -> np.ndarray:
        """Return an arena buffer of at least `size` bytes."""
        if self._arena is None or len(self._arena) < size:
            self._arena = np.zeros(size, dtype=np.uint8)
        return self._arena

    def _bind_inputs(self, graph: Graph, inputs: dict[str, np.ndarray]) -> None:
        """Point input tensors at user-provided arrays."""
        for name in graph.inputs:
            if name not in inputs:
                raise ValueError(f"Missing input tensor: '{name}'")
            graph.tensors[name].buffer = inputs[name]

    def _bind_intermediates(self, graph: Graph, offsets: dict[str, int],
                            arena: np.ndarray) -> None:
        """Point intermediate tensors at their arena views."""
        for name, offset in offsets.items():
            tensor = graph.tensors[name]
            dtype = np.dtype(tensor.dtype)
            size_bytes = int(np.prod(tensor.shape)) * dtype.itemsize
            flat_view = arena[offset:offset + size_bytes]
            tensor.buffer = flat_view.view(dtype).reshape(tensor.shape)

    @abstractmethod
    def compile(self, plan: ExecutionPlan) -> None:
        """Prepare for execution. Called once per plan."""
        ...

    @abstractmethod
    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference. Called many times per compiled plan."""
        ...


# ---------------------------------------------------------------------------
# COpNode struct (mirrors executor.c)
# ---------------------------------------------------------------------------

MAX_INPUTS = 8
MAX_DIMS = 16


class COpNode(ctypes.Structure):
    """Mirrors the OpNode struct in executor.c.

    Pointers are void* (c_void_p) — each dispatch case in C casts to
    the appropriate type. This allows ops with non-float inputs (e.g.,
    int64 indices for embedding, int8 weights for quantization).
    """
    _fields_ = [
        ("op", ctypes.c_int),
        ("n_inputs", ctypes.c_int),
        ("inputs", ctypes.c_void_p * MAX_INPUTS),
        ("output", ctypes.c_void_p),
        ("out_shape", ctypes.c_int * MAX_DIMS),
        ("n_dims", ctypes.c_int),
        ("elem_size", ctypes.c_int),
        ("extra", ctypes.c_int * MAX_DIMS),
    ]


# ---------------------------------------------------------------------------
# C library loading
# ---------------------------------------------------------------------------

def _load_c_executor() -> ctypes.CDLL | None:
    """Try to load the compiled C executor library."""
    csrc_dir = Path(__file__).parent.parent.parent / "csrc"
    if sys.platform == "darwin":
        lib_path = csrc_dir / "libexecutor.dylib"
    else:
        lib_path = csrc_dir / "libexecutor.so"
    if not lib_path.exists():
        return None
    lib = ctypes.CDLL(str(lib_path))
    lib.execute.argtypes = [ctypes.POINTER(COpNode), ctypes.c_int]
    lib.execute.restype = None
    return lib


_c_executor_lib = _load_c_executor()


