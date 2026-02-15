"""Shared types and utilities for both execution paths.

Contains the Executor base class, COpNode struct definition, C library
loading, and arena/buffer management.
"""

import ctypes
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..ir import Graph
from ..planner import ExecutionPlan


# ---------------------------------------------------------------------------
# Executor base class
# ---------------------------------------------------------------------------

class Executor(ABC):
    """Base class for execution backends.

    Subclasses implement compile() to do any one-time preparation,
    and run() to execute inference. The arena is managed here —
    allocated once, reused across inference calls, grown if needed.
    """

    def __init__(self) -> None:
        self._arena: np.ndarray | None = None

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
        ("extra", ctypes.c_int * MAX_DIMS),
    ]


# ---------------------------------------------------------------------------
# C library loading
# ---------------------------------------------------------------------------

def _load_c_executor() -> ctypes.CDLL | None:
    """Try to load the compiled C executor library."""
    csrc_dir = Path(__file__).parent.parent.parent / "csrc_edited"
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


