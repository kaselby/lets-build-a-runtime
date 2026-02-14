"""C-based CPU backend: operator kernels via ctypes.

Loads the compiled shared library from csrc/ and wraps each C kernel
into the standard kernel contract (inputs, output, attrs) -> None.
Falls back gracefully if the library isn't compiled yet.
"""

import ctypes
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..ir import OpType
from ..executor import KernelFn


# Pointer types
FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
INT_PTR = ctypes.POINTER(ctypes.c_int)


def _load_library() -> ctypes.CDLL | None:
    """Try to load the compiled C library."""
    csrc_dir = Path(__file__).parent.parent.parent / "csrc"

    if sys.platform == "darwin":
        lib_path = csrc_dir / "libruntime.dylib"
    else:
        lib_path = csrc_dir / "libruntime.so"

    if not lib_path.exists():
        return None

    lib = ctypes.CDLL(str(lib_path))

    # Declare function signatures so ctypes can type-check.
    # This must match the C declarations in ops.c exactly.
    lib.kernel_matmul.argtypes = [FLOAT_PTR, FLOAT_PTR, FLOAT_PTR,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int]  # batches, trans_b
    lib.kernel_matmul.restype = None

    lib.kernel_matmul_ab.argtypes = [FLOAT_PTR, FLOAT_PTR, FLOAT_PTR,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int,
                                     ctypes.c_float, ctypes.c_float]  # alpha, beta
    lib.kernel_matmul_ab.restype = None

    lib.kernel_matmul_beta.argtypes = [FLOAT_PTR, FLOAT_PTR, FLOAT_PTR,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_float]
    lib.kernel_matmul_beta.restype = None

    lib.kernel_add.argtypes = [FLOAT_PTR, FLOAT_PTR, FLOAT_PTR,
                               ctypes.c_int, ctypes.c_int]
    lib.kernel_add.restype = None

    lib.kernel_relu.argtypes = [FLOAT_PTR, FLOAT_PTR, ctypes.c_int]
    lib.kernel_relu.restype = None

    lib.kernel_transpose.argtypes = [FLOAT_PTR, FLOAT_PTR,
                                     ctypes.c_int, ctypes.c_int]
    lib.kernel_transpose.restype = None

    # Element-wise binary ops: (a, b, out, n)
    for name in ("kernel_div", "kernel_sub", "kernel_mul"):
        fn = getattr(lib, name)
        fn.argtypes = [FLOAT_PTR, FLOAT_PTR, FLOAT_PTR, ctypes.c_int]
        fn.restype = None

    # Element-wise unary: (x, out, n)
    lib.kernel_exp.argtypes = [FLOAT_PTR, FLOAT_PTR, ctypes.c_int]
    lib.kernel_exp.restype = None

    # Reductions: (x, out, outer, axis_size, inner)
    for name in ("kernel_max", "kernel_sum"):
        fn = getattr(lib, name)
        fn.argtypes = [FLOAT_PTR, FLOAT_PTR,
                       ctypes.c_int, ctypes.c_int, ctypes.c_int]
        fn.restype = None

    # Softmax: (x, out, rows, cols)
    lib.kernel_softmax.argtypes = [FLOAT_PTR, FLOAT_PTR,
                                   ctypes.c_int, ctypes.c_int]
    lib.kernel_softmax.restype = None

    # LayerNorm: (x, gamma, beta, out, rows, cols, eps)
    lib.kernel_layernorm.argtypes = [FLOAT_PTR, FLOAT_PTR, FLOAT_PTR, FLOAT_PTR,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_float]
    lib.kernel_layernorm.restype = None

    # Broadcast binary op: (a, b, out, a_strides, b_strides, out_shape, ndim, op)
    lib.kernel_broadcast_binop.argtypes = [
        FLOAT_PTR, FLOAT_PTR, FLOAT_PTR,
        INT_PTR, INT_PTR, INT_PTR,
        ctypes.c_int, ctypes.c_int,
    ]
    lib.kernel_broadcast_binop.restype = None

    # Fused bias+relu: (a, bias, out, M, N)
    lib.kernel_bias_relu.argtypes = [FLOAT_PTR, FLOAT_PTR, FLOAT_PTR,
                                     ctypes.c_int, ctypes.c_int]
    lib.kernel_bias_relu.restype = None

    # Attention: (Q, K, V, output, scratch, batch_heads, seq_len, head_dim, causal)
    lib.kernel_attention.argtypes = [FLOAT_PTR, FLOAT_PTR, FLOAT_PTR,
                                     FLOAT_PTR, FLOAT_PTR,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int]
    lib.kernel_attention.restype = None

    # Scalar binary ops: (a, s, out, n)
    for name in ("kernel_add_scalar", "kernel_div_scalar", "kernel_sub_scalar", "kernel_mul_scalar"):
        fn = getattr(lib, name)
        fn.argtypes = [FLOAT_PTR, ctypes.c_float, FLOAT_PTR, ctypes.c_int]
        fn.restype = None

    # Pow scalar: (x, scalar, out, n)
    lib.kernel_pow_scalar.argtypes = [FLOAT_PTR, ctypes.c_float, FLOAT_PTR, ctypes.c_int]
    lib.kernel_pow_scalar.restype = None

    # Tanh: (x, out, n)
    lib.kernel_tanh.argtypes = [FLOAT_PTR, FLOAT_PTR, ctypes.c_int]
    lib.kernel_tanh.restype = None

    # GELU tanh approximation: (x, out, n)
    lib.kernel_gelu_tanh.argtypes = [FLOAT_PTR, FLOAT_PTR, ctypes.c_int]
    lib.kernel_gelu_tanh.restype = None

    # Embedding: (ids, table, out, n_ids, embed_dim)
    LONG_PTR = ctypes.POINTER(ctypes.c_long)
    lib.kernel_embedding.argtypes = [LONG_PTR, FLOAT_PTR, FLOAT_PTR,
                                     ctypes.c_int, ctypes.c_int]
    lib.kernel_embedding.restype = None

    return lib


def _as_ptr(arr: np.ndarray) -> ctypes.c_void_p:
    """Extract a float pointer from a numpy array."""
    return arr.ctypes.data_as(FLOAT_PTR)


# Broadcast binary op codes (must match the switch in kernel_broadcast_binop)
_BCAST_ADD, _BCAST_SUB, _BCAST_MUL, _BCAST_DIV = 0, 1, 2, 3


def _broadcast_strides(shape: tuple, out_shape: tuple) -> list[int]:
    """Compute broadcast strides for an input shape against an output shape.

    Dims where the input has size 1 (or is absent due to fewer dims) get
    stride 0 — the kernel re-reads the same element, implementing broadcast.
    """
    ndim = len(out_shape)
    offset = ndim - len(shape)
    strides = []
    for d in range(ndim):
        if d < offset or shape[d - offset] == 1:
            strides.append(0)
        else:
            stride = 1
            for k in range(d - offset + 1, len(shape)):
                stride *= shape[k]
            strides.append(stride)
    return strides


def _call_broadcast(lib, a: np.ndarray, b: np.ndarray,
                    output: np.ndarray, op: int) -> None:
    """Call the general broadcast binary op kernel."""
    out_shape = output.shape
    ndim = len(out_shape)
    a_strides = _broadcast_strides(a.shape, out_shape)
    b_strides = _broadcast_strides(b.shape, out_shape)
    c_a_strides = (ctypes.c_int * ndim)(*a_strides)
    c_b_strides = (ctypes.c_int * ndim)(*b_strides)
    c_out_shape = (ctypes.c_int * ndim)(*out_shape)
    lib.kernel_broadcast_binop(
        _as_ptr(a), _as_ptr(b), _as_ptr(output),
        c_a_strides, c_b_strides, c_out_shape,
        ndim, op,
    )


def _wrap_matmul(lib: ctypes.CDLL) -> KernelFn:
    """Wrap kernel_matmul with broadcast batch dim support."""
    def kernel(inputs: list[np.ndarray], output: np.ndarray,
               attrs: dict[str, Any]) -> None:
        a, b = inputs[0], inputs[1]
        trans_b = 1 if attrs.get("transpose_b") else 0
        alpha = attrs.get("alpha", 1.0)
        K = a.shape[-1]
        N = b.shape[-2] if trans_b else b.shape[-1]
        M = a.shape[-2] if a.ndim >= 2 else 1

        def _call_matmul(M_, N_, K_, batches_, trans_b_):
            if alpha != 1.0:
                lib.kernel_matmul_ab(_as_ptr(a), _as_ptr(b), _as_ptr(output),
                                     M_, N_, K_, batches_, trans_b_,
                                     ctypes.c_float(alpha), ctypes.c_float(0.0))
            else:
                lib.kernel_matmul(_as_ptr(a), _as_ptr(b), _as_ptr(output),
                                  M_, N_, K_, batches_, trans_b_)

        if a.ndim > 2 and b.ndim == 2:
            # ND×2D fast path: flatten A's batch dims into M, single sgemm
            M_total = 1
            for d in a.shape[:-1]:
                M_total *= d
            _call_matmul(M_total, N, K, 1, trans_b)
        else:
            a_batch = a.shape[:-2] if a.ndim > 2 else ()
            b_batch = b.shape[:-2] if b.ndim > 2 else ()

            if a_batch == b_batch:
                # Matching batch dims — single C call with batch loop
                batches = 1
                for d in a_batch:
                    batches *= d
                _call_matmul(M, N, K, max(batches, 1), trans_b)
            else:
                # Broadcasting over batch dims — odometer with per-slice sgemm
                _broadcast_matmul(lib, a, b, output, M, N, K, trans_b,
                                  a_batch, b_batch)
    return kernel


def _broadcast_matmul(lib, a, b, output, M, N, K, trans_b,
                      a_batch, b_batch):
    """Batched matmul with broadcasting over batch dimensions."""
    out_batch = output.shape[:-2]
    ndim_batch = len(out_batch)
    a_bstrides = _broadcast_strides(a_batch, out_batch)
    b_bstrides = _broadcast_strides(b_batch, out_batch)

    total_batches = 1
    for d in out_batch:
        total_batches *= d

    a_mat = M * K           # elements per A slice
    b_mat = N * K           # elements per B slice (same whether transposed or not)
    out_mat = M * N         # elements per output slice

    a_base = a.ctypes.data  # integer memory address
    b_base = b.ctypes.data
    out_base = output.ctypes.data

    coords = [0] * ndim_batch
    a_off = 0  # element offset into a
    b_off = 0

    for i in range(total_batches):
        a_ptr = ctypes.cast(a_base + a_off * 4, FLOAT_PTR)
        b_ptr = ctypes.cast(b_base + b_off * 4, FLOAT_PTR)
        out_ptr = ctypes.cast(out_base + i * out_mat * 4, FLOAT_PTR)
        lib.kernel_matmul(a_ptr, b_ptr, out_ptr, M, N, K, 1, trans_b)

        for d in range(ndim_batch - 1, -1, -1):
            coords[d] += 1
            a_off += a_bstrides[d] * a_mat
            b_off += b_bstrides[d] * b_mat
            if coords[d] < out_batch[d]:
                break
            a_off -= coords[d] * a_bstrides[d] * a_mat
            b_off -= coords[d] * b_bstrides[d] * b_mat
            coords[d] = 0


def _wrap_matmul_add(lib: ctypes.CDLL) -> KernelFn:
    """Wrap kernel_matmul_beta for fused matmul + bias add.

    Inputs: [A, B, bias]
    1. Pre-fill output with broadcast bias (each row = bias vector)
    2. Call sgemm with beta=1.0 to accumulate A@B into the bias-filled output
    """
    def kernel(inputs: list[np.ndarray], output: np.ndarray,
               attrs: dict[str, Any]) -> None:
        a, b, bias = inputs[0], inputs[1], inputs[2]
        trans_b = 1 if attrs.get("transpose_b") else 0
        alpha = attrs.get("alpha", 1.0)
        K = a.shape[-1]
        N = b.shape[-2] if trans_b else b.shape[-1]
        M = a.shape[-2] if a.ndim >= 2 else 1

        # Pre-fill output with broadcast bias (handles any number of dims)
        output[...] = bias

        def _call_matmul_add(M_, N_, K_, batches_, trans_b_):
            lib.kernel_matmul_beta(_as_ptr(a), _as_ptr(b), _as_ptr(output),
                                   M_, N_, K_, batches_, trans_b_,
                                   ctypes.c_float(alpha))

        if a.ndim > 2 and b.ndim == 2:
            # ND×2D fast path: flatten A's batch dims into M, single sgemm
            M_total = 1
            for d in a.shape[:-1]:
                M_total *= d
            _call_matmul_add(M_total, N, K, 1, trans_b)
        else:
            a_batch = a.shape[:-2] if a.ndim > 2 else ()
            b_batch = b.shape[:-2] if b.ndim > 2 else ()

            if a_batch == b_batch:
                # Matching batch dims — single C call with batch loop
                batches = 1
                for d in a_batch:
                    batches *= d
                _call_matmul_add(M, N, K, max(batches, 1), trans_b)
            else:
                # Broadcasting over batch dims — odometer with per-slice sgemm
                _broadcast_matmul_add(lib, a, b, output, bias, M, N, K, trans_b,
                                      a_batch, b_batch, alpha)
    return kernel


def _broadcast_matmul_add(lib, a, b, output, bias, M, N, K, trans_b,
                          a_batch, b_batch, alpha):
    """Batched matmul+add with broadcasting over batch dimensions."""
    out_batch = output.shape[:-2]
    ndim_batch = len(out_batch)
    a_bstrides = _broadcast_strides(a_batch, out_batch)
    b_bstrides = _broadcast_strides(b_batch, out_batch)

    total_batches = 1
    for d in out_batch:
        total_batches *= d

    a_mat = M * K           # elements per A slice
    b_mat = N * K           # elements per B slice
    out_mat = M * N         # elements per output slice

    a_base = a.ctypes.data
    b_base = b.ctypes.data
    out_base = output.ctypes.data

    coords = [0] * ndim_batch
    a_off = 0
    b_off = 0

    for i in range(total_batches):
        a_ptr = ctypes.cast(a_base + a_off * 4, FLOAT_PTR)
        b_ptr = ctypes.cast(b_base + b_off * 4, FLOAT_PTR)
        out_ptr = ctypes.cast(out_base + i * out_mat * 4, FLOAT_PTR)

        # Pre-fill this batch slice with broadcast bias
        for r in range(M):
            for c in range(N):
                out_ptr[r * N + c] = bias[c]

        lib.kernel_matmul_beta(a_ptr, b_ptr, out_ptr,
                               M, N, K, 1, trans_b,
                               ctypes.c_float(alpha))

        for d in range(ndim_batch - 1, -1, -1):
            coords[d] += 1
            a_off += a_bstrides[d] * a_mat
            b_off += b_bstrides[d] * b_mat
            if coords[d] < out_batch[d]:
                break
            a_off -= coords[d] * a_bstrides[d] * a_mat
            b_off -= coords[d] * b_bstrides[d] * b_mat
            coords[d] = 0


def _wrap_add(lib: ctypes.CDLL) -> KernelFn:
    """Wrap kernel_add with broadcasting and scalar support."""
    def kernel(inputs: list[np.ndarray], output: np.ndarray,
               attrs: dict[str, Any]) -> None:
        a = inputs[0]
        if "scalar" in attrs:
            lib.kernel_add_scalar(_as_ptr(a), ctypes.c_float(attrs["scalar"]), _as_ptr(output), a.size)
            return
        b = inputs[1]
        if a.shape == b.shape:
            # Same shape — use bias kernel with M=1, N=total (flat loop)
            lib.kernel_add(_as_ptr(a), _as_ptr(b), _as_ptr(output), 1, a.size)
        elif b.size == b.shape[-1] and a.shape[-1] == b.shape[-1]:
            # b is effectively 1D bias [...,N] + [N] — dedicated fast path
            N = b.shape[-1]
            M = a.size // N
            lib.kernel_add(_as_ptr(a), _as_ptr(b), _as_ptr(output), M, N)
        else:
            # General broadcast
            _call_broadcast(lib, a, b, output, _BCAST_ADD)
    return kernel


def _wrap_relu(lib: ctypes.CDLL) -> KernelFn:
    """Wrap kernel_relu into our kernel contract."""
    def kernel(inputs: list[np.ndarray], output: np.ndarray,
               attrs: dict[str, Any]) -> None:
        n = inputs[0].size  # total element count — shape doesn't matter
        lib.kernel_relu(_as_ptr(inputs[0]), _as_ptr(output), n)
    return kernel


def _wrap_transpose(lib: ctypes.CDLL) -> KernelFn:
    """Wrap kernel_transpose into our kernel contract.

    Only handles 2D transposes. Returns None for N-dim so the executor
    falls through to the numpy backend.
    """
    def kernel(inputs: list[np.ndarray], output: np.ndarray,
               attrs: dict[str, Any]) -> None:
        if inputs[0].ndim != 2:
            # N-dim transpose — fall back to numpy
            np.copyto(output, np.swapaxes(inputs[0], attrs["dim0"], attrs["dim1"]))
            return
        rows, cols = inputs[0].shape
        lib.kernel_transpose(_as_ptr(inputs[0]), _as_ptr(output), rows, cols)
    return kernel


_NUMPY_BINOPS = {
    "kernel_div": np.divide,
    "kernel_sub": np.subtract,
    "kernel_mul": np.multiply,
}

_SCALAR_KERNELS = {
    "kernel_div": "kernel_div_scalar",
    "kernel_sub": "kernel_sub_scalar",
    "kernel_mul": "kernel_mul_scalar",
}


def _wrap_elementwise_binary(lib: ctypes.CDLL, fname: str, bcast_op: int) -> KernelFn:
    """Wrap an element-wise binary kernel with broadcasting and scalar support."""
    cfn = getattr(lib, fname)
    scalar_fname = _SCALAR_KERNELS[fname]
    scalar_cfn = getattr(lib, scalar_fname)
    np_fn = _NUMPY_BINOPS[fname]
    def kernel(inputs: list[np.ndarray], output: np.ndarray,
               attrs: dict[str, Any]) -> None:
        a = inputs[0]
        if "scalar" in attrs:
            scalar_cfn(_as_ptr(a), ctypes.c_float(attrs["scalar"]), _as_ptr(output), a.size)
            return
        b = inputs[1]
        if a.shape == b.shape:
            # Same shape — flat kernel, no broadcast overhead
            cfn(_as_ptr(a), _as_ptr(b), _as_ptr(output), a.size)
        else:
            # General broadcast
            _call_broadcast(lib, a, b, output, bcast_op)
    return kernel


def _wrap_exp(lib: ctypes.CDLL) -> KernelFn:
    """Wrap kernel_exp into our kernel contract."""
    def kernel(inputs: list[np.ndarray], output: np.ndarray,
               attrs: dict[str, Any]) -> None:
        n = inputs[0].size
        lib.kernel_exp(_as_ptr(inputs[0]), _as_ptr(output), n)
    return kernel


def _wrap_reduce(lib: ctypes.CDLL, fname: str) -> KernelFn:
    """Wrap a reduction kernel (max, sum) with axis decomposition."""
    cfn = getattr(lib, fname)
    def kernel(inputs: list[np.ndarray], output: np.ndarray,
               attrs: dict[str, Any]) -> None:
        x = inputs[0]
        axis = attrs["axis"]
        outer = 1
        for d in x.shape[:axis]:
            outer *= d
        axis_size = x.shape[axis]
        inner = 1
        for d in x.shape[axis + 1:]:
            inner *= d
        cfn(_as_ptr(x), _as_ptr(output), outer, axis_size, inner)
    return kernel


def _wrap_softmax(lib: ctypes.CDLL) -> KernelFn:
    """Wrap kernel_softmax into our kernel contract."""
    def kernel(inputs: list[np.ndarray], output: np.ndarray,
               attrs: dict[str, Any]) -> None:
        x = inputs[0]
        cols = x.shape[-1]
        rows = x.size // cols
        lib.kernel_softmax(_as_ptr(x), _as_ptr(output), rows, cols)
    return kernel


def _wrap_layernorm(lib: ctypes.CDLL) -> KernelFn:
    """Wrap kernel_layernorm into our kernel contract."""
    def kernel(inputs: list[np.ndarray], output: np.ndarray,
               attrs: dict[str, Any]) -> None:
        x, gamma, beta = inputs[0], inputs[1], inputs[2]
        cols = x.shape[-1]
        rows = x.size // cols
        eps = attrs.get("eps", 1e-5)
        lib.kernel_layernorm(_as_ptr(x), _as_ptr(gamma), _as_ptr(beta),
                             _as_ptr(output), rows, cols, ctypes.c_float(eps))
    return kernel


def _wrap_bias_relu(lib: ctypes.CDLL) -> KernelFn:
    """Wrap kernel_bias_relu: fused bias add + relu in one pass."""
    def kernel(inputs: list[np.ndarray], output: np.ndarray,
               attrs: dict[str, Any]) -> None:
        a, bias = inputs[0], inputs[1]
        N = bias.shape[-1]
        M = a.size // N
        lib.kernel_bias_relu(_as_ptr(a), _as_ptr(bias), _as_ptr(output), M, N)
    return kernel


def _wrap_attention(lib: ctypes.CDLL) -> KernelFn:
    """Wrap kernel_attention: fused multi-head attention.

    Inputs: [Q, K, V, scratch] — scratch is appended by the executor
    from the planner's scratch allocation.
    Q, K, V: [..., seq_len, head_dim] (leading dims are batch)
    """
    def kernel(inputs: list[np.ndarray], output: np.ndarray,
               attrs: dict[str, Any]) -> None:
        Q, K, V, scratch = inputs[0], inputs[1], inputs[2], inputs[3]
        seq_len, head_dim = Q.shape[-2], Q.shape[-1]
        batch_heads = Q.size // (seq_len * head_dim)
        causal = 1 if attrs.get("causal") else 0
        lib.kernel_attention(_as_ptr(Q), _as_ptr(K), _as_ptr(V),
                             _as_ptr(output), _as_ptr(scratch),
                             batch_heads, seq_len, head_dim, causal)
    return kernel


def _wrap_pow_scalar(lib: ctypes.CDLL) -> KernelFn:
    """Wrap kernel_pow_scalar: element-wise x^scalar."""
    def kernel(inputs: list[np.ndarray], output: np.ndarray,
               attrs: dict[str, Any]) -> None:
        lib.kernel_pow_scalar(_as_ptr(inputs[0]),
                              ctypes.c_float(attrs["scalar"]),
                              _as_ptr(output), inputs[0].size)
    return kernel


def _wrap_tanh(lib: ctypes.CDLL) -> KernelFn:
    """Wrap kernel_tanh: element-wise tanh."""
    def kernel(inputs: list[np.ndarray], output: np.ndarray,
               attrs: dict[str, Any]) -> None:
        lib.kernel_tanh(_as_ptr(inputs[0]), _as_ptr(output), inputs[0].size)
    return kernel


def _wrap_gelu(lib: ctypes.CDLL) -> KernelFn:
    """Wrap kernel_gelu_tanh: GELU with tanh approximation."""
    def kernel(inputs: list[np.ndarray], output: np.ndarray,
               attrs: dict[str, Any]) -> None:
        lib.kernel_gelu_tanh(_as_ptr(inputs[0]), _as_ptr(output), inputs[0].size)
    return kernel


def _wrap_embedding(lib: ctypes.CDLL) -> KernelFn:
    """Wrap kernel_embedding: table lookup.

    Inputs: [indices (int64), weight_table (float32)]
    """
    LONG_PTR = ctypes.POINTER(ctypes.c_long)
    def kernel(inputs: list[np.ndarray], output: np.ndarray,
               attrs: dict[str, Any]) -> None:
        ids = inputs[0].astype(np.int64, copy=False)
        table = inputs[1]
        embed_dim = table.shape[-1]
        n_ids = ids.size
        lib.kernel_embedding(ids.ctypes.data_as(LONG_PTR),
                             _as_ptr(table), _as_ptr(output),
                             n_ids, embed_dim)
    return kernel


class CBackend:
    """Backend that dispatches to compiled C kernels."""
    name = "c_cpu"

    def __init__(self) -> None:
        self._lib = _load_library()
        self._kernels: dict[OpType, KernelFn] = {}

        if self._lib is not None:
            self._kernels = {
                OpType.MATMUL: _wrap_matmul(self._lib),
                OpType.MATMUL_ADD: _wrap_matmul_add(self._lib),
                OpType.ADD: _wrap_add(self._lib),
                OpType.RELU: _wrap_relu(self._lib),
                OpType.TRANSPOSE: _wrap_transpose(self._lib),
                OpType.DIV: _wrap_elementwise_binary(self._lib, "kernel_div", _BCAST_DIV),
                OpType.SUB: _wrap_elementwise_binary(self._lib, "kernel_sub", _BCAST_SUB),
                OpType.MUL: _wrap_elementwise_binary(self._lib, "kernel_mul", _BCAST_MUL),
                OpType.EXP: _wrap_exp(self._lib),
                OpType.MAX: _wrap_reduce(self._lib, "kernel_max"),
                OpType.SUM: _wrap_reduce(self._lib, "kernel_sum"),
                OpType.SOFTMAX: _wrap_softmax(self._lib),
                OpType.LAYERNORM: _wrap_layernorm(self._lib),
                OpType.FUSED_BIAS_RELU: _wrap_bias_relu(self._lib),
                OpType.ATTENTION: _wrap_attention(self._lib),
                OpType.POW: _wrap_pow_scalar(self._lib),
                OpType.TANH: _wrap_tanh(self._lib),
                OpType.GELU: _wrap_gelu(self._lib),
                OpType.EMBEDDING: _wrap_embedding(self._lib),
            }

    @property
    def available(self) -> bool:
        return self._lib is not None

    def get_kernel(self, op: OpType) -> KernelFn | None:
        return self._kernels.get(op)
