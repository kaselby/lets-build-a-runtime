"""Numpy-based backend: pure Python fallback for all ops.

Uses numpy's `out=` parameter to write directly into pre-allocated
arena buffers — no temporary allocations, no copies. These are
separate from the constant folding evaluators in ops.py, which
need the return-a-new-array contract.
"""

from typing import Any

import numpy as np

from ..ir import OpType
from ..executor import KernelFn


# In-place kernels: (inputs, output, attrs) -> None
# Each writes directly into the output buffer via out= parameter.

def _matmul(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    b = np.swapaxes(inputs[1], -2, -1) if attrs.get("transpose_b") else inputs[1]
    np.matmul(inputs[0], b, out=output)
    alpha = attrs.get("alpha")
    if alpha is not None and alpha != 1.0:
        np.multiply(output, alpha, out=output)


def _add(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    rhs = attrs["scalar"] if "scalar" in attrs else inputs[1]
    np.add(inputs[0], rhs, out=output)


def _relu(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    np.maximum(inputs[0], 0, out=output)


def _transpose(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    # swapaxes returns a view, so we need to copy into output
    np.copyto(output, np.swapaxes(inputs[0], attrs["dim0"], attrs["dim1"]))


def _permute(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    # transpose returns a view, so we need to copy into output
    np.copyto(output, np.transpose(inputs[0], attrs["axes"]))


def _div(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    rhs = attrs["scalar"] if "scalar" in attrs else inputs[1]
    np.divide(inputs[0], rhs, out=output)


def _sub(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    rhs = attrs["scalar"] if "scalar" in attrs else inputs[1]
    np.subtract(inputs[0], rhs, out=output)


def _mul(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    rhs = attrs["scalar"] if "scalar" in attrs else inputs[1]
    np.multiply(inputs[0], rhs, out=output)


def _exp(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    np.exp(inputs[0], out=output)


def _max(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    result = np.max(inputs[0], axis=attrs["axis"], keepdims=attrs.get("keepdim", False))
    np.copyto(output, result)


def _sum(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    result = np.sum(inputs[0], axis=attrs["axis"], keepdims=attrs.get("keepdim", False))
    np.copyto(output, result)


def _softmax(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    x = inputs[0]
    axis = attrs["axis"]
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    np.divide(e, np.sum(e, axis=axis, keepdims=True), out=output)


def _reshape(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    np.copyto(output, inputs[0].reshape(output.shape))


def _layernorm(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    x, gamma, beta = inputs[0], inputs[1], inputs[2]
    eps = attrs.get("eps", 1e-5)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    np.copyto(output, (x - mean) / np.sqrt(var + eps) * gamma + beta)


def _matmul_add(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    a, b, bias = inputs[0], inputs[1], inputs[2]
    b = np.swapaxes(b, -2, -1) if attrs.get("transpose_b") else b
    np.matmul(a, b, out=output)
    np.add(output, bias, out=output)


def _fused_bias_relu(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    np.add(inputs[0], inputs[1], out=output)
    np.maximum(output, 0, out=output)


def _attention(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    Q, K, V = inputs[0], inputs[1], inputs[2]
    # Q, K, V: [..., seq_len, head_dim]
    group_size = attrs.get("group_size", 1)
    if group_size > 1:
        # GQA: repeat each KV head group_size times to match Q head count
        K = np.repeat(K, group_size, axis=-3)
        V = np.repeat(V, group_size, axis=-3)
    head_dim = Q.shape[-1]
    scale = 1.0 / np.sqrt(head_dim)
    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) * scale
    if attrs.get("causal"):
        seq_len = Q.shape[-2]
        mask = np.triu(np.full((seq_len, seq_len), -np.inf, dtype=np.float32), k=1)
        scores = scores + mask
    # numerically stable softmax along last axis
    scores -= np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores)
    weights = e / np.sum(e, axis=-1, keepdims=True)
    np.matmul(weights, V, out=output)


def _pow(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    np.power(inputs[0], attrs["scalar"], out=output)


def _tanh(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    np.tanh(inputs[0], out=output)


def _gelu(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    x = inputs[0]
    inner = 0.7978845608 * (x + 0.044715 * np.power(x, 3))
    np.multiply(0.5 * x, 1.0 + np.tanh(inner), out=output)


def _slice(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    x = inputs[0]
    dim = attrs["dim"]
    start, end = attrs["start"], attrs["end"]
    slices = tuple(slice(start, end) if d == dim else slice(None) for d in range(x.ndim))
    np.copyto(output, x[slices])


def _embedding(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    ids, table = inputs[0], inputs[1]
    output[:] = table[ids.astype(np.intp)]


def _rsqrt(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    np.divide(1.0, np.sqrt(inputs[0]), out=output)


def _silu(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    x = inputs[0]
    np.divide(x, 1.0 + np.exp(-x), out=output)


def _neg(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    np.negative(inputs[0], out=output)


def _cos(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    np.cos(inputs[0], out=output)


def _sin(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    np.sin(inputs[0], out=output)


def _rmsnorm(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    x, weight = inputs[0], inputs[1]
    eps = attrs.get("eps", 1e-5)
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    np.copyto(output, (x / rms) * weight)


def _cat(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    np.copyto(output, np.concatenate(inputs, axis=attrs["dim"]))


def _gated_act(inputs: list[np.ndarray], output: np.ndarray, attrs: dict[str, Any]) -> None:
    has_bias = attrs.get("has_bias", False)
    act = attrs.get("act", "silu")
    if has_bias:
        x, bias, up = inputs[0], inputs[1], inputs[2]
        v = x + bias
    else:
        x, up = inputs[0], inputs[1]
        v = x
    if act == "silu":
        activated = v / (1.0 + np.exp(-v))
    else:
        activated = 0.5 * v * (1 + np.tanh(0.7978845608 * (v + 0.044715 * v**3)))
    np.multiply(activated, up, out=output)


_KERNELS: dict[OpType, KernelFn] = {
    OpType.MATMUL: _matmul,
    OpType.ADD: _add,
    OpType.RELU: _relu,
    OpType.TRANSPOSE: _transpose,
    OpType.PERMUTE: _permute,
    OpType.DIV: _div,
    OpType.SUB: _sub,
    OpType.MUL: _mul,
    OpType.EXP: _exp,
    OpType.MAX: _max,
    OpType.SUM: _sum,
    OpType.SOFTMAX: _softmax,
    OpType.RESHAPE: _reshape,
    OpType.LAYERNORM: _layernorm,
    OpType.MATMUL_ADD: _matmul_add,
    OpType.FUSED_BIAS_RELU: _fused_bias_relu,
    OpType.ATTENTION: _attention,
    OpType.POW: _pow,
    OpType.TANH: _tanh,
    OpType.GELU: _gelu,
    OpType.EMBEDDING: _embedding,
    OpType.SLICE: _slice,
    OpType.RSQRT: _rsqrt,
    OpType.SILU: _silu,
    OpType.NEG: _neg,
    OpType.COS: _cos,
    OpType.SIN: _sin,
    OpType.RMSNORM: _rmsnorm,
    OpType.CAT: _cat,
    OpType.GATED_ACT: _gated_act,
}


class NumpyBackend:
    """Backend that dispatches to in-place numpy kernels."""
    name = "numpy"

    def get_kernel(self, op: OpType) -> KernelFn | None:
        return _KERNELS.get(op)
