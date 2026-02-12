"""Numpy-based backend: pure Python fallback for all ops.

Uses numpy's `out=` parameter to write directly into pre-allocated
arena buffers — no temporary allocations, no copies. These are
separate from the constant folding evaluators in passes.py, which
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
    # Q, K, V: [batch_heads, seq_len, head_dim]
    head_dim = Q.shape[-1]
    scale = 1.0 / np.sqrt(head_dim)
    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) * scale
    # numerically stable softmax along last axis
    scores -= np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores)
    weights = e / np.sum(e, axis=-1, keepdims=True)
    np.matmul(weights, V, out=output)


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
}


class NumpyBackend:
    """Backend that dispatches to in-place numpy kernels."""
    name = "numpy"

    def get_kernel(self, op: OpType) -> KernelFn | None:
        return _KERNELS.get(op)
