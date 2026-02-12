"""Backend op correctness: each op × both backends × various shapes.

Tests kernels directly (not through the graph pipeline) to isolate
backend-level issues. Each parametrized case is a separate test with
a clear name when it fails.
"""

import numpy as np
import pytest

from conftest import run_kernel


# =====================================================================
# MATMUL
# =====================================================================

MATMUL_CASES = [
    ("2Dx2D",          (4, 8),          (8, 6),          {}),
    ("2Dx2D_transb",   (4, 8),          (6, 8),          {"transpose_b": True}),
    ("3Dx2D",          (2, 16, 64),     (64, 32),        {}),
    ("3Dx2D_transb",   (2, 16, 64),     (32, 64),        {"transpose_b": True}),
    ("4Dx2D",          (2, 4, 16, 64),  (64, 32),        {}),
    ("4Dx2D_transb",   (2, 4, 16, 64),  (32, 64),        {"transpose_b": True}),
    ("3Dx3D_batched",  (2, 16, 64),     (2, 64, 16),     {}),
    ("4Dx4D_batched",  (2, 4, 16, 16),  (2, 4, 16, 16),  {}),
    ("4Dx4D_transb",   (2, 4, 16, 16),  (2, 4, 16, 16),  {"transpose_b": True}),
    ("2Dx3D",          (4, 8),          (3, 8, 6),       {}),
    ("5Dx2D",          (2, 3, 4, 8, 16), (16, 6),        {}),
    ("4Dx3D_bcast",    (2, 4, 8, 16),   (4, 16, 8),      {}),
]


@pytest.mark.parametrize("desc,a_shape,b_shape,attrs",
                         MATMUL_CASES, ids=[c[0] for c in MATMUL_CASES])
def test_matmul(backend, desc, a_shape, b_shape, attrs):
    a = np.random.randn(*a_shape).astype(np.float32)
    b = np.random.randn(*b_shape).astype(np.float32)

    if attrs.get("transpose_b"):
        expected = np.matmul(a, np.swapaxes(b, -2, -1))
    else:
        expected = np.matmul(a, b)

    output = run_kernel(backend, "MATMUL", [a, b], expected.shape, attrs)
    np.testing.assert_allclose(output, expected, atol=1e-4)


# =====================================================================
# ADD (broadcasting)
# =====================================================================

ADD_CASES = [
    ("3D_same",     (2, 16, 64),     (2, 16, 64)),
    ("3D_bias",     (2, 16, 64),     (64,)),
    ("4D_bias",     (2, 4, 16, 64),  (64,)),
    ("4D_keepdim",  (2, 4, 16, 16),  (2, 4, 16, 1)),
]


@pytest.mark.parametrize("desc,a_shape,b_shape",
                         ADD_CASES, ids=[c[0] for c in ADD_CASES])
def test_add(backend, desc, a_shape, b_shape):
    a = np.random.randn(*a_shape).astype(np.float32)
    b = np.random.randn(*b_shape).astype(np.float32)
    expected = a + b
    output = run_kernel(backend, "ADD", [a, b], expected.shape)
    np.testing.assert_allclose(output, expected, atol=1e-5)


# =====================================================================
# Element-wise binary ops (broadcasting)
# =====================================================================

BINOP_CASES = [
    ("same_shape",     (2, 4, 16, 16),    (2, 4, 16, 16)),
    ("keepdim",        (2, 4, 16, 16),    (2, 4, 16, 1)),
    ("5D_weird_bcast", (2, 3, 4, 5, 6),   (2, 1, 4, 1, 1)),
]


@pytest.mark.parametrize("op_name,np_op", [("SUB", "subtract"), ("MUL", "multiply"), ("DIV", "divide")])
@pytest.mark.parametrize("desc,a_shape,b_shape",
                         BINOP_CASES, ids=[c[0] for c in BINOP_CASES])
def test_elementwise_binary(backend, op_name, np_op, desc, a_shape, b_shape):
    a = np.random.randn(*a_shape).astype(np.float32)
    b = np.random.randn(*b_shape).astype(np.float32)
    if op_name == "DIV":
        b = np.abs(b) + 0.1  # avoid division by zero

    expected = getattr(np, np_op)(a, b)
    output = run_kernel(backend, op_name, [a, b], expected.shape)
    np.testing.assert_allclose(output, expected, atol=1e-5)


# =====================================================================
# Unary ops
# =====================================================================

def test_relu(backend):
    x = np.random.randn(2, 16, 64).astype(np.float32)
    expected = np.maximum(x, 0)
    output = run_kernel(backend, "RELU", [x], x.shape)
    np.testing.assert_allclose(output, expected, atol=1e-6)


def test_exp(backend):
    x = np.random.randn(2, 16, 64).astype(np.float32)
    expected = np.exp(x)
    output = run_kernel(backend, "EXP", [x], x.shape)
    np.testing.assert_allclose(output, expected, atol=1e-5)


# =====================================================================
# Reductions
# =====================================================================

REDUCE_CASES = [
    ("4D_last_keepdim",  (2, 4, 16, 16),  3, True),
    ("4D_middle",        (2, 4, 16, 16),  2, False),
    ("4D_axis1",         (2, 4, 16, 16),  1, False),
]


@pytest.mark.parametrize("op_name,np_fn", [("MAX", np.max), ("SUM", np.sum)])
@pytest.mark.parametrize("desc,shape,axis,keepdim",
                         REDUCE_CASES, ids=[c[0] for c in REDUCE_CASES])
def test_reduction(backend, op_name, np_fn, desc, shape, axis, keepdim):
    x = np.random.randn(*shape).astype(np.float32)
    expected = np_fn(x, axis=axis, keepdims=keepdim)
    attrs = {"axis": axis, "keepdim": keepdim}
    output = run_kernel(backend, op_name, [x], expected.shape, attrs)
    np.testing.assert_allclose(output, expected, atol=1e-5)


# =====================================================================
# Softmax
# =====================================================================

@pytest.mark.parametrize("shape,axis", [
    ((2, 4, 16, 16), 3),
    ((2, 16, 64), 2),
])
def test_softmax(backend, shape, axis):
    x = np.random.randn(*shape).astype(np.float32)
    # Reference: numerically stable softmax
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    expected = e / np.sum(e, axis=axis, keepdims=True)

    output = run_kernel(backend, "SOFTMAX", [x], x.shape, {"axis": axis})
    np.testing.assert_allclose(output, expected, atol=1e-5)
