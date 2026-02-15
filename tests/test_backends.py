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


# =====================================================================
# POW (scalar exponent)
# =====================================================================

POW_CASES = [
    ("square",  (4, 16, 64),  2.0),
    ("cube",    (2, 8, 32),   3.0),
    ("sqrt",    (4, 16, 64),  0.5),
    ("inverse", (2, 8, 32),  -1.0),
    ("2D",      (32, 64),     2.0),
]


@pytest.mark.parametrize("desc,shape,exponent",
                         POW_CASES, ids=[c[0] for c in POW_CASES])
def test_pow(backend, desc, shape, exponent):
    x = np.abs(np.random.randn(*shape).astype(np.float32)) + 0.1  # positive for fractional/negative exp
    expected = np.power(x, exponent)
    output = run_kernel(backend, "POW", [x], x.shape, {"scalar": exponent})
    np.testing.assert_allclose(output, expected, atol=1e-4)


# =====================================================================
# TANH
# =====================================================================

@pytest.mark.parametrize("shape", [(4, 16, 64), (2, 8, 32), (128,)])
def test_tanh(backend, shape):
    x = np.random.randn(*shape).astype(np.float32)
    expected = np.tanh(x)
    output = run_kernel(backend, "TANH", [x], x.shape)
    np.testing.assert_allclose(output, expected, atol=1e-5)


# =====================================================================
# GELU (tanh approximation)
# =====================================================================

@pytest.mark.parametrize("shape", [(4, 16, 64), (2, 8, 32), (128,)])
def test_gelu(backend, shape):
    x = np.random.randn(*shape).astype(np.float32)
    # Reference: tanh GELU approximation
    inner = 0.7978845608 * (x + 0.044715 * np.power(x, 3))
    expected = 0.5 * x * (1 + np.tanh(inner))
    output = run_kernel(backend, "GELU", [x], x.shape)
    np.testing.assert_allclose(output, expected, atol=1e-4)


# =====================================================================
# EMBEDDING
# =====================================================================

EMBEDDING_CASES = [
    ("small_1d",    8,  32, (4,)),
    ("small_2d",    8,  16, (2, 3)),
    ("medium",      64, 128, (4, 8)),
]


@pytest.mark.parametrize("desc,vocab,dim,ids_shape",
                         EMBEDDING_CASES, ids=[c[0] for c in EMBEDDING_CASES])
def test_embedding(backend, desc, vocab, dim, ids_shape):
    table = np.random.randn(vocab, dim).astype(np.float32)
    ids = np.random.randint(0, vocab, size=ids_shape).astype(np.int64)
    expected = table[ids]
    output_shape = ids_shape + (dim,)
    output = run_kernel(backend, "EMBEDDING", [ids, table], output_shape)
    np.testing.assert_allclose(output, expected, atol=1e-6)


# =====================================================================
# SLICE (byte offset view)
# =====================================================================

SLICE_CASES = [
    ("chunk0_of_3", (12, 8), 0, 4),   # first 4 rows of a (12,8) tensor
    ("chunk1_of_3", (12, 8), 1, 4),   # middle 4 rows
    ("chunk2_of_3", (12, 8), 2, 4),   # last 4 rows
    ("3D_last_dim", (2, 4, 12), 1, 4),  # split along last dim (contiguous chunks)
]


@pytest.mark.parametrize("desc,input_shape,chunk_idx,chunk_size",
                         SLICE_CASES, ids=[c[0] for c in SLICE_CASES])
def test_slice(np_backend, desc, input_shape, chunk_idx, chunk_size):
    """SLICE correctness via the executor per-op path.

    Alias/in-place sharing only applies to arena intermediates, not graph
    inputs — so SLICE on a graph input always gets arena space and a copy.
    We just check that the output values are correct.
    """
    from runtime.ir import Graph, OpType
    from runtime.planner import plan
    from runtime.executor import InterpretedExecutor
    from runtime.backends.numpy_backend import NumpyBackend

    x = np.random.randn(*input_shape).astype(np.float32)

    ndim = len(input_shape)
    if ndim == 2:
        dim = 0
        trailing = input_shape[1]
        byte_offset = chunk_idx * chunk_size * trailing * 4
        expected = x[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size, :]
        out_shape = (chunk_size, input_shape[1])
    else:
        dim = ndim - 1
        trailing = 1
        byte_offset = chunk_idx * chunk_size * trailing * 4
        expected = x[..., chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
        out_shape = input_shape[:-1] + (chunk_size,)

    start = chunk_idx * chunk_size
    end = start + chunk_size
    attrs = {"byte_offset": byte_offset, "dim": dim, "start": start, "end": end}

    g = Graph()
    g.add_tensor("x", input_shape)
    g.inputs.append("x")
    g.add_tensor("sliced", out_shape)
    g.add_node(OpType.SLICE, ["x"], "sliced", attrs)
    g.add_tensor("out", out_shape)
    g.add_node(OpType.ADD, ["sliced"], "out", {"scalar": 0.0})
    g.outputs.append("out")

    ep = plan(g)

    executor = InterpretedExecutor(backends=[NumpyBackend()])
    executor.compile(ep)
    result = executor.run({"x": x})
    np.testing.assert_allclose(result["out"], expected, atol=1e-6)


def test_slice_alias_on_intermediate():
    """dim=0 SLICE on an arena intermediate should alias (no extra allocation).

    Graph: input → RELU → intermediate → SLICE → sliced → ADD(0) → out

    The SLICE's input is an intermediate (RELU output), so the planner can
    alias it. The sliced tensor should share the intermediate's arena offset
    (shifted by byte_offset), not get its own allocation.
    """
    from runtime.ir import Graph, OpType
    from runtime.planner import plan
    from runtime.executor import InterpretedExecutor
    from runtime.backends.numpy_backend import NumpyBackend

    x = np.random.randn(12, 8).astype(np.float32)
    chunk_size, chunk_idx = 4, 1
    byte_offset = chunk_idx * chunk_size * 8 * 4  # rows × cols × float32

    g = Graph()
    g.add_tensor("x", (12, 8))
    g.inputs.append("x")
    g.add_tensor("relu_out", (12, 8))
    g.add_node(OpType.RELU, ["x"], "relu_out")
    g.add_tensor("sliced", (chunk_size, 8))
    g.add_node(OpType.SLICE, ["relu_out"], "sliced",
               {"byte_offset": byte_offset, "dim": 0, "start": 4, "end": 8})
    g.add_tensor("out", (chunk_size, 8))
    g.add_node(OpType.ADD, ["sliced"], "out", {"scalar": 0.0})
    g.outputs.append("out")

    ep = plan(g)

    # sliced should alias relu_out at byte_offset, no extra allocation
    assert ep.offsets["sliced"] == ep.offsets["relu_out"] + byte_offset, \
        "dim=0 SLICE should point into its input's arena region"
    # Arena should only hold relu_out (384B) + out (128B) = 512, not 640
    assert ep.arena_size == 12 * 8 * 4 + chunk_size * 8 * 4

    # Correctness
    expected = np.maximum(x, 0)[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size, :]
    executor = InterpretedExecutor(backends=[NumpyBackend()])
    executor.compile(ep)
    result = executor.run({"x": x})
    np.testing.assert_allclose(result["out"], expected, atol=1e-6)
