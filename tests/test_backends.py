"""Backend op correctness: each op × both backends × various shapes.

Tests kernels directly (not through the graph pipeline) to isolate
backend-level issues. Each parametrized case is a separate test with
a clear name when it fails.
"""

import numpy as np
import pytest

from runtime.ir import OpType
from runtime.ops import OP_REGISTRY
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
# Simple unary ops (automated: evaluator is the oracle)
# =====================================================================

UNARY_SHAPES = [(4, 16, 64), (2, 8, 32), (128,)]

SIMPLE_UNARY_OPS = [
    # (op,           positive_inputs, atol)
    (OpType.RELU,    False, 1e-6),
    (OpType.EXP,     False, 1e-5),
    (OpType.TANH,    False, 1e-5),
    (OpType.GELU,    False, 1e-4),
    (OpType.RSQRT,   True,  1e-5),
    (OpType.SILU,    False, 1e-5),
    (OpType.NEG,     False, 1e-6),
    (OpType.COS,     False, 1e-5),
    (OpType.SIN,     False, 1e-5),
]

_UNARY_CASES = [
    (op, shape, positive, atol)
    for op, positive, atol in SIMPLE_UNARY_OPS
    for shape in UNARY_SHAPES
]


@pytest.mark.parametrize("op,shape,positive,atol", _UNARY_CASES,
                         ids=[f"{op.name}_{shape}" for op, shape, _, _ in _UNARY_CASES])
def test_unary_op(backend, op, shape, positive, atol):
    """Automated: C kernel vs OP_REGISTRY evaluator for simple unary ops."""
    x = np.abs(np.random.randn(*shape).astype(np.float32)) + 0.01 if positive \
        else np.random.randn(*shape).astype(np.float32)
    expected = OP_REGISTRY[op].evaluator([x], {})
    output = run_kernel(backend, op.name, [x], expected.shape, {})
    np.testing.assert_allclose(output, expected, atol=atol)


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


# =====================================================================
# RMSNORM
# =====================================================================

RMSNORM_CASES = [
    ("2D",       (4, 64)),
    ("3D",       (2, 16, 128)),
    ("large",    (32, 896)),
]


@pytest.mark.parametrize("desc,shape",
                         RMSNORM_CASES, ids=[c[0] for c in RMSNORM_CASES])
def test_rmsnorm(backend, desc, shape):
    x = np.random.randn(*shape).astype(np.float32)
    w = np.random.randn(shape[-1]).astype(np.float32)
    eps = 1e-5
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    expected = (x / rms) * w
    output = run_kernel(backend, "RMSNORM", [x, w], x.shape, {"eps": eps})
    np.testing.assert_allclose(output, expected, atol=1e-5)


# =====================================================================
# CAT
# =====================================================================

CAT_CASES = [
    ("dim0_2",       [(3, 4), (5, 4)],              0),
    ("dim1_2",       [(2, 3), (2, 5)],               1),
    ("dim1_3D",      [(2, 3, 4), (2, 5, 4)],         1),
    ("dim2_3D",      [(1, 16, 32), (1, 16, 32)],     2),
    ("dim0_3",       [(2, 4), (3, 4), (1, 4)],       0),
]


@pytest.mark.parametrize("desc,shapes,dim",
                         CAT_CASES, ids=[c[0] for c in CAT_CASES])
def test_cat(backend, desc, shapes, dim):
    arrays = [np.random.randn(*s).astype(np.float32) for s in shapes]
    expected = np.concatenate(arrays, axis=dim)
    output = run_kernel(backend, "CAT", arrays, expected.shape, {"dim": dim})
    np.testing.assert_allclose(output, expected, atol=1e-6)


# =====================================================================
# GATED_ACT (fused activation-gated multiply)
# =====================================================================

GATED_ACT_CASES = [
    ("silu_nobias_2D",   (8, 64),        "silu", False),
    ("silu_nobias_3D",   (2, 16, 128),   "silu", False),
    ("silu_bias_2D",     (8, 64),        "silu", True),
    ("silu_bias_3D",     (2, 16, 128),   "silu", True),
    ("gelu_nobias_2D",   (8, 64),        "gelu", False),
    ("gelu_nobias_3D",   (2, 16, 128),   "gelu", False),
    ("gelu_bias_2D",     (8, 64),        "gelu", True),
    ("gelu_bias_3D",     (2, 16, 128),   "gelu", True),
]


@pytest.mark.parametrize("desc,shape,act,has_bias",
                         GATED_ACT_CASES, ids=[c[0] for c in GATED_ACT_CASES])
def test_gated_act(c_backend, desc, shape, act, has_bias):
    x = np.random.randn(*shape).astype(np.float32)
    up = np.random.randn(*shape).astype(np.float32)
    attrs = {"act": act, "has_bias": has_bias}

    if has_bias:
        bias = np.random.randn(shape[-1]).astype(np.float32)
        inputs = [x, bias, up]
    else:
        inputs = [x, up]

    expected = OP_REGISTRY[OpType.GATED_ACT].evaluator(inputs, attrs)
    output = run_kernel(c_backend, "GATED_ACT", inputs, expected.shape, attrs)
    np.testing.assert_allclose(output, expected, atol=1e-4)


# =====================================================================
# GQA ATTENTION
# =====================================================================

GQA_CASES = [
    # desc, B, n_q_heads, n_kv_heads, seq_len, head_dim, causal
    ("group2_nocausal",  1, 8,  4, 16, 32, False),
    ("group2_causal",    1, 8,  4, 16, 32, True),
    ("group4",           1, 16, 4, 16, 32, False),
    ("batched_group2",   2, 8,  4, 16, 32, False),
    ("batched_causal",   2, 16, 8, 32, 64, True),
    # Long sequences — exercise flash attention path (seq > 256)
    ("flash_nocausal",   1, 8,  4, 512, 64, False),
    ("flash_causal",     1, 8,  4, 512, 64, True),
]


@pytest.mark.parametrize("desc,B,n_q,n_kv,S,D,causal",
                         GQA_CASES, ids=[c[0] for c in GQA_CASES])
def test_gqa_attention(backend, desc, B, n_q, n_kv, S, D, causal):
    """GQA attention: Q has more heads than K/V, group_size > 1."""
    group_size = n_q // n_kv

    Q = np.random.randn(B, n_q, S, D).astype(np.float32)
    K = np.random.randn(B, n_kv, S, D).astype(np.float32)
    V = np.random.randn(B, n_kv, S, D).astype(np.float32)
    scratch = np.zeros(B * n_q * S * S, dtype=np.float32)

    # Reference: expand K/V then standard attention
    K_exp = np.repeat(K, group_size, axis=1)
    V_exp = np.repeat(V, group_size, axis=1)
    scale = 1.0 / np.sqrt(D)
    scores = np.matmul(Q, np.swapaxes(K_exp, -2, -1)) * scale
    if causal:
        mask = np.triu(np.full((S, S), -np.inf, dtype=np.float32), k=1)
        scores = scores + mask
    scores -= np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores)
    expected = np.matmul(e / np.sum(e, axis=-1, keepdims=True), V_exp)

    attrs = {"group_size": group_size}
    if causal:
        attrs["causal"] = True
    output = run_kernel(backend, "ATTENTION", [Q, K, V, scratch],
                        expected.shape, attrs)
    np.testing.assert_allclose(output, expected, atol=1e-4)
