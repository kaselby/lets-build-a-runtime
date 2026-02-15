"""OpDef registry tests: centralized per-op metadata.

Tests that OP_REGISTRY entries are correct and that the OpDef machinery
(alias predicates, extras packers, evaluators) works as expected.
"""

import struct

import numpy as np
import pytest

from runtime.ir import Graph, Node, OpType
from runtime.ops import OP_REGISTRY, OpDef, _float_bits


# ---------------------------------------------------------------------------
# Registry coverage
# ---------------------------------------------------------------------------

# Ops that should have registry entries (non-exhaustive — core ops)
CORE_OPS = [
    OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV,
    OpType.RELU, OpType.EXP, OpType.TANH, OpType.GELU, OpType.POW,
    OpType.MATMUL, OpType.MATMUL_ADD,
    OpType.SOFTMAX, OpType.LAYERNORM,
    OpType.RESHAPE, OpType.TRANSPOSE, OpType.SLICE,
    OpType.MAX, OpType.SUM,
    OpType.ATTENTION, OpType.FUSED_BIAS_RELU,
    OpType.EMBEDDING,
]


@pytest.mark.parametrize("op", CORE_OPS, ids=[op.name for op in CORE_OPS])
def test_core_ops_registered(op):
    """All core ops should have entries in OP_REGISTRY."""
    assert op in OP_REGISTRY, f"{op.name} missing from OP_REGISTRY"


# ---------------------------------------------------------------------------
# Alias predicate
# ---------------------------------------------------------------------------

class TestAlias:

    def test_reshape_is_always_alias(self):
        """RESHAPE should be an alias for any node."""
        op_def = OP_REGISTRY[OpType.RESHAPE]
        node = Node(id=0, op=OpType.RESHAPE, inputs=["x"], output="y", attrs={"shape": (4, 4)})
        assert op_def.is_alias(node) is True

    def test_slice_dim0_is_alias(self):
        """SLICE with dim=0 should be an alias (contiguous, zero-copy)."""
        op_def = OP_REGISTRY[OpType.SLICE]
        node = Node(id=0, op=OpType.SLICE, inputs=["x"], output="y",
                    attrs={"dim": 0, "byte_offset": 0, "start": 0, "end": 4})
        assert op_def.is_alias(node) is True

    def test_slice_nonzero_dim_not_alias(self):
        """SLICE with dim>0 should NOT be an alias (needs kernel copy)."""
        op_def = OP_REGISTRY[OpType.SLICE]
        node = Node(id=0, op=OpType.SLICE, inputs=["x"], output="y",
                    attrs={"dim": 2, "byte_offset": 0, "start": 0, "end": 4})
        assert op_def.is_alias(node) is False

    def test_matmul_not_alias(self):
        """Compute ops should not be aliases."""
        op_def = OP_REGISTRY[OpType.MATMUL]
        node = Node(id=0, op=OpType.MATMUL, inputs=["a", "b"], output="y", attrs={})
        assert op_def.is_alias(node) is False


# ---------------------------------------------------------------------------
# In-place flag
# ---------------------------------------------------------------------------

INPLACE_OPS = [OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV,
               OpType.RELU, OpType.EXP, OpType.TANH, OpType.POW]
NOT_INPLACE_OPS = [OpType.MATMUL, OpType.SOFTMAX, OpType.LAYERNORM,
                   OpType.ATTENTION, OpType.RESHAPE, OpType.GELU]


@pytest.mark.parametrize("op", INPLACE_OPS, ids=[op.name for op in INPLACE_OPS])
def test_elementwise_ops_are_inplace(op):
    """Element-wise ops should have inplace=True."""
    assert OP_REGISTRY[op].inplace is True


@pytest.mark.parametrize("op", NOT_INPLACE_OPS, ids=[op.name for op in NOT_INPLACE_OPS])
def test_non_elementwise_ops_not_inplace(op):
    """Non-elementwise ops should have inplace=False."""
    assert OP_REGISTRY[op].inplace is False


# ---------------------------------------------------------------------------
# Float bits utility
# ---------------------------------------------------------------------------

def test_float_bits_roundtrip():
    """_float_bits should produce a value that reconstructs the original float."""
    for val in [0.0, 1.0, -1.0, 3.14159, 1e-6, float('inf')]:
        bits = _float_bits(val)
        recovered = struct.unpack('f', struct.pack('i', bits))[0]
        assert recovered == pytest.approx(val), f"Roundtrip failed for {val}"


# ---------------------------------------------------------------------------
# Evaluators (constant folding)
# ---------------------------------------------------------------------------

class TestEvaluators:

    def test_add_evaluator(self):
        """ADD evaluator should compute element-wise addition."""
        op_def = OP_REGISTRY[OpType.ADD]
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)
        result = op_def.evaluator([a, b], {})
        np.testing.assert_allclose(result, [5, 7, 9])

    def test_add_evaluator_scalar(self):
        """ADD evaluator with scalar attr should add a constant."""
        op_def = OP_REGISTRY[OpType.ADD]
        a = np.array([1, 2, 3], dtype=np.float32)
        result = op_def.evaluator([a], {"scalar": 10.0})
        np.testing.assert_allclose(result, [11, 12, 13])

    def test_relu_evaluator(self):
        """RELU evaluator should clamp negatives to zero."""
        op_def = OP_REGISTRY[OpType.RELU]
        x = np.array([-1, 0, 1, -5, 3], dtype=np.float32)
        result = op_def.evaluator([x], {})
        np.testing.assert_allclose(result, [0, 0, 1, 0, 3])

    def test_reshape_evaluator(self):
        """RESHAPE evaluator should change shape without moving data."""
        op_def = OP_REGISTRY[OpType.RESHAPE]
        x = np.arange(12, dtype=np.float32)
        result = op_def.evaluator([x], {"shape": (3, 4)})
        assert result.shape == (3, 4)
        np.testing.assert_allclose(result.ravel(), x)

    def test_exp_evaluator(self):
        """EXP evaluator should compute element-wise exponential."""
        op_def = OP_REGISTRY[OpType.EXP]
        x = np.array([0, 1, -1], dtype=np.float32)
        result = op_def.evaluator([x], {})
        np.testing.assert_allclose(result, np.exp(x), atol=1e-6)


# ---------------------------------------------------------------------------
# Scratch calculators
# ---------------------------------------------------------------------------

class TestScratchCalculators:

    def test_attention_has_scratch(self):
        """ATTENTION should have a scratch calculator."""
        op_def = OP_REGISTRY[OpType.ATTENTION]
        assert op_def.scratch is not None

    def test_attention_scratch_size(self):
        """ATTENTION scratch should be batch_heads * S^2 * sizeof(float)."""
        op_def = OP_REGISTRY[OpType.ATTENTION]
        # Q shape: [B=2, H=4, S=16, D=32]
        in_shapes = [(2, 4, 16, 32), (2, 4, 16, 32), (2, 4, 16, 32)]
        out_shape = (2, 4, 16, 32)
        size = op_def.scratch(in_shapes, out_shape, {})
        # batch_heads = 8, S = 16, sizeof(float) = 4
        expected = 8 * 16 * 16 * 4
        assert size == expected

    def test_matmul_no_scratch(self):
        """MATMUL should not have a scratch calculator."""
        op_def = OP_REGISTRY[OpType.MATMUL]
        assert op_def.scratch is None

    def test_relu_no_scratch(self):
        """Element-wise ops should not need scratch."""
        op_def = OP_REGISTRY[OpType.RELU]
        assert op_def.scratch is None


# ---------------------------------------------------------------------------
# Extras packers
# ---------------------------------------------------------------------------

class TestExtras:

    def test_matmul_extras(self):
        """MATMUL extras should pack K, transpose_b flag, and nd_x_2d flag."""
        op_def = OP_REGISTRY[OpType.MATMUL]
        assert op_def.extras is not None

        g = Graph()
        g.add_tensor("a", (4, 8))
        g.inputs.append("a")
        g.add_tensor("b", (8, 16))
        g.tensors["b"].buffer = np.zeros((8, 16), dtype=np.float32)
        g.constants.append("b")
        g.add_tensor("y", (4, 16))
        g.add_node(OpType.MATMUL, ["a", "b"], "y", {"transpose_b": True, "alpha": 0.5})

        node = list(g)[0]
        extras = op_def.extras(node, g)
        assert isinstance(extras, list)
        assert len(extras) > 0

    def test_attention_extras(self):
        """ATTENTION extras should pack causal flag and other params."""
        op_def = OP_REGISTRY[OpType.ATTENTION]
        assert op_def.extras is not None

        g = Graph()
        g.add_tensor("Q", (2, 4, 16, 32))
        g.inputs.append("Q")
        g.add_tensor("K", (2, 4, 16, 32))
        g.inputs.append("K")
        g.add_tensor("V", (2, 4, 16, 32))
        g.inputs.append("V")
        g.add_tensor("out", (2, 4, 16, 32))
        g.add_node(OpType.ATTENTION, ["Q", "K", "V"], "out", {"causal": True})

        node = list(g)[0]
        extras = op_def.extras(node, g)
        assert isinstance(extras, list)
        assert len(extras) > 0

    def test_ops_without_extras(self):
        """RELU and other simple ops should have no extras packer."""
        assert OP_REGISTRY[OpType.RELU].extras is None
        assert OP_REGISTRY[OpType.EXP].extras is None
