"""Optimization pass tests: structural invariants + correctness preservation.

Each pass is tested for two things:
1. It achieves its structural goal (e.g., TRANSPOSE nodes are absorbed)
2. It preserves correctness (output still matches PyTorch)
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from runtime.exporter import export_model
from runtime.ir import OpType, Graph
from runtime.passes import (
    absorb_into_matmul,
    constant_fold,
    eliminate_dead_code,
    fuse,
    run_pipeline,
    FusionPattern,
    FUSION_PATTERNS,
    absorb_mask_into_attention,
)
from runtime.passes.fusion import fuse_dags
from runtime.planner import plan
from runtime.executor import InterpretedExecutor
from runtime.backends.numpy_backend import NumpyBackend


def _execute(ep, inputs):
    """Helper: run an execution plan via the interpreted executor."""
    executor = InterpretedExecutor(backends=[NumpyBackend()])
    executor.compile(ep)
    return executor.run(inputs)


# ---------------------------------------------------------------------------
# Transpose absorption
# ---------------------------------------------------------------------------

class ManualTransposeModel(nn.Module):
    """Model with explicit x @ w.T — produces TRANSPOSE → MATMUL in the graph."""
    def __init__(self, dim=32):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x):
        return x @ self.weight.T


def test_absorb_removes_transpose():
    """After absorption, no TRANSPOSE should feed into a MATMUL."""
    model = ManualTransposeModel()
    model.eval()
    graph = export_model(model, (torch.randn(4, 32),))

    # Before: should have a TRANSPOSE node
    ops_before = [n.op for n in graph]
    assert OpType.TRANSPOSE in ops_before

    absorb_into_matmul(graph)

    # After: no TRANSPOSE, and MATMUL should have transpose_b=True
    ops_after = [n.op for n in graph]
    assert OpType.TRANSPOSE not in ops_after
    matmuls = [n for n in graph if n.op == OpType.MATMUL]
    assert all(n.attrs.get("transpose_b") for n in matmuls)


def test_absorb_preserves_correctness():
    """Output should match PyTorch after absorption."""
    model = ManualTransposeModel()
    model.eval()
    x = torch.randn(4, 32)

    with torch.no_grad():
        expected = model(x).numpy()

    graph = export_model(model, (x,))
    absorb_into_matmul(graph)
    ep = plan(graph)
    result = _execute(ep, {graph.inputs[0]: x.numpy().copy()})
    output = result[graph.outputs[0]]

    np.testing.assert_allclose(output, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Constant folding
# ---------------------------------------------------------------------------

class ConstantScaleModel(nn.Module):
    """Model where a weight is scaled by a constant — foldable."""
    def __init__(self, dim=16):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc(x) / 2.0


def test_constant_fold_reduces_nodes():
    """Constant folding should evaluate ops with all-constant inputs."""
    model = ConstantScaleModel()
    model.eval()
    graph = export_model(model, (torch.randn(2, 16),))

    n_before = len(graph.nodes)
    constant_fold(graph)
    n_after = len(graph.nodes)

    # The scalar division's constant input gets folded if possible
    # At minimum, node count shouldn't increase
    assert n_after <= n_before


def test_constant_fold_preserves_correctness():
    """Output should match PyTorch after constant folding."""
    model = ConstantScaleModel()
    model.eval()
    x = torch.randn(2, 16)

    with torch.no_grad():
        expected = model(x).numpy()

    graph = export_model(model, (x,))
    constant_fold(graph)
    eliminate_dead_code(graph)
    ep = plan(graph)
    result = _execute(ep, {graph.inputs[0]: x.numpy().copy()})
    output = result[graph.outputs[0]]

    np.testing.assert_allclose(output, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Dead code elimination
# ---------------------------------------------------------------------------

def test_dce_removes_unused_constants():
    """DCE should clean up constants that have no remaining consumers."""
    from conftest import SimpleMLP
    model = SimpleMLP(64)
    model.eval()
    graph = export_model(model, (torch.randn(1, 64),))

    # Constant fold first (creates opportunities for DCE)
    constant_fold(graph)
    n_constants_before = len(graph.constants)
    eliminate_dead_code(graph)
    n_constants_after = len(graph.constants)

    # DCE should not add constants (it only removes)
    assert n_constants_after <= n_constants_before


# ---------------------------------------------------------------------------
# Full pipeline preserves correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch,dim", [(1, 64), (4, 128), (16, 256)])
def test_full_pipeline_correctness(batch, dim):
    """The complete pass pipeline should not change the model's output."""
    from conftest import SimpleMLP
    model = SimpleMLP(dim)
    model.eval()
    x = torch.randn(batch, dim)

    with torch.no_grad():
        expected = model(x).numpy()

    graph = export_model(model, (x,))
    run_pipeline(graph)
    ep = plan(graph)
    result = _execute(ep, {graph.inputs[0]: x.numpy().copy()})
    output = result[graph.outputs[0]]

    np.testing.assert_allclose(output, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(*node_specs, constants=None):
    """Build a small graph from a compact spec for fusion testing.

    Each node_spec is (OpType, [input_names], output_name, output_shape).
    The first input that isn't an output of a previous node becomes a graph input.
    Constants dict maps tensor name -> (shape, numpy_array).
    """
    constants = constants or {}
    g = Graph()
    node_outputs = set()

    for op, inputs, output, shape in node_specs:
        for inp in inputs:
            if inp not in g.tensors:
                inp_shape = constants[inp][0] if inp in constants else shape
                g.add_tensor(inp, inp_shape)
                if inp in constants:
                    g.tensors[inp].buffer = constants[inp][1]
                    g.constants.append(inp)
                elif inp not in node_outputs:
                    g.inputs.append(inp)
        g.add_tensor(output, shape)
        g.add_node(op, inputs, output)
        node_outputs.add(output)

    # Last node's output is the graph output
    g.outputs.append(node_specs[-1][2])
    return g


# ---------------------------------------------------------------------------
# Pattern fusion: FUSED_BIAS_RELU (ADD + RELU)
# ---------------------------------------------------------------------------

def test_bias_relu_fuses():
    """ADD(bias) → RELU should fuse into FUSED_BIAS_RELU."""
    g = _make_graph(
        (OpType.ADD, ["x", "b"], "add_out", (4, 8)),
        (OpType.RELU, ["add_out"], "relu_out", (4, 8)),
        constants={"b": ((8,), np.ones(8, dtype=np.float32))},
    )
    assert fuse(g) is True
    nodes = list(g)
    assert len(nodes) == 1
    assert nodes[0].op == OpType.FUSED_BIAS_RELU


def test_bias_relu_correctness():
    """Fused ADD+RELU should produce the same result as separate ops."""
    x = np.random.randn(4, 8).astype(np.float32)
    b = np.random.randn(8).astype(np.float32)
    expected = np.maximum(x + b, 0)

    g = _make_graph(
        (OpType.ADD, ["x", "b"], "add_out", (4, 8)),
        (OpType.RELU, ["add_out"], "relu_out", (4, 8)),
        constants={"b": ((8,), b)},
    )
    fuse(g)
    ep = plan(g)
    result = _execute(ep, {"x": x})
    np.testing.assert_allclose(result["relu_out"], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Pattern fusion: MATMUL_ADD
# ---------------------------------------------------------------------------

def test_matmul_add_fuses():
    """MATMUL → ADD with 1D bias should fuse into MATMUL_ADD."""
    w = np.random.randn(8, 8).astype(np.float32)
    b = np.random.randn(8).astype(np.float32)
    g = _make_graph(
        (OpType.MATMUL, ["x", "w"], "mm_out", (4, 8)),
        (OpType.ADD, ["mm_out", "b"], "add_out", (4, 8)),
        constants={
            "w": ((8, 8), w),
            "b": ((8,), b),
        },
    )
    assert fuse(g) is True
    nodes = list(g)
    assert len(nodes) == 1
    assert nodes[0].op == OpType.MATMUL_ADD
    assert nodes[0].inputs == ["x", "w", "b"]


def test_matmul_add_preserves_transpose_b():
    """MATMUL_ADD should carry forward transpose_b from the original MATMUL."""
    w = np.random.randn(8, 8).astype(np.float32)
    b = np.random.randn(8).astype(np.float32)
    g = Graph()
    g.add_tensor("x", (4, 8))
    g.inputs.append("x")
    g.add_tensor("w", (8, 8))
    g.tensors["w"].buffer = w
    g.constants.append("w")
    g.add_tensor("b", (8,))
    g.tensors["b"].buffer = b
    g.constants.append("b")
    g.add_tensor("mm_out", (4, 8))
    g.add_node(OpType.MATMUL, ["x", "w"], "mm_out", {"transpose_b": True})
    g.add_tensor("add_out", (4, 8))
    g.add_node(OpType.ADD, ["mm_out", "b"], "add_out")
    g.outputs.append("add_out")

    fuse(g)
    node = list(g)[0]
    assert node.attrs["transpose_b"] is True


def test_matmul_add_no_fuse_2d_bias():
    """ADD with a 2D second input (not a bias vector) should NOT fuse."""
    w = np.random.randn(8, 8).astype(np.float32)
    b2d = np.random.randn(4, 8).astype(np.float32)
    g = _make_graph(
        (OpType.MATMUL, ["x", "w"], "mm_out", (4, 8)),
        (OpType.ADD, ["mm_out", "b2d"], "add_out", (4, 8)),
        constants={
            "w": ((8, 8), w),
            "b2d": ((4, 8), b2d),
        },
    )
    assert fuse(g) is False


def test_matmul_add_no_fuse_multi_consumer():
    """MATMUL → ADD should NOT fuse if MATMUL output has other consumers."""
    w = np.random.randn(8, 8).astype(np.float32)
    b = np.random.randn(8).astype(np.float32)
    g = Graph()
    g.add_tensor("x", (4, 8))
    g.inputs.append("x")
    g.add_tensor("w", (8, 8))
    g.tensors["w"].buffer = w
    g.constants.append("w")
    g.add_tensor("b", (8,))
    g.tensors["b"].buffer = b
    g.constants.append("b")
    g.add_tensor("mm_out", (4, 8))
    g.add_node(OpType.MATMUL, ["x", "w"], "mm_out")
    g.add_tensor("add_out", (4, 8))
    g.add_node(OpType.ADD, ["mm_out", "b"], "add_out")
    # Second consumer of mm_out
    g.add_tensor("relu_out", (4, 8))
    g.add_node(OpType.RELU, ["mm_out"], "relu_out")
    g.outputs.extend(["add_out", "relu_out"])

    assert fuse(g) is False


def test_matmul_add_correctness():
    """Fused MATMUL_ADD should match separate MATMUL + ADD."""
    x = np.random.randn(4, 8).astype(np.float32)
    w = np.random.randn(8, 8).astype(np.float32)
    b = np.random.randn(8).astype(np.float32)
    expected = x @ w + b

    g = _make_graph(
        (OpType.MATMUL, ["x", "w"], "mm_out", (4, 8)),
        (OpType.ADD, ["mm_out", "b"], "add_out", (4, 8)),
        constants={
            "w": ((8, 8), w),
            "b": ((8,), b),
        },
    )
    fuse(g)
    ep = plan(g)
    result = _execute(ep, {"x": x})
    np.testing.assert_allclose(result["add_out"], expected, atol=1e-5)


def test_matmul_add_correctness_transpose_b():
    """Fused MATMUL_ADD with transpose_b should match x @ w.T + b."""
    x = np.random.randn(4, 8).astype(np.float32)
    w = np.random.randn(8, 8).astype(np.float32)
    b = np.random.randn(8).astype(np.float32)
    expected = x @ w.T + b

    g = Graph()
    g.add_tensor("x", (4, 8))
    g.inputs.append("x")
    g.add_tensor("w", (8, 8))
    g.tensors["w"].buffer = w
    g.constants.append("w")
    g.add_tensor("b", (8,))
    g.tensors["b"].buffer = b
    g.constants.append("b")
    g.add_tensor("mm_out", (4, 8))
    g.add_node(OpType.MATMUL, ["x", "w"], "mm_out", {"transpose_b": True})
    g.add_tensor("add_out", (4, 8))
    g.add_node(OpType.ADD, ["mm_out", "b"], "add_out")
    g.outputs.append("add_out")

    fuse(g)
    ep = plan(g)
    result = _execute(ep, {"x": x})
    np.testing.assert_allclose(result["add_out"], expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Priority ordering: ADD+RELU wins over MATMUL+ADD
# ---------------------------------------------------------------------------

def test_priority_bias_relu_wins_over_matmul_add():
    """In MATMUL → ADD → RELU, ADD+RELU (priority 0) should fuse first,
    leaving MATMUL standalone rather than MATMUL+ADD consuming the ADD."""
    w = np.random.randn(8, 8).astype(np.float32)
    b = np.random.randn(8).astype(np.float32)
    g = _make_graph(
        (OpType.MATMUL, ["x", "w"], "mm_out", (4, 8)),
        (OpType.ADD, ["mm_out", "b"], "add_out", (4, 8)),
        (OpType.RELU, ["add_out"], "relu_out", (4, 8)),
        constants={
            "w": ((8, 8), w),
            "b": ((8,), b),
        },
    )
    fuse(g)
    ops = [n.op for n in g]
    assert OpType.FUSED_BIAS_RELU in ops
    assert OpType.MATMUL in ops
    # MATMUL_ADD should NOT appear — ADD was claimed by bias_relu
    assert OpType.MATMUL_ADD not in ops


def test_priority_reversed_matmul_add_wins():
    """With reversed priorities, MATMUL+ADD should win over ADD+RELU."""
    w = np.random.randn(8, 8).astype(np.float32)
    b = np.random.randn(8).astype(np.float32)
    g = _make_graph(
        (OpType.MATMUL, ["x", "w"], "mm_out", (4, 8)),
        (OpType.ADD, ["mm_out", "b"], "add_out", (4, 8)),
        (OpType.RELU, ["add_out"], "relu_out", (4, 8)),
        constants={
            "w": ((8, 8), w),
            "b": ((8,), b),
        },
    )
    # Build reversed-priority patterns
    reversed_patterns = [
        FusionPattern(**{**vars(p), "priority": 1 - p.priority})
        for p in FUSION_PATTERNS
    ]
    fuse(g, patterns=reversed_patterns)
    ops = [n.op for n in g]
    assert OpType.MATMUL_ADD in ops
    assert OpType.RELU in ops
    # FUSED_BIAS_RELU should NOT appear — ADD was claimed by matmul_add
    assert OpType.FUSED_BIAS_RELU not in ops


# ---------------------------------------------------------------------------
# Pattern fusion: GATED_ACT (SILU/GELU × MUL, with optional bias)
# ---------------------------------------------------------------------------

def test_silu_mul_fuses():
    """SILU → MUL should fuse into GATED_ACT with act=silu, has_bias=False."""
    g = _make_graph(
        (OpType.SILU, ["x"], "silu_out", (4, 8)),
        (OpType.MUL, ["silu_out", "up"], "gate_out", (4, 8)),
    )
    assert fuse(g) is True
    nodes = list(g)
    assert len(nodes) == 1
    assert nodes[0].op == OpType.GATED_ACT
    assert nodes[0].attrs["act"] == "silu"
    assert nodes[0].attrs["has_bias"] is False
    assert nodes[0].inputs == ["x", "up"]


def test_gelu_mul_fuses():
    """GELU → MUL should fuse into GATED_ACT with act=gelu, has_bias=False."""
    g = _make_graph(
        (OpType.GELU, ["x"], "gelu_out", (4, 8)),
        (OpType.MUL, ["gelu_out", "up"], "gate_out", (4, 8)),
    )
    assert fuse(g) is True
    nodes = list(g)
    assert len(nodes) == 1
    assert nodes[0].op == OpType.GATED_ACT
    assert nodes[0].attrs["act"] == "gelu"
    assert nodes[0].attrs["has_bias"] is False


def test_bias_silu_mul_fuses():
    """ADD(bias) → SILU → MUL should fuse into GATED_ACT with has_bias=True."""
    b = np.random.randn(8).astype(np.float32)
    g = _make_graph(
        (OpType.ADD, ["x", "b"], "add_out", (4, 8)),
        (OpType.SILU, ["add_out"], "silu_out", (4, 8)),
        (OpType.MUL, ["silu_out", "up"], "gate_out", (4, 8)),
        constants={"b": ((8,), b)},
    )
    assert fuse(g) is True
    nodes = list(g)
    assert len(nodes) == 1
    assert nodes[0].op == OpType.GATED_ACT
    assert nodes[0].attrs["act"] == "silu"
    assert nodes[0].attrs["has_bias"] is True
    assert nodes[0].inputs == ["x", "b", "up"]


def test_bias_gelu_mul_fuses():
    """ADD(bias) → GELU → MUL should fuse into GATED_ACT with has_bias=True."""
    b = np.random.randn(8).astype(np.float32)
    g = _make_graph(
        (OpType.ADD, ["x", "b"], "add_out", (4, 8)),
        (OpType.GELU, ["add_out"], "gelu_out", (4, 8)),
        (OpType.MUL, ["gelu_out", "up"], "gate_out", (4, 8)),
        constants={"b": ((8,), b)},
    )
    assert fuse(g) is True
    nodes = list(g)
    assert len(nodes) == 1
    assert nodes[0].op == OpType.GATED_ACT
    assert nodes[0].attrs["act"] == "gelu"
    assert nodes[0].attrs["has_bias"] is True


def test_priority_bias_silu_wins_over_matmul_add():
    """In MATMUL → ADD → SILU → MUL, ADD+SILU+MUL (priority 0) should fuse
    into GATED_ACT(has_bias=True), leaving MATMUL standalone."""
    w = np.random.randn(8, 8).astype(np.float32)
    b = np.random.randn(8).astype(np.float32)
    g = _make_graph(
        (OpType.MATMUL, ["x", "w"], "mm_out", (4, 8)),
        (OpType.ADD, ["mm_out", "b"], "add_out", (4, 8)),
        (OpType.SILU, ["add_out"], "silu_out", (4, 8)),
        (OpType.MUL, ["silu_out", "up"], "gate_out", (4, 8)),
        constants={
            "w": ((8, 8), w),
            "b": ((8,), b),
        },
    )
    fuse(g)
    ops = [n.op for n in g]
    assert OpType.GATED_ACT in ops
    assert OpType.MATMUL in ops
    # MATMUL_ADD should NOT appear — ADD was claimed by bias_silu_mul
    assert OpType.MATMUL_ADD not in ops
    # Verify the fused node has bias
    gated = [n for n in g if n.op == OpType.GATED_ACT][0]
    assert gated.attrs["has_bias"] is True


# ---------------------------------------------------------------------------
# DAG fusion: SiLU recognition (defensive — aten.silu is normally a single node)
# ---------------------------------------------------------------------------

def test_silu_dag_collapses():
    """The 4-node SiLU decomposition should collapse to a single SILU node."""
    g = Graph()
    shape = (4, 8)
    g.add_tensor("x", shape)
    g.inputs.append("x")
    g.add_tensor("neg_out", shape)
    g.add_node(OpType.NEG, ["x"], "neg_out")
    g.add_tensor("exp_out", shape)
    g.add_node(OpType.EXP, ["neg_out"], "exp_out")
    g.add_tensor("denom", shape)
    g.add_node(OpType.ADD, ["exp_out"], "denom", {"scalar": 1.0})
    g.add_tensor("out", shape)
    g.add_node(OpType.DIV, ["x", "denom"], "out")
    g.outputs.append("out")

    assert len(g.nodes) == 4
    assert fuse_dags(g) is True
    nodes = list(g)
    assert len(nodes) == 1
    assert nodes[0].op == OpType.SILU


# ---------------------------------------------------------------------------
# MLP end-to-end with fusion
# ---------------------------------------------------------------------------

def test_mlp_fusion_structure():
    """Full pipeline on MLP: should produce FUSED_BIAS_RELU for inner layers
    and MATMUL_ADD for the final layer (no RELU after it)."""
    from conftest import SimpleMLP
    model = SimpleMLP(64)
    model.eval()
    graph = export_model(model, (torch.randn(4, 64),))
    run_pipeline(graph)

    ops = [n.op for n in graph]
    # Inner layers: MATMUL → FUSED_BIAS_RELU (ADD+RELU fused, priority 0)
    assert ops.count(OpType.FUSED_BIAS_RELU) == 2
    # Final layer: MATMUL → ADD with no RELU, so MATMUL_ADD (priority 1)
    assert ops.count(OpType.MATMUL_ADD) == 1
    # No standalone ADDs should remain
    assert OpType.ADD not in ops


def test_mlp_fusion_correctness():
    """Full pipeline on MLP: fused output should match PyTorch."""
    from conftest import SimpleMLP
    model = SimpleMLP(64)
    model.eval()
    x = torch.randn(4, 64)

    with torch.no_grad():
        expected = model(x).numpy()

    graph = export_model(model, (x,))
    run_pipeline(graph)
    ep = plan(graph)
    result = _execute(ep, {graph.inputs[0]: x.numpy().copy()})
    np.testing.assert_allclose(result[graph.outputs[0]], expected, atol=1e-4)


# ---------------------------------------------------------------------------
# DAG fusion: GELU recognition
# ---------------------------------------------------------------------------

def _build_gelu_graph():
    """Build a graph with the 8-node GELU tanh approximation pattern.

    Pattern: x -> pow(3) -> mul(0.044715) -> add(x) -> mul(sqrt(2/pi))
             -> tanh -> add(1.0) -> mul(half_x)
    where half_x = mul(x, 0.5)
    """
    g = Graph()
    shape = (4, 8)

    # Graph input
    g.add_tensor("x", shape)
    g.inputs.append("x")

    # pow(x, 3)
    g.add_tensor("pow_out", shape)
    g.add_node(OpType.POW, ["x"], "pow_out", {"scalar": 3.0})

    # mul(pow_out, 0.044715)
    g.add_tensor("mul_coeff_out", shape)
    g.add_node(OpType.MUL, ["pow_out"], "mul_coeff_out", {"scalar": 0.044715})

    # add(x, mul_coeff_out)  — tensor-tensor add
    g.add_tensor("add_inner_out", shape)
    g.add_node(OpType.ADD, ["x", "mul_coeff_out"], "add_inner_out")

    # mul(add_inner_out, sqrt(2/pi))
    g.add_tensor("mul_sqrt2pi_out", shape)
    g.add_node(OpType.MUL, ["add_inner_out"], "mul_sqrt2pi_out", {"scalar": 0.7978845608})

    # tanh(mul_sqrt2pi_out)
    g.add_tensor("tanh_out", shape)
    g.add_node(OpType.TANH, ["mul_sqrt2pi_out"], "tanh_out")

    # add(tanh_out, 1.0)
    g.add_tensor("add_one_out", shape)
    g.add_node(OpType.ADD, ["tanh_out"], "add_one_out", {"scalar": 1.0})

    # mul(x, 0.5) — half_x
    g.add_tensor("half_x", shape)
    g.add_node(OpType.MUL, ["x"], "half_x", {"scalar": 0.5})

    # mul(add_one_out, half_x) — final_output
    g.add_tensor("final", shape)
    g.add_node(OpType.MUL, ["add_one_out", "half_x"], "final")
    g.outputs.append("final")

    return g


def test_gelu_recognition_collapses():
    """The 8-node GELU pattern should collapse to a single GELU node."""
    g = _build_gelu_graph()
    assert len(g.nodes) == 8

    changed = fuse_dags(g)
    assert changed is True

    nodes = list(g)
    assert len(nodes) == 1
    assert nodes[0].op == OpType.GELU
    assert nodes[0].inputs == ["x"]
    assert nodes[0].output == "final"


def test_gelu_recognition_correctness():
    """Collapsed GELU should produce the same result as the 8-node pattern."""
    g = _build_gelu_graph()

    # Compute reference from the formula
    x = np.random.randn(4, 8).astype(np.float32)
    inner = 0.7978845608 * (x + 0.044715 * np.power(x, 3))
    expected = 0.5 * x * (1 + np.tanh(inner))

    fuse_dags(g)
    eliminate_dead_code(g)
    ep = plan(g)
    result = _execute(ep, {"x": x})
    np.testing.assert_allclose(result["final"], expected, atol=1e-5)


def test_gelu_recognition_negative_partial():
    """A partial GELU pattern (missing the tanh) should NOT be recognized."""
    g = Graph()
    shape = (4, 8)
    g.add_tensor("x", shape)
    g.inputs.append("x")
    # Just pow -> mul -> add (incomplete pattern)
    g.add_tensor("pow_out", shape)
    g.add_node(OpType.POW, ["x"], "pow_out", {"scalar": 3.0})
    g.add_tensor("mul_out", shape)
    g.add_node(OpType.MUL, ["pow_out"], "mul_out", {"scalar": 0.044715})
    g.add_tensor("add_out", shape)
    g.add_node(OpType.ADD, ["x", "mul_out"], "add_out")
    g.outputs.append("add_out")

    changed = fuse_dags(g)
    assert changed is False
    assert len(g.nodes) == 3  # unchanged


def test_gelu_recognition_negative_wrong_scalar():
    """GELU pattern with wrong scalar should NOT match."""
    g = _build_gelu_graph()
    # Tamper with the sqrt(2/pi) scalar to make it wrong
    for node in g.nodes.values():
        if node.op == OpType.MUL and node.attrs.get("scalar") == 0.7978845608:
            node.attrs["scalar"] = 0.5  # wrong value
            break

    changed = fuse_dags(g)
    assert changed is False


def test_gelu_in_full_pipeline():
    """GELU recognition should integrate with the full pipeline.

    Build a graph: x -> MATMUL -> ADD (bias) -> GELU_pattern -> output
    After full pipeline: should have MATMUL_ADD and GELU nodes.
    """
    g = Graph()
    shape = (4, 8)
    dim = 8

    g.add_tensor("x", shape)
    g.inputs.append("x")

    w = np.random.randn(dim, dim).astype(np.float32)
    g.add_tensor("w", (dim, dim))
    g.tensors["w"].buffer = w
    g.constants.append("w")

    b = np.random.randn(dim).astype(np.float32)
    g.add_tensor("b", (dim,))
    g.tensors["b"].buffer = b
    g.constants.append("b")

    # MATMUL(x, w)
    g.add_tensor("mm", shape)
    g.add_node(OpType.MATMUL, ["x", "w"], "mm")

    # ADD(mm, b) — bias
    g.add_tensor("biased", shape)
    g.add_node(OpType.ADD, ["mm", "b"], "biased")

    # GELU pattern on biased
    g.add_tensor("pow_out", shape)
    g.add_node(OpType.POW, ["biased"], "pow_out", {"scalar": 3.0})
    g.add_tensor("mul_coeff_out", shape)
    g.add_node(OpType.MUL, ["pow_out"], "mul_coeff_out", {"scalar": 0.044715})
    g.add_tensor("add_inner_out", shape)
    g.add_node(OpType.ADD, ["biased", "mul_coeff_out"], "add_inner_out")
    g.add_tensor("mul_sqrt2pi_out", shape)
    g.add_node(OpType.MUL, ["add_inner_out"], "mul_sqrt2pi_out", {"scalar": 0.7978845608})
    g.add_tensor("tanh_out", shape)
    g.add_node(OpType.TANH, ["mul_sqrt2pi_out"], "tanh_out")
    g.add_tensor("add_one_out", shape)
    g.add_node(OpType.ADD, ["tanh_out"], "add_one_out", {"scalar": 1.0})
    g.add_tensor("half_x", shape)
    g.add_node(OpType.MUL, ["biased"], "half_x", {"scalar": 0.5})
    g.add_tensor("final", shape)
    g.add_node(OpType.MUL, ["add_one_out", "half_x"], "final")
    g.outputs.append("final")

    run_pipeline(g)
    ops = [n.op for n in g]
    assert OpType.GELU in ops
    # MATMUL + ADD should be fused into MATMUL_ADD (priority 1, after GELU recognition)
    assert OpType.MATMUL_ADD in ops


# ---------------------------------------------------------------------------
# Causal attention fusion: MATMUL → ADD(mask) → SOFTMAX → MATMUL
# ---------------------------------------------------------------------------

def test_causal_attention_fusion_structure():
    """CausalNaiveTransformerBlock should produce fused ATTENTION with causal=True.

    The model applies an explicit upper-triangular -inf mask via ADD before
    softmax. The causal_attention fusion should collapse MATMUL→ADD→SOFTMAX→MATMUL
    into a single ATTENTION node with causal=True, dropping the mask constant.
    """
    from conftest import CausalNaiveTransformerBlock
    model = CausalNaiveTransformerBlock(d_model=64, n_heads=4, seq_len=16)
    model.eval()
    graph = export_model(model, (torch.randn(2, 16, 64),))
    run_pipeline(graph)

    attention_nodes = [n for n in graph if n.op == OpType.ATTENTION]
    assert len(attention_nodes) == 1
    assert attention_nodes[0].attrs.get("causal") is True


@pytest.mark.parametrize("batch,seq,d_model,n_heads", [
    (1, 8, 64, 4),
    (2, 16, 64, 4),
])
def test_causal_attention_fusion_correctness(batch, seq, d_model, n_heads):
    """Causal attention fusion should produce the same result as PyTorch."""
    from conftest import CausalNaiveTransformerBlock
    model = CausalNaiveTransformerBlock(d_model=d_model, n_heads=n_heads, seq_len=seq)
    model.eval()
    x = torch.randn(batch, seq, d_model)

    with torch.no_grad():
        expected = model(x).numpy()

    graph = export_model(model, (x,))
    run_pipeline(graph)
    ep = plan(graph)
    result = _execute(ep, {graph.inputs[0]: x.numpy().copy()})
    np.testing.assert_allclose(result[graph.outputs[0]], expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Attention fusion: MATMUL → SOFTMAX → MATMUL
# ---------------------------------------------------------------------------

def test_attention_fusion_structure():
    """NaiveTransformerBlock should produce fused ATTENTION (non-causal).

    The model uses F.softmax (no explicit mask), which produces the 3-node
    pattern MATMUL→SOFTMAX→MATMUL that fuses into ATTENTION.
    """
    from conftest import NaiveTransformerBlock
    model = NaiveTransformerBlock(d_model=64, n_heads=4)
    model.eval()
    graph = export_model(model, (torch.randn(2, 16, 64),))
    run_pipeline(graph)

    attention_nodes = [n for n in graph if n.op == OpType.ATTENTION]
    assert len(attention_nodes) == 1
    assert not attention_nodes[0].attrs.get("causal")


# ---------------------------------------------------------------------------
# Causal mask absorption (standalone pass test)
# ---------------------------------------------------------------------------

def test_absorb_causal_mask_standalone():
    """absorb_mask_into_attention should detect and absorb a causal mask constant."""
    g = Graph()
    g.add_tensor("Q", (2, 4, 16, 32))
    g.inputs.append("Q")
    g.add_tensor("K", (2, 4, 16, 32))
    g.inputs.append("K")
    g.add_tensor("V", (2, 4, 16, 32))
    g.inputs.append("V")

    # Build a causal mask constant (upper-triangular -inf)
    mask_data = np.triu(np.full((16, 16), -np.inf, dtype=np.float32), k=1)
    g.add_tensor("mask", (16, 16))
    g.tensors["mask"].buffer = mask_data
    g.constants.append("mask")

    g.add_tensor("attn_out", (2, 4, 16, 32))
    g.add_node(OpType.ATTENTION, ["Q", "K", "V", "mask"], "attn_out")
    g.outputs.append("attn_out")

    changed = absorb_mask_into_attention(g)
    assert changed is True

    attn = list(g)[0]
    assert attn.attrs.get("causal") is True
    assert "mask" not in attn.inputs
    assert len(attn.inputs) == 3  # Q, K, V only


# ---------------------------------------------------------------------------
# Automated fusion correctness: unfused evaluator chain vs fused execution
#
# For each registered fusion pattern, build the ideal-case graph, compute
# the reference by chaining numpy evaluators, then fuse + plan + execute
# and compare. This is a safety net — hand-written tests above cover
# structural invariants, negative cases, and priority ordering.
# ---------------------------------------------------------------------------

from runtime.ops import OP_REGISTRY
from runtime.passes.fusion import FUSION_PATTERNS, DAG_FUSION_PATTERNS

# Each entry defines how to build valid inputs for one fusion pattern.
# (pattern_name, graph_builder, input_generator)
# graph_builder returns a Graph; input_generator returns {name: ndarray}.

def _fusion_bias_relu():
    shape = (4, 64)
    g = _make_graph(
        (OpType.ADD, ["x", "b"], "add_out", shape),
        (OpType.RELU, ["add_out"], "relu_out", shape),
        constants={"b": ((64,), np.random.randn(64).astype(np.float32))},
    )
    return g, {"x": np.random.randn(*shape).astype(np.float32)}


def _fusion_matmul_add():
    shape = (4, 8)
    w = np.random.randn(16, 8).astype(np.float32)
    b = np.random.randn(8).astype(np.float32)
    g = _make_graph(
        (OpType.MATMUL, ["x", "w"], "mm_out", shape),
        (OpType.ADD, ["mm_out", "b"], "add_out", shape),
        constants={"w": ((16, 8), w), "b": ((8,), b)},
    )
    return g, {"x": np.random.randn(4, 16).astype(np.float32)}


def _fusion_silu_mul():
    shape = (4, 64)
    g = _make_graph(
        (OpType.SILU, ["x"], "silu_out", shape),
        (OpType.MUL, ["silu_out", "up"], "gate_out", shape),
    )
    return g, {
        "x": np.random.randn(*shape).astype(np.float32),
        "up": np.random.randn(*shape).astype(np.float32),
    }


def _fusion_gelu_mul():
    shape = (4, 64)
    g = _make_graph(
        (OpType.GELU, ["x"], "gelu_out", shape),
        (OpType.MUL, ["gelu_out", "up"], "gate_out", shape),
    )
    return g, {
        "x": np.random.randn(*shape).astype(np.float32),
        "up": np.random.randn(*shape).astype(np.float32),
    }


def _fusion_bias_silu_mul():
    shape = (4, 64)
    b = np.random.randn(64).astype(np.float32)
    g = _make_graph(
        (OpType.ADD, ["x", "b"], "add_out", shape),
        (OpType.SILU, ["add_out"], "silu_out", shape),
        (OpType.MUL, ["silu_out", "up"], "gate_out", shape),
        constants={"b": ((64,), b)},
    )
    return g, {
        "x": np.random.randn(*shape).astype(np.float32),
        "up": np.random.randn(*shape).astype(np.float32),
    }


def _fusion_bias_gelu_mul():
    shape = (4, 64)
    b = np.random.randn(64).astype(np.float32)
    g = _make_graph(
        (OpType.ADD, ["x", "b"], "add_out", shape),
        (OpType.GELU, ["add_out"], "gelu_out", shape),
        (OpType.MUL, ["gelu_out", "up"], "gate_out", shape),
        constants={"b": ((64,), b)},
    )
    return g, {
        "x": np.random.randn(*shape).astype(np.float32),
        "up": np.random.randn(*shape).astype(np.float32),
    }


FUSION_CORRECTNESS_CASES = [
    ("bias_relu",     _fusion_bias_relu),
    ("matmul_add",    _fusion_matmul_add),
    ("silu_mul",      _fusion_silu_mul),
    ("gelu_mul",      _fusion_gelu_mul),
    ("bias_silu_mul", _fusion_bias_silu_mul),
    ("bias_gelu_mul", _fusion_bias_gelu_mul),
]


@pytest.mark.parametrize("name,builder", FUSION_CORRECTNESS_CASES,
                         ids=[c[0] for c in FUSION_CORRECTNESS_CASES])
def _execute_with_c(ep, inputs):
    """Helper: run an execution plan with C + numpy backends."""
    from runtime.backends.c_backend import CBackend
    executor = InterpretedExecutor(backends=[CBackend(), NumpyBackend()])
    executor.compile(ep)
    return executor.run(inputs)


# ---------------------------------------------------------------------------
# GQA absorption: RESHAPE → EXPAND → RESHAPE → ATTENTION
# ---------------------------------------------------------------------------

def _build_gqa_graph(B=2, n_q=8, n_kv=4, S=16, D=32):
    """Build a graph with GQA expand chain on K feeding into ATTENTION."""
    group_size = n_q // n_kv
    g = Graph()

    # Inputs: Q, K, V
    g.add_tensor("Q", (B, n_q, S, D)); g.inputs.append("Q")
    g.add_tensor("K", (B, n_kv, S, D)); g.inputs.append("K")
    g.add_tensor("V", (B, n_kv, S, D)); g.inputs.append("V")

    # K expand chain: unsqueeze → expand → flatten
    g.add_tensor("K_unsq", (B, n_kv, 1, S, D))
    g.add_node(OpType.RESHAPE, ["K"], "K_unsq", {"shape": (B, n_kv, 1, S, D)})

    g.add_tensor("K_exp", (B, n_kv, group_size, S, D))
    g.add_node(OpType.EXPAND, ["K_unsq"], "K_exp", {"shape": (B, n_kv, group_size, S, D)})

    g.add_tensor("K_flat", (B, n_q, S, D))
    g.add_node(OpType.RESHAPE, ["K_exp"], "K_flat", {"shape": (B, n_q, S, D)})

    # ATTENTION(Q, K_flat, V) → output
    g.add_tensor("attn_out", (B, n_q, S, D))
    g.add_node(OpType.ATTENTION, ["Q", "K_flat", "V"], "attn_out")

    g.outputs.append("attn_out")
    return g, group_size


def test_gqa_fusion_structure():
    """RESHAPE→EXPAND→RESHAPE→ATTENTION should fuse, setting group_size."""
    g, expected_gs = _build_gqa_graph()

    assert fuse(g) is True

    attn_nodes = [n for n in g if n.op == OpType.ATTENTION]
    assert len(attn_nodes) == 1
    assert attn_nodes[0].attrs.get("group_size") == expected_gs

    # K input should be rewired to the original (pre-unsqueeze) tensor
    assert attn_nodes[0].inputs[1] == "K"

    # Expand chain should be dead (no consumers) — DCE would clean it
    expand_nodes = [n for n in g if n.op == OpType.EXPAND]
    assert len(expand_nodes) == 0


def test_gqa_fusion_both_kv():
    """When both K and V have expand chains, both should be absorbed."""
    B, n_q, n_kv, S, D = 2, 8, 4, 16, 32
    group_size = n_q // n_kv
    g = Graph()

    g.add_tensor("Q", (B, n_q, S, D)); g.inputs.append("Q")
    g.add_tensor("K", (B, n_kv, S, D)); g.inputs.append("K")
    g.add_tensor("V", (B, n_kv, S, D)); g.inputs.append("V")

    # K expand chain
    g.add_tensor("K_unsq", (B, n_kv, 1, S, D))
    g.add_node(OpType.RESHAPE, ["K"], "K_unsq", {"shape": (B, n_kv, 1, S, D)})
    g.add_tensor("K_exp", (B, n_kv, group_size, S, D))
    g.add_node(OpType.EXPAND, ["K_unsq"], "K_exp", {"shape": (B, n_kv, group_size, S, D)})
    g.add_tensor("K_flat", (B, n_q, S, D))
    g.add_node(OpType.RESHAPE, ["K_exp"], "K_flat", {"shape": (B, n_q, S, D)})

    # V expand chain
    g.add_tensor("V_unsq", (B, n_kv, 1, S, D))
    g.add_node(OpType.RESHAPE, ["V"], "V_unsq", {"shape": (B, n_kv, 1, S, D)})
    g.add_tensor("V_exp", (B, n_kv, group_size, S, D))
    g.add_node(OpType.EXPAND, ["V_unsq"], "V_exp", {"shape": (B, n_kv, group_size, S, D)})
    g.add_tensor("V_flat", (B, n_q, S, D))
    g.add_node(OpType.RESHAPE, ["V_exp"], "V_flat", {"shape": (B, n_q, S, D)})

    # ATTENTION(Q, K_flat, V_flat)
    g.add_tensor("attn_out", (B, n_q, S, D))
    g.add_node(OpType.ATTENTION, ["Q", "K_flat", "V_flat"], "attn_out")
    g.outputs.append("attn_out")

    assert fuse(g) is True

    attn_nodes = [n for n in g if n.op == OpType.ATTENTION]
    assert len(attn_nodes) == 1
    assert attn_nodes[0].attrs.get("group_size") == group_size
    assert attn_nodes[0].inputs[0] == "Q"
    assert attn_nodes[0].inputs[1] == "K"
    assert attn_nodes[0].inputs[2] == "V"


@pytest.mark.parametrize("desc,builder", FUSION_CORRECTNESS_CASES,
                         ids=[c[0] for c in FUSION_CORRECTNESS_CASES])
def test_fusion_correctness_auto(desc, builder):
    """Automated: fused execution matches unfused evaluator chain."""
    # Seed RNG so both builder() calls produce identical constants
    np.random.seed(42)
    g, inputs = builder()

    # Reference: run unfused (numpy is sufficient for primitive ops)
    ep_unfused = plan(g)
    ref = _execute(ep_unfused, inputs)
    expected = ref[g.outputs[0]]

    # Rebuild with same seed (constants must match)
    np.random.seed(42)
    g, inputs = builder()
    fuse(g)
    eliminate_dead_code(g)
    ep_fused = plan(g)
    # Use C backend — fused ops like GATED_ACT only have C kernels
    result = _execute_with_c(ep_fused, inputs)
    actual = result[g.outputs[0]]

    np.testing.assert_allclose(actual, expected, atol=1e-4)
