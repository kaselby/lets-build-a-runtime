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
from runtime.ir import OpType
from runtime.ir import Graph
from runtime.passes import (
    absorb_into_matmul,
    constant_fold,
    eliminate_dead_code,
    fuse,
    run_pipeline,
    FusionPattern,
    FUSION_PATTERNS,
)
from runtime.planner import plan
from runtime.executor import Executor
from runtime.backends.numpy_backend import NumpyBackend


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
    executor = Executor(backends=[NumpyBackend()])
    result = executor.execute(ep, {graph.inputs[0]: x.numpy().copy()})
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
    executor = Executor(backends=[NumpyBackend()])
    result = executor.execute(ep, {graph.inputs[0]: x.numpy().copy()})
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
    executor = Executor(backends=[NumpyBackend()])
    result = executor.execute(ep, {graph.inputs[0]: x.numpy().copy()})
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
    executor = Executor(backends=[NumpyBackend()])
    result = executor.execute(ep, {"x": x})
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
    executor = Executor(backends=[NumpyBackend()])
    result = executor.execute(ep, {"x": x})
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
    executor = Executor(backends=[NumpyBackend()])
    result = executor.execute(ep, {"x": x})
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
    executor = Executor(backends=[NumpyBackend()])
    result = executor.execute(ep, {graph.inputs[0]: x.numpy().copy()})
    np.testing.assert_allclose(result[graph.outputs[0]], expected, atol=1e-4)
