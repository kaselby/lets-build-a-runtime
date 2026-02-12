"""End-to-end oracle tests: full pipeline compared against PyTorch.

These are the highest-leverage tests — a single test exercises the exporter,
optimization passes, memory planner, and executor all at once. If PyTorch
and our runtime agree on the output, the whole pipeline is correct.
"""

import numpy as np
import pytest
import torch

from runtime.backends.c_backend import CBackend
from runtime.backends.numpy_backend import NumpyBackend
from runtime.exporter import export_model
from runtime.executor import Executor
from runtime.passes import run_pipeline
from runtime.planner import plan

from conftest import SimpleMLP, NaiveTransformerBlock, SDPATransformerBlock


# (batch_size, hidden_dim) configs — covers small to large
MLP_CONFIGS = [(1, 64), (4, 128), (32, 512), (128, 2048), (32, 4096)]

# (batch, seq_len, d_model, n_heads) configs for transformer tests
TRANSFORMER_CONFIGS = [
    (1, 8, 64, 4),
    (2, 16, 64, 4),
    (4, 32, 128, 8),
    (2, 64, 256, 8),
]


@pytest.mark.parametrize("batch,dim", MLP_CONFIGS)
@pytest.mark.parametrize("backend_name", ["numpy", "c"])
def test_mlp_per_op_executor(batch, dim, backend_name):
    """Per-op dispatch path: export → optimize → plan → execute per-node."""
    model = SimpleMLP(dim)
    model.eval()
    x = torch.randn(batch, dim)

    with torch.no_grad():
        expected = model(x).numpy()

    graph = export_model(model, (x,))
    run_pipeline(graph)
    ep = plan(graph)

    if backend_name == "numpy":
        executor = Executor(backends=[NumpyBackend()])
    else:
        executor = Executor(backends=[CBackend(), NumpyBackend()])

    result = executor.execute(ep, {graph.inputs[0]: x.numpy().copy()})
    output = result[graph.outputs[0]]

    np.testing.assert_allclose(output, expected, atol=1e-4)


@pytest.mark.parametrize("batch,dim", MLP_CONFIGS)
def test_mlp_compiled_executor(batch, dim):
    """Compiled C executor path: single ctypes call for the whole plan."""
    model = SimpleMLP(dim)
    model.eval()
    x = torch.randn(batch, dim)

    with torch.no_grad():
        expected = model(x).numpy()

    graph = export_model(model, (x,))
    run_pipeline(graph)
    ep = plan(graph)

    executor = Executor(backends=[CBackend(), NumpyBackend()])
    compiled = executor.compile_plan(ep)
    result = executor.execute_compiled(compiled, {graph.inputs[0]: x.numpy().copy()})
    output = result[graph.outputs[0]]

    np.testing.assert_allclose(output, expected, atol=1e-4)


@pytest.mark.parametrize("batch,dim", MLP_CONFIGS)
def test_compiled_reuse_across_inputs(batch, dim):
    """Compiled plan should produce correct results across multiple different inputs."""
    model = SimpleMLP(dim)
    model.eval()

    graph = export_model(model, (torch.randn(batch, dim),))
    run_pipeline(graph)
    ep = plan(graph)

    executor = Executor(backends=[CBackend(), NumpyBackend()])
    compiled = executor.compile_plan(ep)

    for _ in range(5):
        x = torch.randn(batch, dim)
        with torch.no_grad():
            expected = model(x).numpy()

        result = executor.execute_compiled(compiled, {graph.inputs[0]: x.numpy().copy()})
        output = result[graph.outputs[0]]
        np.testing.assert_allclose(output, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Transformer end-to-end tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch,seq,d_model,n_heads", TRANSFORMER_CONFIGS)
@pytest.mark.parametrize("backend_name", ["numpy", "c"])
def test_transformer_per_op_executor(batch, seq, d_model, n_heads, backend_name):
    """Per-op dispatch: transformer block matches PyTorch."""
    model = NaiveTransformerBlock(d_model, n_heads)
    model.eval()
    x = torch.randn(batch, seq, d_model)

    with torch.no_grad():
        expected = model(x).numpy()

    graph = export_model(model, (x,))
    run_pipeline(graph)
    ep = plan(graph)

    if backend_name == "numpy":
        executor = Executor(backends=[NumpyBackend()])
    else:
        executor = Executor(backends=[CBackend(), NumpyBackend()])

    result = executor.execute(ep, {graph.inputs[0]: x.numpy().copy()})
    output = result[graph.outputs[0]]
    np.testing.assert_allclose(output, expected, atol=1e-4)


@pytest.mark.parametrize("batch,seq,d_model,n_heads", TRANSFORMER_CONFIGS)
def test_transformer_compiled_executor(batch, seq, d_model, n_heads):
    """Compiled C executor: transformer block matches PyTorch."""
    model = NaiveTransformerBlock(d_model, n_heads)
    model.eval()
    x = torch.randn(batch, seq, d_model)

    with torch.no_grad():
        expected = model(x).numpy()

    graph = export_model(model, (x,))
    run_pipeline(graph)
    ep = plan(graph)

    executor = Executor(backends=[CBackend(), NumpyBackend()])
    compiled = executor.compile_plan(ep)
    result = executor.execute_compiled(compiled, {graph.inputs[0]: x.numpy().copy()})
    output = result[graph.outputs[0]]
    np.testing.assert_allclose(output, expected, atol=1e-4)


@pytest.mark.parametrize("batch,seq,d_model,n_heads", TRANSFORMER_CONFIGS)
def test_transformer_compiled_reuse(batch, seq, d_model, n_heads):
    """Compiled plan reuse: correct across multiple inputs."""
    model = NaiveTransformerBlock(d_model, n_heads)
    model.eval()

    graph = export_model(model, (torch.randn(batch, seq, d_model),))
    run_pipeline(graph)
    ep = plan(graph)

    executor = Executor(backends=[CBackend(), NumpyBackend()])
    compiled = executor.compile_plan(ep)

    for _ in range(5):
        x = torch.randn(batch, seq, d_model)
        with torch.no_grad():
            expected = model(x).numpy()
        result = executor.execute_compiled(compiled, {graph.inputs[0]: x.numpy().copy()})
        output = result[graph.outputs[0]]
        np.testing.assert_allclose(output, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# SDPA transformer end-to-end tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch,seq,d_model,n_heads", TRANSFORMER_CONFIGS)
@pytest.mark.parametrize("backend_name", ["numpy", "c"])
def test_sdpa_transformer_per_op_executor(batch, seq, d_model, n_heads, backend_name):
    """Per-op dispatch: SDPA transformer block matches PyTorch."""
    model = SDPATransformerBlock(d_model, n_heads)
    model.eval()
    x = torch.randn(batch, seq, d_model)

    with torch.no_grad():
        expected = model(x).numpy()

    graph = export_model(model, (x,))
    run_pipeline(graph)
    ep = plan(graph)

    if backend_name == "numpy":
        executor = Executor(backends=[NumpyBackend()])
    else:
        executor = Executor(backends=[CBackend(), NumpyBackend()])

    result = executor.execute(ep, {graph.inputs[0]: x.numpy().copy()})
    output = result[graph.outputs[0]]
    np.testing.assert_allclose(output, expected, atol=1e-4)


@pytest.mark.parametrize("batch,seq,d_model,n_heads", TRANSFORMER_CONFIGS)
def test_sdpa_transformer_compiled_executor(batch, seq, d_model, n_heads):
    """Compiled C executor: SDPA transformer block matches PyTorch."""
    model = SDPATransformerBlock(d_model, n_heads)
    model.eval()
    x = torch.randn(batch, seq, d_model)

    with torch.no_grad():
        expected = model(x).numpy()

    graph = export_model(model, (x,))
    run_pipeline(graph)
    ep = plan(graph)

    executor = Executor(backends=[CBackend(), NumpyBackend()])
    compiled = executor.compile_plan(ep)
    result = executor.execute_compiled(compiled, {graph.inputs[0]: x.numpy().copy()})
    output = result[graph.outputs[0]]
    np.testing.assert_allclose(output, expected, atol=1e-4)


@pytest.mark.parametrize("batch,seq,d_model,n_heads", TRANSFORMER_CONFIGS)
def test_sdpa_transformer_compiled_reuse(batch, seq, d_model, n_heads):
    """Compiled plan reuse: SDPA transformer correct across multiple inputs."""
    model = SDPATransformerBlock(d_model, n_heads)
    model.eval()

    graph = export_model(model, (torch.randn(batch, seq, d_model),))
    run_pipeline(graph)
    ep = plan(graph)

    executor = Executor(backends=[CBackend(), NumpyBackend()])
    compiled = executor.compile_plan(ep)

    for _ in range(5):
        x = torch.randn(batch, seq, d_model)
        with torch.no_grad():
            expected = model(x).numpy()
        result = executor.execute_compiled(compiled, {graph.inputs[0]: x.numpy().copy()})
        output = result[graph.outputs[0]]
        np.testing.assert_allclose(output, expected, atol=1e-4)
