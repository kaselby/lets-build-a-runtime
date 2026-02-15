"""End-to-end oracle tests: full pipeline compared against PyTorch.

These are the highest-leverage tests — a single test exercises the exporter,
optimization passes, memory planner, and executor all at once. If PyTorch
and our runtime agree on the output, the whole pipeline is correct.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from runtime.backends.c_backend import CBackend
from runtime.backends.numpy_backend import NumpyBackend
from runtime.exporter import export_model
from runtime.executor import InterpretedExecutor, CompiledExecutor
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


# ---------------------------------------------------------------------------
# MLP tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch,dim", MLP_CONFIGS)
@pytest.mark.parametrize("backend_name", ["numpy", "c"])
def test_mlp_interpreted(batch, dim, backend_name):
    """Interpreted dispatch path: export → optimize → plan → per-node execute."""
    model = SimpleMLP(dim)
    model.eval()
    x = torch.randn(batch, dim)

    with torch.no_grad():
        expected = model(x).numpy()

    graph = export_model(model, (x,))
    run_pipeline(graph)
    ep = plan(graph)

    if backend_name == "numpy":
        executor = InterpretedExecutor(backends=[NumpyBackend()])
    else:
        executor = InterpretedExecutor(backends=[CBackend(), NumpyBackend()])

    executor.compile(ep)
    result = executor.run({graph.inputs[0]: x.numpy().copy()})
    output = result[graph.outputs[0]]

    np.testing.assert_allclose(output, expected, atol=1e-4)


@pytest.mark.parametrize("batch,dim", MLP_CONFIGS)
def test_mlp_compiled(batch, dim):
    """Compiled C executor path: single ctypes call for the whole plan."""
    model = SimpleMLP(dim)
    model.eval()
    x = torch.randn(batch, dim)

    with torch.no_grad():
        expected = model(x).numpy()

    graph = export_model(model, (x,))
    run_pipeline(graph)
    ep = plan(graph)

    executor = CompiledExecutor()
    executor.compile(ep)
    result = executor.run({graph.inputs[0]: x.numpy().copy()})
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

    executor = CompiledExecutor()
    executor.compile(ep)

    for _ in range(5):
        x = torch.randn(batch, dim)
        with torch.no_grad():
            expected = model(x).numpy()

        result = executor.run({graph.inputs[0]: x.numpy().copy()})
        output = result[graph.outputs[0]]
        np.testing.assert_allclose(output, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Transformer end-to-end tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch,seq,d_model,n_heads", TRANSFORMER_CONFIGS)
@pytest.mark.parametrize("backend_name", ["numpy", "c"])
def test_transformer_interpreted(batch, seq, d_model, n_heads, backend_name):
    """Interpreted dispatch: transformer block matches PyTorch."""
    model = NaiveTransformerBlock(d_model, n_heads)
    model.eval()
    x = torch.randn(batch, seq, d_model)

    with torch.no_grad():
        expected = model(x).numpy()

    graph = export_model(model, (x,))
    run_pipeline(graph)
    ep = plan(graph)

    if backend_name == "numpy":
        executor = InterpretedExecutor(backends=[NumpyBackend()])
    else:
        executor = InterpretedExecutor(backends=[CBackend(), NumpyBackend()])

    executor.compile(ep)
    result = executor.run({graph.inputs[0]: x.numpy().copy()})
    output = result[graph.outputs[0]]
    np.testing.assert_allclose(output, expected, atol=1e-4)


@pytest.mark.parametrize("batch,seq,d_model,n_heads", TRANSFORMER_CONFIGS)
def test_transformer_compiled(batch, seq, d_model, n_heads):
    """Compiled C executor: transformer block matches PyTorch."""
    model = NaiveTransformerBlock(d_model, n_heads)
    model.eval()
    x = torch.randn(batch, seq, d_model)

    with torch.no_grad():
        expected = model(x).numpy()

    graph = export_model(model, (x,))
    run_pipeline(graph)
    ep = plan(graph)

    executor = CompiledExecutor()
    executor.compile(ep)
    result = executor.run({graph.inputs[0]: x.numpy().copy()})
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

    executor = CompiledExecutor()
    executor.compile(ep)

    for _ in range(5):
        x = torch.randn(batch, seq, d_model)
        with torch.no_grad():
            expected = model(x).numpy()
        result = executor.run({graph.inputs[0]: x.numpy().copy()})
        output = result[graph.outputs[0]]
        np.testing.assert_allclose(output, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# SDPA transformer end-to-end tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch,seq,d_model,n_heads", TRANSFORMER_CONFIGS)
@pytest.mark.parametrize("backend_name", ["numpy", "c"])
def test_sdpa_transformer_interpreted(batch, seq, d_model, n_heads, backend_name):
    """Interpreted dispatch: SDPA transformer block matches PyTorch."""
    model = SDPATransformerBlock(d_model, n_heads)
    model.eval()
    x = torch.randn(batch, seq, d_model)

    with torch.no_grad():
        expected = model(x).numpy()

    graph = export_model(model, (x,))
    run_pipeline(graph)
    ep = plan(graph)

    if backend_name == "numpy":
        executor = InterpretedExecutor(backends=[NumpyBackend()])
    else:
        executor = InterpretedExecutor(backends=[CBackend(), NumpyBackend()])

    executor.compile(ep)
    result = executor.run({graph.inputs[0]: x.numpy().copy()})
    output = result[graph.outputs[0]]
    np.testing.assert_allclose(output, expected, atol=1e-4)


@pytest.mark.parametrize("batch,seq,d_model,n_heads", TRANSFORMER_CONFIGS)
def test_sdpa_transformer_compiled(batch, seq, d_model, n_heads):
    """Compiled C executor: SDPA transformer block matches PyTorch."""
    model = SDPATransformerBlock(d_model, n_heads)
    model.eval()
    x = torch.randn(batch, seq, d_model)

    with torch.no_grad():
        expected = model(x).numpy()

    graph = export_model(model, (x,))
    run_pipeline(graph)
    ep = plan(graph)

    executor = CompiledExecutor()
    executor.compile(ep)
    result = executor.run({graph.inputs[0]: x.numpy().copy()})
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

    executor = CompiledExecutor()
    executor.compile(ep)

    for _ in range(5):
        x = torch.randn(batch, seq, d_model)
        with torch.no_grad():
            expected = model(x).numpy()
        result = executor.run({graph.inputs[0]: x.numpy().copy()})
        output = result[graph.outputs[0]]
        np.testing.assert_allclose(output, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Causal naive transformer end-to-end tests (explicit mask fusion)
# ---------------------------------------------------------------------------

from conftest import CausalNaiveTransformerBlock

CAUSAL_CONFIGS = [
    (1, 8, 64, 4),
    (2, 16, 64, 4),
]


@pytest.mark.parametrize("batch,seq,d_model,n_heads", CAUSAL_CONFIGS)
@pytest.mark.parametrize("backend_name", ["numpy", "c"])
def test_causal_transformer_interpreted(batch, seq, d_model, n_heads, backend_name):
    """Interpreted dispatch: causal naive transformer matches PyTorch."""
    model = CausalNaiveTransformerBlock(d_model, n_heads, seq_len=seq)
    model.eval()
    x = torch.randn(batch, seq, d_model)

    with torch.no_grad():
        expected = model(x).numpy()

    graph = export_model(model, (x,))
    run_pipeline(graph)
    ep = plan(graph)

    if backend_name == "numpy":
        executor = InterpretedExecutor(backends=[NumpyBackend()])
    else:
        executor = InterpretedExecutor(backends=[CBackend(), NumpyBackend()])

    executor.compile(ep)
    result = executor.run({graph.inputs[0]: x.numpy().copy()})
    output = result[graph.outputs[0]]
    np.testing.assert_allclose(output, expected, atol=1e-4)


@pytest.mark.parametrize("batch,seq,d_model,n_heads", CAUSAL_CONFIGS)
def test_causal_transformer_compiled(batch, seq, d_model, n_heads):
    """Compiled C executor: causal naive transformer matches PyTorch.

    The model uses explicit mask addition (not SDPA), which exercises
    the causal_attention fusion pattern and the C kernel's causal flag.
    """
    model = CausalNaiveTransformerBlock(d_model, n_heads, seq_len=seq)
    model.eval()
    x = torch.randn(batch, seq, d_model)

    with torch.no_grad():
        expected = model(x).numpy()

    graph = export_model(model, (x,))
    run_pipeline(graph)
    ep = plan(graph)

    executor = CompiledExecutor()
    executor.compile(ep)
    result = executor.run({graph.inputs[0]: x.numpy().copy()})
    output = result[graph.outputs[0]]
    np.testing.assert_allclose(output, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Interpreted vs compiled agreement
# ---------------------------------------------------------------------------

def test_interpreted_vs_compiled_mlp():
    """Both execution paths should produce identical results for an MLP."""
    model = SimpleMLP(128)
    model.eval()
    x = torch.randn(4, 128)

    graph = export_model(model, (x,))
    run_pipeline(graph)
    ep = plan(graph)
    inp = {graph.inputs[0]: x.numpy().copy()}

    interp = InterpretedExecutor(backends=[CBackend(), NumpyBackend()])
    interp.compile(ep)
    result_interp = interp.run(inp)

    compiled = CompiledExecutor()
    compiled.compile(ep)
    result_compiled = compiled.run(inp)

    for name in graph.outputs:
        np.testing.assert_allclose(
            result_interp[name], result_compiled[name], atol=1e-6,
            err_msg=f"Interpreted vs compiled mismatch for output '{name}'"
        )


def test_interpreted_vs_compiled_transformer():
    """Both execution paths should produce identical results for a transformer."""
    model = NaiveTransformerBlock(64, 4)
    model.eval()
    x = torch.randn(2, 16, 64)

    graph = export_model(model, (x,))
    run_pipeline(graph)
    ep = plan(graph)
    inp = {graph.inputs[0]: x.numpy().copy()}

    interp = InterpretedExecutor(backends=[CBackend(), NumpyBackend()])
    interp.compile(ep)
    result_interp = interp.run(inp)

    compiled = CompiledExecutor()
    compiled.compile(ep)
    result_compiled = compiled.run(inp)

    for name in graph.outputs:
        np.testing.assert_allclose(
            result_interp[name], result_compiled[name], atol=1e-5,
            err_msg=f"Interpreted vs compiled mismatch for output '{name}'"
        )


# ---------------------------------------------------------------------------
# GPT-2 body end-to-end tests
# ---------------------------------------------------------------------------

from transformers import GPT2LMHeadModel, GPT2Config


class GPT2Body(nn.Module):
    """GPT-2 transformer body: blocks + final layernorm. No embeddings or LM head."""
    def __init__(self, model):
        super().__init__()
        self.h = model.transformer.h
        self.ln_f = model.transformer.ln_f

    def forward(self, hidden_states):
        for block in self.h:
            hidden_states = block(hidden_states)[0]
        return self.ln_f(hidden_states)


def _make_gpt2_body():
    """Create a GPT-2 body model with 2 layers, 12 heads, 768 dim."""
    config = GPT2Config(
        n_layer=2,
        n_head=12,
        n_embd=768,
        vocab_size=50257,
    )
    full_model = GPT2LMHeadModel(config)
    full_model.eval()
    body = GPT2Body(full_model)
    body.eval()
    return body


GPT2_SEQ_LENGTHS = [1, 16, 64]


@pytest.mark.parametrize("seq_len", GPT2_SEQ_LENGTHS)
@pytest.mark.parametrize("backend_name", ["numpy", "c"])
def test_gpt2_body_interpreted(seq_len, backend_name):
    """Interpreted dispatch: GPT-2 body matches PyTorch across sequence lengths."""
    model = _make_gpt2_body()
    x = torch.randn(1, seq_len, 768)

    with torch.no_grad():
        expected = model(x).numpy()

    graph = export_model(model, (x,))
    run_pipeline(graph)
    ep = plan(graph)

    if backend_name == "numpy":
        executor = InterpretedExecutor(backends=[NumpyBackend()])
    else:
        executor = InterpretedExecutor(backends=[CBackend(), NumpyBackend()])

    executor.compile(ep)
    result = executor.run({graph.inputs[0]: x.numpy().copy()})
    output = result[graph.outputs[0]]
    np.testing.assert_allclose(output, expected, atol=1e-4)


@pytest.mark.parametrize("seq_len", GPT2_SEQ_LENGTHS)
def test_gpt2_body_compiled(seq_len):
    """Compiled C executor: GPT-2 body matches PyTorch."""
    model = _make_gpt2_body()
    x = torch.randn(1, seq_len, 768)

    with torch.no_grad():
        expected = model(x).numpy()

    graph = export_model(model, (x,))
    run_pipeline(graph)
    ep = plan(graph)

    executor = CompiledExecutor()
    executor.compile(ep)
    result = executor.run({graph.inputs[0]: x.numpy().copy()})
    output = result[graph.outputs[0]]
    np.testing.assert_allclose(output, expected, atol=1e-4)
