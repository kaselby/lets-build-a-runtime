"""Shared fixtures and helpers for the test suite.

pytest discovers conftest.py automatically — fixtures defined here are
available to all test files in this directory without explicit imports.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from runtime.backends.numpy_backend import NumpyBackend
from runtime.backends.c_backend import CBackend
from runtime.exporter import export_model
from runtime.executor import InterpretedExecutor, CompiledExecutor
from runtime.ir import OpType
from runtime.passes import run_pipeline
from runtime.planner import plan


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    """3-layer MLP: Linear→ReLU→Linear→ReLU→Linear."""
    def __init__(self, dim=128):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


class NaiveTransformerBlock(nn.Module):
    """Single transformer block with manual multi-head attention (F.softmax).

    No nn.MultiheadAttention — uses explicit Q/K/V projections, reshape
    into heads, matmul attention, and output projection. This is what our
    runtime can compile.
    """
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Linear(d_model, d_model * 4)
        self.ffn2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        B, S, D = x.shape
        h = self.ln1(x)
        q = self.wq(h).reshape(B, S, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        k = self.wk(h).reshape(B, S, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        v = self.wv(h).reshape(B, S, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = F.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        attn = attn.permute(0, 2, 1, 3).reshape(B, S, D)
        x = x + self.wo(attn)
        x = x + self.ffn2(torch.relu(self.ffn1(self.ln2(x))))
        return x


class CausalNaiveTransformerBlock(nn.Module):
    """Same as NaiveTransformerBlock but with explicit causal mask addition.

    Uses a registered buffer for the upper-triangular -inf mask and adds it
    to the attention scores before softmax. This produces the 4-node pattern
    MATMUL→ADD(mask)→SOFTMAX→MATMUL that the causal_attention fusion targets.

    seq_len is fixed at construction (must match input) since the mask is a
    registered buffer with a concrete shape.
    """
    def __init__(self, d_model=64, n_heads=4, seq_len=16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Linear(d_model, d_model * 4)
        self.ffn2 = nn.Linear(d_model * 4, d_model)
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, S, D = x.shape
        h = self.ln1(x)
        q = self.wq(h).reshape(B, S, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        k = self.wk(h).reshape(B, S, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        v = self.wv(h).reshape(B, S, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores + self.causal_mask
        weights = F.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        attn = attn.permute(0, 2, 1, 3).reshape(B, S, D)
        x = x + self.wo(attn)
        x = x + self.ffn2(torch.relu(self.ffn1(self.ln2(x))))
        return x


class SDPATransformerBlock(nn.Module):
    """Same architecture as NaiveTransformerBlock but using F.scaled_dot_product_attention.

    The exporter maps aten.scaled_dot_product_attention directly to our
    fused ATTENTION op — no fusion pass needed.
    """
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Linear(d_model, d_model * 4)
        self.ffn2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        B, S, D = x.shape
        h = self.ln1(x)
        q = self.wq(h).reshape(B, S, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        k = self.wk(h).reshape(B, S, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        v = self.wv(h).reshape(B, S, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 2, 1, 3).reshape(B, S, D)
        x = x + self.wo(attn)
        x = x + self.ffn2(torch.relu(self.ffn1(self.ln2(x))))
        return x


# ---------------------------------------------------------------------------
# Backend fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def np_backend():
    return NumpyBackend()


@pytest.fixture
def c_backend():
    return CBackend()


@pytest.fixture(params=["numpy", "c"], ids=["numpy", "C"])
def backend(request, np_backend, c_backend):
    """Parametrized fixture: tests using this run once per backend."""
    if request.param == "numpy":
        return np_backend
    return c_backend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_kernel(backend, op_name, inputs, output_shape, attrs=None):
    """Run a single kernel and return the output array."""
    op = getattr(OpType, op_name)
    kernel = backend.get_kernel(op)
    assert kernel is not None, f"No kernel for {op_name} in {backend.name}"
    output = np.zeros(output_shape, dtype=np.float32)
    kernel(inputs, output, attrs or {})
    return output


def run_interpreted(model, x_torch, backend_name="c"):
    """Full pipeline: export → optimize → plan → interpreted execute."""
    graph = export_model(model, (x_torch,))
    run_pipeline(graph)
    ep = plan(graph)
    if backend_name == "numpy":
        executor = InterpretedExecutor(backends=[NumpyBackend()])
    else:
        executor = InterpretedExecutor(backends=[CBackend(), NumpyBackend()])
    executor.compile(ep)
    return executor.run({graph.inputs[0]: x_torch.numpy().copy()})


def run_compiled(model, x_torch):
    """Full pipeline: export → optimize → plan → compiled execute."""
    graph = export_model(model, (x_torch,))
    run_pipeline(graph)
    ep = plan(graph)
    executor = CompiledExecutor()
    executor.compile(ep)
    return executor.run({graph.inputs[0]: x_torch.numpy().copy()})
