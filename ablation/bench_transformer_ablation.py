"""
Transformer ablation benchmark: attention / layernorm / linear / element-wise / data movement / other time breakdown.

Measures performance across:
  - PyTorch baselines (naive F.softmax + SDPA)
  - ONNX Runtime baselines (no optimizations, full optimizations, multi-threaded)
  - Python executor (numpy + C backends, with fusion)
  - Compiled C executor with incremental fusion
  - Attention kernel variants (scalar, SIMD, SIMD+GCD)
  - LayerNorm kernel variants (scalar, SIMD, SIMD+GCD)

For each variant, reports total time and six-way breakdown:
attention, layernorm, linear, element-wise, data movement, and everything else.

Run:  python bench_transformer_ablation.py [--configs toy,small,...] [--skip-large]
"""

import argparse
import ctypes
import math
import os
import sys
import tempfile
import warnings
from pathlib import Path

# Ensure the project root is on the path so 'runtime' is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from runtime.backends.c_backend import CBackend
from runtime.backends.numpy_backend import NumpyBackend
from runtime.exporter import export_model
from runtime.executor import (
    COpNode, CompiledExecutor, InterpretedExecutor, MAX_DIMS, MAX_INPUTS,
)
from runtime.ir import Graph, OpType
from runtime.passes import (
    FUSION_PATTERNS,
    absorb_into_matmul,
    absorb_mask_into_attention,
    constant_fold,
    eliminate_dead_code,
    fuse,
    fuse_dags,
    merge_parallel_matmuls,
)
from runtime.ops import OP_REGISTRY
from runtime.planner import plan

# ---------------------------------------------------------------------------
# Scratch override for ablation
#
# The main planner allocates flash-sized scratch (BR*BC) for long sequences,
# but ablation needs to run both standard (S*S) and flash (various tile sizes)
# from the same plan. We temporarily override the attention scratch calculator
# to allocate max(S*S, largest_tile) per slice, ensuring every variant fits.
# ---------------------------------------------------------------------------
_MAX_ABLATION_TILE = 256 * 512  # largest flash block size in ablation


def _ablation_attention_scratch(input_shapes, output_shape, attrs) -> int:
    q_shape = input_shapes[0]
    batch_heads = 1
    for d in q_shape[:-2]:
        batch_heads *= d
    seq_len = q_shape[-2]
    per_slice = max(seq_len * seq_len, _MAX_ABLATION_TILE)
    return batch_heads * per_slice * 4  # float32


def _plan_for_ablation(graph):
    """Plan with scratch large enough for all attention variants."""
    attn_def = OP_REGISTRY[OpType.ATTENTION]
    original_scratch = attn_def.scratch
    attn_def.scratch = _ablation_attention_scratch
    try:
        return plan(graph)
    finally:
        attn_def.scratch = original_scratch


try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    from transformers import GPT2LMHeadModel, GPT2Config
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from transformers import AutoConfig, AutoModelForCausalLM
    HAS_QWEN3 = True
except ImportError:
    HAS_QWEN3 = False


# =====================================================================
# Models (duplicated from conftest to keep benchmark self-contained)
# =====================================================================

class NaiveTransformerBlock(nn.Module):
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


class SDPATransformerBlock(nn.Module):
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


def _make_gpt2(n_layer=2, n_head=12, n_embd=768):
    """Create a full GPT-2 model. Returns None if transformers unavailable."""
    if not HAS_TRANSFORMERS:
        return None
    config = GPT2Config(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        vocab_size=50257,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    model.config.use_cache = False
    return model


class CausalLMWrapper(nn.Module):
    """Wrapper that returns logits tensor directly (no CausalLMOutput)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids, use_cache=False).logits


def _make_qwen3(variant="0.6B", n_layer=2):
    """Create a Qwen3 model with random weights. Returns (wrapper, raw_model) or None."""
    if not HAS_QWEN3:
        return None
    config = AutoConfig.from_pretrained(f"Qwen/Qwen3-{variant}", trust_remote_code=True)
    config.num_hidden_layers = n_layer
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True,
                                              torch_dtype=torch.float32)
    model.eval()
    model.config.use_cache = False
    wrapper = CausalLMWrapper(model)
    wrapper.eval()
    return wrapper, model


# =====================================================================
# Configs
# =====================================================================

@dataclass
class Config:
    name: str
    d_model: int
    n_heads: int
    seq_len: int
    batch: int = 1
    warmup: int = 20
    iters: int = 100
    model: str = "naive"  # "naive", "gpt2", "qwen3", or "qwen3-4B"

ALL_CONFIGS = [
    Config("Toy",     64,   4,   32,   warmup=5, iters=10),
    Config("Small",   256,  4,   128,  warmup=5, iters=10),
    Config("Medium",  512,  8,   256,  warmup=5, iters=10),
    Config("GPT-2",   768,  12,  512,  warmup=5, iters=10),
    Config("1B",      2048, 16,  512,  warmup=5, iters=10),
    Config("3B",      3072, 24,  1024, warmup=5, iters=10),
    Config("7B",      4096, 32,  1024, warmup=5, iters=10),
    Config("7B-4K",   4096, 32,  4096, warmup=3, iters=10),
    Config("1B-8K",   2048, 16,  8192, warmup=3, iters=10),
    # GPT-2 body (HuggingFace, 2-layer, causal attention)
    Config("gpt2-s16",  768, 12,  16,  warmup=5, iters=10, model="gpt2"),
    Config("gpt2-s64",  768, 12,  64,  warmup=5, iters=10, model="gpt2"),
    Config("gpt2-s256", 768, 12, 256,  warmup=5, iters=10, model="gpt2"),
    Config("gpt2-s1024", 768, 12, 1024, warmup=3, iters=5,  model="gpt2"),
    # Qwen3-0.6B body (HuggingFace, 2-layer, GQA 2:1 + RoPE + SiLU gated MLP)
    Config("q3-0.6B-s16",   1024, 16,   16,  warmup=5, iters=10, model="qwen3"),
    Config("q3-0.6B-s256",  1024, 16,  256,  warmup=5, iters=10, model="qwen3"),
    Config("q3-0.6B-s1024", 1024, 16,  1024, warmup=3, iters=10, model="qwen3"),
    Config("q3-0.6B-s4K",   1024, 16,  4096, warmup=2, iters=5,  model="qwen3"),
    Config("q3-0.6B-s8K",   1024, 16,  8192, warmup=2, iters=3,  model="qwen3"),
    Config("q3-0.6B-s16K",  1024, 16, 16384, warmup=2, iters=3,  model="qwen3"),
    # Qwen3-4B body (HuggingFace, 2-layer, GQA 4:1 + RoPE + SiLU gated MLP)
    Config("q3-4B-s16",     2560, 32,   16,  warmup=5, iters=10, model="qwen3-4B"),
    Config("q3-4B-s256",    2560, 32,  256,  warmup=3, iters=10, model="qwen3-4B"),
    Config("q3-4B-s1024",   2560, 32,  1024, warmup=3, iters=5,  model="qwen3-4B"),
    Config("q3-4B-s4K",     2560, 32,  4096, warmup=2, iters=3,  model="qwen3-4B"),
    Config("q3-4B-s8K",     2560, 32,  8192, warmup=2, iters=3,  model="qwen3-4B"),
    Config("q3-4B-s16K",    2560, 32, 16384, warmup=2, iters=3,  model="qwen3-4B"),
]

# Configs with d_model >= 2048 use the reduced variant set
LARGE_THRESHOLD = 2048


# =====================================================================
# Pass pipelines (incremental)
# =====================================================================

_patterns = {p.name: p for p in FUSION_PATTERNS}
P_MATMUL_ADD = _patterns["matmul_add"]
P_BIAS_RELU = _patterns["bias_relu"]
P_ATTENTION = _patterns["attention"]


def pipeline_none(graph):
    pass

def pipeline_fold_dce(graph):
    constant_fold(graph)
    eliminate_dead_code(graph)

def pipeline_absorb(graph):
    absorb_into_matmul(graph)
    constant_fold(graph)
    eliminate_dead_code(graph)

def pipeline_matmul_add(graph):
    absorb_into_matmul(graph)
    constant_fold(graph)
    fuse(graph, patterns=[P_MATMUL_ADD])
    eliminate_dead_code(graph)

def pipeline_bias_relu(graph):
    absorb_into_matmul(graph)
    constant_fold(graph)
    fuse(graph, patterns=[P_MATMUL_ADD, P_BIAS_RELU])
    eliminate_dead_code(graph)

def pipeline_full_no_merge(graph):
    absorb_into_matmul(graph)
    constant_fold(graph)
    absorb_mask_into_attention(graph)
    fuse_dags(graph)
    fuse(graph)
    eliminate_dead_code(graph)

def pipeline_full(graph):
    absorb_into_matmul(graph)
    constant_fold(graph)
    absorb_mask_into_attention(graph)
    fuse_dags(graph)
    merge_parallel_matmuls(graph)
    fuse(graph)
    eliminate_dead_code(graph)


# =====================================================================
# Load ablation C library
# =====================================================================

def _load_ablation_lib():
    ablation_dir = Path(__file__).parent
    if sys.platform == "darwin":
        lib_path = ablation_dir / "libablation.dylib"
    else:
        lib_path = ablation_dir / "libablation.so"
    if not lib_path.exists():
        raise RuntimeError(f"Ablation library not found at {lib_path} — run make in ablation/")
    lib = ctypes.CDLL(str(lib_path))

    lib.execute_ablation.argtypes = [
        ctypes.POINTER(COpNode), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.execute_ablation.restype = None

    lib.timed_execute_ablation.argtypes = [
        ctypes.POINTER(COpNode), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.timed_execute_ablation.restype = None

    return lib


# =====================================================================
# Op classification for timing breakdown
# =====================================================================

def classify_op_indices(graph, exec_order):
    """Return dict of category name -> set of indices into the COpNode array.

    Must match the compiled executor's skip logic exactly: skip alias ops
    whose first input is internal (not a graph input or constant). This
    ensures indices here correspond 1:1 with COpNode array positions.

    Categories:
      "attn"    — fused ATTENTION nodes, or SOFTMAX + neighboring MATMULs in unfused graphs
      "ln"      — LAYERNORM or RMSNORM
      "linear"  — MATMUL, MATMUL_ADD that are NOT in the attn set
      "elemwise" — ADD, SUB, MUL, DIV, RELU, EXP, TANH, POW, GELU, SILU, NEG, COS, SIN,
                   RSQRT, FUSED_BIAS_RELU, GATED_ACT
      "move"    — TRANSPOSE, PERMUTE, SLICE, EMBEDDING
    """
    from runtime.ops import OP_REGISTRY

    ELEMWISE_OPS = {
        OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV,
        OpType.RELU, OpType.EXP, OpType.TANH, OpType.POW,
        OpType.GELU, OpType.SILU, OpType.NEG, OpType.COS, OpType.SIN,
        OpType.RSQRT, OpType.FUSED_BIAS_RELU, OpType.GATED_ACT,
    }
    MOVE_OPS = {OpType.TRANSPOSE, OpType.PERMUTE, OpType.SLICE}
    LN_OPS = {OpType.LAYERNORM, OpType.RMSNORM}
    LINEAR_OPS = {OpType.MATMUL, OpType.MATMUL_ADD}

    # First pass: identify attention node IDs
    attn_ids = set()
    for node in exec_order:
        if node.op == OpType.ATTENTION:
            attn_ids.add(node.id)
        elif node.op == OpType.SOFTMAX:
            attn_ids.add(node.id)
            producer = graph.producer(node.inputs[0])
            if producer and producer.op == OpType.MATMUL:
                attn_ids.add(producer.id)
            for c in graph.consumers(node.output):
                if c.op == OpType.MATMUL:
                    attn_ids.add(c.id)

    # Second pass: classify nodes using the same skip logic as CompiledExecutor
    external = set(graph.inputs) | set(graph.constants)
    dispatched = []
    for n in exec_order:
        op_def = OP_REGISTRY.get(n.op)
        if op_def is not None and op_def.is_alias(n) and n.inputs[0] not in external:
            continue
        dispatched.append(n)

    categories = {"attn": set(), "ln": set(), "linear": set(),
                  "elemwise": set(), "move": set()}

    for i, n in enumerate(dispatched):
        if n.id in attn_ids:
            categories["attn"].add(i)
        elif n.op in LN_OPS:
            categories["ln"].add(i)
        elif n.op in LINEAR_OPS:
            categories["linear"].add(i)
        elif n.op in ELEMWISE_OPS:
            categories["elemwise"].add(i)
        elif n.op in MOVE_OPS:
            categories["move"].add(i)
        # else: uncategorized → implicit "other" (EMBEDDING, non-alias RESHAPE, etc.)

    return categories


# =====================================================================
# Timing helpers
# =====================================================================

def _patch_inputs(executor, inputs):
    """Patch graph input pointers into the compiled executor's COpNode array."""
    for name, slots in executor._input_slots.items():
        ptr = inputs[name].ctypes.data
        for node_idx, slot_idx in slots:
            executor._nodes[node_idx].inputs[slot_idx] = ptr


def bench_compiled_timed(lib, executor, inputs, n_nodes,
                         categories,
                         softmax_mode, attn_mode, layernorm_mode,
                         warmup, iters):
    """Benchmark a compiled executor with per-op timing via the ablation library.

    Returns a Result with per-category timing — median across iterations.
    """
    _patch_inputs(executor, inputs)
    nodes = executor._nodes
    times_buf = (ctypes.c_double * n_nodes)()

    # Pre-compute the union of all categorized indices for "other" calculation
    all_categorized = set()
    for idx_set in categories.values():
        all_categorized |= idx_set

    # Warmup (no timing)
    for _ in range(warmup):
        lib.execute_ablation(nodes, n_nodes,
                             softmax_mode, attn_mode, layernorm_mode)

    # Timed iterations — collect per-iteration aggregates
    cat_keys = ["attn", "ln", "linear", "elemwise", "move"]
    per_iter = {k: [] for k in cat_keys + ["other", "total"]}

    for _ in range(iters):
        lib.timed_execute_ablation(nodes, n_nodes, times_buf,
                                   softmax_mode, attn_mode, layernorm_mode)
        cat_ns = {}
        for key in cat_keys:
            cat_ns[key] = sum(times_buf[i] for i in categories.get(key, set()))
        other_ns = sum(times_buf[i] for i in range(n_nodes)
                       if i not in all_categorized)
        total_ns = sum(cat_ns.values()) + other_ns

        per_iter["total"].append(total_ns / 1000)
        for key in cat_keys:
            per_iter[key].append(cat_ns[key] / 1000)
        per_iter["other"].append(other_ns / 1000)

    return Result(
        total_us=_median(per_iter["total"]),
        attn_us=_median(per_iter["attn"]),
        ln_us=_median(per_iter["ln"]),
        linear_us=_median(per_iter["linear"]),
        elemwise_us=_median(per_iter["elemwise"]),
        move_us=_median(per_iter["move"]),
        other_us=_median(per_iter["other"]),
    )


def bench_pytorch_naive(model, x, warmup, iters):
    """Benchmark NaiveTransformerBlock with attention + layernorm + linear timing."""
    B, S, D = x.shape
    m = model

    with torch.no_grad():
        for _ in range(warmup):
            m(x)

    per_iter = {k: [] for k in ["total", "attn", "ln", "linear", "elem", "move"]}
    with torch.no_grad():
        for _ in range(iters):
            t_start = time.perf_counter_ns()

            tln = time.perf_counter_ns()
            h = m.ln1(x)
            ln_time = time.perf_counter_ns() - tln

            tl = time.perf_counter_ns()
            q_raw = m.wq(h)
            k_raw = m.wk(h)
            v_raw = m.wv(h)
            linear_time = time.perf_counter_ns() - tl

            tmv = time.perf_counter_ns()
            q = q_raw.reshape(B, S, m.n_heads, m.d_k).permute(0, 2, 1, 3)
            k = k_raw.reshape(B, S, m.n_heads, m.d_k).permute(0, 2, 1, 3)
            v = v_raw.reshape(B, S, m.n_heads, m.d_k).permute(0, 2, 1, 3)
            move_time = time.perf_counter_ns() - tmv

            ta = time.perf_counter_ns()
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(m.d_k)
            weights = F.softmax(scores, dim=-1)
            attn = torch.matmul(weights, v)
            attn_time = time.perf_counter_ns() - ta

            tmv2 = time.perf_counter_ns()
            attn = attn.permute(0, 2, 1, 3).reshape(B, S, D)
            move_time += time.perf_counter_ns() - tmv2

            tl2 = time.perf_counter_ns()
            wo_out = m.wo(attn)
            linear_time += time.perf_counter_ns() - tl2

            te = time.perf_counter_ns()
            x_out = x + wo_out
            elem_time = time.perf_counter_ns() - te

            tln2 = time.perf_counter_ns()
            ln2_out = m.ln2(x_out)
            ln_time += time.perf_counter_ns() - tln2

            tl3 = time.perf_counter_ns()
            ffn1_out = m.ffn1(ln2_out)
            linear_time += time.perf_counter_ns() - tl3

            te2 = time.perf_counter_ns()
            relu_out = torch.relu(ffn1_out)
            elem_time += time.perf_counter_ns() - te2

            tl4 = time.perf_counter_ns()
            ffn2_out = m.ffn2(relu_out)
            linear_time += time.perf_counter_ns() - tl4

            te3 = time.perf_counter_ns()
            x_out = x_out + ffn2_out
            elem_time += time.perf_counter_ns() - te3

            total_ns = time.perf_counter_ns() - t_start
            per_iter["total"].append(total_ns / 1000)
            per_iter["attn"].append(attn_time / 1000)
            per_iter["ln"].append(ln_time / 1000)
            per_iter["linear"].append(linear_time / 1000)
            per_iter["elem"].append(elem_time / 1000)
            per_iter["move"].append(move_time / 1000)

    med = {k: _median(v) for k, v in per_iter.items()}
    other = med["total"] - med["attn"] - med["ln"] - med["linear"] - med["elem"] - med["move"]
    return Result(
        med["total"], med["attn"], med["ln"],
        med["linear"], med["elem"], med["move"], max(other, 0.0),
    )


def bench_pytorch_sdpa(model, x, warmup, iters):
    """Benchmark SDPATransformerBlock with attention + layernorm + linear timing."""
    B, S, D = x.shape
    m = model

    with torch.no_grad():
        for _ in range(warmup):
            m(x)

    per_iter = {k: [] for k in ["total", "attn", "ln", "linear", "elem", "move"]}
    with torch.no_grad():
        for _ in range(iters):
            t_start = time.perf_counter_ns()

            tln = time.perf_counter_ns()
            h = m.ln1(x)
            ln_time = time.perf_counter_ns() - tln

            tl = time.perf_counter_ns()
            q_raw = m.wq(h)
            k_raw = m.wk(h)
            v_raw = m.wv(h)
            linear_time = time.perf_counter_ns() - tl

            tmv = time.perf_counter_ns()
            q = q_raw.reshape(B, S, m.n_heads, m.d_k).permute(0, 2, 1, 3)
            k = k_raw.reshape(B, S, m.n_heads, m.d_k).permute(0, 2, 1, 3)
            v = v_raw.reshape(B, S, m.n_heads, m.d_k).permute(0, 2, 1, 3)
            move_time = time.perf_counter_ns() - tmv

            ta = time.perf_counter_ns()
            attn = F.scaled_dot_product_attention(q, k, v)
            attn_time = time.perf_counter_ns() - ta

            tmv2 = time.perf_counter_ns()
            attn = attn.permute(0, 2, 1, 3).reshape(B, S, D)
            move_time += time.perf_counter_ns() - tmv2

            tl2 = time.perf_counter_ns()
            wo_out = m.wo(attn)
            linear_time += time.perf_counter_ns() - tl2

            te = time.perf_counter_ns()
            x_out = x + wo_out
            elem_time = time.perf_counter_ns() - te

            tln2 = time.perf_counter_ns()
            ln2_out = m.ln2(x_out)
            ln_time += time.perf_counter_ns() - tln2

            tl3 = time.perf_counter_ns()
            ffn1_out = m.ffn1(ln2_out)
            linear_time += time.perf_counter_ns() - tl3

            te2 = time.perf_counter_ns()
            relu_out = torch.relu(ffn1_out)
            elem_time += time.perf_counter_ns() - te2

            tl4 = time.perf_counter_ns()
            ffn2_out = m.ffn2(relu_out)
            linear_time += time.perf_counter_ns() - tl4

            te3 = time.perf_counter_ns()
            x_out = x_out + ffn2_out
            elem_time += time.perf_counter_ns() - te3

            total_ns = time.perf_counter_ns() - t_start
            per_iter["total"].append(total_ns / 1000)
            per_iter["attn"].append(attn_time / 1000)
            per_iter["ln"].append(ln_time / 1000)
            per_iter["linear"].append(linear_time / 1000)
            per_iter["elem"].append(elem_time / 1000)
            per_iter["move"].append(move_time / 1000)

    med = {k: _median(v) for k, v in per_iter.items()}
    other = med["total"] - med["attn"] - med["ln"] - med["linear"] - med["elem"] - med["move"]
    return Result(
        med["total"], med["attn"], med["ln"],
        med["linear"], med["elem"], med["move"], max(other, 0.0),
    )


def bench_pytorch_gpt2(model, x, warmup, iters):
    """Benchmark full GPT2LMHeadModel with per-component timing.

    Manually decomposes the forward pass to instrument each component:
      - LN: ln_1, ln_2 per block + ln_f
      - Linear: c_attn, c_proj (attn), c_fc, c_proj (mlp), lm_head
      - Attn: scaled_dot_product_attention call
      - Elem: GELU activations + residual adds
      - Move: head split/merge reshapes + transposes
      - Other: token/position embeddings
    """
    m = model
    tfm = m.transformer
    B, S = x.shape
    position_ids = torch.arange(S, device=x.device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(warmup):
            m(x)

    per_iter = {k: [] for k in ["total", "attn", "ln", "linear", "elem", "move", "other"]}
    with torch.no_grad():
        for _ in range(iters):
            ln_time = 0
            linear_time = 0
            attn_time = 0
            elem_time = 0
            move_time = 0
            other_time = 0

            t_start = time.perf_counter_ns()

            # Embeddings (other)
            t0 = time.perf_counter_ns()
            inputs_embeds = tfm.wte(x)
            position_embeds = tfm.wpe(position_ids)
            other_time += time.perf_counter_ns() - t0

            # Embedding add (elem)
            t0 = time.perf_counter_ns()
            h = inputs_embeds + position_embeds
            elem_time += time.perf_counter_ns() - t0

            for block in tfm.h:
                n_heads = block.attn.num_heads
                head_dim = block.attn.head_dim

                # LayerNorm 1
                t0 = time.perf_counter_ns()
                ln_out = block.ln_1(h)
                ln_time += time.perf_counter_ns() - t0

                # QKV projection (linear)
                t0 = time.perf_counter_ns()
                qkv = block.attn.c_attn(ln_out)
                linear_time += time.perf_counter_ns() - t0

                # Split Q/K/V + reshape to heads (move)
                t0 = time.perf_counter_ns()
                q, k, v = qkv.split(block.attn.split_size, dim=2)
                q = q.view(B, S, n_heads, head_dim).transpose(1, 2)
                k = k.view(B, S, n_heads, head_dim).transpose(1, 2)
                v = v.view(B, S, n_heads, head_dim).transpose(1, 2)
                move_time += time.perf_counter_ns() - t0

                # Attention (SDPA)
                t0 = time.perf_counter_ns()
                attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                attn_time += time.perf_counter_ns() - t0

                # Merge heads (move)
                t0 = time.perf_counter_ns()
                attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, n_heads * head_dim)
                move_time += time.perf_counter_ns() - t0

                # Output projection (linear)
                t0 = time.perf_counter_ns()
                attn_out = block.attn.c_proj(attn_out)
                linear_time += time.perf_counter_ns() - t0

                # Residual (elem)
                t0 = time.perf_counter_ns()
                h = h + attn_out
                elem_time += time.perf_counter_ns() - t0

                residual = h

                # LayerNorm 2
                t0 = time.perf_counter_ns()
                ln_out = block.ln_2(h)
                ln_time += time.perf_counter_ns() - t0

                # FFN: fc (linear)
                t0 = time.perf_counter_ns()
                mlp_h = block.mlp.c_fc(ln_out)
                linear_time += time.perf_counter_ns() - t0

                # GELU (elem)
                t0 = time.perf_counter_ns()
                mlp_h = block.mlp.act(mlp_h)
                elem_time += time.perf_counter_ns() - t0

                # FFN: proj (linear)
                t0 = time.perf_counter_ns()
                mlp_h = block.mlp.c_proj(mlp_h)
                linear_time += time.perf_counter_ns() - t0

                # Residual (elem)
                t0 = time.perf_counter_ns()
                h = residual + mlp_h
                elem_time += time.perf_counter_ns() - t0

            # Final layernorm
            t0 = time.perf_counter_ns()
            h = tfm.ln_f(h)
            ln_time += time.perf_counter_ns() - t0

            # LM head (linear)
            t0 = time.perf_counter_ns()
            logits = m.lm_head(h)
            linear_time += time.perf_counter_ns() - t0

            total_ns = time.perf_counter_ns() - t_start
            per_iter["total"].append(total_ns / 1000)
            per_iter["attn"].append(attn_time / 1000)
            per_iter["ln"].append(ln_time / 1000)
            per_iter["linear"].append(linear_time / 1000)
            per_iter["elem"].append(elem_time / 1000)
            per_iter["move"].append(move_time / 1000)
            per_iter["other"].append(other_time / 1000)

    med = {k: _median(v) for k, v in per_iter.items()}
    accounted = med["attn"] + med["ln"] + med["linear"] + med["elem"] + med["move"] + med["other"]
    overhead = max(med["total"] - accounted, 0.0)
    return Result(
        med["total"], med["attn"], med["ln"],
        med["linear"], med["elem"], med["move"], med["other"] + overhead,
    )


def bench_pytorch_qwen3(wrapper, x, warmup, iters):
    """Benchmark Qwen3 CausalLM with per-component timing.

    Manually decomposes the forward pass to instrument each component:
      - LN: input_layernorm, post_attention_layernorm per block + final norm (RMSNorm)
      - Linear: q/k/v/o_proj (attn), gate/up/down_proj (mlp), lm_head
      - Attn: scaled_dot_product_attention call
      - Elem: SiLU activations, gate*up multiply, residual adds
      - Move: head reshape/transpose, merge heads
      - Other: token embedding, RoPE (rotary position encoding), QK head norms
    """
    m = wrapper.model  # unwrap CausalLMWrapper → Qwen3ForCausalLM
    tfm = m.model      # Qwen3Model
    B, S = x.shape

    # Pre-compute rotary embeddings (shared across iterations)
    with torch.no_grad():
        # rotary_emb needs a dummy hidden to infer device/dtype
        dummy_h = tfm.embed_tokens(x)
        position_ids = torch.arange(S, device=x.device).unsqueeze(0)
        cos, sin = tfm.rotary_emb(dummy_h, position_ids)

    # Helper: apply rotary position embedding
    def _rotate_half(t):
        t1 = t[..., :t.shape[-1] // 2]
        t2 = t[..., t.shape[-1] // 2:]
        return torch.cat((-t2, t1), dim=-1)

    def _apply_rope(q, k, cos, sin):
        q_out = (q * cos) + (_rotate_half(q) * sin)
        k_out = (k * cos) + (_rotate_half(k) * sin)
        return q_out, k_out

    # Helper: repeat KV heads for GQA
    def _repeat_kv(hidden_states, n_rep):
        if n_rep == 1:
            return hidden_states
        batch, n_kv_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, n_kv_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, n_kv_heads * n_rep, slen, head_dim)

    with torch.no_grad():
        for _ in range(warmup):
            wrapper(x)

    per_iter = {k: [] for k in ["total", "attn", "ln", "linear", "elem", "move", "other"]}
    with torch.no_grad():
        for _ in range(iters):
            ln_time = 0
            linear_time = 0
            attn_time = 0
            elem_time = 0
            move_time = 0
            other_time = 0

            t_start = time.perf_counter_ns()

            # Token embedding (other)
            t0 = time.perf_counter_ns()
            h = tfm.embed_tokens(x)
            other_time += time.perf_counter_ns() - t0

            for layer in tfm.layers:
                sa = layer.self_attn
                head_dim = sa.head_dim
                n_kv_groups = sa.num_key_value_groups
                n_heads = sa.config.num_attention_heads
                n_kv_heads = sa.config.num_key_value_heads
                n_rep = n_kv_groups  # == n_heads // n_kv_heads

                # Input layernorm (LN)
                t0 = time.perf_counter_ns()
                ln_out = layer.input_layernorm(h)
                ln_time += time.perf_counter_ns() - t0

                # Q/K/V projections (linear)
                t0 = time.perf_counter_ns()
                q = sa.q_proj(ln_out)
                k = sa.k_proj(ln_out)
                v = sa.v_proj(ln_out)
                linear_time += time.perf_counter_ns() - t0

                # Reshape to heads (move)
                t0 = time.perf_counter_ns()
                q = q.view(B, S, n_heads, head_dim).transpose(1, 2)
                k = k.view(B, S, n_kv_heads, head_dim).transpose(1, 2)
                v = v.view(B, S, n_kv_heads, head_dim).transpose(1, 2)
                move_time += time.perf_counter_ns() - t0

                # QK head norms (other — Qwen3-specific)
                t0 = time.perf_counter_ns()
                q = sa.q_norm(q)
                k = sa.k_norm(k)
                other_time += time.perf_counter_ns() - t0

                # RoPE (other)
                t0 = time.perf_counter_ns()
                q, k = _apply_rope(q, k, cos, sin)
                other_time += time.perf_counter_ns() - t0

                # GQA: repeat KV heads (move)
                t0 = time.perf_counter_ns()
                k = _repeat_kv(k, n_rep)
                v = _repeat_kv(v, n_rep)
                move_time += time.perf_counter_ns() - t0

                # Attention (SDPA)
                t0 = time.perf_counter_ns()
                attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                attn_time += time.perf_counter_ns() - t0

                # Merge heads (move)
                t0 = time.perf_counter_ns()
                attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, S, -1)
                move_time += time.perf_counter_ns() - t0

                # Output projection (linear)
                t0 = time.perf_counter_ns()
                attn_out = sa.o_proj(attn_out)
                linear_time += time.perf_counter_ns() - t0

                # Residual (elem)
                t0 = time.perf_counter_ns()
                h = h + attn_out
                elem_time += time.perf_counter_ns() - t0

                residual = h

                # Post-attention layernorm (LN)
                t0 = time.perf_counter_ns()
                ln_out = layer.post_attention_layernorm(h)
                ln_time += time.perf_counter_ns() - t0

                # MLP: gate + up projections (linear)
                t0 = time.perf_counter_ns()
                gate = layer.mlp.gate_proj(ln_out)
                up = layer.mlp.up_proj(ln_out)
                linear_time += time.perf_counter_ns() - t0

                # SiLU activation + gate*up multiply (elem)
                t0 = time.perf_counter_ns()
                gate = layer.mlp.act_fn(gate)
                mlp_h = gate * up
                elem_time += time.perf_counter_ns() - t0

                # MLP: down projection (linear)
                t0 = time.perf_counter_ns()
                mlp_h = layer.mlp.down_proj(mlp_h)
                linear_time += time.perf_counter_ns() - t0

                # Residual (elem)
                t0 = time.perf_counter_ns()
                h = residual + mlp_h
                elem_time += time.perf_counter_ns() - t0

            # Final norm (LN)
            t0 = time.perf_counter_ns()
            h = tfm.norm(h)
            ln_time += time.perf_counter_ns() - t0

            # LM head (linear)
            t0 = time.perf_counter_ns()
            logits = m.lm_head(h)
            linear_time += time.perf_counter_ns() - t0

            total_ns = time.perf_counter_ns() - t_start
            per_iter["total"].append(total_ns / 1000)
            per_iter["attn"].append(attn_time / 1000)
            per_iter["ln"].append(ln_time / 1000)
            per_iter["linear"].append(linear_time / 1000)
            per_iter["elem"].append(elem_time / 1000)
            per_iter["move"].append(move_time / 1000)
            per_iter["other"].append(other_time / 1000)

    med = {k: _median(v) for k, v in per_iter.items()}
    accounted = med["attn"] + med["ln"] + med["linear"] + med["elem"] + med["move"] + med["other"]
    overhead = max(med["total"] - accounted, 0.0)
    return Result(
        med["total"], med["attn"], med["ln"],
        med["linear"], med["elem"], med["move"], med["other"] + overhead,
    )


def bench_python_executor(executor, ep, graph, warmup, iters):
    """Benchmark per-op Python executor with per-op timing."""
    inp_name = graph.inputs[0]
    x_np = graph.tensors[inp_name].buffer

    ELEMWISE_OPS = {
        OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV,
        OpType.RELU, OpType.EXP, OpType.TANH, OpType.POW,
        OpType.GELU, OpType.SILU, OpType.NEG, OpType.COS, OpType.SIN,
        OpType.RSQRT, OpType.FUSED_BIAS_RELU, OpType.GATED_ACT,
    }
    MOVE_OPS = {OpType.TRANSPOSE, OpType.PERMUTE}
    LN_OPS = {OpType.LAYERNORM, OpType.RMSNORM}
    LINEAR_OPS = {OpType.MATMUL, OpType.MATMUL_ADD}

    # Classify attention node IDs (same logic as classify_op_indices)
    attn_ids = set()
    for node in ep.order:
        if node.op == OpType.ATTENTION:
            attn_ids.add(node.id)
        elif node.op == OpType.SOFTMAX:
            attn_ids.add(node.id)
            producer = graph.producer(node.inputs[0])
            if producer and producer.op == OpType.MATMUL:
                attn_ids.add(producer.id)
            for c in graph.consumers(node.output):
                if c.op == OpType.MATMUL:
                    attn_ids.add(c.id)

    # Compile and warmup via the standard API
    executor.compile(ep)
    for _ in range(warmup):
        executor.run({inp_name: x_np})

    # Timed — manually dispatch with per-op timing
    # (reach into internals for fine-grained measurement)
    from runtime.ops import OP_REGISTRY
    arena = executor._get_arena(ep.arena_size)
    external = set(graph.inputs) | set(graph.constants)
    cat_keys = ["attn", "ln", "linear", "elemwise", "move", "other"]
    per_iter = {k: [] for k in cat_keys + ["total"]}

    for _ in range(iters):
        executor._bind_inputs(graph, {inp_name: x_np})
        executor._bind_intermediates(graph, ep.offsets, arena)

        cat_ns = {k: 0 for k in cat_keys}
        for node in ep.order:
            op_def = OP_REGISTRY.get(node.op)
            if op_def is not None and op_def.is_alias(node) and node.inputs[0] not in external:
                continue
            kernel = executor._resolve_kernel(node.op)
            input_buffers = [graph.tensors[inp].buffer for inp in node.inputs]
            if node.id in ep.scratch:
                offset, size_bytes = ep.scratch[node.id]
                input_buffers.append(arena[offset:offset + size_bytes])
            output_buffer = graph.tensors[node.output].buffer

            t0 = time.perf_counter_ns()
            kernel(input_buffers, output_buffer, node.attrs)
            dt = time.perf_counter_ns() - t0

            if node.id in attn_ids:
                cat_ns["attn"] += dt
            elif node.op in LN_OPS:
                cat_ns["ln"] += dt
            elif node.op in LINEAR_OPS:
                cat_ns["linear"] += dt
            elif node.op in ELEMWISE_OPS:
                cat_ns["elemwise"] += dt
            elif node.op in MOVE_OPS:
                cat_ns["move"] += dt
            else:
                cat_ns["other"] += dt

        total = sum(cat_ns.values())
        per_iter["total"].append(total / 1000)
        for k in cat_keys:
            per_iter[k].append(cat_ns[k] / 1000)

    return Result(
        total_us=_median(per_iter["total"]),
        attn_us=_median(per_iter["attn"]),
        ln_us=_median(per_iter["ln"]),
        linear_us=_median(per_iter["linear"]),
        elemwise_us=_median(per_iter["elemwise"]),
        move_us=_median(per_iter["move"]),
        other_us=_median(per_iter["other"]),
    )


def _median(values):
    s = sorted(values)
    return s[len(s) // 2]


def _speedup(our, baseline):
    """Format speedup: '1.33x' if we're faster, '0.75x' if slower."""
    if baseline <= 0:
        return "     —"
    ratio = baseline / our
    return f"{ratio:5.2f}x"


# =====================================================================
# ONNX Runtime helpers
# =====================================================================

def export_to_onnx(model: nn.Module, x: torch.Tensor) -> str:
    """Export a PyTorch model to a temporary ONNX file. Returns the path."""
    import logging
    fd, path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    # Suppress verbose torch.onnx export logging (stdout progress + stderr warnings)
    import logging
    loggers = ["torch.onnx", "onnxscript"]
    old_levels = {name: logging.getLogger(name).level for name in loggers}
    for name in loggers:
        logging.getLogger(name).setLevel(logging.ERROR)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    devnull = open(os.devnull, "w")
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(model, (x,), path,
                              input_names=["input"], output_names=["output"],
                              opset_version=18)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()
        for name, level in old_levels.items():
            logging.getLogger(name).setLevel(level)
    return path


def bench_ort(onnx_path: str, x_np: np.ndarray,
              opt_level, n_threads: int,
              warmup: int, iters: int):
    """Benchmark an ONNX Runtime session. Returns Result with only total_us populated.

    No per-op breakdown available — only total inference time.
    """
    opts = ort.SessionOptions()
    opts.graph_optimization_level = opt_level
    opts.intra_op_num_threads = n_threads
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3  # suppress warnings
    session = ort.InferenceSession(onnx_path, opts,
                                   providers=["CPUExecutionProvider"])

    feed = {"input": x_np}
    # Warmup
    for _ in range(warmup):
        session.run(None, feed)

    # Timed iterations
    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        session.run(None, feed)
        times.append((time.perf_counter_ns() - t0) / 1000)

    total = _median(times)
    return Result(total, 0.0, 0.0, 0.0, 0.0, 0.0, total)


# =====================================================================
# Variant definitions
# =====================================================================

@dataclass
class Result:
    total_us: float
    attn_us: float
    ln_us: float
    linear_us: float
    elemwise_us: float
    move_us: float
    other_us: float


@dataclass
class Variant:
    name: str
    # "pytorch_naive", "pytorch_sdpa", "python_numpy", "python_c", "compiled", "ort"
    mode: str
    pipeline: object = None  # callable or None
    softmax_mode: int = 0    # 0=SIMD, 1=scalar
    attn_mode: int = 2       # 0=scalar, 1=SIMD, 2=SIMD+GCD
    layernorm_mode: int = 2  # 0=scalar, 1=SIMD, 2=SIMD+GCD
    group: str = ""          # for display grouping
    large: bool = True       # include in large-config runs
    slow: bool = False       # run with reduced warmup/iters
    qwen3: bool = True       # include in Qwen3-config runs
    ort_opt_level: object = None   # ort.GraphOptimizationLevel (set at runtime)
    ort_threads: int = 1           # intra_op_num_threads for ORT


VARIANTS = [
    # --- Baselines ---
    Variant("PT naive",           "pytorch_naive",  group="Baselines", slow=True),
    Variant("PT SDPA",            "pytorch_sdpa",   group="Baselines"),]

# ORT variants are appended at runtime once we know ort is available
def _build_ort_variants():
    if not HAS_ORT:
        return []
    return [
        Variant("ORT no-opt 1T",      "ort", group="ONNX Runtime",
                ort_opt_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                ort_threads=1, large=False),
        Variant("ORT optimized 1T",   "ort", group="ONNX Runtime",
                ort_opt_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                ort_threads=1),
        Variant("ORT optimized MT",   "ort", group="ONNX Runtime",
                ort_opt_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                ort_threads=0),  # 0 = ORT picks based on cores
    ]

VARIANTS += _build_ort_variants()

VARIANTS += [

    # --- Python executor ---
    Variant("Numpy per-op",       "python_numpy", pipeline_full, group="Python exec",
            large=False, qwen3=False),
    Variant("C per-op",           "python_c",     pipeline_full, group="Python exec",
            large=False),

    # --- Compiled C, incremental fusion (attention stays unfused) ---
    Variant("No passes",          "compiled", pipeline_none,      group="Incremental fusion",
            slow=True, qwen3=False),
    Variant("+ fold/DCE",         "compiled", pipeline_fold_dce,  group="Incremental fusion",
            qwen3=False),
    Variant("+ BLAS absorb",      "compiled", pipeline_absorb,    group="Incremental fusion",
            qwen3=False),
    Variant("+ MATMUL_ADD",       "compiled", pipeline_matmul_add,group="Incremental fusion",
            qwen3=False),
    Variant("+ BIAS_RELU",        "compiled", pipeline_bias_relu, group="Incremental fusion",
            qwen3=False),

    # --- Attention kernel variants (all non-attn fusions active) ---
    Variant("Attn: prim scalar",  "compiled", pipeline_bias_relu,
            softmax_mode=1, group="Attention variants",
            large=False, qwen3=False),
    Variant("Attn: fused scalar", "compiled", pipeline_full,
            attn_mode=0, layernorm_mode=0, group="Attention variants", slow=True),
    Variant("Attn: fused SIMD",   "compiled", pipeline_full,
            attn_mode=1, layernorm_mode=0, group="Attention variants",
            large=False),
    Variant("Attn: fused GCD",    "compiled", pipeline_full,
            attn_mode=4, layernorm_mode=0, group="Attention variants"),

    # --- LayerNorm kernel variants (all fusions active, attention at full) ---
    Variant("LN: scalar",         "compiled", pipeline_full,
            layernorm_mode=0, group="LayerNorm variants",
            large=False),
    Variant("LN: SIMD",           "compiled", pipeline_full,
            layernorm_mode=1, group="LayerNorm variants",
            large=False),
    Variant("LN: SIMD+GCD",      "compiled", pipeline_full,
            layernorm_mode=2, group="LayerNorm variants"),

    # --- Merge parallel matmuls ablation ---
    Variant("No merge",           "compiled", pipeline_full_no_merge,
            attn_mode=2, layernorm_mode=2, group="Merge parallel matmuls"),
    Variant("+ merge matmuls",    "compiled", pipeline_full,
            attn_mode=2, layernorm_mode=2, group="Merge parallel matmuls"),

    # --- Full optimization (adaptive flash + optimized LayerNorm) ---
    Variant("Full optimization",  "compiled", pipeline_full,
            attn_mode=2, layernorm_mode=2, group="Full optimization"),

    # --- Flash block size ablation (all use parameterized kernel with GCD) ---
    Variant("Flash 32x32",        "compiled", pipeline_full,
            attn_mode=5, layernorm_mode=2, group="Flash block sizes"),
    Variant("Flash 64x128",       "compiled", pipeline_full,
            attn_mode=6, layernorm_mode=2, group="Flash block sizes"),
    Variant("Flash 128x256",      "compiled", pipeline_full,
            attn_mode=7, layernorm_mode=2, group="Flash block sizes"),
    Variant("Flash 256x512",      "compiled", pipeline_full,
            attn_mode=8, layernorm_mode=2, group="Flash block sizes"),
]


# =====================================================================
# Main benchmark loop
# =====================================================================

def _fmt(us):
    if us >= 1_000_000:
        return f"{us/1_000_000:7.2f}s "
    if us >= 1000:
        return f"{us/1000:7.2f}ms"
    return f"{us:7.1f}us"


def run_config(cfg: Config, lib):
    """Run ablation variants for a single config.

    Large configs (d_model >= LARGE_THRESHOLD) skip variants marked large=False
    to keep runtime manageable.
    """
    is_large = cfg.d_model >= LARGE_THRESHOLD
    is_qwen3_cfg = cfg.model.startswith("qwen3")
    variants = [v for v in VARIANTS if v.large or not is_large]
    if is_qwen3_cfg:
        variants = [v for v in variants if v.qwen3]

    print(f"\n{'='*120}")
    print(f"  {cfg.name}: d_model={cfg.d_model}, n_heads={cfg.n_heads}, "
          f"seq={cfg.seq_len}, batch={cfg.batch}"
          + (f"  [large — {len(variants)} variants]" if is_large else ""))
    head_dim = cfg.d_model // cfg.n_heads
    scratch_mb = cfg.n_heads * cfg.seq_len**2 * 4 / 1e6
    print(f"  head_dim={head_dim}, attn scratch={scratch_mb:.1f}MB")
    print(f"{'='*120}")

    # Build models
    is_gpt2 = cfg.model == "gpt2"
    is_qwen3 = cfg.model.startswith("qwen3")
    is_hf = is_gpt2 or is_qwen3  # HuggingFace model (no naive/sdpa baselines)
    if is_qwen3:
        qwen3_variant = cfg.model.split("-", 1)[1] if "-" in cfg.model else "0.6B"
        result = _make_qwen3(variant=qwen3_variant)
        if result is None:
            print(f"  SKIPPED: transformers + Qwen3 config required")
            return {}
        qwen3_wrapper, qwen3_raw = result
        naive_model = qwen3_wrapper  # wrapper used for export
        sdpa_model = None
        vocab_size = qwen3_raw.config.vocab_size
        x_torch = torch.randint(0, vocab_size, (cfg.batch, cfg.seq_len))
        x_np = x_torch.numpy().astype(np.int64).copy()
    elif is_gpt2:
        gpt2_model = _make_gpt2(n_head=cfg.n_heads, n_embd=cfg.d_model)
        if gpt2_model is None:
            print(f"  SKIPPED: transformers package required for GPT-2 configs")
            return {}
        naive_model = gpt2_model
        sdpa_model = None
        x_torch = torch.randint(0, 50257, (cfg.batch, cfg.seq_len))
        x_np = x_torch.numpy().astype(np.int64).copy()
    else:
        naive_model = NaiveTransformerBlock(cfg.d_model, cfg.n_heads)
        naive_model.eval()
        sdpa_model = SDPATransformerBlock(cfg.d_model, cfg.n_heads)
        sdpa_model.load_state_dict(naive_model.state_dict())
        sdpa_model.eval()
        x_torch = torch.randn(cfg.batch, cfg.seq_len, cfg.d_model)
        x_np = x_torch.numpy().copy()

    # Export ONNX model once for all ORT variants
    # Skip ORT for long sequences — CPU EP doesn't scale (>15x slower at 8K)
    skip_ort = cfg.seq_len >= 4096
    onnx_path = None
    has_ort_variants = any(v.mode == "ort" for v in variants)
    if has_ort_variants and HAS_ORT and not skip_ort:
        onnx_path = export_to_onnx(naive_model, x_torch)

    # Export our graph once for all runtime variants (save/load to get fresh copies)
    needs_our_graph = any(v.mode in ("compiled", "python_numpy", "python_c")
                         for v in variants)
    graph_path = None
    if needs_our_graph:
        base_graph = export_model(naive_model, (x_torch,))
        fd, graph_path = tempfile.mkstemp(prefix="ablation_graph_")
        os.close(fd)
        base_graph.save(graph_path)
        del base_graph

    results: dict[str, Result] = {}

    header = (f"  {'Variant':<24} {'Total':>9} {'Attn':>9} {'Linear':>9} "
              f"{'LN':>9} {'Elem':>9} {'Move':>9} {'Other':>9} {'vs PT':>7}")
    prev_group = None

    for v in variants:
        if v.group != prev_group:
            if prev_group is not None:
                print()
            print(f"  --- {v.group} ---")
            print(header)
            prev_group = v.group

        try:
            # Skip PyTorch SDPA baseline for HF model configs (no matching model)
            if is_hf and v.mode == "pytorch_sdpa":
                continue
            if skip_ort and v.mode == "ort":
                continue

            # Reduce warmup/iters for slow variants
            warmup = max(cfg.warmup // 2, 2) if v.slow else cfg.warmup
            iters = max(cfg.iters // 3, 3) if v.slow else cfg.iters

            if v.mode == "pytorch_naive":
                if is_qwen3:
                    result = bench_pytorch_qwen3(
                        qwen3_wrapper, x_torch, warmup, iters)
                elif is_gpt2:
                    result = bench_pytorch_gpt2(
                        naive_model, x_torch, warmup, iters)
                else:
                    result = bench_pytorch_naive(
                        naive_model, x_torch, warmup, iters)

            elif v.mode == "pytorch_sdpa":
                result = bench_pytorch_sdpa(
                    sdpa_model, x_torch, warmup, iters)

            elif v.mode.startswith("python_"):
                backend_name = v.mode.split("_")[1]
                graph = Graph.load(graph_path)
                if v.pipeline:
                    v.pipeline(graph)
                ep = plan(graph)
                if backend_name == "numpy":
                    executor = InterpretedExecutor(backends=[NumpyBackend()])
                else:
                    executor = InterpretedExecutor(backends=[CBackend(), NumpyBackend()])
                graph.tensors[graph.inputs[0]].buffer = x_np
                result = bench_python_executor(
                    executor, ep, graph, warmup, iters)

            elif v.mode == "compiled":
                graph = Graph.load(graph_path)
                if v.pipeline:
                    v.pipeline(graph)
                ep = _plan_for_ablation(graph)
                executor = CompiledExecutor()
                executor.compile(ep)

                exec_order = list(ep.order)
                categories = classify_op_indices(graph, exec_order)
                n_nodes = executor._n_nodes

                result = bench_compiled_timed(
                    lib, executor, {graph.inputs[0]: x_np}, n_nodes,
                    categories,
                    v.softmax_mode, v.attn_mode, v.layernorm_mode,
                    warmup, iters)

            elif v.mode == "ort":
                result = bench_ort(
                    onnx_path, x_np, v.ort_opt_level, v.ort_threads,
                    warmup, iters)

            results[v.name] = result

            pt_total = results.get("PT naive", Result(1, 0, 0, 0, 0, 0, 0)).total_us
            vs_pt = _speedup(result.total_us, pt_total)

            display_name = v.name
            if is_hf and v.name == "PT naive":
                display_name = "PyTorch"

            if v.mode == "ort":
                # No per-op breakdown for ORT
                print(f"  {display_name:<24} {_fmt(result.total_us):>9} {'—':>9} "
                      f"{'—':>9} {'—':>9} {'—':>9} {'—':>9} {'—':>9} {vs_pt}")
            else:
                print(f"  {display_name:<24} {_fmt(result.total_us):>9} "
                      f"{_fmt(result.attn_us):>9} {_fmt(result.linear_us):>9} "
                      f"{_fmt(result.ln_us):>9} {_fmt(result.elemwise_us):>9} "
                      f"{_fmt(result.move_us):>9} {_fmt(result.other_us):>9} {vs_pt}")

        except Exception as e:
            print(f"  {v.name:<24} FAILED: {e}")

    # Cleanup temp ONNX file and force GC to avoid onnxscript segfault
    # (stale onnx_ir objects with C extension pointers cause use-after-free
    # when GC triggers during a later config's export)
    if onnx_path and os.path.exists(onnx_path):
        os.unlink(onnx_path)
    if graph_path:
        for ext in (".json", ".weights"):
            p = Path(graph_path).with_suffix(ext)
            if p.exists():
                p.unlink()
    import gc; gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser(description="Transformer ablation benchmark")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config names to run (e.g. 'Toy,Small,GPT-2')")
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip configs with seq_len >= 4096")
    args = parser.parse_args()

    configs = ALL_CONFIGS
    if args.configs:
        names = {n.strip().lower() for n in args.configs.split(",")}
        configs = [c for c in configs if c.name.lower() in names]
    if args.skip_large:
        configs = [c for c in configs if c.seq_len < 4096]

    if not configs:
        print("No configs selected. Available:", ", ".join(c.name for c in ALL_CONFIGS))
        return

    # Filter out HF model configs if dependencies are unavailable
    gpt2_configs = [c for c in configs if c.model == "gpt2"]
    if gpt2_configs and not HAS_TRANSFORMERS:
        print(f"Warning: skipping {len(gpt2_configs)} GPT-2 configs "
              f"(transformers package not installed)")
        configs = [c for c in configs if c.model != "gpt2"]

    qwen3_configs = [c for c in configs if c.model.startswith("qwen3")]
    if qwen3_configs and not HAS_QWEN3:
        print(f"Warning: skipping {len(qwen3_configs)} Qwen3 configs "
              f"(transformers package not installed)")
        configs = [c for c in configs if not c.model.startswith("qwen3")]

    lib = _load_ablation_lib()
    print(f"Loaded ablation library")
    print(f"ONNX Runtime: {'v' + ort.__version__ if HAS_ORT else 'not available'}")
    print(f"Transformers (GPT-2): {'available' if HAS_TRANSFORMERS else 'not available'}")
    print(f"Transformers (Qwen3): {'available' if HAS_QWEN3 else 'not available'}")
    print(f"Running {len(configs)} configs x {len(VARIANTS)} variants")

    import gc
    all_results = {}
    for cfg in configs:
        all_results[cfg.name] = run_config(cfg, lib)
        gc.collect()

    # Summary
    print(f"\n{'='*120}")
    print("  SUMMARY: Best compiled variant vs baselines")
    print(f"{'='*120}")
    print(f"  {'Config':<12} {'PT naive':>10} {'PT SDPA':>10} {'ORT opt':>10} "
          f"{'Best ours':>10} {'vs PT':>7} {'vs SDPA':>7} {'vs ORT':>7}")
    print(f"  {'-'*82}")
    _default = Result(0, 0, 0, 0, 0, 0, 0)
    for cfg in configs:
        r = all_results.get(cfg.name, {})
        pt_naive = r.get("PT naive", _default).total_us
        pt_sdpa = r.get("PT SDPA", _default).total_us
        ort_opt = r.get("ORT optimized 1T", r.get("ORT optimized MT", _default)).total_us
        # Find best compiled variant (exclude baselines, python exec, and ORT)
        best_name, best_total = "", float("inf")
        for name, res in r.items():
            if name.startswith("PT ") or name.startswith("ORT ") \
                    or name in ("Numpy per-op", "C per-op"):
                continue
            if res.total_us < best_total:
                best_total = res.total_us
                best_name = name
        if best_total < float("inf"):
            vs_naive = _speedup(best_total, pt_naive)
            vs_sdpa = _speedup(best_total, pt_sdpa)
            vs_ort = _speedup(best_total, ort_opt)
            ort_str = _fmt(ort_opt) if ort_opt > 0 else "—"
            print(f"  {cfg.name:<12} {_fmt(pt_naive):>10} {_fmt(pt_sdpa):>10} "
                  f"{ort_str:>10} {_fmt(best_total):>10} "
                  f"{vs_naive} {vs_sdpa} {vs_ort}"
                  f"  ({best_name})")


if __name__ == "__main__":
    main()
