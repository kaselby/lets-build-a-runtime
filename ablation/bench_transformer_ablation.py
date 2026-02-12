"""
Transformer ablation benchmark: attention / layernorm / other time breakdown.

Measures performance across:
  - PyTorch baselines (naive F.softmax + SDPA)
  - Python executor (numpy + C backends, with fusion)
  - Compiled C executor with incremental fusion
  - Attention kernel variants (scalar, SIMD, SIMD+GCD)
  - LayerNorm kernel variants (scalar, SIMD, SIMD+GCD)

For each variant, reports total time and three-way breakdown:
attention, layernorm, and everything else.

Run:  python bench_transformer_ablation.py [--configs toy,small,...] [--skip-large]
"""

import argparse
import ctypes
import math
import sys
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
from runtime.executor import COpNode, Executor, FLOAT_PTR, MAX_DIMS, MAX_INPUTS
from runtime.ir import OpType
from runtime.passes import (
    FUSION_PATTERNS,
    absorb_into_matmul,
    constant_fold,
    eliminate_dead_code,
    fuse,
)
from runtime.planner import plan


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

def pipeline_full(graph):
    absorb_into_matmul(graph)
    constant_fold(graph)
    fuse(graph, patterns=[P_MATMUL_ADD, P_BIAS_RELU, P_ATTENTION])
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
    """Return (attn_indices, ln_indices) for the RESHAPE-stripped execution order.

    Attention ops: fused ATTENTION nodes, or the SOFTMAX + its neighboring
    MATMULs (Q@K^T and W@V) in unfused graphs.
    LayerNorm ops: any LAYERNORM node.
    """
    attn_ids = set()
    ln_ids = set()

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
        elif node.op == OpType.LAYERNORM:
            ln_ids.add(node.id)

    stripped = [n for n in exec_order if n.op != OpType.RESHAPE]
    attn_set = {i for i, n in enumerate(stripped) if n.id in attn_ids}
    ln_set = {i for i, n in enumerate(stripped) if n.id in ln_ids}
    return attn_set, ln_set


# =====================================================================
# Timing helpers
# =====================================================================

def _patch_inputs(compiled_plan, inputs):
    """Patch graph input pointers into the compiled plan's COpNode array."""
    for name, slots in compiled_plan.input_slots.items():
        ptr = inputs[name].ctypes.data_as(FLOAT_PTR)
        for node_idx, slot_idx in slots:
            compiled_plan.nodes[node_idx].inputs[slot_idx] = ptr


def bench_compiled_timed(lib, compiled_plan, inputs, n_nodes,
                         attn_indices, ln_indices,
                         softmax_mode, attn_mode, layernorm_mode,
                         warmup, iters):
    """Benchmark a compiled plan with per-op timing via the ablation library.

    Returns (total_us, attn_us, ln_us, other_us) — median across iterations.
    """
    _patch_inputs(compiled_plan, inputs)
    times_buf = (ctypes.c_double * n_nodes)()

    # Warmup (no timing)
    for _ in range(warmup):
        lib.execute_ablation(compiled_plan.nodes, n_nodes,
                             softmax_mode, attn_mode, layernorm_mode)

    # Timed iterations — collect per-iteration aggregates
    totals, attns, lns, others = [], [], [], []
    for _ in range(iters):
        lib.timed_execute_ablation(compiled_plan.nodes, n_nodes, times_buf,
                                   softmax_mode, attn_mode, layernorm_mode)
        attn_ns = sum(times_buf[i] for i in attn_indices)
        ln_ns = sum(times_buf[i] for i in ln_indices)
        other_ns = sum(times_buf[i] for i in range(n_nodes)
                       if i not in attn_indices and i not in ln_indices)
        total_ns = attn_ns + ln_ns + other_ns
        totals.append(total_ns / 1000)
        attns.append(attn_ns / 1000)
        lns.append(ln_ns / 1000)
        others.append(other_ns / 1000)

    return _median(totals), _median(attns), _median(lns), _median(others)


def bench_pytorch_naive(model, x, warmup, iters):
    """Benchmark NaiveTransformerBlock with attention + layernorm timing."""
    B, S, D = x.shape
    m = model

    with torch.no_grad():
        for _ in range(warmup):
            m(x)

    total_ns_all, attn_ns_all, ln_ns_all = [], [], []
    with torch.no_grad():
        for _ in range(iters):
            t_start = time.perf_counter_ns()

            tln = time.perf_counter_ns()
            h = m.ln1(x)
            ln_time = time.perf_counter_ns() - tln

            q = m.wq(h).reshape(B, S, m.n_heads, m.d_k).permute(0, 2, 1, 3)
            k = m.wk(h).reshape(B, S, m.n_heads, m.d_k).permute(0, 2, 1, 3)
            v = m.wv(h).reshape(B, S, m.n_heads, m.d_k).permute(0, 2, 1, 3)

            ta = time.perf_counter_ns()
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(m.d_k)
            weights = F.softmax(scores, dim=-1)
            attn = torch.matmul(weights, v)
            attn_ns_all.append(time.perf_counter_ns() - ta)

            attn = attn.permute(0, 2, 1, 3).reshape(B, S, D)
            x_out = x + m.wo(attn)

            tln2 = time.perf_counter_ns()
            ln2_out = m.ln2(x_out)
            ln_time += time.perf_counter_ns() - tln2

            x_out = x_out + m.ffn2(torch.relu(m.ffn1(ln2_out)))
            total_ns_all.append(time.perf_counter_ns() - t_start)
            ln_ns_all.append(ln_time)

    total_us = [t / 1000 for t in total_ns_all]
    attn_us = [t / 1000 for t in attn_ns_all]
    ln_us = [t / 1000 for t in ln_ns_all]
    other_us = [t - a - l for t, a, l in zip(total_us, attn_us, ln_us)]
    return _median(total_us), _median(attn_us), _median(ln_us), _median(other_us)


def bench_pytorch_sdpa(model, x, warmup, iters):
    """Benchmark SDPATransformerBlock with attention + layernorm timing."""
    B, S, D = x.shape
    m = model

    with torch.no_grad():
        for _ in range(warmup):
            m(x)

    total_ns_all, attn_ns_all, ln_ns_all = [], [], []
    with torch.no_grad():
        for _ in range(iters):
            t_start = time.perf_counter_ns()

            tln = time.perf_counter_ns()
            h = m.ln1(x)
            ln_time = time.perf_counter_ns() - tln

            q = m.wq(h).reshape(B, S, m.n_heads, m.d_k).permute(0, 2, 1, 3)
            k = m.wk(h).reshape(B, S, m.n_heads, m.d_k).permute(0, 2, 1, 3)
            v = m.wv(h).reshape(B, S, m.n_heads, m.d_k).permute(0, 2, 1, 3)

            ta = time.perf_counter_ns()
            attn = F.scaled_dot_product_attention(q, k, v)
            attn_ns_all.append(time.perf_counter_ns() - ta)

            attn = attn.permute(0, 2, 1, 3).reshape(B, S, D)
            x_out = x + m.wo(attn)

            tln2 = time.perf_counter_ns()
            ln2_out = m.ln2(x_out)
            ln_time += time.perf_counter_ns() - tln2

            x_out = x_out + m.ffn2(torch.relu(m.ffn1(ln2_out)))
            total_ns_all.append(time.perf_counter_ns() - t_start)
            ln_ns_all.append(ln_time)

    total_us = [t / 1000 for t in total_ns_all]
    attn_us = [t / 1000 for t in attn_ns_all]
    ln_us = [t / 1000 for t in ln_ns_all]
    other_us = [t - a - l for t, a, l in zip(total_us, attn_us, ln_us)]
    return _median(total_us), _median(attn_us), _median(ln_us), _median(other_us)


def bench_python_executor(executor, ep, graph, warmup, iters):
    """Benchmark per-op Python executor with per-op timing."""
    inp_name = graph.inputs[0]
    x_np = graph.tensors[inp_name].buffer

    # Classify nodes
    attn_ids = set()
    ln_ids = set()
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
        elif node.op == OpType.LAYERNORM:
            ln_ids.add(node.id)

    # Warmup
    for _ in range(warmup):
        executor.execute(ep, {inp_name: x_np})

    # Timed — manually dispatch with per-op timing
    arena = executor._get_arena(ep.arena_size)
    totals, attns, lns, others = [], [], [], []

    for _ in range(iters):
        executor._bind_inputs(graph, {inp_name: x_np})
        executor._bind_intermediates(graph, ep.offsets, arena)

        attn_ns = 0
        ln_ns = 0
        other_ns = 0
        for node in ep.order:
            if node.op == OpType.RESHAPE:
                inp_buf = graph.tensors[node.inputs[0]].buffer
                graph.tensors[node.output].buffer = inp_buf.reshape(
                    graph.tensors[node.output].shape)
                continue
            kernel = executor._resolve_kernel(node.op)
            input_buffers = [graph.tensors[inp].buffer for inp in node.inputs]
            if node.id in ep.scratch:
                offset, size_bytes = ep.scratch[node.id]
                input_buffers.append(arena[offset:offset + size_bytes].view(np.float32))
            output_buffer = graph.tensors[node.output].buffer

            t0 = time.perf_counter_ns()
            kernel(input_buffers, output_buffer, node.attrs)
            dt = time.perf_counter_ns() - t0

            if node.id in attn_ids:
                attn_ns += dt
            elif node.id in ln_ids:
                ln_ns += dt
            else:
                other_ns += dt

        totals.append((attn_ns + ln_ns + other_ns) / 1000)
        attns.append(attn_ns / 1000)
        lns.append(ln_ns / 1000)
        others.append(other_ns / 1000)

    return _median(totals), _median(attns), _median(lns), _median(others)


def _median(values):
    s = sorted(values)
    return s[len(s) // 2]


# =====================================================================
# Variant definitions
# =====================================================================

@dataclass
class Result:
    total_us: float
    attn_us: float
    ln_us: float
    other_us: float


@dataclass
class Variant:
    name: str
    # "pytorch_naive", "pytorch_sdpa", "python_numpy", "python_c", "compiled"
    mode: str
    pipeline: object = None  # callable or None
    softmax_mode: int = 0    # 0=SIMD, 1=scalar
    attn_mode: int = 2       # 0=scalar, 1=SIMD, 2=SIMD+GCD
    layernorm_mode: int = 2  # 0=scalar, 1=SIMD, 2=SIMD+GCD
    group: str = ""          # for display grouping
    large: bool = True       # include in large-config runs


VARIANTS = [
    # --- Baselines ---
    Variant("PT naive",           "pytorch_naive",  group="Baselines"),
    Variant("PT SDPA",            "pytorch_sdpa",   group="Baselines"),

    # --- Python executor ---
    Variant("Numpy per-op",       "python_numpy", pipeline_full, group="Python exec",
            large=False),
    Variant("C per-op",           "python_c",     pipeline_full, group="Python exec",
            large=False),

    # --- Compiled C, incremental fusion (attention stays unfused) ---
    Variant("No passes",          "compiled", pipeline_none,      group="Incremental fusion"),
    Variant("+ fold/DCE",         "compiled", pipeline_fold_dce,  group="Incremental fusion",
            large=False),
    Variant("+ BLAS absorb",      "compiled", pipeline_absorb,    group="Incremental fusion",
            large=False),
    Variant("+ MATMUL_ADD",       "compiled", pipeline_matmul_add,group="Incremental fusion",
            large=False),
    Variant("+ BIAS_RELU",        "compiled", pipeline_bias_relu, group="Incremental fusion",
            large=False),

    # --- Attention kernel variants (all non-attn fusions active) ---
    Variant("Attn: prim scalar",  "compiled", pipeline_bias_relu,
            softmax_mode=1, group="Attention variants",
            large=False),
    Variant("Attn: fused scalar", "compiled", pipeline_full,
            attn_mode=0, layernorm_mode=0, group="Attention variants"),
    Variant("Attn: fused SIMD",   "compiled", pipeline_full,
            attn_mode=1, layernorm_mode=0, group="Attention variants",
            large=False),
    Variant("Attn: fused GCD",    "compiled", pipeline_full,
            attn_mode=2, layernorm_mode=0, group="Attention variants"),

    # --- Flash attention (all fusions active, SIMD+GCD) ---
    Variant("Attn: flash GCD",    "compiled", pipeline_full,
            attn_mode=3, layernorm_mode=0, group="Attention variants"),

    # --- LayerNorm kernel variants (all fusions active, attention at full) ---
    Variant("LN: scalar",         "compiled", pipeline_full,
            layernorm_mode=0, group="LayerNorm variants",
            large=False),
    Variant("LN: SIMD",           "compiled", pipeline_full,
            layernorm_mode=1, group="LayerNorm variants",
            large=False),
    Variant("LN: SIMD+GCD",      "compiled", pipeline_full,
            layernorm_mode=2, group="LayerNorm variants"),

    # --- Flash + optimized LayerNorm (the full package) ---
    Variant("Flash + LN opt",     "compiled", pipeline_full,
            attn_mode=3, layernorm_mode=2, group="Full optimization"),
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
    variants = [v for v in VARIANTS if v.large or not is_large]

    print(f"\n{'='*100}")
    print(f"  {cfg.name}: d_model={cfg.d_model}, n_heads={cfg.n_heads}, "
          f"seq={cfg.seq_len}, batch={cfg.batch}"
          + (f"  [large — {len(variants)} variants]" if is_large else ""))
    head_dim = cfg.d_model // cfg.n_heads
    scratch_mb = cfg.n_heads * cfg.seq_len**2 * 4 / 1e6
    print(f"  head_dim={head_dim}, attn scratch={scratch_mb:.1f}MB")
    print(f"{'='*100}")

    # Build models
    naive_model = NaiveTransformerBlock(cfg.d_model, cfg.n_heads)
    naive_model.eval()
    sdpa_model = SDPATransformerBlock(cfg.d_model, cfg.n_heads)
    sdpa_model.load_state_dict(naive_model.state_dict())
    sdpa_model.eval()

    x_torch = torch.randn(cfg.batch, cfg.seq_len, cfg.d_model)
    x_np = x_torch.numpy().copy()

    results: dict[str, Result] = {}

    header = (f"  {'Variant':<24} {'Total':>9} {'Attn':>9} {'LN':>9} "
              f"{'Other':>9} {'Attn%':>6} {'LN%':>5} {'vs PT':>7}")
    prev_group = None

    for v in variants:
        if v.group != prev_group:
            if prev_group is not None:
                print()
            print(f"  --- {v.group} ---")
            print(header)
            prev_group = v.group

        try:
            if v.mode == "pytorch_naive":
                total, attn, ln, other = bench_pytorch_naive(
                    naive_model, x_torch, cfg.warmup, cfg.iters)

            elif v.mode == "pytorch_sdpa":
                total, attn, ln, other = bench_pytorch_sdpa(
                    sdpa_model, x_torch, cfg.warmup, cfg.iters)

            elif v.mode.startswith("python_"):
                backend_name = v.mode.split("_")[1]
                graph = export_model(naive_model, (x_torch,))
                if v.pipeline:
                    v.pipeline(graph)
                ep = plan(graph)
                if backend_name == "numpy":
                    executor = Executor(backends=[NumpyBackend()])
                else:
                    executor = Executor(backends=[CBackend(), NumpyBackend()])
                graph.tensors[graph.inputs[0]].buffer = x_np
                total, attn, ln, other = bench_python_executor(
                    executor, ep, graph, cfg.warmup, cfg.iters)

            elif v.mode == "compiled":
                graph = export_model(naive_model, (x_torch,))
                if v.pipeline:
                    v.pipeline(graph)
                # Tag ATTENTION nodes for flash scratch allocation
                if v.attn_mode == 3:
                    for node in graph.nodes.values():
                        if node.op == OpType.ATTENTION:
                            node.attrs["flash"] = True
                ep = plan(graph)
                executor = Executor(backends=[CBackend(), NumpyBackend()])
                compiled = executor.compile_plan(ep)

                exec_order = list(ep.order)
                attn_indices, ln_indices = classify_op_indices(graph, exec_order)
                n_nodes = compiled.n_nodes

                total, attn, ln, other = bench_compiled_timed(
                    lib, compiled, {graph.inputs[0]: x_np}, n_nodes,
                    attn_indices, ln_indices,
                    v.softmax_mode, v.attn_mode, v.layernorm_mode,
                    cfg.warmup, cfg.iters)

            results[v.name] = Result(total, attn, ln, other)

            pt_total = results.get("PT naive", Result(1, 0, 0, 0)).total_us
            attn_pct = (attn / total * 100) if total > 0 else 0
            ln_pct = (ln / total * 100) if total > 0 else 0
            vs_pt = total / pt_total if pt_total > 0 else 0

            print(f"  {v.name:<24} {_fmt(total):>9} {_fmt(attn):>9} "
                  f"{_fmt(ln):>9} {_fmt(other):>9} "
                  f"{attn_pct:5.1f}% {ln_pct:4.1f}% {vs_pt:6.2f}x")

        except Exception as e:
            print(f"  {v.name:<24} FAILED: {e}")

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

    lib = _load_ablation_lib()
    print(f"Loaded ablation library")
    print(f"Running {len(configs)} configs x {len(VARIANTS)} variants")

    all_results = {}
    for cfg in configs:
        all_results[cfg.name] = run_config(cfg, lib)

    # Summary
    print(f"\n{'='*100}")
    print("  SUMMARY: Best compiled variant vs PyTorch baselines")
    print(f"{'='*100}")
    print(f"  {'Config':<12} {'PT naive':>10} {'PT SDPA':>10} {'Best ours':>10} "
          f"{'vs naive':>8} {'vs SDPA':>8}")
    print(f"  {'-'*62}")
    for cfg in configs:
        r = all_results.get(cfg.name, {})
        pt_naive = r.get("PT naive", Result(0, 0, 0, 0)).total_us
        pt_sdpa = r.get("PT SDPA", Result(0, 0, 0, 0)).total_us
        best_name, best_total = "", float("inf")
        for name, res in r.items():
            if name.startswith("PT ") or name in ("Numpy per-op", "C per-op"):
                continue
            if res.total_us < best_total:
                best_total = res.total_us
                best_name = name
        if best_total < float("inf"):
            vs_naive = best_total / pt_naive if pt_naive > 0 else 0
            vs_sdpa = best_total / pt_sdpa if pt_sdpa > 0 else 0
            print(f"  {cfg.name:<12} {_fmt(pt_naive):>10} {_fmt(pt_sdpa):>10} "
                  f"{_fmt(best_total):>10} {vs_naive:7.2f}x {vs_sdpa:7.2f}x  ({best_name})")


if __name__ == "__main__":
    main()
