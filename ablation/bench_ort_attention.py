"""
Benchmark ORT with different attention graph structures.

ORT's attention fusion depends on recognizing specific ONNX graph patterns.
Different PyTorch attention implementations produce different ONNX graphs,
and ORT may fuse some but not others. This script exports several attention
variants and benchmarks each to find what ORT likes.

Variants:
  1. Naive manual attention (matmul → div → softmax → matmul)
  2. SDPA (F.scaled_dot_product_attention — decomposed by ONNX export)
  3. nn.MultiheadAttention (PyTorch's built-in, may export to fused ONNX ops)
  4. Pre-scaled Q (multiply Q by 1/sqrt(d_k) before matmul, changes graph shape)
  5. Packed QKV (single Linear → 3*d_model, split — matches BERT/GPT-2 pattern)
  6. ORT Python optimizer applied to the naive export

Run:  python ablation/bench_ort_attention.py [--configs toy,small,...]
"""

import argparse
import math
import os
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import onnxruntime as ort
except ImportError:
    print("onnxruntime not installed — pip install onnxruntime")
    sys.exit(1)

try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    from onnxruntime.transformers.optimizer import optimize_model
    HAS_ORT_OPTIMIZER = True
except ImportError:
    HAS_ORT_OPTIMIZER = False


# =====================================================================
# Model variants — same transformer block, different attention patterns
# =====================================================================

class NaiveAttentionBlock(nn.Module):
    """Manual attention: Q@K^T / sqrt(d_k) → softmax → P@V"""
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


class SDPAAttentionBlock(nn.Module):
    """F.scaled_dot_product_attention — PyTorch decomposes during ONNX export"""
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


class MHABlock(nn.Module):
    """nn.MultiheadAttention — PyTorch's built-in MHA"""
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.wo = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Linear(d_model, d_model * 4)
        self.ffn2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        h = self.ln1(x)
        attn, _ = self.mha(h, h, h, need_weights=False)
        x = x + self.wo(attn)
        x = x + self.ffn2(torch.relu(self.ffn1(self.ln2(x))))
        return x


class PreScaledAttentionBlock(nn.Module):
    """Pre-scale Q by 1/sqrt(d_k) before matmul — different graph shape"""
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)
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
        q = q * self.scale  # scale Q instead of dividing scores
        k = self.wk(h).reshape(B, S, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        v = self.wv(h).reshape(B, S, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1))
        weights = F.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        attn = attn.permute(0, 2, 1, 3).reshape(B, S, D)
        x = x + self.wo(attn)
        x = x + self.ffn2(torch.relu(self.ffn1(self.ln2(x))))
        return x


class PackedQKVBlock(nn.Module):
    """Packed QKV: single Linear → 3*d_model, then split.

    This matches the BERT/GPT-2 pattern that ORT's attention fusion
    is designed to recognize. One projection produces all of Q, K, V
    at once, which creates a different (and more recognizable) ONNX
    graph topology than separate Q/K/V projections.
    """
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Linear(d_model, d_model * 4)
        self.ffn2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        B, S, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, S, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, D_k]
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = F.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
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
    iters: int = 50

ALL_CONFIGS = [
    Config("Toy",     64,   4,   32,   warmup=10, iters=50),
    Config("Small",   256,  4,   128,  warmup=10, iters=50),
    Config("Medium",  512,  8,   256,  warmup=10, iters=30),
    Config("GPT-2",   768,  12,  512,  warmup=5,  iters=20),
    Config("1B",      2048, 16,  512,  warmup=5,  iters=10),
]


# =====================================================================
# Helpers
# =====================================================================

def export_onnx(model, x, path):
    """Export model to ONNX, suppressing all noise."""
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


def inspect_onnx_graph(path):
    """Print a summary of the ONNX graph's op types."""
    if not HAS_ONNX:
        print(f"    (install 'onnx' package for graph inspection)")
        return
    model = onnx.load(path)
    op_counts = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    sorted_ops = sorted(op_counts.items(), key=lambda x: -x[1])
    ops_str = ", ".join(f"{op}:{count}" for op, count in sorted_ops)
    print(f"    ONNX graph ({len(model.graph.node)} nodes): {ops_str}")


def inspect_ort_optimized(path, opt_level, n_threads):
    """Save ORT's optimized graph and report what changed. Returns the optimized path."""
    if not HAS_ONNX:
        return None
    opts = ort.SessionOptions()
    opts.graph_optimization_level = opt_level
    opts.intra_op_num_threads = n_threads
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3

    opt_path = path.replace(".onnx", f"_ort_optimized.onnx")
    opts.optimized_model_filepath = opt_path

    ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])

    if os.path.exists(opt_path):
        model = onnx.load(opt_path)
        op_counts = {}
        for node in model.graph.node:
            op_type = node.op_type
            if node.domain and node.domain != "":
                op_type = f"{node.domain}:{node.op_type}"
            op_counts[op_type] = op_counts.get(op_type, 0) + 1
        sorted_ops = sorted(op_counts.items(), key=lambda x: -x[1])
        ops_str = ", ".join(f"{op}:{count}" for op, count in sorted_ops)
        print(f"    ORT optimized ({len(model.graph.node)} nodes): {ops_str}")
        # Check if attention was fused
        fused_attn = any("Attention" in op or "MultiHeadAttention" in op
                         for op in op_counts)
        if fused_attn:
            print(f"    *** ATTENTION FUSED ***")
        os.unlink(opt_path)
        return fused_attn
    return False


def run_ort_python_optimizer(input_path, n_heads):
    """Run ORT's Python-based transformer optimizer. Returns optimized path or None."""
    if not HAS_ORT_OPTIMIZER:
        return None
    try:
        fd, opt_path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        # The Python optimizer has its own, more flexible pattern matching
        # for attention, layernorm, etc.
        m = optimize_model(
            input_path,
            model_type="bert",  # tells it to look for BERT-style attention
            num_heads=n_heads,
            hidden_size=0,  # auto-detect
            optimization_options=None,
        )
        m.save_model_to_file(opt_path)

        if HAS_ONNX:
            model = onnx.load(opt_path)
            op_counts = {}
            for node in model.graph.node:
                op_type = node.op_type
                if node.domain and node.domain != "":
                    op_type = f"{node.domain}:{node.op_type}"
                op_counts[op_type] = op_counts.get(op_type, 0) + 1
            sorted_ops = sorted(op_counts.items(), key=lambda x: -x[1])
            ops_str = ", ".join(f"{op}:{count}" for op, count in sorted_ops)
            print(f"    Python optimizer ({len(model.graph.node)} nodes): {ops_str}")
            fused_attn = any("Attention" in op or "MultiHeadAttention" in op
                             for op in op_counts)
            if fused_attn:
                print(f"    *** ATTENTION FUSED ***")

        return opt_path
    except Exception as e:
        print(f"    Python optimizer failed: {e}")
        if os.path.exists(opt_path):
            os.unlink(opt_path)
        return None


def bench_session(path, x_np, opt_level, n_threads, warmup, iters):
    """Time an ORT session, return median us."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = opt_level
    opts.intra_op_num_threads = n_threads
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])

    # Figure out input name from the session
    input_name = session.get_inputs()[0].name
    feed = {input_name: x_np}
    for _ in range(warmup):
        session.run(None, feed)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        session.run(None, feed)
        times.append((time.perf_counter_ns() - t0) / 1000)

    return sorted(times)[len(times) // 2]


def bench_pytorch(model, x, warmup, iters):
    """Time a PyTorch model, return median us."""
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            model(x)
            times.append((time.perf_counter_ns() - t0) / 1000)
    return sorted(times)[len(times) // 2]


def _fmt(us):
    if us >= 1_000_000:
        return f"{us/1_000_000:7.2f}s "
    if us >= 1000:
        return f"{us/1000:7.2f}ms"
    return f"{us:7.1f}us"


# =====================================================================
# Main
# =====================================================================

def run_config(cfg):
    print(f"\n{'='*90}")
    print(f"  {cfg.name}: d_model={cfg.d_model}, n_heads={cfg.n_heads}, "
          f"seq={cfg.seq_len}, batch={cfg.batch}")
    print(f"{'='*90}")

    x = torch.randn(cfg.batch, cfg.seq_len, cfg.d_model)
    x_np = x.numpy().copy()

    # Build model variants
    naive = NaiveAttentionBlock(cfg.d_model, cfg.n_heads).eval()
    sdpa = SDPAAttentionBlock(cfg.d_model, cfg.n_heads).eval()
    sdpa.load_state_dict(naive.state_dict())
    prescaled = PreScaledAttentionBlock(cfg.d_model, cfg.n_heads).eval()
    prescaled.load_state_dict(naive.state_dict())
    mha = MHABlock(cfg.d_model, cfg.n_heads).eval()
    packed = PackedQKVBlock(cfg.d_model, cfg.n_heads).eval()

    variants = [
        ("Naive (Q@K^T/sqrt)", naive),
        ("SDPA",               sdpa),
        ("nn.MHA",             mha),
        ("Pre-scaled Q",       prescaled),
        ("Packed QKV",         packed),
    ]

    # PyTorch baseline
    pt_us = bench_pytorch(naive, x, cfg.warmup, cfg.iters)
    print(f"\n  PyTorch eager baseline: {_fmt(pt_us)}")

    # Results table header
    print(f"\n  {'Variant':<22} {'no-opt 1T':>10} {'opt 1T':>10} {'opt MT':>10} "
          f"{'opt/PT':>8} {'MT/PT':>8}")
    print(f"  {'-'*72}")

    for name, model in variants:
        fd, path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)

        try:
            export_onnx(model, x, path)
        except Exception as e:
            print(f"  {name:<22} EXPORT FAILED: {e}")
            if os.path.exists(path):
                os.unlink(path)
            continue

        # Inspect graphs
        print(f"\n  {name}:")
        inspect_onnx_graph(path)
        inspect_ort_optimized(path, ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 1)

        # Benchmark
        try:
            no_opt = bench_session(path, x_np,
                                   ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1,
                                   cfg.warmup, cfg.iters)
            opt_1t = bench_session(path, x_np,
                                   ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 1,
                                   cfg.warmup, cfg.iters)
            opt_mt = bench_session(path, x_np,
                                   ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 0,
                                   cfg.warmup, cfg.iters)

            vs_opt = opt_1t / pt_us if pt_us > 0 else 0
            vs_mt = opt_mt / pt_us if pt_us > 0 else 0
            print(f"  {name:<22} {_fmt(no_opt):>10} {_fmt(opt_1t):>10} "
                  f"{_fmt(opt_mt):>10} {vs_opt:7.2f}x {vs_mt:7.2f}x")
        except Exception as e:
            print(f"  {name:<22} BENCH FAILED: {e}")

        # Also try ORT's Python transformer optimizer on this variant
        if HAS_ORT_OPTIMIZER:
            opt_path = run_ort_python_optimizer(path, cfg.n_heads)
            if opt_path:
                try:
                    pyopt_1t = bench_session(opt_path, x_np,
                                             ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 1,
                                             cfg.warmup, cfg.iters)
                    pyopt_mt = bench_session(opt_path, x_np,
                                             ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 0,
                                             cfg.warmup, cfg.iters)
                    vs_pyopt = pyopt_1t / pt_us if pt_us > 0 else 0
                    vs_pyopt_mt = pyopt_mt / pt_us if pt_us > 0 else 0
                    print(f"    + Python opt        {' ':>10} {_fmt(pyopt_1t):>10} "
                          f"{_fmt(pyopt_mt):>10} {vs_pyopt:7.2f}x {vs_pyopt_mt:7.2f}x")
                except Exception as e:
                    print(f"    + Python opt        BENCH FAILED: {e}")
                os.unlink(opt_path)

        os.unlink(path)


def main():
    parser = argparse.ArgumentParser(description="ORT attention variant benchmark")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config names (e.g. 'Toy,Small,GPT-2')")
    args = parser.parse_args()

    configs = ALL_CONFIGS
    if args.configs:
        names = {n.strip().lower() for n in args.configs.split(",")}
        configs = [c for c in configs if c.name.lower() in names]

    if not configs:
        print("No configs. Available:", ", ".join(c.name for c in ALL_CONFIGS))
        return

    print(f"ONNX Runtime v{ort.__version__}")
    print(f"onnx package: {'v' + onnx.__version__ if HAS_ONNX else 'not installed'}")
    print(f"ORT Python optimizer: {'available' if HAS_ORT_OPTIMIZER else 'not available'}")
    print(f"Configs: {', '.join(c.name for c in configs)}")

    for cfg in configs:
        run_config(cfg)


if __name__ == "__main__":
    main()
