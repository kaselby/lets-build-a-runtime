#!/usr/bin/env python3
"""Memory ablation: planner strategies + baselines (PyTorch, ORT).

For each model config, measures:
  1. Arena size across all planner strategy combinations (order × fit × sharing)
  2. PyTorch peak-alive activation memory (torch.profiler)
  3. ORT peak activation memory (AllocatorGetStats via C API, minus weights)

Run:  python ablation/bench_memory_ablation.py [--configs toy,small,...] [--all]
"""

import argparse
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from runtime.exporter import export_model
from runtime.passes import run_pipeline
from runtime.planner import plan, PlannerConfig, OrderStrategy, FitStrategy

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
    from ablation.ort_arena_stats import get_ort_arena_stats
    HAS_ORT_CAPI = True
except Exception:
    HAS_ORT_CAPI = False


# =====================================================================
# Models
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
        import math
        import torch.nn.functional as F
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


def _make_gpt2(n_layer=2, n_head=12, n_embd=768):
    if not HAS_TRANSFORMERS:
        return None
    config = GPT2Config(n_layer=n_layer, n_head=n_head, n_embd=n_embd, vocab_size=50257)
    model = GPT2LMHeadModel(config)
    model.eval()
    model.config.use_cache = False
    return model


# =====================================================================
# Configs
# =====================================================================

class Config:
    def __init__(self, name, d_model, n_heads, seq_len, batch=1, model="naive"):
        self.name = name
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.batch = batch
        self.model = model


ALL_CONFIGS = [
    Config("Toy",     64,   4,   32),
    Config("Small",   256,  4,   128),
    Config("Medium",  512,  8,   256),
    Config("GPT-2",   768,  12,  512),
    Config("1B",      2048, 16,  512),
    Config("3B",      3072, 24,  1024),
    Config("7B",      4096, 32,  1024),
    Config("gpt2-s16",  768, 12,  16,  model="gpt2"),
    Config("gpt2-s64",  768, 12,  64,  model="gpt2"),
    Config("gpt2-s256", 768, 12, 256,  model="gpt2"),
]


# =====================================================================
# Planner strategies
# =====================================================================

class Strategy:
    def __init__(self, name, order, fit, enable_inplace, enable_aliases):
        self.name = name
        self.config = PlannerConfig(
            order=order, fit=fit,
            enable_inplace=enable_inplace, enable_aliases=enable_aliases,
        )


STRATEGIES = [
    Strategy("no-opt",        OrderStrategy.NAIVE,             FitStrategy.FIRST_FIT, False, False),
    Strategy("naive+share",   OrderStrategy.NAIVE,             FitStrategy.FIRST_FIT, True,  True),
    Strategy("v1 (default)",  OrderStrategy.MEMORY_AWARE_V1,   FitStrategy.FIRST_FIT, True,  True),
    Strategy("v2",            OrderStrategy.MEMORY_AWARE_V2,   FitStrategy.FIRST_FIT, True,  True),
    Strategy("v3",            OrderStrategy.MEMORY_AWARE_V3,   FitStrategy.FIRST_FIT, True,  True),
    Strategy("v1+best-fit",   OrderStrategy.MEMORY_AWARE_V1,   FitStrategy.BEST_FIT,  True,  True),
    Strategy("v2+best-fit",   OrderStrategy.MEMORY_AWARE_V2,   FitStrategy.BEST_FIT,  True,  True),
    Strategy("v1-no-inplace", OrderStrategy.MEMORY_AWARE_V1,   FitStrategy.FIRST_FIT, False, True),
    Strategy("v1-no-alias",   OrderStrategy.MEMORY_AWARE_V1,   FitStrategy.FIRST_FIT, True,  False),
]


# =====================================================================
# Baselines
# =====================================================================

def _export_onnx(model, x_torch):
    """Export model to ONNX, suppressing all noise. Returns temp file path."""
    fd, onnx_path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    old_level = logging.getLogger("torch.onnx").level
    logging.getLogger("torch.onnx").setLevel(logging.ERROR)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    devnull = open(os.devnull, "w")
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(model, (x_torch,), onnx_path,
                              input_names=["input"], output_names=["output"],
                              opset_version=18)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()
        logging.getLogger("torch.onnx").setLevel(old_level)
    return onnx_path


def pytorch_peak_alive(model, x_torch):
    """Peak simultaneously-alive activation memory during PyTorch eager execution.

    Uses torch.profiler with profile_memory=True, tracking self_cpu_memory_usage
    (not cpu_memory_usage) to avoid double-counting nested ops (e.g. linear->addmm).
    Includes deallocation events, so the running sum is the true peak-alive watermark.
    Weights excluded — profiler only sees allocations during the forward pass.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            profile_memory=True,
        ) as prof:
            with torch.no_grad():
                model(x_torch)

    running = 0
    peak = 0
    for ev in prof.events():
        running += ev.self_cpu_memory_usage
        peak = max(peak, running)
    return peak


def ort_peak_activations(model, x_torch):
    """ORT peak activation memory via AllocatorGetStats C API.

    Calls ORT's C API directly to get MaxInUse from the session's arena allocator.
    ORT's arena includes weights, so we subtract model parameter bytes.
    Returns activation bytes, or None on failure.
    """
    if not HAS_ORT or not HAS_ORT_CAPI:
        return None
    try:
        onnx_path = _export_onnx(model, x_torch)
        stats = get_ort_arena_stats(onnx_path, tuple(x_torch.shape))
        os.unlink(onnx_path)
        if "MaxInUse" not in stats:
            return None
        max_in_use = int(stats["MaxInUse"])
        weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        return max_in_use - weight_bytes
    except Exception as e:
        print(f"    (ORT C API failed: {e})")
        return None


def intermediates_total(graph):
    """Sum of all intermediate tensor sizes (no reuse) — upper bound."""
    external = set(graph.inputs) | set(graph.constants)
    total = 0
    for name, info in graph.tensors.items():
        if name not in external:
            total += int(np.prod(info.shape)) * np.dtype(info.dtype).itemsize
    return total


# =====================================================================
# Benchmark
# =====================================================================

def _fmt(nbytes):
    if nbytes is None:
        return "n/a"
    return f"{nbytes / 1024:,.1f}"


def run_config(cfg):
    """Measure arena size across all strategies + baselines for a single config."""

    is_gpt2 = cfg.model == "gpt2"
    if is_gpt2:
        model = _make_gpt2(n_head=cfg.n_heads, n_embd=cfg.d_model)
        if model is None:
            print(f"  SKIPPED: transformers package required for GPT-2 configs")
            return {}
        x_torch = torch.randint(0, 50257, (cfg.batch, cfg.seq_len))
    else:
        model = NaiveTransformerBlock(cfg.d_model, cfg.n_heads)
        model.eval()
        x_torch = torch.randn(cfg.batch, cfg.seq_len, cfg.d_model)

    # Export and optimize graph once
    graph = export_model(model, (x_torch,))
    run_pipeline(graph)

    n_intermediates = sum(
        1 for name in graph.tensors
        if name not in (set(graph.inputs) | set(graph.constants))
    )
    total_intermed = intermediates_total(graph)

    # Print header
    print(f"\n{'='*90}")
    print(f"  Config: {cfg.name} (d={cfg.d_model}, h={cfg.n_heads}, s={cfg.seq_len})")
    print(f"  Graph: {len(graph.nodes)} nodes, {n_intermediates} intermediates,"
          f" {_fmt(total_intermed)} KB total")
    print(f"{'='*90}")

    # --- Strategy ablation ---
    results = {}
    print(f"\n  Planner strategies:")
    print(f"  {'Strategy':<18} {'Arena (KB)':>12} {'vs default':>10} {'vs no-opt':>10}")
    print(f"  {'-'*55}")

    no_opt_size = None
    default_size = None

    for strategy in STRATEGIES:
        try:
            ep = plan(graph, strategy.config)
            arena_kb = ep.arena_size / 1024
            results[strategy.name] = ep.arena_size

            if strategy.name == "no-opt":
                no_opt_size = ep.arena_size
            if strategy.name == "v1 (default)":
                default_size = ep.arena_size

            # vs default
            if strategy.name == "v1 (default)":
                vs_default = "baseline"
            elif default_size and default_size > 0:
                pct = (ep.arena_size - default_size) / default_size * 100
                vs_default = f"{pct:+.1f}%"
            else:
                vs_default = "—"

            # vs no-opt
            if no_opt_size and no_opt_size > 0:
                vs_no_opt = f"{ep.arena_size / no_opt_size:.2f}x"
            else:
                vs_no_opt = "—"

            print(f"  {strategy.name:<18} {arena_kb:>12.1f} {vs_default:>10} {vs_no_opt:>10}")

        except Exception as e:
            print(f"  {strategy.name:<18} FAILED: {e}")

    # --- Baselines ---
    print(f"\n  Baselines (peak activation memory, no weights):")
    print(f"  {'-'*55}")

    # PyTorch
    pt_peak = pytorch_peak_alive(model, x_torch)
    if default_size and default_size > 0:
        pt_ratio = f"({pt_peak / default_size:.2f}x ours)"
    else:
        pt_ratio = ""
    print(f"  {'PyTorch (peak-alive)':<35} {_fmt(pt_peak):>10} KB  {pt_ratio}")

    # ORT
    ort_activ = ort_peak_activations(model, x_torch)
    if ort_activ is not None:
        if default_size and default_size > 0:
            ort_ratio = f"({ort_activ / default_size:.2f}x ours)"
        else:
            ort_ratio = ""
        print(f"  {'ORT (C API, activations only)':<35} {_fmt(ort_activ):>10} KB  {ort_ratio}")
    elif HAS_ORT:
        print(f"  {'ORT':<35} {'n/a':>10}")

    # Our default for easy reference
    if default_size:
        print(f"  {'Ours (v1 default)':<35} {_fmt(default_size):>10} KB")

    # Reuse efficiency
    if default_size and total_intermed:
        print(f"\n  Reuse: {default_size/total_intermed:.1%} of no-reuse"
              f" ({_fmt(total_intermed)} KB -> {_fmt(default_size)} KB)")

    results["__pt_peak"] = pt_peak
    results["__ort_activ"] = ort_activ
    return results


def main():
    parser = argparse.ArgumentParser(description="Memory ablation benchmark")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config names (e.g. 'Toy,Small,GPT-2')")
    parser.add_argument("--all", action="store_true",
                        help="Run all configs (default: toy,small,gpt2-s16)")
    args = parser.parse_args()

    if args.all:
        configs = ALL_CONFIGS
    elif args.configs:
        names = {n.strip().lower() for n in args.configs.split(",")}
        configs = [c for c in ALL_CONFIGS if c.name.lower() in names]
    else:
        configs = [c for c in ALL_CONFIGS if c.name.lower() in {"toy", "small", "gpt2-s16"}]

    if not configs:
        print("No configs selected. Available:", ", ".join(c.name for c in ALL_CONFIGS))
        return

    gpt2_configs = [c for c in configs if c.model == "gpt2"]
    if gpt2_configs and not HAS_TRANSFORMERS:
        print(f"Warning: skipping {len(gpt2_configs)} GPT-2 configs (transformers not installed)")
        configs = [c for c in configs if c.model != "gpt2"]

    print("Memory Ablation: Planner Strategies + Baselines")
    print("=" * 90)
    print(f"  ONNX Runtime:  {'v' + ort.__version__ if HAS_ORT else 'not available'}")
    print(f"  ORT C API:     {'available' if HAS_ORT_CAPI else 'not available'}")
    print(f"  Transformers:  {'available' if HAS_TRANSFORMERS else 'not available'}")
    print(f"  Configs:       {len(configs)}  x  {len(STRATEGIES)} strategies")
    print()
    print("  All metrics are activation memory only (weights excluded).")
    print("  PyTorch: torch.profiler self_cpu_memory_usage watermark")
    print("  ORT:     AllocatorGetStats MaxInUse via C API, minus model weights")

    all_results = {}
    for cfg in configs:
        all_results[cfg.name] = run_config(cfg)

    # Summary table
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Config':<12} {'no-opt':>10} {'v1':>10} {'best':>10}"
          f" {'PyTorch':>10} {'ORT':>10}")
    print(f"  {'-'*65}")
    for cfg in configs:
        r = all_results.get(cfg.name, {})
        if not r:
            continue
        no_opt = r.get("no-opt", 0)
        v1 = r.get("v1 (default)", 0)
        best_size = min(v for v in r.values() if v > 0) if r.values() else 0

        pt = r.get("__pt_peak") or 0
        ort_a = r.get("__ort_activ") or 0

        fmt = lambda x: f"{x/1024:.0f}K" if x > 0 else "—"
        print(f"  {cfg.name:<12} {fmt(no_opt):>10} {fmt(v1):>10} {fmt(best_size):>10}"
              f" {fmt(pt):>10} {fmt(ort_a):>10}")


if __name__ == "__main__":
    main()
