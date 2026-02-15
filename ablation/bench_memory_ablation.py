#!/usr/bin/env python3
"""Memory allocation ablation: arena size across different planner strategies.

Measures peak arena size for various planner configurations and compares against
PyTorch and ONNX Runtime baselines. Tests incremental strategy improvements to
identify which optimizations have the most impact.

Run:  python ablation/bench_memory_ablation.py [--configs toy,small,...] [--all]
"""

import argparse
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def _make_gpt2_body(n_layer=2, n_head=12, n_embd=768):
    """Create a GPT-2 body model. Returns None if transformers unavailable."""
    if not HAS_TRANSFORMERS:
        return None
    config = GPT2Config(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        vocab_size=50257,
    )
    full_model = GPT2LMHeadModel(config)
    full_model.eval()
    body = GPT2Body(full_model)
    body.eval()
    return body


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
    # GPT-2 body (HuggingFace, 2-layer, causal attention)
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
            order=order,
            fit=fit,
            enable_inplace=enable_inplace,
            enable_aliases=enable_aliases,
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
# PyTorch and ORT baselines
# =====================================================================

def pytorch_baseline(graph):
    """Sum of all intermediate tensor sizes (no reuse)."""
    external = set(graph.inputs) | set(graph.constants)
    total = 0
    for name, info in graph.tensors.items():
        if name not in external:
            total += int(np.prod(info.shape)) * np.dtype(info.dtype).itemsize
    return total


def ort_baseline(model, x):
    """Extract peak memory from ONNX Runtime profiling.

    Uses ORT's enable_mem_pattern + enable_profiling to capture allocation
    events, then parses the profile JSON for peak memory usage.
    """
    if not HAS_ORT:
        return None

    try:
        import json
        import logging

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
                torch.onnx.export(model, (x,), onnx_path,
                                  input_names=["input"], output_names=["output"],
                                  opset_version=18)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            devnull.close()
            logging.getLogger("torch.onnx").setLevel(old_level)

        # Run with profiling to capture memory allocation events
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern = True
        opts.enable_profiling = True
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3

        session = ort.InferenceSession(onnx_path, opts,
                                       providers=["CPUExecutionProvider"])

        x_np = x.numpy() if isinstance(x, torch.Tensor) else x
        session.run(None, {"input": x_np})

        profile_path = session.end_profiling()
        os.unlink(onnx_path)

        # Parse the profile JSON for memory allocation events
        with open(profile_path) as f:
            events = json.load(f)
        os.unlink(profile_path)

        # Track allocations: sum up all "alloc" sizes, subtract "free" sizes,
        # keep running peak. ORT profile events have "cat" and "args" fields.
        peak = 0
        current = 0
        for event in events:
            args = event.get("args", {})
            # ORT memory events have "size" in args
            if "size" in args:
                size = int(args["size"])
                name = event.get("name", "")
                if "Alloc" in name or "alloc" in name:
                    current += size
                    peak = max(peak, current)
                elif "Free" in name or "free" in name:
                    current -= size

        # If we didn't find alloc/free events, try alternative: look for
        # MemoryPattern or activation_memory fields
        if peak == 0:
            for event in events:
                args = event.get("args", {})
                for key in ("activation_size", "total_size", "peak_size"):
                    if key in args:
                        val = int(args[key])
                        peak = max(peak, val)

        return peak if peak > 0 else None

    except Exception:
        return None


# =====================================================================
# Benchmark
# =====================================================================

def run_config(cfg):
    """Measure arena size across all strategies for a single config."""

    # Build model
    is_gpt2 = cfg.model == "gpt2"
    if is_gpt2:
        model = _make_gpt2_body(n_head=cfg.n_heads, n_embd=cfg.d_model)
        if model is None:
            print(f"  SKIPPED: transformers package required for GPT-2 configs")
            return {}
    else:
        model = NaiveTransformerBlock(cfg.d_model, cfg.n_heads)
        model.eval()

    x_torch = torch.randn(cfg.batch, cfg.seq_len, cfg.d_model)

    # Export and optimize graph once
    graph = export_model(model, (x_torch,))
    run_pipeline(graph)

    # Compute PyTorch baseline (sum of all intermediates)
    pt_baseline = pytorch_baseline(graph)

    # Print header
    print(f"\n{'='*90}")
    print(f"  Config: {cfg.name} (d={cfg.d_model}, h={cfg.n_heads}, s={cfg.seq_len})")
    print(f"  Graph: {len(graph.nodes)} nodes, {sum(1 for name in graph.tensors if name not in (set(graph.inputs) | set(graph.constants)))} intermediates")
    print(f"{'='*90}")

    # Compute results
    results = {}
    print(f"  {'Strategy':<18} {'Arena (KB)':>12} {'Savings':>10} {'vs no-opt':>10}")
    print(f"  {'-'*50}")

    no_opt_size = None
    default_size = None

    for strategy in STRATEGIES:
        try:
            ep = plan(graph, strategy.config)
            arena_kb = ep.arena_size / 1024
            results[strategy.name] = ep.arena_size

            # Track baseline for percentage calculations
            if strategy.name == "no-opt":
                no_opt_size = ep.arena_size
            if strategy.name == "v1 (default)":
                default_size = ep.arena_size

            # Compute savings
            if default_size and default_size > 0:
                savings_pct = (default_size - ep.arena_size) / default_size * 100
            else:
                savings_pct = 0.0

            # Compute vs no-opt
            if no_opt_size and no_opt_size > 0:
                ratio = ep.arena_size / no_opt_size
                vs_no_opt = f"{ratio:.2f}x"
            else:
                vs_no_opt = "—"

            # Format savings
            if strategy.name == "v1 (default)":
                savings_str = "—*"
            elif default_size and default_size > 0:
                savings_str = f"{savings_pct:+.1f}%"
            else:
                savings_str = "—"

            print(f"  {strategy.name:<18} {arena_kb:>12.1f} {savings_str:>10} {vs_no_opt:>10}")

        except Exception as e:
            print(f"  {strategy.name:<18} FAILED: {e}")

    # Summary
    print()
    print(f"  PyTorch baseline (no reuse): {pt_baseline/1024:.1f} KB")
    if no_opt_size:
        pt_ratio = pt_baseline / no_opt_size
        print(f"  no-opt arena / PyTorch: {pt_ratio:.2f}x")

    # ORT baseline
    ort_mem = ort_baseline(model, x_torch)
    if ort_mem is not None and default_size:
        ort_kb = ort_mem / 1024
        ort_ratio = ort_mem / default_size
        print(f"  ORT arena (profiled): {ort_kb:.1f} KB ({ort_ratio:.2f}x vs our default)")
    elif HAS_ORT:
        print(f"  ORT arena: profiling data not available")

    return results


def main():
    parser = argparse.ArgumentParser(description="Memory allocation ablation benchmark")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config names (e.g. 'Toy,Small,GPT-2')")
    parser.add_argument("--all", action="store_true",
                        help="Run all configs (default: toy,small,gpt2-s16)")
    args = parser.parse_args()

    # Determine which configs to run
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

    # Filter GPT-2 if transformers unavailable
    gpt2_configs = [c for c in configs if c.model == "gpt2"]
    if gpt2_configs and not HAS_TRANSFORMERS:
        print(f"Warning: skipping {len(gpt2_configs)} GPT-2 configs (transformers not installed)")
        configs = [c for c in configs if c.model != "gpt2"]

    print("Memory Allocation Ablation")
    print("="*90)
    print(f"ONNX Runtime: {'v' + ort.__version__ if HAS_ORT else 'not available'}")
    print(f"Transformers (GPT-2): {'available' if HAS_TRANSFORMERS else 'not available'}")
    print(f"Running {len(configs)} configs x {len(STRATEGIES)} strategies")

    all_results = {}
    for cfg in configs:
        all_results[cfg.name] = run_config(cfg)

    # Summary
    print()
    print("="*90)
    print("SUMMARY: Arena size improvements")
    print("="*90)
    print(f"  {'Config':<12} {'no-opt':>10} {'v1':>10} {'v2':>10} {'v3':>10} {'best':>15}")
    print(f"  {'-'*70}")
    for cfg in configs:
        r = all_results.get(cfg.name, {})
        no_opt = r.get("no-opt", 0)
        v1 = r.get("v1 (default)", 0)
        v2 = r.get("v2", 0)
        v3 = r.get("v3", 0)

        # Find best
        best_size = min(v for v in r.values() if v > 0) if r.values() else 0
        best_name = next((k for k, v in r.items() if v == best_size), "—")

        fmt = lambda x: f"{x/1024:.1f}KB" if x > 0 else "—"
        print(f"  {cfg.name:<12} {fmt(no_opt):>10} {fmt(v1):>10} {fmt(v2):>10} {fmt(v3):>10} {best_name:>15}")


if __name__ == "__main__":
    main()
