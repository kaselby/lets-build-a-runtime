#!/usr/bin/env python3
"""Three-way memory comparison: our runtime vs PyTorch eager vs ONNX Runtime.

Metrics (activations only, no weights):
  1. Peak activation memory with reuse — what each runtime actually allocates:
       - Ours:    planner arena size (exact)
       - PyTorch: torch.profiler peak-alive via self_cpu_memory_usage
       - ORT:     AllocatorGetStats MaxInUse via C API, minus weight bytes
  2. Total intermediates (no reuse) — upper bound, for context.

Run:  python ablation/bench_memory_comparison.py [--configs toy,small,...] [--all]
"""

import argparse
import logging
import os
import platform
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
from runtime.planner import plan

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
    import onnx
    import onnx.shape_inference
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

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
# ONNX export helper
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
            torch.onnx.export(
                model, (x_torch,), onnx_path,
                input_names=["input"], output_names=["output"],
                opset_version=18,
            )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()
        logging.getLogger("torch.onnx").setLevel(old_level)
    return onnx_path


# =====================================================================
# Metric 1: Peak activation memory (with reuse) — what each runtime uses
# =====================================================================

def our_arena(graph):
    """Our planner's arena size (with reuse, in-place, aliases)."""
    return plan(graph).arena_size


def pytorch_peak_alive(model, x_torch):
    """Peak simultaneously-alive activation memory during PyTorch eager execution.

    Uses torch.profiler with profile_memory=True. We track self_cpu_memory_usage
    (not cpu_memory_usage) to avoid double-counting nested ops like linear→addmm.
    Deallocation events ([memory] with negative deltas) are included, so the
    running sum reflects the actual peak-alive watermark.

    Weights are excluded — the profiler only sees allocations during the forward
    pass, and weights are already allocated before profiling starts.
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
    """ORT's peak activation memory via AllocatorGetStats C API.

    Calls ORT's C API directly (bypassing Python bindings) to get MaxInUse from
    the session's arena allocator. ORT's arena includes both weights and activations,
    so we subtract the model's parameter bytes to isolate activation memory.

    Returns (activation_bytes, raw_max_in_use, weight_bytes) or None on failure.
    """
    if not HAS_ORT or not HAS_ORT_CAPI:
        return None

    try:
        onnx_path = _export_onnx(model, x_torch)
        input_shape = tuple(x_torch.shape)
        stats = get_ort_arena_stats(onnx_path, input_shape)
        os.unlink(onnx_path)

        if "MaxInUse" not in stats:
            return None

        max_in_use = int(stats["MaxInUse"])
        weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        activation_bytes = max_in_use - weight_bytes

        return activation_bytes, max_in_use, weight_bytes
    except Exception as e:
        print(f"    (ORT C API failed: {e})")
        return None


# =====================================================================
# Metric 2: Analytical intermediate sizes (no reuse, for context)
# =====================================================================

def our_intermediates(graph):
    """Sum of all intermediate tensor sizes from our optimized graph (no reuse)."""
    external = set(graph.inputs) | set(graph.constants)
    total = 0
    for name, info in graph.tensors.items():
        if name not in external:
            total += int(np.prod(info.shape)) * np.dtype(info.dtype).itemsize
    return total


def ort_intermediates(model, x_torch):
    """Sum of intermediate tensor sizes from ORT's optimized ONNX graph."""
    if not HAS_ORT or not HAS_ONNX:
        return None

    try:
        onnx_path = _export_onnx(model, x_torch)
        fd2, opt_path = tempfile.mkstemp(suffix="_opt.onnx")
        os.close(fd2)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.optimized_model_filepath = opt_path
        opts.log_severity_level = 3
        ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])

        opt_model = onnx.load(opt_path)
        opt_model = onnx.shape_inference.infer_shapes(opt_model)
        g = opt_model.graph

        init_names = {init.name for init in g.initializer}
        node_outputs = set()
        for node in g.node:
            node_outputs.update(node.output)
        intermediate_names = node_outputs - init_names

        type_map = {}
        for vi in g.value_info:
            type_map[vi.name] = vi.type
        for out in g.output:
            type_map[out.name] = out.type

        dtype_sizes = {1: 4, 2: 1, 3: 1, 5: 2, 6: 4, 7: 8, 10: 2, 11: 8}

        def tensor_size(type_proto):
            tt = type_proto.tensor_type
            itemsize = dtype_sizes.get(tt.elem_type, 4)
            if tt.shape is None:
                return 0
            total = itemsize
            for dim in tt.shape.dim:
                if dim.dim_value > 0:
                    total *= dim.dim_value
                else:
                    return 0
            return total

        total = sum(tensor_size(type_map[n]) for n in intermediate_names if n in type_map)

        os.unlink(onnx_path)
        if os.path.exists(opt_path):
            os.unlink(opt_path)

        return total if total > 0 else None
    except Exception:
        return None


# =====================================================================
# Formatting helpers
# =====================================================================

def _fmt_kb(nbytes):
    if nbytes is None:
        return "n/a"
    return f"{nbytes / 1024:,.1f} KB"


def _ratio_str(value, baseline):
    if value is None or baseline is None or baseline == 0:
        return ""
    return f"({value / baseline:.2f}x)"


# =====================================================================
# Run a single config
# =====================================================================

def run_config(cfg):
    """Measure memory metrics for a single config."""

    # Build model
    is_gpt2 = cfg.model == "gpt2"
    if is_gpt2:
        model = _make_gpt2_body(n_head=cfg.n_heads, n_embd=cfg.d_model)
        if model is None:
            print(f"\n  SKIPPED {cfg.name}: transformers package required")
            return
        model_desc = "GPT-2 body (2-layer)"
    else:
        model = NaiveTransformerBlock(cfg.d_model, cfg.n_heads)
        model.eval()
        model_desc = "NaiveTransformerBlock"

    x_torch = torch.randn(cfg.batch, cfg.seq_len, cfg.d_model)

    # Export and optimize our graph
    graph = export_model(model, (x_torch,))
    run_pipeline(graph)

    # ---- Peak activation memory (with reuse) ----
    ours_planned = our_arena(graph)
    pt_peak = pytorch_peak_alive(model, x_torch)
    ort_result = ort_peak_activations(model, x_torch)

    # ---- Analytical intermediates (no reuse, for context) ----
    ours_intermed = our_intermediates(graph)
    ort_intermed = ort_intermediates(model, x_torch)

    # ---- Print results ----
    w = 60
    print()
    print("=" * w)
    print(f"  Config: {cfg.name} (d={cfg.d_model}, h={cfg.n_heads}, s={cfg.seq_len})")
    print(f"  Model: {model_desc}")
    print("=" * w)

    # Peak activation memory (the main comparison)
    print()
    print("  Peak activation memory (with reuse):")
    print(f"    {'Ours (planner arena):':<35} {_fmt_kb(ours_planned):>12}")
    pt_ratio = _ratio_str(pt_peak, ours_planned)
    print(f"    {'PyTorch (profiler peak-alive):':<35} {_fmt_kb(pt_peak):>12}  {pt_ratio}")
    if ort_result is not None:
        ort_activ, ort_raw, ort_weights = ort_result
        ort_ratio = _ratio_str(ort_activ, ours_planned)
        print(f"    {'ORT (C API MaxInUse - weights):':<35} {_fmt_kb(ort_activ):>12}  {ort_ratio}")
    elif HAS_ORT:
        print(f"    {'ORT:':<35} {'not available':>12}")

    # Intermediates for context
    print()
    print("  Total intermediates (no reuse, for context):")
    print(f"    {'Ours (optimized graph):':<35} {_fmt_kb(ours_intermed):>12}")
    if ort_intermed is not None:
        print(f"    {'ORT (optimized graph):':<35} {_fmt_kb(ort_intermed):>12}")

    # Reuse efficiency
    print()
    reuse_ratio = ours_planned / ours_intermed if ours_intermed else 0
    print(f"  Reuse efficiency (arena / intermediates): {reuse_ratio:.1%}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Memory comparison: us vs PyTorch vs ORT"
    )
    parser.add_argument(
        "--configs", type=str, default=None,
        help="Comma-separated config names (e.g. 'Toy,Small,gpt2-s16')",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all configs (default: Toy, Small, gpt2-s16)",
    )
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

    print("Memory Comparison: Ours vs PyTorch vs ORT")
    print("=" * 60)
    print(f"  ONNX Runtime:  {'v' + ort.__version__ if HAS_ORT else 'not available'}")
    print(f"  ORT C API:     {'available' if HAS_ORT_CAPI else 'not available'}")
    print(f"  Transformers:  {'available' if HAS_TRANSFORMERS else 'not available'}")
    print(f"  Platform:      {platform.system()} ({platform.machine()})")
    print(f"  Configs:       {len(configs)}")
    print()
    print("  Methodology:")
    print("    Ours:    exact arena size from memory planner (activations + scratch)")
    print("    PyTorch: torch.profiler peak-alive (self_cpu_memory_usage watermark)")
    print("    ORT:     AllocatorGetStats MaxInUse via C API, minus model weights")
    print("    All metrics exclude model weights — activation memory only.")

    for cfg in configs:
        run_config(cfg)

    print()


if __name__ == "__main__":
    main()
