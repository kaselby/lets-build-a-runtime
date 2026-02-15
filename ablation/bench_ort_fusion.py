"""
Investigate ORT attention fusion with the transformer-specific optimizer.

ORT has two optimization paths:
  1. Session-level (ORT_ENABLE_ALL) — basic op fusion
  2. onnxruntime.transformers.optimizer.optimize_model() — transformer-specific
     pattern matching for attention, layernorm, etc.

This script tests both paths on several model/export variants to find a
configuration where ORT successfully fuses attention.

Run:  python ablation/bench_ort_fusion.py
"""

import math
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
from onnxruntime.transformers.optimizer import optimize_model


# =====================================================================
# Models
# =====================================================================

class NaiveTransformerBlock(nn.Module):
    """Our custom transformer block (manual attention)."""
    def __init__(self, d_model=768, n_heads=12):
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


# =====================================================================
# Helpers
# =====================================================================

def export_onnx_quiet(model, args, path, opset=18, dynamo=None, **kwargs):
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
            export_kwargs = dict(opset_version=opset, **kwargs)
            if dynamo is not None:
                export_kwargs["dynamo"] = dynamo
            torch.onnx.export(model, args, path, **export_kwargs)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()
        for name, level in old_levels.items():
            logging.getLogger(name).setLevel(level)


def graph_summary(model_or_path):
    """Return (total_nodes, op_counts_dict)."""
    if isinstance(model_or_path, str):
        model = onnx.load(model_or_path)
    else:
        model = model_or_path
    counts = {}
    for node in model.graph.node:
        op = node.op_type
        if node.domain and node.domain != "":
            op = f"{node.domain}:{op}"
        counts[op] = counts.get(op, 0) + 1
    return len(model.graph.node), counts


def print_graph(model_or_path, label):
    """Print graph summary."""
    n, counts = graph_summary(model_or_path)
    sorted_ops = sorted(counts.items(), key=lambda x: -x[1])
    ops_str = ", ".join(f"{op}:{c}" for op, c in sorted_ops)
    print(f"    {label} ({n} nodes): {ops_str}")


def fusion_stats_str(onnx_model_obj):
    """Get fusion statistics from an OnnxModel object."""
    from onnxruntime.transformers.optimizer import get_fusion_statistics
    stats = get_fusion_statistics(onnx_model_obj.model)
    return dict(stats)


def try_optimize(raw_path, model_type, num_heads, hidden_size, label):
    """Try optimize_model and report results. Returns opt_path or None."""
    fd, opt_path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    try:
        opt = optimize_model(raw_path, model_type=model_type,
                             num_heads=num_heads, hidden_size=hidden_size)
        stats = fusion_stats_str(opt)
        opt.save_model_to_file(opt_path)
        print(f"    {label} fusion stats: {stats}")
        print_graph(opt_path, f"{label} graph")
        return opt_path
    except Exception as e:
        print(f"    {label} FAILED: {e}")
        if os.path.exists(opt_path):
            os.unlink(opt_path)
        return None


def bench_ort_session(path, feed, warmup=20, iters=50, threads=1,
                      opt_level=None):
    """Return median inference time in us."""
    if opt_level is None:
        opt_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts = ort.SessionOptions()
    opts.graph_optimization_level = opt_level
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])

    for _ in range(warmup):
        session.run(None, feed)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        session.run(None, feed)
        times.append((time.perf_counter_ns() - t0) / 1000)
    return sorted(times)[len(times) // 2]


def bench_pt(model, args, warmup=20, iters=50):
    """Return median PyTorch inference time in us."""
    if not isinstance(args, tuple):
        args = (args,)
    with torch.no_grad():
        for _ in range(warmup):
            model(*args)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            model(*args)
            times.append((time.perf_counter_ns() - t0) / 1000)
    return sorted(times)[len(times) // 2]


def _fmt(us):
    if us >= 1_000_000:
        return f"{us/1_000_000:.2f}s"
    if us >= 1000:
        return f"{us/1000:.2f}ms"
    return f"{us:.1f}us"


# =====================================================================
# Test 1: Custom model with optimize_model (gpt2 + bert model types)
# =====================================================================

def test_custom_model():
    print("\n" + "="*80)
    print("  TEST 1: Custom NaiveTransformerBlock — model_type variants")
    print("="*80)

    d_model, n_heads, seq = 768, 12, 128
    model = NaiveTransformerBlock(d_model, n_heads).eval()
    x = torch.randn(1, seq, d_model)
    x_np = x.numpy().copy()
    feed = {"input": x_np}

    fd, raw_path = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    export_onnx_quiet(model, (x,), raw_path,
                      input_names=["input"], output_names=["output"])
    print_graph(raw_path, "Raw ONNX")

    pt_us = bench_pt(model, x)
    raw_us = bench_ort_session(raw_path, feed)
    print(f"\n    PyTorch:  {_fmt(pt_us)}")
    print(f"    ORT raw:  {_fmt(raw_us)} ({raw_us/pt_us:.2f}x PT)")

    # Try session-level optimization
    session_us = bench_ort_session(raw_path, feed,
                                   opt_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    print(f"    ORT session opt: {_fmt(session_us)} ({session_us/pt_us:.2f}x PT)")

    # Try different model types with optimize_model
    for mtype in ['gpt2', 'bert']:
        opt_path = try_optimize(raw_path, mtype, n_heads, d_model,
                                f"model_type={mtype}")
        if opt_path:
            opt_us = bench_ort_session(opt_path, feed)
            print(f"    ORT {mtype} optimized: {_fmt(opt_us)} ({opt_us/pt_us:.2f}x PT)")
            os.unlink(opt_path)

    os.unlink(raw_path)


# =====================================================================
# Test 2: Custom model with TorchScript export (dynamo=False)
# =====================================================================

def test_torchscript_export():
    print("\n" + "="*80)
    print("  TEST 2: Custom model — TorchScript vs Dynamo ONNX export")
    print("="*80)

    d_model, n_heads, seq = 768, 12, 128
    model = NaiveTransformerBlock(d_model, n_heads).eval()
    x = torch.randn(1, seq, d_model)
    x_np = x.numpy().copy()
    feed = {"input": x_np}

    for dynamo_flag, label in [(False, "TorchScript"), (True, "Dynamo")]:
        print(f"\n  --- {label} export ---")
        fd, raw_path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        try:
            export_onnx_quiet(model, (x,), raw_path,
                              input_names=["input"], output_names=["output"],
                              dynamo=dynamo_flag)
            print_graph(raw_path, "Raw")

            for mtype in ['gpt2', 'bert']:
                opt_path = try_optimize(raw_path, mtype, n_heads, d_model,
                                        f"{label}+{mtype}")
                if opt_path:
                    opt_us = bench_ort_session(opt_path, feed)
                    pt_us = bench_pt(model, x, warmup=5, iters=20)
                    print(f"    Bench: {_fmt(opt_us)} ({opt_us/pt_us:.2f}x PT)")
                    os.unlink(opt_path)
        except Exception as e:
            print(f"    {label} export FAILED: {e}")
        if os.path.exists(raw_path):
            os.unlink(raw_path)


# =====================================================================
# Test 3: HuggingFace GPT2 (the gold standard for ORT's GPT-2 fusion)
# =====================================================================

def test_hf_gpt2():
    print("\n" + "="*80)
    print("  TEST 3: HuggingFace GPT2Model — ORT's intended target")
    print("="*80)

    try:
        from transformers import GPT2Config, GPT2Model
    except ImportError:
        print("    transformers not installed — skipping")
        return

    config = GPT2Config(
        n_layer=1,
        n_head=12,
        n_embd=768,
        use_cache=False,
        attn_implementation="eager",
    )
    model = GPT2Model(config).eval()
    seq = 128
    input_ids = torch.randint(0, config.vocab_size, (1, seq))
    ids_np = input_ids.numpy()
    feed = {"input_ids": ids_np}

    pt_us = bench_pt(model, input_ids)
    print(f"    PyTorch: {_fmt(pt_us)}")

    # Try both TorchScript and Dynamo export
    for dynamo_flag, label in [(False, "TorchScript"), (True, "Dynamo")]:
        print(f"\n  --- HF GPT2 + {label} export ---")
        fd, raw_path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        try:
            export_onnx_quiet(model, (input_ids,), raw_path,
                              input_names=["input_ids"],
                              output_names=["last_hidden_state"],
                              dynamo=dynamo_flag)
            print_graph(raw_path, "Raw")

            raw_us = bench_ort_session(raw_path, feed)
            print(f"    ORT raw: {_fmt(raw_us)} ({raw_us/pt_us:.2f}x PT)")

            # Session-level optimization
            session_us = bench_ort_session(raw_path, feed,
                                           opt_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
            print(f"    ORT session opt: {_fmt(session_us)} ({session_us/pt_us:.2f}x PT)")

            # Transformer optimizer
            for mtype in ['gpt2', 'bert']:
                opt_path = try_optimize(raw_path, mtype, 12, 768,
                                        f"{label}+{mtype}")
                if opt_path:
                    opt_us = bench_ort_session(opt_path, feed)
                    print(f"    Bench: {_fmt(opt_us)} ({opt_us/pt_us:.2f}x PT)")
                    os.unlink(opt_path)

        except Exception as e:
            print(f"    {label} export FAILED: {e}")
            import traceback
            traceback.print_exc()

        if os.path.exists(raw_path):
            os.unlink(raw_path)


# =====================================================================
# Test 4: BERT-style encoder (ORT's most mature fusion target)
# =====================================================================

def test_hf_bert():
    print("\n" + "="*80)
    print("  TEST 4: HuggingFace BertModel — ORT's most mature target")
    print("="*80)

    try:
        from transformers import BertConfig, BertModel
    except ImportError:
        print("    transformers not installed — skipping")
        return

    config = BertConfig(
        num_hidden_layers=1,
        num_attention_heads=12,
        hidden_size=768,
        intermediate_size=3072,
    )
    model = BertModel(config).eval()
    seq = 128
    input_ids = torch.randint(0, config.vocab_size, (1, seq))
    ids_np = input_ids.numpy()
    feed = {"input_ids": ids_np}

    pt_us = bench_pt(model, input_ids)
    print(f"    PyTorch: {_fmt(pt_us)}")

    for dynamo_flag, label in [(False, "TorchScript"), (True, "Dynamo")]:
        print(f"\n  --- BERT + {label} export ---")
        fd, raw_path = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        try:
            export_onnx_quiet(model, (input_ids,), raw_path,
                              input_names=["input_ids"],
                              output_names=["last_hidden_state"],
                              dynamo=dynamo_flag)
            print_graph(raw_path, "Raw")

            raw_us = bench_ort_session(raw_path, feed)
            print(f"    ORT raw: {_fmt(raw_us)} ({raw_us/pt_us:.2f}x PT)")

            session_us = bench_ort_session(raw_path, feed,
                                           opt_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
            print(f"    ORT session opt: {_fmt(session_us)} ({session_us/pt_us:.2f}x PT)")

            opt_path = try_optimize(raw_path, 'bert', 12, 768, f"{label}+bert")
            if opt_path:
                opt_us = bench_ort_session(opt_path, feed)
                print(f"    Bench: {_fmt(opt_us)} ({opt_us/pt_us:.2f}x PT)")
                os.unlink(opt_path)

        except Exception as e:
            print(f"    {label} export FAILED: {e}")
            import traceback
            traceback.print_exc()

        if os.path.exists(raw_path):
            os.unlink(raw_path)


# =====================================================================
# Test 5: Export with older opset (ORT fusion was designed for opset 11-14)
# =====================================================================

def test_older_opsets():
    print("\n" + "="*80)
    print("  TEST 5: Custom model — older opset versions (11-17)")
    print("="*80)

    d_model, n_heads, seq = 768, 12, 128
    model = NaiveTransformerBlock(d_model, n_heads).eval()
    x = torch.randn(1, seq, d_model)

    for opset in [11, 12, 13, 14, 15, 17]:
        for dynamo_flag, dlabel in [(False, "TS"), (True, "Dy")]:
            fd, path = tempfile.mkstemp(suffix=".onnx")
            os.close(fd)
            try:
                export_onnx_quiet(model, (x,), path,
                                  input_names=["input"], output_names=["output"],
                                  opset=opset, dynamo=dynamo_flag)
                n, counts = graph_summary(path)
                opt = optimize_model(path, model_type='gpt2',
                                     num_heads=n_heads, hidden_size=d_model)
                stats = fusion_stats_str(opt)
                attn_count = stats.get("Attention", 0) + stats.get("MultiHeadAttention", 0)
                marker = " <<<< ATTENTION FUSED!" if attn_count > 0 else ""
                print(f"    opset {opset:2d} {dlabel}: {n:3d} nodes → "
                      f"Attention={attn_count}, stats={stats}{marker}")
            except Exception as e:
                err = str(e)[:80]
                print(f"    opset {opset:2d} {dlabel}: FAILED — {err}")
            if os.path.exists(path):
                os.unlink(path)


def main():
    print(f"ONNX Runtime v{ort.__version__}")
    print(f"ONNX v{onnx.__version__}")
    print(f"PyTorch v{torch.__version__}")
    try:
        import transformers
        print(f"Transformers v{transformers.__version__}")
    except ImportError:
        print("Transformers: not installed")

    test_custom_model()
    test_torchscript_export()
    test_hf_gpt2()
    test_hf_bert()
    test_older_opsets()


if __name__ == "__main__":
    main()
