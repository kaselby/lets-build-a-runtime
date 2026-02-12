"""Performance ablation: PyTorch vs all runtime execution paths.

Run with: pytest -m benchmark tests/test_benchmark.py -s

Reports timing for each configuration — no pass/fail assertions on speed.
The -s flag is needed to see the printed table.
"""

import time

import numpy as np
import pytest
import torch

from runtime.backends.c_backend import CBackend
from runtime.backends.numpy_backend import NumpyBackend
from runtime.exporter import export_model
from runtime.executor import Executor
from runtime.passes import (
    absorb_into_matmul, constant_fold,
    eliminate_dead_code, fuse,
)
from runtime.planner import plan

from conftest import SimpleMLP, NaiveTransformerBlock, SDPATransformerBlock


def _bench(fn, warmup=50, iters=500):
    for _ in range(warmup):
        fn()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t) / iters * 1e6


def _median(values):
    s = sorted(values)
    return s[len(s) // 2]


def _fmt(us):
    return f"{us/1000:7.2f}ms" if us >= 1000 else f"{us:7.1f}us"


CONFIGS = [
    (1, 512),
    (32, 512),
    (128, 512),
    (1, 2048),
    (32, 2048),
    (128, 2048),
    (1, 4096),
    (32, 4096),
    (128, 4096),
]

TRIALS = 3


@pytest.mark.benchmark
def test_ablation():
    """Full ablation: PyTorch vs numpy vs per-op C vs compiled C ± fusion.

    Columns:
      C exec   — compiled C, no fusion (absorb + fold + DCE only)
      +fusion  — adds pattern fusion (bias+relu, matmul+add)
    """
    base = [absorb_into_matmul, constant_fold, eliminate_dead_code]
    pipelines = {
        "C exec":  base,
        "+fusion": base[:2] + [fuse] + base[2:],
    }
    pipe_keys = list(pipelines.keys())

    header = (
        f"{'config':>12}  |  {'PyTorch':>9}  {'numpy':>9}  {'C perop':>9}"
        + "".join(f"  {k:>9}" for k in pipe_keys)
        + f"  |  {'np/PT':>6}  {'Cop/PT':>6}"
        + "".join(f"  {k+'/PT':>7}" for k in pipe_keys)
    )
    print(f"\n{header}")
    print("-" * len(header))

    for batch, dim in CONFIGS:
        model = SimpleMLP(dim)
        model.eval()
        x_torch = torch.randn(batch, dim)
        x_np = x_torch.numpy().copy()

        def setup(pipeline):
            graph = export_model(model, (x_torch,))
            for p in pipeline:
                p(graph)
            ep = plan(graph)
            return graph, ep

        # --- Numpy backend, per-op (full pipeline) ---
        graph_np, ep_np = setup(pipelines["+fusion"])
        exec_np = Executor(backends=[NumpyBackend()])

        # --- C backend, per-op (full pipeline) ---
        graph_c, ep_c = setup(pipelines["+fusion"])
        exec_c = Executor(backends=[CBackend(), NumpyBackend()])

        # --- Compiled C, each fusion level ---
        compiled = {}
        executors = {}
        for key, pipeline in pipelines.items():
            g, ep = setup(pipeline)
            ex = Executor(backends=[CBackend(), NumpyBackend()])
            compiled[key] = ex.compile_plan(ep)
            executors[key] = ex

        inp = graph_np.inputs[0]

        # Collect timings
        times = {"pt": [], "np": [], "cop": []}
        for k in pipe_keys:
            times[k] = []

        for _ in range(TRIALS):
            with torch.no_grad():
                times["pt"].append(_bench(lambda: model(x_torch)))
            times["np"].append(_bench(lambda: exec_np.execute(ep_np, {inp: x_np})))
            times["cop"].append(_bench(lambda: exec_c.execute(ep_c, {inp: x_np})))
            for k in pipe_keys:
                ex, cp = executors[k], compiled[k]
                times[k].append(_bench(
                    lambda ex=ex, cp=cp: ex.execute_compiled(cp, {inp: x_np})))

        med = {k: _median(v) for k, v in times.items()}
        pt = med["pt"]

        label = f"{batch}x{dim}"
        print(
            f"{label:>12}  |  {_fmt(pt):>9}  {_fmt(med['np']):>9}  {_fmt(med['cop']):>9}"
            + "".join(f"  {_fmt(med[k]):>9}" for k in pipe_keys)
            + f"  |  {med['np']/pt:6.2f}  {med['cop']/pt:6.2f}"
            + "".join(f"  {med[k]/pt:7.2f}" for k in pipe_keys)
        )


# (batch, seq_len, d_model, n_heads)
TRANSFORMER_CONFIGS = [
    (1, 16, 64, 4),
    (4, 16, 64, 4),
    (1, 64, 128, 8),
    (4, 64, 128, 8),
    (1, 128, 256, 8),
    (4, 128, 256, 8),
]

TRANSFORMER_TRIALS = 3


@pytest.mark.benchmark
def test_transformer_ablation():
    """Transformer ablation: PyTorch baselines vs compiled C ± fusion.

    Columns:
      PT naive — eager PyTorch with F.softmax attention
      PT SDPA  — eager PyTorch with scaled_dot_product_attention
      C exec   — compiled C, no fusion (absorb + fold + DCE only)
      +fusion  — adds pattern fusion (attention, bias+relu, matmul+add)
    """
    base = [absorb_into_matmul, constant_fold, eliminate_dead_code]
    pipelines = {
        "C exec":  base,
        "+fusion": base[:2] + [fuse] + base[2:],
    }
    pipe_keys = list(pipelines.keys())

    header = (
        f"{'config':>16}  |  {'PT naive':>9}  {'PT SDPA':>9}"
        + "".join(f"  {k:>9}" for k in pipe_keys)
        + f"  |  {'naive/PT':>7}  {'fused/PT':>8}  {'fused/SDPA':>10}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for batch, seq, d_model, n_heads in TRANSFORMER_CONFIGS:
        # Build models with shared weights for fair comparison
        naive = NaiveTransformerBlock(d_model, n_heads)
        naive.eval()
        sdpa = SDPATransformerBlock(d_model, n_heads)
        sdpa.load_state_dict(naive.state_dict())
        sdpa.eval()

        x_torch = torch.randn(batch, seq, d_model)
        x_np = x_torch.numpy().copy()

        # Compiled C at each fusion level
        compiled = {}
        executors = {}
        inp_name = None
        for key, pipeline in pipelines.items():
            graph = export_model(naive, (x_torch,))
            for p in pipeline:
                p(graph)
            ep = plan(graph)
            ex = Executor(backends=[CBackend(), NumpyBackend()])
            compiled[key] = ex.compile_plan(ep)
            executors[key] = ex
            if inp_name is None:
                inp_name = graph.inputs[0]

        times = {"pt_naive": [], "pt_sdpa": []}
        for k in pipe_keys:
            times[k] = []

        for _ in range(TRANSFORMER_TRIALS):
            with torch.no_grad():
                times["pt_naive"].append(_bench(lambda: naive(x_torch), warmup=20, iters=200))
                times["pt_sdpa"].append(_bench(lambda: sdpa(x_torch), warmup=20, iters=200))
            for k in pipe_keys:
                ex, cp = executors[k], compiled[k]
                times[k].append(_bench(
                    lambda ex=ex, cp=cp: ex.execute_compiled(cp, {inp_name: x_np}),
                    warmup=20, iters=200))

        med = {k: _median(v) for k, v in times.items()}
        pt_naive = med["pt_naive"]
        pt_sdpa = med["pt_sdpa"]
        fused = med["+fusion"]

        label = f"{batch}x{seq}x{d_model}"
        print(
            f"{label:>16}  |  {_fmt(pt_naive):>9}  {_fmt(pt_sdpa):>9}"
            + "".join(f"  {_fmt(med[k]):>9}" for k in pipe_keys)
            + f"  |  {med['C exec']/pt_naive:7.2f}  {fused/pt_naive:8.2f}  {fused/pt_sdpa:10.2f}"
        )
