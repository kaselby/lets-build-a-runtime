# Inference Runtime

An interpreter-style inference runtime built from scratch. Takes a PyTorch model, compiles it to an optimized graph, plans memory, and executes via C kernels — all in one pipeline. Beats eager PyTorch on custom transformers, GPT-2, and Qwen3.


## What it does

```
torch.nn.Module
  → torch.export          Capture ATen-level graph
  → exporter              Map to our IR, load weights
  → optimization passes   Absorb, fold, fuse (DAG + pattern), DCE
  → memory planner        Lifetime analysis, arena layout, scratch buffers
  → compiled executor     One C call for the whole graph
```

The speedup comes from eliminating overhead between ops, not from faster kernels:

- **No per-op allocation.** Eager PyTorch allocates a fresh tensor for every intermediate. We pre-plan memory and reuse arena regions via lifetime analysis.
- **No Python dispatch overhead.** The compiled executor is a single `ctypes` call that dispatches a flat C struct array — no Python between ops.
- **Fused ops.** Pattern-matched fusion eliminates memory round-trips (MATMUL+ADD via sgemm beta, ADD+RELU in one pass, full attention as one kernel, gated activations).
- **DAG fusion.** Recognizes complex multi-input patterns like GELU, RMSNorm, and SiLU from their decomposed forms.

## Performance

Benchmarked on Apple M4 Max. Times are median over 100 iterations.

**Custom transformer** (single-layer block with multi-head attention):

| Config | d_model | seq_len | PyTorch | Ours | vs PT | vs ORT |
|--------|---------|---------|---------|------|-------|--------|
| Toy | 64 | 32 | 264 us | 25 us | 10.8x | 1.6x |
| Small | 256 | 128 | 758 us | 444 us | 1.7x | 1.0x |
| Medium | 512 | 256 | 2.86 ms | 1.84 ms | 1.6x | 1.2x |
| GPT-2 | 768 | 512 | 7.39 ms | 5.06 ms | 1.5x | 1.8x |
| 1B | 2048 | 512 | 29.7 ms | 23.7 ms | 1.3x | 2.4x |
| 7B | 4096 | 1024 | 185 ms | 165 ms | 1.1x | 2.8x |
| 7B-4K | 4096 | 4096 | 1.12 s | 713 ms | 1.6x | — |
| 1B-8K | 2048 | 8192 | 1.45 s | 584 ms | 2.5x | — |

**GPT-2** (2-layer HuggingFace body, 768d/12h):

| seq_len | PyTorch | Ours | vs PT | vs ORT |
|---------|---------|------|-------|--------|
| 16 | 4.90 ms | 4.14 ms | 1.2x | 0.9x |
| 64 | 8.36 ms | 5.50 ms | 1.5x | 1.5x |
| 256 | 18.5 ms | 14.1 ms | 1.3x | 2.3x |
| 1024 | 61.6 ms | 47.5 ms | 1.3x | 3.3x |

**Qwen3** (2-layer HuggingFace body with GQA):

| Model | seq_len | PyTorch | Ours | vs PT | vs ORT |
|-------|---------|---------|------|-------|--------|
| 0.6B | 256 | 54.4 ms | 42.2 ms | 1.3x | 2.5x |
| 0.6B | 1024 | 170 ms | 142 ms | 1.2x | 3.1x |
| 4B | 256 | 146 ms | 131 ms | 1.1x | 2.4x |
| 4B | 1024 | 498 ms | 462 ms | 1.1x | 2.8x |

**Memory** (activation arena, weights excluded):

| Config | Ours | PyTorch | ORT | vs PT | vs ORT |
|--------|------|---------|-----|-------|--------|
| GPT-2 768d s512 | 9 MB | 45 MB | 24 MB | 5.0x | 2.7x |
| 7B 4096d s1024 | 96 MB | 480 MB | 192 MB | 5.0x | 2.0x |
| 7B-4K s4096 | 384 MB | 5.0 GB | 2.3 GB | 13.0x | 6.0x |
| GPT-2 s1024 | 202 MB | 199 MB | 315 MB | 1.0x | 1.6x |
| Qwen3-4B s1024 | 614 MB | 604 MB | 867 MB | 1.0x | 1.4x |

Our planner achieves large savings on custom transformers (lifetime reuse + inplace). On real HuggingFace models the graph structure limits reuse, but we still match or beat PyTorch and consistently beat ORT.

## Architecture

**Python + C hybrid.** Python handles the graph layer (IR, passes, planner, executor orchestration). C handles the kernels (CBLAS matmul via Accelerate/OpenBLAS, hand-written element-wise ops, fused attention with GCD parallelism). Connected via `ctypes` with zero-copy pointer passing through numpy.

**Single graph IR throughout.** No multi-level IR lowering. Optimization passes mutate the graph in-place, the planner assigns arena offsets, the executor builds a C struct array and dispatches it. Closest in spirit to ONNX Runtime's CPU execution provider.

**Memory model.** Weights live outside the arena (permanent, loaded once). The arena is a flat numpy byte buffer for intermediate activations and kernel scratch workspace, sized by the planner. RESHAPE and contiguous SLICE ops are zero-copy aliases into the same arena region.

**Two execution paths.** Compiled executor (one ctypes call, zero dispatch overhead) for production. Interpreted executor (Python loop with backend chain) for debugging and per-op profiling.

**Dynamic shapes.** Export once with symbolic dimensions, resolve to concrete values without re-exporting or re-optimizing. Session handles rebinding automatically.

## Project structure

```
runtime/
  ir.py              — Graph IR: OpType enum, TensorInfo, Node, Graph
  ops.py             — Op registry: evaluators, scratch, alias, inplace, shape inference
  planner.py         — Lifetime analysis, first-fit arena allocation, scratch buffers
  session.py         — Session API: optimize → plan → compile → run
  exporter/
    torch/
      exporter.py    — torch.export → our Graph (dynamic shape support)
      handlers.py    — ATen op handler registry (~30 handlers)
  passes/
    passes.py        — Pass infrastructure + core passes (absorption, folding, DCE)
    fusion.py        — Pattern-based fusion + DAG fusion (GELU, RMSNorm, SiLU)
  executor/
    common.py        — Executor ABC, COpNode struct, profiling types
    compiled.py      — CompiledExecutor: single ctypes call for whole graph
    interpreted.py   — InterpretedExecutor: Python loop with backend chain
  backends/
    numpy_backend.py — In-place numpy kernels (fallback)
    c_backend.py     — ctypes wrappers for C shared library
  validation/
    core.py          — Validator types, Phase, Severity, registry, runner
    graph.py         — Graph-phase validators (POST_EXPORT through POST_RESOLVE)
    plan.py          — Memory plan validators (POST_PLAN)
    execution.py     — Execution plan validators (PRE_EXECUTE)
csrc/
  executor.c         — C dispatch loop (function pointer table, per-op dispatch)
  runtime.h          — Shared C types (OpNode struct, op enum)
  ops.h              — Kernel declarations, PARALLEL_FOR macro (GCD on macOS)
  Makefile           — Accelerate on macOS, OpenBLAS on Linux
  ops/
    matmul.c         — CBLAS sgemm/sgemv, batched matmul
    attn.c           — Standard + flash attention, causal masking, GQA, GCD parallel
    elementwise.c    — Unary/binary ops, scalar/broadcast variants
    reduce.c         — Max, sum, softmax (numerically stable)
    norm.c           — LayerNorm, RMSNorm
    shape.c          — Transpose, slice, cat, embedding
tests/
  conftest.py          — Shared fixtures (backends, models, helpers)
  test_backends.py     — Op correctness: every op × both backends × various shapes
  test_end_to_end.py   — Full pipeline oracle tests (per-op + compiled vs PyTorch)
  test_passes.py       — Optimization pass invariants and correctness
  test_planner.py      — Arena no-overlap, reuse savings, reshape aliasing
  test_session.py      — Session API: dynamic shapes, rebind, validation
  test_dynamic_shapes.py — Dynamic shape export, resolve, re-resolve
  test_ops.py          — OpDef registry coverage (evaluators, alias, extras, shape inference)
  test_benchmark.py    — Performance ablation (pytest -m benchmark -s)
docs/
  DESIGN.md            — Final-state design decisions and rationale
  DESIGN_LOG_FULL.md   — Full iterative design log
ablation/              — Standalone benchmark scripts and results
```

## Getting started

**Prerequisites:** Python 3.13, PyTorch, numpy. macOS (Accelerate) or Linux (OpenBLAS). Optional: `transformers` for GPT-2/Qwen3 models.

```bash
# Build the C libraries
cd csrc && make && cd ..

# Run tests (202 tests, excludes benchmarks by default)
python -m pytest tests/

# Run benchmarks
python -m pytest tests/test_benchmark.py -m benchmark -s
```

**Quick example:**

```python
import torch
from runtime.exporter import export_model
from runtime.session import Session

# Define a model
model = torch.nn.Sequential(
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 512),
)
x = torch.randn(32, 512)

# Compile and run (one C call)
graph = export_model(model, (x,))
session = Session(graph)
session.create()                    # optimize → plan → compile
result = session.run({"x": x.numpy()})
```

**With dynamic shapes:**

```python
graph = export_model(model, (x,), dynamic_dims={'L': [('x', 0)]})
session = Session(graph)
session.create(bindings={'L': 32})  # resolve shapes, plan, compile
result = session.run({"x": x.numpy()})

# Rebind to a different batch size without re-optimizing
session.create(bindings={'L': 64})
```

## Optimization passes

1. **BLAS flag absorption** — Folds adjacent transposes into `sgemm` `CblasTrans` flag. Folds scalar multiply/divide into matmul `alpha`. Keeps `nn.Linear` weights in their original `[out, in]` layout for better BLAS packing performance.

2. **Constant folding** — Evaluates subgraphs with all-constant inputs at compile time using numpy evaluators. Eliminates infrastructure ops (ARANGE, CAST, comparisons) that can't reach the executor.

3. **Causal mask absorption** — Detects constant causal masks and absorbs them into `ATTENTION(causal=True)`. Eliminates the mask tensor entirely; the C kernel generates the pattern on-the-fly.

4. **DAG fusion** — Recognizes complex multi-input patterns from their decomposed forms:
   - GELU (tanh approximation): `mul → pow → mul → add → tanh → add → mul`
   - RMSNorm: `pow → sum → div → add → rsqrt → mul`
   - SiLU: `neg → exp → add → div`

5. **Pattern fusion** — Registry of `FusionPattern` objects matched greedily by priority:
   - `ATTENTION`: MATMUL→SOFTMAX→MATMUL (with causal mask absorption, GQA detection)
   - `GATED_ACT`: SiLU/GELU + MUL (with/without bias) for gated FFN blocks
   - `BIAS_RELU`: ADD+RELU in one pass
   - `MATMUL_ADD`: sgemm with `beta=1.0` for fused bias

6. **Dead code elimination** — Removes nodes with no consumers. Cleans up dead constants.

## Attention

Three recognition paths, all converging to the same fused C kernel:

| Source | How it's recognized |
|--------|-------------------|
| `F.scaled_dot_product_attention` | Exporter maps directly to `ATTENTION` |
| `F.softmax(Q @ K^T / sqrt(d))` | Absorption folds scale into alpha, fusion matches MATMUL→SOFTMAX→MATMUL |
| Manual softmax decomposition | DAG fusion can recognize GELU/RMSNorm patterns; softmax DAG not yet implemented |

Two kernel implementations:
- **Standard:** Materializes full S×S attention matrix. Uses sgemm for Q@K^T and P@V.
- **Flash:** Tiled online-softmax (B_r=128, B_c=256). Never materializes full S×S. Planner selects automatically based on seq_len.

Both support:
- **Causal masking** via `causal=True` attribute (upper-triangular -inf pattern generated on-the-fly)
- **Grouped query attention** via `group_size` parameter (K/V strided by group size)
- **GCD `dispatch_apply`** for parallel execution across batch×head slices on macOS

## Design decisions

See [docs/DESIGN.md](docs/DESIGN.md) for detailed rationale on:
- Single graph IR (no multi-level lowering)
- Weight transpose handling (CblasTrans vs pre-transpose)
- Pattern-matched fusion over interpreter fusion (bespoke kernels won)
- MATMUL+ADD via sgemm beta trick
- Compiled executor extras encoding (float-to-int bit-cast)
- Memory planner design (lifetime analysis, memory-aware ordering, inplace reuse)
- Validation at pipeline boundaries
- Registry-everywhere extensibility pattern
