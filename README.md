# Inference Runtime

An interpreter-style inference runtime built from scratch. Takes a PyTorch model, compiles it to an optimized graph, plans memory, and executes via C kernels — all in one pipeline. Beats eager PyTorch (including `torch.sdpa`) on small transformers and MLPs.

Built as an interactive learning project and interview prep artifact for the ONNX Runtime team.

## What it does

```
torch.nn.Module
  → torch.export          Capture ATen-level graph
  → exporter              Map to our IR, load weights
  → optimization passes   Absorb, fold, fuse, DCE
  → memory planner        Lifetime analysis, arena layout
  → compiled executor     One C call for the whole graph
```

The speedup comes from eliminating overhead between ops, not from faster kernels:

- **No per-op allocation.** Eager PyTorch allocates a fresh tensor for every intermediate. We pre-plan memory and reuse arena regions via lifetime analysis.
- **No Python dispatch overhead.** The compiled executor is a single `ctypes` call that dispatches a flat C struct array — no Python between ops.
- **Fused ops.** Pattern-matched fusion eliminates memory round-trips (MATMUL+ADD via sgemm beta, ADD+RELU in one pass, full attention as one kernel).

## Performance

Benchmarked on Apple M4 Max. Times are median over 100 iterations.

**MLP** (3-layer, Linear→ReLU→Linear→ReLU→Linear):

| Config    | PyTorch  | Compiled C | Ratio  |
|-----------|----------|------------|--------|
| 1×512     | 20 us    | 13 us      | 0.63x  |
| 128×512   | 395 us   | 195 us     | 0.49x  |
| 32×2048   | 1.07 ms  | 848 us     | 0.79x  |

**Transformer** (single-layer block with multi-head attention):

| Config       | PT (SDPA) | Compiled C | Ratio  |
|--------------|-----------|------------|--------|
| 1×16×64      | 179 us    | 21 us      | 0.12x  |
| 4×64×128     | 460 us    | 335 us     | 0.73x  |
| 4×128×256    | 1.50 ms   | 1.20 ms    | 0.80x  |

## Architecture

**Python + C hybrid.** Python handles the graph layer (IR, passes, planner, executor orchestration). C handles the kernels (CBLAS matmul via Accelerate/OpenBLAS, hand-written element-wise ops, SIMD softmax). Connected via `ctypes` with zero-copy pointer passing through numpy.

**Single graph IR throughout.** No multi-level IR lowering. Optimization passes mutate the graph in-place, the planner assigns arena offsets, the executor builds a C struct array and dispatches it. Closest in spirit to ONNX Runtime's CPU execution provider.

**Memory model.** Weights live outside the arena (permanent, loaded once). The arena is a flat numpy byte buffer for intermediate activations and kernel scratch workspace, sized by the planner. RESHAPE ops are zero-copy aliases into the same arena region.

## Project structure

```
runtime/
  ir.py          — Graph IR: OpType enum, TensorInfo, Node, Graph
  exporter.py    — torch.export → our Graph (ATen handler registry)
  passes.py      — Optimization passes + fusion pattern registry
  planner.py     — Lifetime analysis, first-fit arena allocation, scratch buffers
  executor.py    — Per-op dispatch + compiled C dispatch, Backend protocol
  backends/
    numpy_backend.py — In-place numpy kernels (fallback)
    c_backend.py     — ctypes wrappers for C shared library
csrc/
  ops.c          — C kernels (CBLAS matmul, element-wise, reductions, softmax, attention)
  executor.c     — C dispatch loop (OpNode struct, switch-based dispatch)
  Makefile       — Accelerate on macOS, OpenBLAS on Linux
tests/
  conftest.py      — Shared fixtures (backends, model definitions, helpers)
  test_backends.py — Op correctness: every op × both backends × various shapes
  test_end_to_end.py — Full pipeline oracle tests (per-op + compiled vs PyTorch)
  test_passes.py   — Optimization pass invariants and correctness
  test_planner.py  — Arena no-overlap, reuse savings, reshape aliasing
  test_benchmark.py — Performance ablation (pytest -m benchmark -s)
```

## Getting started

**Prerequisites:** Python 3.13, PyTorch, numpy. macOS (Accelerate) or Linux (OpenBLAS).

```bash
# Build the C libraries
cd csrc && make && cd ..

# Run tests (excludes benchmarks by default)
python -m pytest tests/

# Run benchmarks
python -m pytest tests/test_benchmark.py -m benchmark -s
```

**Quick example:**

```python
import torch
from runtime.exporter import export_model
from runtime.passes import run_pipeline
from runtime.planner import plan
from runtime.executor import Executor
from runtime.backends.c_backend import CBackend
from runtime.backends.numpy_backend import NumpyBackend

# Define a model
model = torch.nn.Sequential(
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 512),
)
x = torch.randn(32, 512)

# Compile
graph = export_model(model, (x,))
run_pipeline(graph)           # optimize: absorb, fold, fuse, DCE
ep = plan(graph)              # lifetime analysis + arena layout

# Execute (one C call)
executor = Executor(backends=[CBackend(), NumpyBackend()])
compiled = executor.compile_plan(ep)
result = executor.execute_compiled(compiled, {graph.inputs[0]: x.numpy()})
```

## Optimization passes

1. **BLAS flag absorption** — Folds adjacent transposes into `sgemm` `CblasTrans` flag. Folds scalar multiply/divide into matmul `alpha`. Keeps `nn.Linear` weights in their original `[out, in]` layout for better BLAS packing performance.

2. **Constant folding** — Evaluates subgraphs with all-constant inputs at compile time using numpy evaluators. Results become new constants.

3. **Pattern fusion** — Registry of `FusionPattern` objects matched greedily by priority:
   - Priority 0: `ATTENTION` (MATMUL→SOFTMAX→MATMUL), `BIAS_RELU` (ADD+RELU)
   - Priority 1: `MATMUL_ADD` (sgemm with beta=1.0 for fused bias)

4. **Dead code elimination** — Removes nodes with no consumers. Cleans up dead constants.

## Attention

Three recognition paths, all converging to the same fused C kernel:

| Source | How it's recognized |
|--------|-------------------|
| `F.scaled_dot_product_attention` | Exporter maps directly to `ATTENTION` |
| `F.softmax(Q @ K^T / sqrt(d))` | Absorption folds scale into alpha, fusion matches MATMUL→SOFTMAX→MATMUL |
| Manual softmax decomposition | Not yet implemented (DAG pattern) |

Two kernel implementations:
- **Standard:** Materializes full S×S attention matrix. Uses sgemm for Q@K^T and P@V.
- **Flash:** Tiled online-softmax (32×32 blocks). Never materializes full S×S. Better for long sequences.

Both use GCD `dispatch_apply` for parallel execution across batch×head slices on macOS.

## Design decisions

See [DESIGN.md](DESIGN.md) for detailed rationale on:
- Weight transpose handling (CblasTrans vs pre-transpose)
- Pattern-matched fusion over interpreter fusion (bespoke kernels won)
- MATMUL+ADD via sgemm beta trick
- Compiled executor extras encoding (float-to-int bit-cast)
