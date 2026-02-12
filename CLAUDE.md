# Inference Runtime

## Project Goal

Build an interpreted inference runtime that can compile and run basic MLPs and simple transformers with measurable speedup over naive eager PyTorch. The speedup comes from eliminating overhead between ops (fusion, memory reuse, reduced dispatch overhead), not from writing faster kernels.

This is an interactive learning project and interview prep artifact for the ONNX Runtime team.

## Architecture

**Interpreter-style runtime.** One graph representation throughout — no multi-level IR lowering. Optimization passes mutate the graph in-place, then we plan memory and execute. Closest in spirit to ONNX Runtime's CPU execution provider.

**Python + C hybrid.** Python for the graph layer (IR, passes, memory planner, executor orchestration). C for operator kernels called via ctypes.

**Memory model.** Weights are stored separately from the execution arena. The arena is a numpy byte buffer for intermediate activations and kernel scratch workspace — sized by the memory planner, reused across ops via lifetime analysis. Scratch buffers for fused kernels (e.g., the S×S attention matrix) are planner-allocated arena regions with single-step lifetimes, passed to kernels as extra inputs. Weights are loaded once and persist across inference calls. C operators receive raw float pointers via numpy's ctypes interface.

## Design Principles

- **Build it like a real runtime wherever we can**, but don't overengineer. This doesn't need to be production code, but the architecture should reflect how real runtimes work.
- **Primitive ops, optimized by passes.** The exporter emits fine-grained ops (matmul, add, relu, transpose, permute). Optimization passes transform the graph: BLAS flag absorption folds transposes into matmul flags, constant folding evaluates static subgraphs, pattern-based fusion handles specific patterns (MATMUL+ADD+RELU → FUSED_BIAS_RELU, MATMUL+ADD → MATMUL_ADD, MATMUL+SOFTMAX+MATMUL → ATTENTION), element-wise fusion chains arbitrary element-wise ops into interpreter-dispatched fused kernels, and DCE cleans up dead nodes.
- **Planner owns all memory.** No kernel allocates internally. The arena holds both intermediate activations and kernel scratch workspace, unified through the same lifetime analysis and first-fit allocation.
- **Extensibility through registries, not if/else chains.** Prefer registries and dispatch over hardcoing and if-else chains wherever possible - see the optimization passes, fusion chains and ATen mappings as examples.
- **Named tensors as connective tissue.** Nodes reference data by tensor name strings. TensorInfo objects hold shape/dtype/buffer metadata. Nodes are purely about computation, tensors are purely about data.
- **Inputs and constants are tensors, not nodes.** Every node in the graph is a real compute operation. Graph inputs and weights are entries in the tensor registry with no producer node. This is how ONNX does it.
- **Memory-conscious.** Avoid unnecessary duplication of weight data. Prefer approaches that minimize peak memory. The motivation here is on-device settings.

## Project Structure

```
runtime/
  ir.py          — Graph IR: OpType, TensorInfo, Node, Graph
  exporter.py    — torch.export → our Graph (handler registry)
  passes.py      — Optimization passes: BLAS flag absorption, constant folding, fusion, DCE
  planner.py     — Memory planner: lifetime analysis, arena offset assignment, scratch buffers, ExecutionPlan
  executor.py    — Executor (per-op dispatch + compiled C dispatch), Backend protocol
  backends/
    numpy_backend.py — In-place numpy kernels (fallback)
    c_backend.py     — ctypes bindings to C shared library
csrc/
  ops.c          — C operator kernels (matmul via CBLAS, add, relu, attention, etc.)
  executor.c     — C dispatch loop (OpNode struct, switch-based dispatch, calls into ops.c)
  Makefile       — Build system (Accelerate on macOS, OpenBLAS on Linux)
tests/
  conftest.py      — Shared fixtures (backends, models, helpers)
  test_backends.py — Op correctness: every op × both backends × various shapes
  test_end_to_end.py — Full pipeline oracle tests (per-op + compiled vs PyTorch)
  test_passes.py   — Optimization pass invariants + correctness
  test_planner.py  — Arena no-overlap, reuse savings, reshape aliasing
  test_benchmark.py — Performance ablation (pytest -m benchmark -s)
```

## Pipeline Flow

```
torch.nn.Module
  → torch.export.export()
  → exporter.export_model() → Graph (with weight data loaded)
  → passes.run_pipeline() → optimized Graph
      absorb_into_matmul             (BLAS flag absorption + scalar folding into alpha)
      constant_fold                  (evaluate static subgraphs)
      fuse                           (pattern-based fusion: MATMUL+SOFTMAX+MATMUL → ATTENTION,
                                      ADD+RELU → FUSED_BIAS_RELU, MATMUL+ADD → MATMUL_ADD)
      eliminate_dead_code             (remove unused nodes/tensors)
  → planner.plan() → ExecutionPlan (arena layout + scratch allocations + execution order)
  → executor.compile_plan() → CompiledPlan (C struct array, scratch pointers bound)
  → executor.execute_compiled() → inference results (one C call)
```

## Conventions

- Op handlers in the exporter: `(fx_node, Graph, node_map) -> None`
- Passes: `(Graph) -> bool` (returns whether graph was modified)
- Fusion patterns registered via `register_fusion(FusionPattern(...))`
- Scratch calculators registered via `register_scratch(OpType, calc)` — calc takes `(input_shapes, output_shape) -> bytes`
- Scratch buffers are appended to kernel inputs by the executor (last input = scratch)
- Tensor names from torch.export are preserved for debuggability
- `venv/` contains the project's Python environment (Python 3.13, PyTorch, numpy)
- See `DESIGN.md` for detailed technical decisions and rationale
