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
- **Extensibility through registries, not if/else chains.** Prefer registries and dispatch over hardcoding and if-else chains wherever possible — see the optimization passes, fusion chains, ATen mappings, and validation framework as examples.
- **Named tensors as connective tissue.** Nodes reference data by tensor name strings. TensorInfo objects hold shape/dtype/buffer metadata. Nodes are purely about computation, tensors are purely about data.
- **Inputs and constants are tensors, not nodes.** Every node in the graph is a real compute operation. Graph inputs and weights are entries in the tensor registry with no producer node. This is how ONNX does it.
- **Memory-conscious.** Avoid unnecessary duplication of weight data. Prefer approaches that minimize peak memory. The motivation here is on-device settings.
- **Validate at pipeline boundaries.** Catch errors early with clear diagnostics. Validation runs at each pipeline stage transition (post-export, post-optimize, post-resolve, post-plan, pre-execute) via a registry of tagged validators.

## Project Structure

```
runtime/
  ir.py              — Graph IR: OpType, TensorInfo, Node, Graph (summary, dump, save/load)
  ops.py             — Op registry: evaluators, scratch calculators, alias/inplace flags, extras packers, shape inference
  planner.py         — Memory planner: lifetime analysis, arena offsets, scratch buffers, MemoryPlan, ExecutionPlan
  session.py         — Session API: optimize → validate → plan → validate → compile → run
  exporter/
    exporter.py      — torch.export → our Graph (three-phase pipeline)
    handlers.py      — ATen op handler registry + handler utilities
  passes/
    passes.py        — Pass infrastructure + core passes (absorption, folding, DCE)
    fusion.py        — Pattern-based fusion + DAG fusion (GELU recognition)
  executor/
    common.py        — Executor ABC, COpNode struct, profiling types (OpTiming, RunProfile)
    compiled.py      — CompiledExecutor: single ctypes call for whole graph
    interpreted.py   — InterpretedExecutor: Python loop with backend chain
  backends/
    numpy_backend.py — In-place numpy kernels (fallback)
    c_backend.py     — ctypes bindings to C shared library
  validation/
    core.py          — Validator types, Phase, Severity, registry, runner
    __init__.py      — Re-exports from core, triggers submodule registration
    graph.py         — Graph-phase validators (POST_EXPORT through POST_RESOLVE_OPTIMIZE)
    plan.py          — Memory plan validators (POST_PLAN)
    execution.py     — Execution plan validators (PRE_EXECUTE)
csrc/
  ops.c              — C operator kernels (matmul via CBLAS, add, relu, attention, etc.)
  executor.c         — C dispatch loop (function pointer table, calls into ops.c)
  runtime.h          — Shared C types (OpNode struct, op enum)
  Makefile           — Build system (Accelerate on macOS, OpenBLAS on Linux)
tests/
  conftest.py        — Shared fixtures (backends, models, helpers)
  test_backends.py   — Op correctness: every op × both backends × various shapes
  test_end_to_end.py — Full pipeline oracle tests (per-op + compiled vs PyTorch)
  test_passes.py     — Optimization pass invariants + correctness
  test_planner.py    — Arena no-overlap, reuse savings, reshape aliasing
  test_session.py    — Session API tests (dynamic shapes, rebind, validation)
  test_dynamic_shapes.py — Dynamic shape export, resolve, re-resolve
  test_ops.py        — OpDef registry coverage (evaluators, alias, extras, shape inference)
  test_benchmark.py  — Performance ablation (pytest -m benchmark -s)
docs/
  DESIGN.md          — Final-state design decisions and rationale
  DESIGN_LOG_FULL.md — Full iterative design log with tradeoffs and removed features
ablation/            — Standalone benchmark scripts (transformer ablation, memory comparison, ORT)
```

## Pipeline Flow

```
torch.nn.Module
  → torch.export.export()
  → exporter.export_model() → Graph (with weight data loaded)
  → passes.run_pipeline() → optimized Graph
      absorb_into_matmul             (BLAS flag absorption + scalar folding into alpha)
      constant_fold                  (evaluate static subgraphs)
      absorb_mask_into_attention     (constant causal masks → causal=True attr)
      fuse_dags                      (GELU recognition)
      fuse                           (pattern-based fusion: ATTENTION, BIAS_RELU, MATMUL_ADD)
      eliminate_dead_code            (remove unused nodes/tensors)
  → [validation: POST_OPTIMIZE]
  → resolve_graph(bindings) → concrete Graph (if dynamic shapes)
  → [validation: POST_RESOLVE_OPTIMIZE]
  → planner.plan() → MemoryPlan (arena layout + scratch allocations + execution order)
  → [validation: POST_PLAN]
  → ExecutionPlan (MemoryPlan + executor type + backend)
  → [validation: PRE_EXECUTE]
  → Executor.compile(memory_plan) → ready for inference
  → Executor.run(inputs) → results (one C call for compiled path)
```

## Conventions

- Op handlers in the exporter: `(fx_node, Graph, node_map, symbol_map) -> None`
- Passes: `(Graph) -> bool` (returns whether graph was modified)
- Validators: `(target) -> list[ValidationResult]`, registered via `@register_validator(name, phase)`
- Fusion patterns registered via `register_fusion(FusionPattern(...))`
- Scratch calculators defined on `OpDef.scratch` in `OP_REGISTRY`
- Scratch buffers are appended to kernel inputs by the executor (last input = scratch)
- Tensor names from torch.export are preserved for debuggability
- `venv/` contains the project's Python environment (Python 3.13, PyTorch, numpy)

## Documentation

- **`CLAUDE.md`** (this file) — Project overview, structure, conventions. Keep up to date.
- **`docs/DESIGN.md`** — Final-state design decisions and rationale. Describes what we built and why, in its current form. Update when adding features or making design changes.
- **`docs/DESIGN_LOG_FULL.md`** — Full iterative design log. Contains the complete history of design discussions, tradeoff analyses, alternatives considered, bugs found, and features that were tried and removed. Append new entries when making significant design decisions. Never delete old entries — this is a historical record.
- When making a design decision or adding a significant feature: add a concise section to DESIGN.md describing the final state, and a detailed section to DESIGN_LOG_FULL.md covering the tradeoffs and process.
