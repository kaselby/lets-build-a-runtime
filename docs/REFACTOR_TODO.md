# Refactor TODO

Code review and cleanup pass. Goal: understand every component, fix design issues, polish for interview presentation.

Detailed change log in `runtime_edited/REFACTOR_NOTES.md`.

## Reviewed and ported

- [x] **IR** (`ir.py`) — no changes needed
- [x] **Ops** (`ops.py`) — NEW file, centralizes per-op metadata (evaluator, scratch, alias, inplace, extras)
- [x] **Exporter core** (`exporter/exporter.py`) — pipeline logic, three-phase graph walk
- [x] **Exporter handlers** (`exporter/handlers.py`) — new utilities (_output_meta, _emit, _get_axis), handler factories, all non-deferred handlers
- [x] **Passes: absorption** — decomposed into focused helpers, added _pretranspose_constant_b
- [x] **Passes: constant folding** — evaluators now go through OP_REGISTRY
- [x] **Passes: mask absorption** — unified _is_causal_mask (was three separate functions)
- [x] **Passes: DCE** — ported as-is
- [x] **Passes: fusion framework** — FusionPattern, registry, match/apply engine (no patterns yet)
- [x] **Planner** — unified alias+inplace sharing, memory-aware ordering, scratch from OP_REGISTRY
- [x] **Memory-aware ordering** — three variants compared (v1 simple, v2 lazy re-score, v3 event-driven), leaning v2
- [x] **Executor: common** (`executor/common.py`) — Executor ABC, COpNode struct, C lib loading, arena management
- [x] **Executor: compiled** (`executor/compiled.py`) — struct building, one-call execution, extras via OP_REGISTRY
- [x] **Executor: interpreted** (`executor/interpreted.py`) — backend chain dispatch for ablations
- [x] **SLICE redesign** — per-node alias (callable), contiguous=alias, non-contiguous=C kernel, segmented execution removed
- [x] **Extras packing** — `_fill_extras` if/elif chain replaced by `extras` field on OpDef (registry pattern)

## Still to review

- [ ] **C backend** (`backends/c_backend.py`) — ctypes kernel wrappers, stride computation, ND matmul flattening
- [ ] **C kernels** (`csrc/ops.c`) — CBLAS matmul, SIMD softmax/layernorm/gelu, attention, flash attention
- [ ] **C executor** (`csrc/executor.c`) — OpNode struct, dispatch switch, execute loop

## Deferred (come back to after full review)

### Fusion patterns
- [ ] bias_relu (ADD+RELU, priority 0)
- [ ] matmul_add (MATMUL+ADD, priority 1)
- [ ] attention (MATMUL+SOFTMAX+MATMUL, priority 0)
- [ ] causal_attention (MATMUL+ADD(mask)+SOFTMAX+MATMUL, priority 0)
- [ ] GELU recognition (8-node DAG matcher, outside fusion framework)

### Exporter handlers
- [ ] split/getitem — rethink _pending_splits (module-level mutable state is a bug risk)
- [ ] sdpa — review alongside attention fusion pass
- [ ] Constant-folded infrastructure ops (arange, new_ones, expand, slice_tensor, diff, cumsum, to_dtype, index)

### Evaluators
- [ ] GPT-2 fold-only evaluators (CAST, EXPAND, SLICE_TENSOR, DIFF, CMP_NE, CMP_LE, CMP_EQ, CUMSUM, BITWISE_AND, INDEX)

### Pipeline assembly
- [ ] DEFAULT_PIPELINE population (needs all passes finalized)
- [ ] Pick memory-aware ordering version (v1 vs v2) and delete the others

### C-side work
- [ ] `kernel_slice` in ops.c (strided copy for non-contiguous SLICE)
- [ ] `case OP_SLICE` in executor.c dispatch switch

### Validation / robustness
- [ ] Post-optimization check for ops with no kernel and no evaluator (catch fold-only ops that didn't fold)
- [ ] Consider RESHAPE→RESHAPE chain collapsing pass

### Benchmarking
- [ ] Memory allocation ablation (naive vs first-fit, FIFO vs memory-aware, with/without in-place)
- [ ] Compare arena size against PyTorch and ORT on MLP, transformer, GPT-2
