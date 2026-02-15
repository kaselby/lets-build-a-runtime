# Refactor Audit: What's Left

Comprehensive gap analysis between `runtime_edited/` and a fully working runtime.
Generated Feb 15, organized by work area with effort estimates and dependency notes.

---

## 1. Package Infrastructure

**Status: DONE**

### ~~Missing `__init__.py` files~~

- [x] `runtime_edited/__init__.py`
- [x] `runtime_edited/passes/__init__.py`
- [x] `runtime_edited/executor/__init__.py`
- [x] `runtime_edited/backends/__init__.py`

---

## 2. Numpy Backend

**Status: DONE** — ported to `runtime_edited/backends/numpy_backend.py`.

---

## 3. C Backend (`backends/c_backend.py`)

**Status: DONE** — straight copy from `runtime/`, only change is library path (`csrc` → `csrc_edited`).

The original `c_backend.py` is the most complex backend file — it's a ctypes wrapper layer with significant Python-side logic for:
- Broadcasting (general N-D broadcast via stride computation)
- Batch matmul dispatch (ND×2D fast path, odometer for general broadcast)
- Shape decomposition (reductions → outer/axis_size/inner)
- Scalar vs tensor dispatch per op

### What needs to happen

The C backend wrapper code is largely **independent of the refactoring changes** — it's a bridge between Python shapes and C kernel signatures. The main question is how it fits into the new architecture:

- **Original:** `CBackend` class with `get_kernel(op) -> KernelFn`, loaded as part of `Executor(backends=[CBackend(), NumpyBackend()])`
- **New:** `InterpretedExecutor(backends=[...])` uses same Backend protocol

The `CBackend` class should port cleanly since the Backend protocol is the same. The wrapper functions (`_wrap_matmul`, `_wrap_add`, etc.) are unchanged.

**What to check:** The original c_backend loads `libruntime.dylib` for per-op dispatch. The executor loads `libexecutor.dylib` for compiled dispatch. Both are the same C kernels but different shared libraries. This separation should still work with the new split.

**Effort:** M — large file (~600 lines) but mostly a clean transfer. No architectural changes needed.

---

## 4. Fusion Patterns

**Status: DONE** — all 4 chain patterns registered in `passes/fusion.py`:

- [x] **bias_relu** — `ADD + RELU → FUSED_BIAS_RELU` (priority 0)
- [x] **matmul_add** — `MATMUL + ADD → MATMUL_ADD` (priority 1)
- [x] **attention** — `MATMUL + SOFTMAX + MATMUL → ATTENTION` (priority 0)
- [x] **causal_attention** — `MATMUL + ADD(mask) + SOFTMAX + MATMUL → ATTENTION` (priority 0)

---

## 5. GELU Recognition

**Status: DONE** — implemented via DAG fusion framework (option 3, not over-engineered)

Instead of porting the hand-written `recognize_gelu` pass, we extended the fusion framework with **DAG pattern matching** (`DAGNode`, `DAGFusionPattern`, `fuse_dags`). The GELU tanh approximation is registered as a declarative 8-node DAG pattern in `passes/fusion.py`. The framework handles:
- Anchored backward matching from root node
- Commutative input ordering (ADD, MUL)
- Scalar constraints with tolerance
- Same-tensor constraints (x0 appears 3 times)
- Consumer count verification (no external consumers on internal nodes)

The `fuse_dags` pass is in DEFAULT_PIPELINE before chain `fuse`. Adding future DAG patterns (SiLU, GeGLU, etc.) is ~10-15 lines each.

---

## 6. Missing Exporter Handlers

**Status: Several handler groups not yet ported**

### ~~6a. split/getitem (needed for GPT-2 QKV split)~~

**DONE** — redesigned to eliminate the `_pending_splits` module-level dict. Split is a no-op (like dropout). Getitem self-serves split metadata by inspecting the source FX node's args directly (`source_fx.target == torch.ops.aten.split.Tensor`), then emits a SLICE node with computed byte offset. No shared mutable state between handlers. See DESIGN_LOG_FULL.md "Single-Output Nodes vs Multi-Output" for the full design discussion.

### 6b. sdpa handler

**DONE** — ported to `exporter/handlers.py`.

### 6c. Constant-folded infrastructure ops

**DONE** — all handlers ported to `exporter/handlers.py`:

- [x] `_handle_arange`, `_handle_arange_start`, `_handle_new_ones` — constant materializers via shared `_materialize_constant` helper
- [x] `_handle_to_dtype` — alias if same dtype, CAST node otherwise
- [x] `_handle_expand`, `_handle_cumsum` — trivial `_emit` calls
- [x] `_handle_slice_tensor`, `_handle_diff`, `_handle_index` — direct ports using `_emit`
- [x] `_handle_ne_scalar` — was already ported

---

## 7. GPT-2 Fold-Only Evaluators

**Status: DONE** — all 10 fold-only ops registered in `OP_REGISTRY` with evaluators.

Ops at 100+ are infrastructure ops (mask generation, type conversion) that must be eliminated by constant folding before execution. Both executors enforce this boundary — fold-only ops reaching execution raise a clear `RuntimeError`. The C dispatch table enforces it naturally via `DISPATCH_TABLE_SIZE = 100`.

- [x] `CAST`, `EXPAND`, `CMP_NE`, `CMP_LE`, `CMP_EQ`, `CUMSUM`, `BITWISE_AND` — one-liner lambdas
- [x] `SLICE_TENSOR`, `DIFF`, `INDEX` — helper functions in ops.py

---

## 8. C Layer

**Status: Enum + dispatch table DONE, SLICE kernel and c_backend.py remaining**

### ~~8a. C-side SLICE kernel~~

**DONE** — `kernel_slice` in `csrc_edited/ops.c`, `dispatch_slice` updated in `executor.c`, ctypes wrapper + `_wrap_slice` in `c_backend.py`. Defensive guard (`extra[0] == 0`) for alias slices that leak through.

### ~~8b. Function pointer dispatch table~~

**DONE** — `csrc_edited/executor.c` refactored from 250-line `switch` to `dispatch_fn` table with C99 designated initializers. Each op is a self-contained `dispatch_xxx(OpNode*)` function. Three inline helpers (`total_elements`, `extra_float`, `leading_dims`) eliminate repeated patterns. Adding a new op: add enum value, write dispatch function, add one table line.

### ~~8c. OpType enum sync~~

**DONE** — Both `runtime_edited/ir.py` and `csrc_edited/executor.c` use range-based numbering (10–19 unary, 20–29 binary, 30–39 reductions, 40–49 BLAS, 50–59 shape, 60–69 normalization, 70–79 fused, 100+ fold-only). C uses a proper `enum OpType` instead of `#define`s. Sync test still TODO.

### 8d. ctypes declarations in c_backend.py

If `kernel_slice` is added to ops.c, c_backend.py needs a matching ctypes declaration and wrapper function. Same for any new kernels.

**Effort:** S — follows established pattern.

---

## 9. Pipeline Assembly

### 9a. DEFAULT_PIPELINE

**DONE** — pipeline is:
```python
DEFAULT_PIPELINE = [absorb_into_matmul, constant_fold, absorb_mask_into_attention, fuse_dags, fuse, eliminate_dead_code]
```

`fuse_dags` runs before chain `fuse`, handling GELU and future DAG patterns.

### 9b. Memory-aware ordering decision

Three versions exist in `order.py`: v1 (original), v2 (lazy re-score, recommended), v3 (event-driven). Need to:
- [ ] Pick v2 (as recommended in REFACTOR_NOTES)
- [ ] Wire it into planner.py as the ordering function
- [ ] Delete v1 and v3 (or move to an appendix file)

**Effort:** S — decision already made, just needs the wiring.

**Current state of order.py:** The file has **zero imports** — it will crash on import. It also references `_tensor_size()` which is only defined in planner.py. The v1 `_memory_aware_order` body is duplicated in both order.py and planner.py. Once we pick v2, wire it into planner.py (either inline or import) and delete order.py.

---

## 10. Test Compatibility

**Status: Major API incompatibilities with existing test suite**

### 10a. API shape mismatch

The tests import and use the **original** API:
```python
from runtime.executor import Executor
ex = Executor(backends=[CBackend(), NumpyBackend()])
ex.execute(plan, inputs)              # per-op dispatch
compiled = ex.compile_plan(plan)      # compile
ex.execute_compiled(compiled, inputs) # compiled dispatch
```

The refactored code has a **different** API:
```python
# Interpreted (per-op)
from runtime_edited.executor.interpreted import InterpretedExecutor
ex = InterpretedExecutor(backends=[...])
ex.compile(plan)
ex.run(inputs)

# Compiled
from runtime_edited.executor.compiled import CompiledExecutor
ex = CompiledExecutor()
ex.compile(plan)
ex.run(inputs)
```

Key differences:
- Single `Executor` class → split into `InterpretedExecutor` + `CompiledExecutor`
- `execute(plan, inputs)` → `compile(plan)` + `run(inputs)` (separate steps)
- `compile_plan(plan)` + `execute_compiled(compiled, inputs)` → `compile(plan)` + `run(inputs)`
- No `CompiledPlan` data class exposed to callers

**Options:**
1. **Write new tests** for runtime_edited API (clean but doubles test maintenance)
2. **Compatibility shim** — thin `Executor` class in runtime_edited that wraps both executors to match old API
3. **Update test imports** — make tests work with new API, conditioned on which runtime they target
4. **Replace runtime/ with runtime_edited/ and update tests** — the end goal anyway

**Recommendation:** Option 4 when ready. Until then, the `test_planner_edited.py` pattern (importing from `runtime_edited` directly) shows the approach for incremental testing.

### 10b. Import paths

Tests import from `runtime`, not `runtime_edited`. The final swap means either:
- Rename `runtime_edited/` → `runtime/` (and archive old `runtime/`)
- Or update all test imports

### 10c. Planner API differences

Tests import internal planner helpers:
```python
from runtime.planner import _compute_lifetimes, _find_reshape_aliases, SCRATCH_CALCULATORS, register_scratch
```

The refactored planner:
- `_find_reshape_aliases` → replaced by `_resolve_alias` (different API: walks producer chain vs precomputed dict)
- `SCRATCH_CALCULATORS` → removed, now on `OP_REGISTRY[op].scratch`
- `register_scratch` → no longer exists (add to OP_REGISTRY instead)
- `_compute_lifetimes` → similar but returns `(lifetimes, memory_root)` instead of just lifetimes

**Impact:** `test_planner.py` tests will need updating. `test_planner_edited.py` already tests the new API.

### 10d. Passes API differences

Tests import:
```python
from runtime.passes import fuse, FusionPattern, FUSION_PATTERNS, recognize_gelu
```

In refactored code:
- `fuse` is in `passes/fusion.py`, not `passes/passes.py`
- `FUSION_PATTERNS` is in `passes/fusion.py`
- `recognize_gelu` doesn't exist yet

Need `__init__.py` re-exports to match expected import paths.

**Effort:** M for the full test migration. Much of it is mechanical import path changes, but the planner and executor API changes require substantive test rewrites.

---

## 11. Summary: Dependency Graph and Suggested Order

### Critical path to GPT-2 end-to-end:

```
Package infra (__init__.py files)
  ↓
GPT-2 exporter handlers (split/getitem, sdpa, infrastructure ops)
  + GPT-2 evaluators (fold-only ops in OP_REGISTRY)
  ↓
Fusion patterns (bias_relu, matmul_add, attention, causal_attention)
  + GELU recognition
  ↓
Backends (numpy_backend, c_backend)
  + C-side SLICE kernel
  ↓
Pipeline assembly (DEFAULT_PIPELINE, ordering decision)
  ↓
Test migration (API shim or test rewrites)
  ↓
Swap runtime_edited/ → runtime/
```

### By effort:

| Item | Effort | Transfer? | Blocks | Status |
|------|--------|-----------|--------|--------|
| Package `__init__.py` | S | New | Everything | **DONE** |
| Fold-only evaluators | S | Direct | GPT-2 export | **DONE** |
| GELU recognition | M | New (DAG framework) | GPT-2 pipeline | **DONE** |
| Fusion patterns (×4) | M | Direct + review | End-to-end correctness | **DONE** |
| Exporter: sdpa | S | Direct | SDPA transformer tests | **DONE** |
| Exporter: split/getitem | M | Redesign | GPT-2 tests | **DONE** |
| Exporter: infrastructure | M | Direct | GPT-2 export | **DONE** |
| Numpy backend | M | Direct | InterpretedExecutor | **DONE** |
| C backend (c_backend.py) | M | Direct | Per-op C dispatch | **DONE** |
| C SLICE kernel | S | New | Compiled GPT-2 | **DONE** |
| C function pointer table | M | Refactor | None (polish) | **DONE** |
| OpType enum sync | S | New | None (robustness) | **DONE** |
| Ordering decision | S | Pick v2 | None | TODO |
| Pipeline assembly | S | Config | End-to-end | **DONE** |
| Test migration | M-L | Rewrite | Validation | TODO |

### Minimum viable path (MLP + transformer only, no GPT-2):

1. Package infrastructure
2. Fusion patterns (bias_relu, matmul_add, attention)
3. One backend (numpy OR c_backend)
4. Pipeline assembly
5. Basic test adaptation

### Full GPT-2 path (adds to above):

6. split/getitem handler
7. Infrastructure op handlers + evaluators
8. GELU recognition
9. Causal attention fusion
10. C SLICE kernel
11. sdpa handler

---

## 12. Open Questions

1. ~~**Numpy backend: adapter vs port?**~~ Resolved — ported as separate in-place kernels.

2. ~~**split/getitem redesign:**~~ Resolved — eliminated `_pending_splits` entirely. Getitem reads split metadata from the FX graph directly.

3. ~~**C function pointer table:**~~ Resolved — done. Range-based enum + dispatch table in `csrc_edited/executor.c`.

4. **Test strategy:** Write new tests for runtime_edited API first, or go straight to swapping and updating the existing 200+ tests?

5. ~~**GELU:**~~ Resolved — built DAG fusion framework. Handles GELU and extensible to SiLU, GeGLU, etc.
