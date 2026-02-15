# Refactor Notes

Ongoing notes on changes made during the code review and cleanup pass.

## Exporter

### Structural changes
- Split `exporter.py` into `exporter/exporter.py` (core pipeline) and `exporter/handlers.py` (op handlers + registry)

### New utilities
- `_output_meta(fx_node)` — extracts shape/dtype from tracing metadata (was duplicated in every handler)
- `_emit(fx_node, graph, node_map, op, inputs, attrs)` — registers tensor, adds node, updates node_map (was duplicated in every handler)
- `_get_axis(fx_node, arg_index, default)` — extracts and normalizes axis args, handles negative dims (via `% ndim`), single-element lists, kwargs fallback

### Handler cleanup
- Consolidated no-op handlers (contiguous, dropout, alias) into single `_handle_noop`
- Replaced hand-written exp/tanh handlers with `_make_simple_handler` (they were identical to relu)
- Merged `_handle_reshape` and `_handle_reshape_from_meta` — always read target shape from output metadata (guaranteed fully resolved, no unresolved `-1` dims)
- Unified negative dim normalization to `% ndim` everywhere

### Design changes
- **Linear**: no longer sets `transpose_b` directly. Emits TRANSPOSE + MATMUL + ADD as primitives. BLAS absorption pass handles the rest.
- **Addmm (Conv1D)**: no longer pre-transposes weight buffer at export time. Emits MATMUL + ADD. The absorption pass needs new logic to pre-transpose constant weights and set `transpose_b=True` for performance.

### Deferred handlers (not yet ported)
- **split/getitem**: The two-phase pattern with module-level `_pending_splits` dict needs rethinking — shared mutable state across calls is a bug risk
- **sdpa**: Direct ATTENTION mapping with optional mask and causal flag — straightforward but wanted to review alongside the attention fusion pass
- **Constant-folded infrastructure ops**: arange, new_ones, expand, slice_tensor, diff, cumsum, to_dtype, index — only needed for GPT-2 causal mask subgraph, all folded away before execution

## Passes

### Structural changes
- Split `passes.py` into two files:
  - `passes/passes.py` — pipeline infrastructure + absorption + constant folding + mask absorption + DCE
  - `passes/fusion.py` — FusionPattern framework + matching/rewriting engine + registered patterns
- `passes/folding.py` deleted — constant folding moved into passes.py (was 137 lines of evaluator registry + pass, now 30 lines using OP_REGISTRY)

### MATMUL absorption cleanup
- Decomposed monolithic `absorb_into_matmul` into focused helpers:
  - `_absorb_transpose_b` — absorbs TRANSPOSE on B input into `transpose_b=True`
  - `_pretranspose_constant_b` — **new**: pre-transposes constant B weights (e.g. Conv1D) and sets `transpose_b=True` (replaces exporter-side pre-transpose in `_handle_addmm`)
  - `_fold_scalar_into_alpha` — shared math for folding scalar MUL/DIV into alpha (was duplicated between backward and forward cases)
  - `_absorb_input_scalars` — backward scalar absorption on inputs
  - `_absorb_output_scalar` — forward scalar absorption on output
- Main function is now a tight loop calling four helpers

### Mask absorption cleanup
- Unified causal mask detection into single `_is_causal_mask` function (was three separate functions: `_is_bool_causal_mask`, `_is_float_causal_mask` in mask absorption, and `_is_causal_mask` in causal attention fusion)
- Handles both bool (lower-triangular True) and float (upper-triangular -inf) formats

### Constant folding
- Evaluator lookups now go through `OP_REGISTRY` instead of a local dict
- GPT-2 fold-only op evaluators (CAST, EXPAND, SLICE_TENSOR, etc.) not yet ported — deferred with their exporter handlers

### Fusion
- Framework ported (FusionPattern, registry, fuse/match/apply engine)
- No patterns registered yet — deferring individual pattern review

### DCE
- Ported as-is, no changes needed

### Design discussions noted
- Fold-only ops (CAST, EXPAND, etc.) aren't inherently constant — they just happen to have constant inputs in GPT-2. If a future model uses them with runtime inputs, execution would fail. Consider adding a post-optimization validation check for unimplemented ops.
- GELU recognition is a hand-written 8-node DAG matcher outside the fusion framework. Works, but can't use FusionPattern because the subgraph isn't a linear chain (x fans out to pow and mul paths that reconverge). Worth being prepared to explain in interview.

## Op definitions (NEW: ops.py)

### Motivation
Per-op metadata was scattered across four separate registries: evaluators in folding.py, scratch calculators in planner.py, alias detection hardcoded in planner, and C kernel bindings in c_backend.py. Adding a new op meant touching all of them and hoping you didn't forget one.

### New `OpDef` dataclass
Centralizes all Python-side per-op knowledge in one place:
- `evaluator` — numpy implementation (used by constant folding + fallback execution)
- `scratch` — scratch buffer size calculator (used by planner)
- `alias` — `bool | Callable[[Node], bool]`, whether output shares input memory. `is_alias(node)` method handles both. Used by planner and executor to skip alias ops.
- `inplace` — bool, whether the kernel can safely write output into first input's buffer (all elementwise ops)
- `extras` — packs op-specific params into `COpNode.extra[]` for compiled C executor. Replaces the monolithic `_fill_extras` if/elif chain (~90 lines → 4-line registry dispatch).

### What it replaced
- `NUMPY_EVALUATORS` dict in folding.py → `op_def.evaluator`
- `SCRATCH_CALCULATORS` dict in planner.py → `op_def.scratch`
- Hardcoded `if node.op == OpType.RESHAPE` checks in planner → `op_def.is_alias(node)`
- `_find_reshape_aliases()` function in planner → `_resolve_alias()` using `op_def.is_alias()`
- `_fill_extras()` if/elif chain in executor → `op_def.extras(node, graph)` per-op packers
- `_float_bits()` helper for float→int bit-casting shared across packers

### Alias handling redesign
- **Old**: planner pre-computed an alias dict mapping tensor names to (root, byte_offset) tuples, threaded it through lifetime computation. Tracked byte offsets everywhere even though only the executor needs them.
- **New**: `alias` field on OpDef supports both `bool` and `Callable[[Node], bool]`. `OpDef.is_alias(node)` method handles both. `_resolve_alias(name, graph)` walks the producer chain using `is_alias()`. No byte offsets in the planner — the executor reads `byte_offset` from node attrs directly.
- Alias chains (RESHAPE→RESHAPE) should be collapsed by passes before planning; `_resolve_alias` handles them as a backstop but they shouldn't occur in practice
- **SLICE alias is now per-node.** `_slice_alias(node)` returns True for contiguous slices (dim=0) and False for non-contiguous (dim>0). Contiguous slices share input memory (zero-copy). Non-contiguous slices are regular compute nodes dispatched to a C kernel (strided copy).
- **TODO**: Consider adding a pass to collapse RESHAPE→RESHAPE chains. RESHAPE→SLICE chains are trickier and probably not worth solving unless they come up in practice.

## Planner

### Structural changes
- Single file `planner.py` (same as original, no split needed)
- Scratch calculators and alias detection now read from `OP_REGISTRY` — no local registries

### Unified alias + in-place memory sharing
- **Old**: Two separate passes — `_compute_lifetimes` handled aliases via a precomputed alias dict, then `_apply_inplace` retroactively merged lifetimes. Chained in-place (e.g., EXP → RELU) didn't work because the second op couldn't find the merged lifetime.
- **New**: Single pass in `_compute_lifetimes` handles both alias and in-place sharing through the same mechanism. For each node in execution order:
  1. Alias op → unconditionally share input's memory (`memory_root[output] = get_root(input)`)
  2. In-place eligible (dying input, same size) → conditionally share (`memory_root[output] = get_root(input)`)
  3. Otherwise → new lifetime
- Returns `(lifetimes, memory_root)` where `memory_root` maps shared tensors to their arena-owning root
- `get_root()` follows both alias and in-place chains, so chained in-place works naturally
- Consumer counting uses `_resolve_alias()` (graph-level alias-only resolution) — stable regardless of in-place decisions, avoids circular dependency
- `_resolve_root` renamed to `_resolve_alias` to clarify it only follows graph-level aliases
- `_apply_inplace` removed entirely

### Memory-aware topological ordering (moved to separate file, decision pending)

Ordering functions moved to their own file for comparison. Three implementations exist:

#### v1: Original (compute-once-at-push)
- Kahn's algorithm with max-heap, priority = `freed_bytes` only
- `_freed_bytes` computed at push time, never updated
- **Staleness problem**: scores can only be *under*-estimated, never over-estimated. When `remaining_consumers[t]` drops from 2→1 after another node is scheduled, a node that would now free `t` is stuck in the heap with a stale lower score. The node never gets popped to discover its improved priority.
- Simple (~60 lines), but misses optimization opportunities in graphs with multi-consumer tensors (residual connections, attention)

#### v2: Lazy re-score (recommended)
- Kahn's + max-heap, priority = `freed_bytes - output_alloc_bytes + inplace_bonus`
- On pop, recomputes score with current `remaining_consumers`. If changed, pushes back and pops next candidate
- **Alias-aware**: resolves to alias roots for consumer counting, caches resolutions
- **Net delta heuristic**: accounts for both freeing inputs and allocating output. A node freeing 1MB but allocating 2MB is a net memory increase.
- **Inplace bonus**: small forward-looking bonus (output_size/8) for nodes whose output will be consumed in-place by a downstream op. Nudges the scheduler to set up in-place chains.
- Lazy re-score handles under-estimation: stale nodes get popped, re-scored higher, pushed back to correct position. Extra pop-push cycles are O(log N) each and rare in practice.
- ~115 lines, straightforward to understand and explain

#### v3: Event-driven (from Codex)
- Kahn's + max-heap with version-based stale entry skipping, priority = `freed_bytes - output_alloc_bytes`
- **Proactive updates**: when `remaining_consumers[root]` drops to 1, immediately finds the last consumer via `remaining_by_root_node` index and pushes an updated heap entry
- In-place-aware output estimation: approximates planner's in-place decision in the score model
- Handles duplicate inputs correctly via per-node edge counts (`node_root_uses`)
- **Edge case gap**: event trigger fires at `remaining_consumers == 1`, but `_freed_bytes` checks `remaining_consumers == use_count`. For multi-edge consumers (e.g., `add(x, x)` with `use_count == 2`), the trigger at `== 1` misses the case where remaining drops from 4→2 and a single node now holds all edges. Rare in practice.
- ~140 lines, significant bookkeeping (version counters, ready_ids, remaining_by_root_node)

#### Comparison summary
| | v1 | v2 (lazy) | v3 (event-driven) |
|---|---|---|---|
| Staleness handling | None | Lazy re-score on pop | Proactive push on refcount event |
| Score metric | freed only | net delta + inplace bonus | net delta + inplace-aware output |
| Alias-aware | No | Yes (cached) | Yes |
| Duplicate inputs | Not handled | Handled (seen_roots set) | Handled (edge counts) |
| Complexity | O(N log N) | O(N log N) amortized | O((N+E) log N) |
| Code complexity | ~60 lines | ~115 lines | ~140 lines |
| Edge cases | Staleness | None known | Multi-edge trigger gap |

**Leaning toward v2** — best balance of correctness, simplicity, and heuristic quality. The lazy re-score is "good enough" for realistic graphs, the inplace bonus is a genuinely useful addition, and it's much easier to explain in an interview than the event-driven machinery.

### Interview talking points
- Can discuss tradeoffs of first-fit vs best-fit offset assignment
- Can discuss NP-hardness of optimal topological ordering for minimum memory
- Memory-aware ordering and in-place reuse are the two main heuristics real runtimes use
- The planner runs once at compile time, so O(n²) algorithms are fine for realistic graph sizes
- Can walk through the staleness analysis: why freed_bytes can only be under-estimated, why lazy re-eval works, why event-driven is theoretically better but practically overkill
- Net delta vs freed-only: peak memory is about live set size, need to account for output allocation not just input freeing

## Executor

### Structural changes
- Split monolithic `executor.py` into three files:
  - `executor/common.py` — `Executor` ABC, `COpNode` struct, C library loading, arena management, buffer binding
  - `executor/compiled.py` — `CompiledExecutor`: builds COpNode struct array, one C call per inference
  - `executor/interpreted.py` — `InterpretedExecutor`: Python loop with backend chain dispatch (for ablations)

### New `Executor` ABC
- `compile(plan)` — one-time preparation (build struct array or stash plan)
- `run(inputs)` — per-call inference (patch pointers + C call, or Python dispatch loop)
- Shared: `_get_arena()`, `_bind_inputs()`, `_bind_intermediates()`
- Both executors subclass this — same interface, easy to swap in tests/benchmarks

### Compiled executor cleanup
- `_fill_extras` replaced by `OP_REGISTRY` dispatch (4 lines vs ~90 lines of if/elif)
- SLICE handling simplified: contiguous slices are aliases (skipped), non-contiguous slices dispatch to C kernel. Eliminates the segmented execution pattern (C segment → Python SLICE → C segment) entirely.
- `_build_node` populates one COpNode struct — input pointers, scratch, output, shape, extras
- `run()` is minimal: patch input pointers, one C call, copy outputs

### Interpreted executor
- Backend protocol + kernel resolution preserved for ablation use
- Same compile/run interface as compiled executor
- Alias ops skipped via `OP_REGISTRY` (no hardcoded op checks)

### SLICE redesign
- **Old**: contiguous and non-contiguous SLICEs both handled as special cases in the executor. Non-contiguous SLICEs required segmented C execution (pause C, run Python SLICE, resume C) with complex pointer patching.
- **New**: `alias` on OpDef is `Callable[[Node], bool]` for SLICE — `_slice_alias(node)` returns True for dim=0, False otherwise. Contiguous slices are zero-copy aliases (like RESHAPE). Non-contiguous slices are regular C-dispatched compute nodes with a `_slice_extras` packer providing `[outer, orig_dim_size, start, slice_len, inner]`.
- Net result: ~60 lines of complex segmented execution code replaced by ~30 lines of straightforward extras packing + a C kernel (not yet implemented).

## C layer

### Not yet reviewed
- `backends/c_backend.py` — ctypes kernel wrappers
- `csrc/ops.c` — C kernel implementations
- `csrc/executor.c` — C dispatch loop

### C-side work needed for SLICE
- `kernel_slice` in ops.c — strided copy using `[outer, orig_dim_size, start, slice_len, inner]` from extras
- `case OP_SLICE` in executor.c dispatch switch

## IR

- No changes (reviewed, happy with it)

## Overall progress

### Reviewed and ported
- `ir.py` — no changes needed
- `ops.py` — centralizes per-op metadata (evaluator, scratch, alias, inplace, extras)
- `exporter/exporter.py` — core pipeline (clean)
- `exporter/handlers.py` — all handlers except deferred ones
- `passes/passes.py` — pipeline infra + absorption + constant folding + mask absorption + DCE
- `passes/fusion.py` — framework only, no patterns registered yet
- `planner.py` — unified alias+inplace sharing, memory-aware ordering, scratch from OP_REGISTRY
- `executor/common.py` — Executor ABC, COpNode struct, arena management
- `executor/compiled.py` — struct building, one-call execution (SLICE segmentation removed)
- `executor/interpreted.py` — backend chain dispatch for ablations

### Not yet reviewed
- `backends/c_backend.py` — ctypes kernel wrappers
- `csrc/ops.c` — C kernel implementations
- `csrc/executor.c` — C dispatch loop

### Deferred items
- Fusion patterns (bias_relu, matmul_add, attention, causal_attention, GELU recognition)
- Exporter handlers: split/getitem, sdpa, constant-folded infrastructure ops
- GPT-2 fold-only evaluators (CAST, EXPAND, SLICE_TENSOR, DIFF, CMP_NE, CMP_LE, CMP_EQ, CUMSUM, BITWISE_AND, INDEX)
- DEFAULT_PIPELINE population (needs all passes finalized first)
- Pick memory-aware ordering version (v1 vs v2) and delete the others
- C kernel for non-contiguous SLICE (`kernel_slice` + executor.c dispatch case)
- Memory allocation benchmarking: measure arena size with various ablations (naive vs first-fit, FIFO order vs memory-aware, with/without in-place reuse) on MLP, transformer, and GPT-2 graphs. Compare peak memory against PyTorch (`torch.cuda.max_memory_allocated` / manual tracking) and ONNX Runtime (`SessionOptions` memory stats). Strong interview talking point if we can show concrete numbers.
