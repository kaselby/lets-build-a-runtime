# Design Log

Detailed technical decisions, tradeoff analyses, and iteration history. See `DESIGN.md` for the high-level summary.

## Op Representation

**TRANSPOSE vs PERMUTE are separate ops.** A 2D axis swap has different optimization opportunities (BLAS transpose flags, fusion into matmul) than a general N-dimensional permute (which requires data movement). The exporter inspects the permutation axes and emits the appropriate op.

**addmm decomposition.** `torch.export` emits `aten.addmm(bias, input, weight)` for linear layers. We decompose this into separate MATMUL + ADD nodes so the fusion pass has something to work with. Originally the exporter pre-transposed Conv1D weights and set `transpose_b` directly; now it emits primitives and the absorption pass handles the optimization (see Exporter Refactoring).

## Single-Output Nodes vs Multi-Output

Each node in our IR produces exactly one output tensor. This is a simplifying design choice, not a fundamental constraint.

### The motivating case: split

GPT-2's QKV projection does `split(projected, hidden_dim, dim=-1)` to split a `[B, S, 3*D]` tensor into three `[B, S, D]` chunks. `torch.export` decomposes this into `aten.split.Tensor` (which produces a tuple) followed by three `operator.getitem` calls (which unpack it). Two approaches for representing this in our IR:

**Option A: Multi-output split node.** One node, three output tensors. The graph is cleaner (1 node instead of 3+1), and the IR more directly represents the semantics. ONNX uses this approach — their Split op lists multiple outputs.

**Option B: Decompose into SLICE nodes.** The split handler is a no-op; each getitem emits an individual SLICE node with one output that aliases or copies a chunk of the source. More nodes in the graph, but each node fits the single-output model.

We chose Option B.

### What multi-output would cost

We traced every reference to `node.output` across the codebase (~50 sites) and categorized the changes:

**Mechanical changes (~20 sites).** Replace `node.output` with a loop over `node.outputs`. Same logic, just iterated. Covers `ir.py` (producer tracking, toposort, validation, remove_node), `passes/passes.py` (DCE dead check), `passes/fusion.py` (DAG consumer counts), `order.py` (ordering heuristics), and `planner.py` (birth-time tracking). All trivial — "node is dead" becomes "all outputs are unused."

**Zero changes (~25 sites).** If `Node` exposes a `.output` convenience property that returns `outputs[0]` (asserting length 1), every site that manipulates single-output ops — fusion validators, pass logic, the compiled executor — stays untouched.

**Real design work (1 thing).** The planner's alias model currently supports identity aliases: "this output IS the same buffer as that input" (used by RESHAPE and dim-0 SLICE). Multi-output split needs *offset* aliases: "output_i = source buffer + byte_offset_i." The planner would need to track both the alias source and a byte offset per tensor, changing `memory_root: dict[str, str]` to something like `memory_root: dict[str, tuple[str, int]]`.

### Why SLICE decomposition wins here

The irony is that SLICE decomposition already solves the offset alias problem without teaching the planner anything new. Each SLICE node independently carries its byte offset in `attrs["byte_offset"]`, and the planner's existing machinery handles it — alias SLICEs (dim=0, contiguous) share the source buffer directly, non-alias SLICEs (dim>0, non-contiguous) get their own arena allocation and a C kernel copies the strided data.

Multi-output split would move the same offset information from "N independent SLICE nodes with per-node attrs" into "the planner understands offset aliases as a first-class concept." Same information, different home. The SLICE approach keeps the planner simpler at the cost of a noisier graph.

### When multi-output would earn its keep

If we were supporting hundreds of ops (like ONNX Runtime), multi-output would be worth the planner complexity because many ops naturally produce multiple outputs (TopK, LSTM, BatchNormalization, etc.) and decomposing each into single-output nodes adds graph noise and pass complexity. For our focused runtime with ~25 ops where only split needs it, the decomposition is the simpler choice.

### Implementation: split/getitem as self-serving SLICE emission

The exporter handles the two-phase `aten.split.Tensor` → `operator.getitem` pattern:

1. `_handle_split` is a no-op — it maps the split's fx_name to its input tensor in node_map (like dropout).
2. `_handle_getitem` checks if its source is a split by inspecting the FX graph directly — reading chunk_size, dim, and input shape from the source node's args. It then computes the byte offset and emits a SLICE node.

No shared mutable state between handlers. The getitem handler is self-contained — it reads split metadata from the FX graph rather than relying on a stash dict. The original design used a module-level `_pending_splits` dictionary shared between the split and getitem handlers — this was replaced because shared mutable state across handler calls is a bug risk.

## Exporter Refactoring

The exporter was split from a single `exporter.py` into `exporter/exporter.py` (core pipeline) and `exporter/handlers.py` (op handlers + registry).

### Core pipeline

Three-phase graph walk: placeholders (graph inputs) → compute nodes → outputs. Three passes is deliberate — a single loop would need to handle forward references from outputs to not-yet-processed compute nodes. The clarity is worth the negligible performance cost.

### Handler utilities

Per-handler boilerplate was extracted into shared utilities:
- **`_output_meta(fx_node)`** — extracts shape/dtype from tracing metadata (was duplicated in every handler)
- **`_emit(fx_node, graph, node_map, op, inputs, attrs)`** — registers tensor, adds node, updates node_map
- **`_get_axis(fx_node, arg_index, default)`** — extracts and normalizes axis args, handles negative dims (via `% ndim`), single-element lists, kwargs fallback

Handler factories (`_make_simple_handler`, `_make_binary_handler`, `_make_reduction_handler`) cover common patterns. No-op handlers (contiguous, dropout, alias) were consolidated into a single `_handle_noop`.

### Design change: exporter emits primitives

**Old:** `_handle_linear` set `transpose_b=True` directly on the MATMUL node. `_handle_addmm` pre-transposed the weight buffer at export time. The exporter was embedding BLAS optimization knowledge.

**New:** `_handle_linear` emits `TRANSPOSE(weight) + MATMUL + ADD` as primitives. `_handle_addmm` emits `MATMUL + ADD` without pre-transposing. The absorption pass handles everything: `_absorb_transpose_b` absorbs the explicit TRANSPOSE into `transpose_b=True`, and `_pretranspose_constant_b` pre-transposes Conv1D constant weights and sets `transpose_b=True`.

This is a cleaner separation of concerns: the exporter does mechanical ATen→IR mapping, passes do optimization. Each can be tested independently.

## OpDef Registry

### Motivation

Per-op metadata was scattered across four separate locations: evaluators in a `NUMPY_EVALUATORS` dict in `folding.py`, scratch calculators in `SCRATCH_CALCULATORS` in `planner.py`, alias detection hardcoded in `_find_reshape_aliases` in the planner, and C kernel extras packing in a ~90-line `_fill_extras` if/elif chain in the executor. Adding a new op meant touching all four files and hoping you didn't forget one.

### The OpDef dataclass

Centralizes all Python-side per-op knowledge in one registry entry:

- **`evaluator`** — numpy implementation (constant folding + fallback execution)
- **`scratch`** — scratch buffer size calculator (planner)
- **`alias`** — `bool | Callable[[Node], bool]`, whether output shares input memory
- **`inplace`** — bool, whether the kernel can write output into first input's buffer
- **`extras`** — packs op-specific params into `COpNode.extra[]` for compiled C dispatch

### What it replaced

| Before | After |
|--------|-------|
| `NUMPY_EVALUATORS` dict in folding.py | `op_def.evaluator` |
| `SCRATCH_CALCULATORS` dict in planner.py | `op_def.scratch` |
| Hardcoded `if node.op == OpType.RESHAPE` in planner | `op_def.is_alias(node)` |
| `_find_reshape_aliases()` function | `_resolve_alias()` using `op_def.is_alias()` |
| `_fill_extras()` 90-line if/elif chain | per-op `op_def.extras(node, graph)` packers |
| `_float_bits()` helper for float→int bit-cast | shared utility in `ops.py` |

### Alias handling redesign

**Old approach:** The planner pre-computed an alias dict mapping tensor names to `(root, byte_offset)` tuples, threaded it through lifetime computation. Byte offsets were tracked everywhere even though only the executor needs them. Only RESHAPE was an alias — hardcoded check.

**New approach:** `alias` field on OpDef supports both `bool` and `Callable[[Node], bool]`. The `OpDef.is_alias(node)` method handles both. `_resolve_alias(name, graph)` walks the producer chain using `is_alias()`. No byte offsets in the planner — the executor reads `byte_offset` from node attrs directly.

**SLICE alias is per-node.** `_slice_alias(node)` returns True for contiguous slices (dim=0) and False for non-contiguous (dim>0). Contiguous slices share input memory (zero-copy). Non-contiguous slices are regular compute nodes dispatched to a C kernel.

## Unified Alias + In-Place Memory Sharing

### The bug in the original design

The original planner had two separate passes: `_compute_lifetimes` handled aliases via a precomputed alias dict, then `_apply_inplace` retroactively merged lifetimes for elementwise ops. **Chained in-place didn't work** — e.g., EXP → RELU where both could write into the same buffer. The second op couldn't find the merged lifetime from the first because `_apply_inplace` processed ops independently without propagating its decisions.

### The fix: single-pass unification

A single pass in `_compute_lifetimes` handles both alias and in-place sharing through the same `memory_root` mechanism. For each node in execution order:

1. **Alias op** → unconditionally share input's memory: `memory_root[output] = get_root(input)`
2. **In-place eligible** (dying input, same size) → conditionally share: `memory_root[output] = get_root(input)`
3. **Otherwise** → new lifetime

Returns `(lifetimes, memory_root)` where `memory_root` maps shared tensors to their arena-owning root. `get_root()` follows both alias and in-place chains, so chained in-place works naturally.

### Consumer counting subtlety

Consumer counting uses `_resolve_alias()` (graph-structural alias resolution only) rather than `get_root()` (which also follows in-place decisions). This is intentional: consumer counts must be stable regardless of in-place decisions to avoid a circular dependency where in-place choices affect consumer counts which affect in-place choices.

## Memory-Aware Topological Ordering

The execution order affects peak memory — scheduling a node that frees large tensors before one that allocates large outputs reduces the live set. Three implementations were compared:

### v1: Original (compute-once-at-push)

Kahn's algorithm with max-heap, priority = `freed_bytes` only. `_freed_bytes` computed at push time, never updated.

**Staleness problem:** scores can only be *under*-estimated, never over-estimated. When `remaining_consumers[t]` drops from 2→1 after another node is scheduled, a node that would now free `t` is stuck in the heap with a stale lower score. The node never gets re-evaluated. Simple (~60 lines), but misses optimization opportunities in graphs with multi-consumer tensors (residual connections, attention).

### v2: Lazy re-score (chosen)

Kahn's + max-heap, priority = `freed_bytes - output_alloc_bytes + inplace_bonus`.

- **On pop**, recomputes score with current `remaining_consumers`. If changed, pushes back and pops next candidate.
- **Alias-aware**: resolves to alias roots for consumer counting, caches resolutions.
- **Net delta heuristic**: accounts for both freeing inputs and allocating output. A node freeing 1MB but allocating 2MB is a net memory increase.
- **Inplace bonus**: small forward-looking bonus (output_size/8) for nodes whose output will be consumed in-place by a downstream op. Nudges the scheduler to set up in-place chains.

Lazy re-score handles the under-estimation problem: stale nodes get popped, re-scored higher, pushed back to correct position. Extra pop-push cycles are O(log N) each and rare in practice. ~115 lines.

### v3: Event-driven (from Codex)

Kahn's + max-heap with version-based stale entry skipping, priority = `freed_bytes - output_alloc_bytes`.

- **Proactive updates**: when `remaining_consumers[root]` drops to 1, immediately finds the last consumer and pushes an updated heap entry.
- In-place-aware output estimation.
- **Edge case gap**: event trigger fires at `remaining_consumers == 1`, but `_freed_bytes` checks `remaining_consumers == use_count`. For multi-edge consumers (e.g., `add(x, x)` with `use_count == 2`), the trigger can miss cases. Rare in practice.
- ~140 lines, significant bookkeeping (version counters, ready_ids, remaining_by_root_node).

### Comparison

| | v1 | v2 (lazy) | v3 (event-driven) |
|---|---|---|---|
| Staleness handling | None | Lazy re-score on pop | Proactive push on refcount event |
| Score metric | freed only | net delta + inplace bonus | net delta + inplace-aware output |
| Alias-aware | No | Yes (cached) | Yes |
| Duplicate inputs | Not handled | Handled (seen_roots set) | Handled (edge counts) |
| Complexity | O(N log N) | O(N log N) amortized | O((N+E) log N) |
| Code complexity | ~60 lines | ~115 lines | ~140 lines |
| Edge cases | Staleness | None known | Multi-edge trigger gap |

**v2 was chosen** for its theoretical properties — best balance of correctness, simplicity, and heuristic quality. The lazy re-score handles staleness cleanly, the net-delta scoring is more principled, and it's easy to explain in an interview. However, empirical benchmarks showed that **ordering strategy has negligible impact on peak memory** for our workloads (see below).

### Empirical results: ordering doesn't matter

We benchmarked all three ordering strategies (plus naive toposort and various sharing ablations) across transformer blocks of increasing size and 2-layer GPT-2 at multiple sequence lengths. Arena sizes are activation memory only (weights excluded).

**Transformer block (single layer, 20 nodes after optimization):**

| Config | no-opt | naive+share | v1 | v2 | v3 | best-fit | PyTorch | ORT |
|--------|--------|-------------|----|----|----|---------:|--------:|----:|
| Toy (d=64, s=32) | 80 KB | 48 KB | 48 KB | 48 KB | 48 KB | 48 KB | 144 KB | 55 KB |
| Small (d=256, s=128) | 1.3 MB | 768 KB | 768 KB | 768 KB | 768 KB | 768 KB | 2.3 MB | 1.0 MB |
| Medium (d=512, s=256) | 5.0 MB | 4.0 MB | 4.0 MB | 4.0 MB | 4.0 MB | 4.0 MB | 11.0 MB | 4.0 MB |
| GPT-2 dims (d=768, s=512) | 18.0 MB | 18.0 MB | 18.0 MB | 18.0 MB | 18.0 MB | 18.0 MB | 45.0 MB | 24.0 MB |

All ordering strategies produce **identical** arena sizes at every config. The only lever that moves the needle is in-place reuse (Toy: 80→48 KB, a 40% reduction). At GPT-2 dimensions, even in-place makes no difference — attention scratch (S² per batch×head) dominates the arena.

**GPT-2 (2-layer HuggingFace model, 64 nodes after optimization):**

| Config | no-opt | naive+share | v1 | v2 | v3 | no-inplace | PyTorch |
|--------|--------|-------------|----|----|----|-----------:|--------:|
| s=16 | 3189 KB | 3189 KB | 3237 KB | 3237 KB | 3189 KB | 3189 KB | 3189 KB |
| s=64 | 12.5 MB | 12.5 MB | 12.6 MB | 12.6 MB | 12.5 MB | 12.5 MB | 12.5 MB |
| s=256 | 49.8 MB | 49.8 MB | 50.6 MB | 50.6 MB | 49.8 MB | 49.8 MB | 49.8 MB |

The GPT-2 results are surprising: memory-aware ordering (v1, v2) is **1.5% worse** than naive toposort. In-place reuse is also counterproductive — disabling it matches the optimal result. v3 fixes the staleness issue well enough to match naive, but doesn't beat it.

### Why ordering doesn't matter here

Two factors conspire to make ordering irrelevant for these graphs:

1. **Attention scratch dominates.** The fused attention kernel's S² scratch buffer is by far the largest single allocation. At s=512, one scratch buffer is `12 × 512 × 512 × 4 = 12 MB` — larger than all other intermediates combined. No ordering heuristic can reduce this; it's a fixed cost determined by the algorithm.

2. **Graph structure limits reordering freedom.** After optimization (fusion, DCE), the transformer block graph is a nearly linear chain with only a few branch points (residual connections). The topological ordering has very few valid permutations, so different heuristics converge to the same or similar schedules.

### In-place reuse: helps most configs, slight penalty on GPT-2

In-place reuse delivers clear savings on smaller graphs: 40% arena reduction at Toy/Small, 20% at Medium. At GPT-2 transformer-block dimensions it has no effect (attention scratch dominates). On the actual 2-layer GPT-2 model, it's slightly counterproductive — a 1.5% arena increase. The mechanism: chaining an output to an input's arena slot extends the root tensor's lifetime, blocking that arena region longer and potentially forcing later allocations to higher offsets. For GPT-2's specific graph structure (64 nodes, multiple residual connections), this lifetime extension costs slightly more than the allocation it saves.

This is a known tradeoff in real runtimes. In-place reuse is a local optimization (this tensor can share that slot) that can have negative global effects (longer lifetimes reduce packing opportunities). For our workloads, in-place is net positive overall and the GPT-2 penalty is small enough to keep it enabled by default.

### Decision: v1 as default, v2/v3 retained for reference

v1 is the default ordering strategy. It's simple (~60 lines), the staleness issue is theoretical rather than practical for our workloads, and empirically it matches or nearly matches all alternatives. v2 and v3 are retained in `order.py` as research implementations — they demonstrate more sophisticated approaches and provide good interview discussion material about the tradeoffs.

The real memory savings come from the planner's core capabilities (lifetime analysis, first-fit packing, alias resolution, in-place sharing) rather than from ordering heuristics. Our arena is 2.5-3x smaller than PyTorch's peak-alive memory and competitive with ORT's arena allocator across all tested configs.

### Interview talking points

- Tradeoffs of first-fit vs best-fit offset assignment
- NP-hardness of optimal topological ordering for minimum memory
- Memory-aware ordering and in-place reuse are the two main heuristics real runtimes use
- **Empirical finding: ordering heuristics make no measurable difference for transformer graphs** — attention scratch dominates, graph structure limits reordering freedom
- **In-place reuse can be counterproductive** — lifetime extension vs allocation savings tradeoff
- The planner runs once at compile time, so O(n²) algorithms are fine for realistic graph sizes
- Can walk through the staleness analysis: why freed_bytes can only be under-estimated, why lazy re-eval works
- Net delta vs freed-only: peak memory is about live set size, need to account for output allocation not just input freeing
- The value of implementing, measuring, and making an empirically-grounded decision rather than assuming a theoretical improvement translates to practice

## Weight Transpose Handling

`nn.Linear` stores weights as `[out_features, in_features]` and computes `input @ weight.T`. After `torch.export`, the graph contains an explicit TRANSPOSE node followed by MATMUL. Three options for handling this:

1. **Constant folding** — pre-transpose at load time, call `cblas_sgemm` with `CblasNoTrans`. Materializes a contiguous `[in, out]` copy. Seems like the obvious choice, but is actually a **pessimization at large dimensions** (see below).
2. **Strided views** — reinterpret the `[out, in]` buffer with swapped strides. No copy, but C kernels can't handle non-contiguous data without extra logic.
3. **BLAS transpose flags** — fuse TRANSPOSE into MATMUL, keep weights in original `[out, in]` layout, call `cblas_sgemm` with `CblasTrans`. This is what `nn.Linear` does internally and is the fastest option at large dimensions.

**We use option 3.** The absorption pass pattern-matches TRANSPOSE → MATMUL and absorbs the transpose as a `transpose_b` attribute on the MATMUL node. The C kernel passes `CblasTrans` to `cblas_sgemm`.

### Why CblasTrans is faster (the BLAS packing story)

Benchmarks on Apple M4 Max, batch=32:

| Dim  | NoTrans (pre-transposed) | CblasTrans (original layout) | Speedup |
|------|--------------------------|------------------------------|---------|
| 512  | 16.7 us                  | 28.2 us                      | 0.59x   |
| 2048 | 328.0 us                 | 234.4 us                     | 1.40x   |
| 4096 | 1493.5 us                | 987.5 us                     | 1.51x   |

The cause is **memory access patterns during BLAS tile packing**. Before computing, `cblas_sgemm` copies tiles of the source matrices into cache-aligned packed buffers. For the B matrix, BLAS reads columns (in the mathematical sense):

- **CblasNoTrans on [K×N] row-major:** Column j requires elements at offsets `j, N+j, 2N+j, ...` — stride N. At dim=4096, that's 16KB between consecutive reads. Cache-hostile.
- **CblasTrans on [N×K] row-major:** "Column j" of the logical transposed matrix is row j of the stored matrix — offsets `j*K, j*K+1, j*K+2, ...` — stride 1, perfectly sequential. Prefetcher-friendly.

The effect reverses at small dimensions because the entire matrix fits in L2 cache regardless of access pattern, and CblasTrans has slight overhead from the transposed packing code path.

This was the root cause of PyTorch outperforming our runtime at dim=2048+. PyTorch's `nn.Linear` naturally uses the CblasTrans path; our constant folding was inadvertently switching to the slow CblasNoTrans path.

Full analysis in Obsidian: `INTERVIEW PREP/Inference Runtime Project/BLAS Transpose Performance Deep Dive.md`

## BLAS Flag Absorption Pass

The `absorb_into_matmul` pass absorbs adjacent ops into sgemm parameters. Originally a monolithic function, now decomposed into focused helpers:

- **`_absorb_transpose_b`** — absorbs TRANSPOSE on B input into `transpose_b=True`. Only absorbs when `{dim0, dim1} == {ndim-2, ndim-1}` (the last-two-dims swap), not head-reshape permutes.
- **`_pretranspose_constant_b`** — **new in refactor**: for constant B inputs not already transposed (e.g., Conv1D weights stored as `[in, out]`), pre-transposes the weight buffer in-place and sets `transpose_b=True`. This replaces the old approach where `_handle_addmm` in the exporter did the pre-transpose.
- **`_fold_scalar_into_alpha`** — shared math for folding scalar MUL/DIV into alpha (was duplicated between backward and forward cases)
- **`_absorb_input_scalars`** — backward scalar absorption on inputs
- **`_absorb_output_scalar`** — forward scalar absorption on output

The main function is a tight loop calling four helpers.

**This is not a fusion.** Traditional fusion combines ops into a single fused kernel. This is flag absorption — telling BLAS to handle a data layout concern internally. The generic FusionPattern machinery doesn't fit because it collects external inputs in chain order, which would scramble the A/B input ordering. A dedicated pass is cleaner.

**Critical ordering constraint.** This pass must run before constant folding. If constant folding runs first, it evaluates TRANSPOSE(constant_weight) eagerly, materializing a pre-transposed copy. The TRANSPOSE node disappears, the absorption pass finds nothing to do, and we end up on the slow CblasNoTrans path.

## Causal Mask Handling

### Unified mask detection

The original code had three separate functions for detecting causal masks: `_is_bool_causal_mask`, `_is_float_causal_mask` in the mask absorption pass, and a third `_is_causal_mask` in the causal attention fusion pattern. These were unified into a single `_is_causal_mask` function that handles both formats:

- **Bool format:** lower-triangular True matrix (from `torch.tril(torch.ones(..., dtype=torch.bool))`)
- **Float format:** upper-triangular `-inf` matrix (from `torch.where(causal_mask, 0.0, -inf)`)

### Mask absorption pass

`absorb_causal_mask` detects when a constant mask feeding an ATTENTION node is causal, replaces it with a `causal=True` attribute, and removes the mask input. The causal flag flows through the full stack: `OpDef.extras` packer → `COpNode.extra[]` → C `dispatch_attention` → `kernel_attention`/`kernel_attention_flash`.

## RESHAPE as Zero-Copy Alias

RESHAPE doesn't move data — it reinterprets dimension boundaries on the same contiguous memory. This is fundamentally different from TRANSPOSE/PERMUTE, which rearrange elements into a new physical layout. A `[4, 3]` tensor reshaped to `[3, 4]` has identical memory; the same tensor transposed to `[3, 4]` has elements in completely different positions.

The runtime handles RESHAPE as a zero-copy alias:

1. **Planner:** `OpDef.alias = True` for RESHAPE. `_resolve_alias()` walks producer chains. RESHAPE outputs get no arena allocation. The root tensor's lifetime is extended to cover all alias consumers.
2. **Executor:** When it encounters a RESHAPE node during dispatch, it creates a numpy view of the input buffer with the new shape and assigns it to the output tensor. No kernel is dispatched, no data moves.
3. **Constant folding:** If RESHAPE's input is a constant, constant folding handles it naturally — `np.reshape` returns a view, which gets promoted to a constant. The RESHAPE node is then removed by DCE.

This matters for transformer models where RESHAPE is used heavily for head splitting (`[B, S, D] → [B, S, H, D_k]`) and merging (`[B, S, H, D_k] → [B, S, D]`). Six reshapes per attention layer, all zero-cost.

## Non-Contiguous SLICE

**The original problem.** The compiled C execution path is supposed to be a single `execute()` call. But graphs with non-contiguous SLICE ops (notably GPT-2's QKV split along dim=2) broke this into segmented execution: run C nodes → pause → do a SLICE copy in Python → resume C nodes. The C executor had no kernel for strided copies.

**The workaround.** `compile_plan` pulled non-alias SLICEs out of the C execution order and recorded their positions. `execute_compiled` then ran C segments between them, executing each SLICE in Python via numpy before resuming. This added Python re-entry overhead and complicated the single-call invariant.

**The fix.** The `OpDef.alias` field for SLICE was changed from `True` to `Callable[[Node], bool]` — `_slice_alias(node)` returns True for dim=0 (contiguous, zero-copy alias) and False for dim>0 (non-contiguous, needs a kernel). A `kernel_slice` was implemented in `csrc/ops.c` that copies a non-contiguous slice into a contiguous output buffer using `[outer, orig_dim_size, start, slice_len, inner]` parameters from the extras array. The segmented execution code was removed entirely, restoring the single-call invariant. ~60 lines of complex segmented execution replaced by ~30 lines of straightforward extras packing + a C kernel.

## Memory Planning

The arena is for intermediate activations only. Weights live in their own buffers loaded from the model. Key reason: weights are permanent (alive across all inference calls, never freed), while intermediates are ephemeral (produced and consumed within a single pass). Mixing them would bloat the arena and prevent reuse.

The planner uses greedy first-fit offset assignment: for each tensor in birth order, filter to only temporally-alive allocations, then find the lowest offset that fits in the gaps.

## Scratch Buffers (Kernel Workspace)

Some fused kernels need temporary workspace that isn't an input or output — e.g., fused attention needs an S×S buffer for the attention matrix. This is the first case in our runtime where a kernel needs an intermediate buffer, and it raised a design question: where does that memory come from?

**Design principle: the planner owns all memory.** Every other kernel operates on pre-allocated buffers — inputs, outputs, and arena views. Letting kernels malloc internally would violate this and add per-call allocation overhead. Instead, scratch buffers go through the same arena allocation as regular intermediates.

**Scratch is a planner concern, not a graph concern.** The graph IR describes computation semantics: "compute attention on Q, K, V, produce O." The scratch buffer is an implementation detail of *how* the kernel executes — it depends on the op type, the tensor shapes, and potentially which backend runs it (a flash attention kernel needs much less scratch than the standard kernel). Different backends could have different scratch requirements for the same op. The graph shouldn't know about any of this.

### How it works

**Registry.** Scratch calculators are registered on `OpDef.scratch`:

```python
ScratchCalculator = Callable[[list[tuple[int, ...]], tuple[int, ...]], int]
#                              input_shapes            output_shape     -> bytes
```

The signature takes only the tensor shapes — no graph coupling. Most ops have no scratch (field is `None`). Attention returns `batch_heads * S * S * sizeof(float)`.

**Planner.** During planning, `_compute_scratch()` queries `OP_REGISTRY` for each node. If scratch is needed, it creates a `Lifetime` entry with `born=step, dies=step` (single-step). These lifetimes go through the same `_assign_offsets` first-fit as regular intermediates. The arena is unified.

**Executor.** When dispatching a node (per-op or compiled), the executor checks `plan.scratch` for that node. If present, it creates a flat `float32` arena view at the recorded offset and appends it to the kernel's input list. The kernel receives `[Q, K, V, scratch]` and doesn't care where the scratch came from.

### Scratch vs parallelism tradeoff

The standard attention kernel can process (batch, head) slices in parallel via GCD `dispatch_apply`. Each concurrent slice needs its own scratch buffer.

**The current approach:** The scratch calculator allocates `batch_heads * S * S * sizeof(float)`. Each GCD thread indexes into its region at `scratch + bh * scratch_per_slice`. Zero synchronization, zero allocation in the kernel. This over-allocates when `batch_heads >> num_cores`, but the scratch has a single-step lifetime so the planner reclaims it immediately.

**Flash attention scratch is tiny regardless.** The flash kernel's per-slice scratch is `32×32×4` = 4KB (one tile, not the full S×S matrix), so even at `batch_heads=128` it's only 512KB total.

## Optimization Pass Pipeline

Passes are plain callables `(Graph) -> bool`. The pipeline is a list, run in order. `run_until_stable()` runs the pipeline repeatedly until no pass reports changes.

**Default ordering: MATMUL absorption → constant folding → causal mask absorption → pattern fusion → DAG fusion → dead code elimination.** MATMUL absorption runs first so constant folding doesn't eagerly materialize transposes. Causal mask absorption runs before fusion so the causal flag is available for the CAUSAL_ATTENTION pattern. DAG fusion runs after chain fusion since it handles patterns (like GELU) that chain fusion can't.

**Pattern-based fusion** uses a registry of `FusionPattern` objects with priority-based grouping. Within each priority level, patterns are matched greedily longest-first. Patterns include optional validators and attr builders. Currently registered:
- **Priority 0:** `ATTENTION`, `CAUSAL_ATTENTION`, `BIAS_RELU`
- **Priority 1:** `MATMUL_ADD`

**DAG fusion** handles non-linear subgraph patterns. Currently recognizes the GELU tanh approximation — an 8-node subgraph where `x` fans out to `x^3` and `x` paths that reconverge through multiplication. This can't use `FusionPattern` because the chain matcher requires sole-consumer constraints at each step, and DAG patterns have multi-consumer intermediate nodes by definition. The `fuse_dags` pass is extensible to SiLU, GeGLU, etc. (~10-15 lines per new pattern).

**Constant folding and DCE** work as a pair. Folding promotes a node's output to a constant and removes the node, but leaves the input constants in place. DCE cleans up any that become dead. Constant folding reads evaluators from `OP_REGISTRY` instead of maintaining its own evaluator dict.

**Fold-only ops** (CAST, EXPAND, SLICE_TENSOR, CMP_NE, etc., enum values >= 100) are infrastructure ops that must be eliminated by constant folding. They have evaluators in `OP_REGISTRY` but no C kernels. Both executors raise `RuntimeError` if one reaches execution. These exist for subgraphs like GPT-2's causal mask generation — fully constant at export time, but the exporter needs to represent the ops so constant folding can evaluate them. If a future model uses these ops with runtime inputs, execution would fail; this is a known limitation.

## Numpy Evaluators

Constant folding evaluators now live in `OP_REGISTRY` as `OpDef.evaluator` fields, replacing the standalone `NUMPY_EVALUATORS` dict in the old `folding.py`. These are separate from the numpy backend's in-place kernels — the evaluators return new arrays (what constant folding needs), while the backend kernels write into pre-allocated output buffers via numpy's `out=` parameter (what the executor needs). The two have different memory contracts and should not be conflated.

## Executor Architecture

### Structural evolution

The executor was originally a single `executor.py` containing both the compiled and per-op dispatch paths, the `COpNode` struct definition, C library loading, arena management, and all the buffer binding logic. This was split into:

- **`executor/common.py`** — `Executor` ABC, `COpNode` struct, C library loading, arena management, buffer binding
- **`executor/compiled.py`** — `CompiledExecutor`: builds COpNode struct array, one C call per inference
- **`executor/interpreted.py`** — `InterpretedExecutor`: Python loop with backend chain dispatch

### Executor ABC

- `compile(plan)` — one-time preparation (build struct array or stash plan)
- `run(inputs)` — per-call inference (patch pointers + C call, or Python dispatch loop)
- Shared: `_get_arena()`, `_bind_inputs()`, `_bind_intermediates()`

Both executors subclass this — same interface, easy to swap in tests/benchmarks.

### CompiledExecutor cleanup

- `_fill_extras` replaced by `OP_REGISTRY` dispatch (4 lines vs ~90 lines of if/elif)
- SLICE handling simplified: contiguous slices are aliases (skipped), non-contiguous slices dispatch to C kernel. Segmented execution eliminated entirely.
- `_build_node` populates one COpNode struct — input pointers, scratch, output, shape, extras
- `run()` is minimal: patch input pointers, one C call, copy outputs

### InterpretedExecutor

Backend protocol + kernel resolution preserved for ablation use. Same compile/run interface as compiled executor. Alias ops skipped via `OP_REGISTRY` (no hardcoded op checks).

### Session API

`session.py` wraps the full pipeline: `InferenceSession(model, sample_inputs)` → `session.run(inputs)`. Mirrors ONNX Runtime's `InferenceSession` pattern. Handles export, optimization, planning, and executor creation internally.

## C Dispatch: Switch to Function Pointer Table

### The problem with switch dispatch

The original `executor.c` used a ~250-line `switch` statement for op dispatch. Every new op meant adding a case, and the cases mixed dispatch logic (extracting dimensions, handling flags) with kernel calls. No way to add an op without touching the switch.

### Function pointer table

Replaced with a `dispatch_fn` array indexed by `OpType` enum value, using C99 designated initializers:

```c
typedef void (*dispatch_fn)(const OpNode*);

static dispatch_fn DISPATCH_TABLE[DISPATCH_TABLE_SIZE] = {
    [OP_ADD]       = dispatch_add,
    [OP_RELU]      = dispatch_relu,
    [OP_MATMUL]    = dispatch_matmul,
    // ...
};
```

Each op is a self-contained `dispatch_xxx(OpNode*)` function that extracts its own parameters and calls the kernel. Three inline helpers eliminate repeated patterns:
- `total_elements(node)` — product of output shape
- `extra_float(node, idx)` — bit-cast extra[idx] from int to float
- `leading_dims(node, trailing)` — product of all dims except trailing

Adding a new op: add enum value in ir.py, write a dispatch function, add one table line. No monolithic switch to edit.

### Range-based enum numbering

OpType values are grouped by category: 10-19 unary, 20-29 binary, 30-39 reductions, 40-49 BLAS, 50-59 shape, 60-69 normalization, 70-79 fused, 100+ fold-only. Both Python (`ir.py`) and C (`executor.c`) use the same `enum OpType`. The C dispatch table has `DISPATCH_TABLE_SIZE = 100`, naturally enforcing that fold-only ops (>= 100) can't be dispatched.

## C Operators

C kernels live in `csrc/ops.c`, compiled to a shared library via the `csrc/Makefile`. MATMUL uses `cblas_sgemm` from Accelerate (macOS) or OpenBLAS (Linux). Element-wise ops (ADD, RELU, DIV, SUB, MUL, EXP, TANH, GELU), reductions (MAX, SUM), SOFTMAX, TRANSPOSE, SLICE, and LAYERNORM are hand-written loops. Fused attention kernels (standard and flash) combine sgemm + softmax + sgemm into single calls with SIMD softmax and GCD threading on macOS.

The C backend (`runtime/backends/c_backend.py`) loads the library via ctypes. Each wrapper extracts float pointers from numpy arrays using `.ctypes.data_as()` and passes dimensions as ints. Zero-copy: C operates directly on the same memory backing the numpy arrays.

**N-dim handling in wrappers.** C kernels are flat/2D (they take M, N, K etc.). The Python wrappers handle higher-dimensional inputs. Fast paths for common cases, general broadcast for everything else:
- **MATMUL ND×2D** (e.g., `[B, S, D] @ [D, N]^T` for linear layers): flatten A's batch dims into M, call a single sgemm.
- **MATMUL matching batch dims** (e.g., `[B, H, S, D] @ [B, H, D, S]`): loop over batch slices, one sgemm per slice.
- **MATMUL broadcast batch dims** (e.g., `[B, H, S, D] @ [H, D, S]`): odometer loop over the broadcast batch shape, one sgemm per slice.
- **ADD bias** (`[..., N] + [N]`): dedicated kernel with M×N loop.
- **ADD/SUB/MUL/DIV same shape**: flat kernel, no broadcast overhead.
- **ADD/SUB/MUL/DIV general broadcast**: `kernel_broadcast_binop` in C.

## N-dim Broadcasting

General broadcasting for binary ops is handled by a single C kernel, `kernel_broadcast_binop`. The Python wrapper computes "broadcast strides" for each input — normal strides for real dimensions, 0 for broadcast dimensions (size 1 or absent). The C kernel iterates over the output elements using coordinate increment with carry (odometer pattern), indexing into each input via its broadcast strides.

The odometer approach avoids division and modulo per element. The inner-most dimension increments and breaks immediately for most elements, making the coordinate tracking O(1) amortized. The per-element switch on op code (ADD/SUB/MUL/DIV) is branch-predictor-friendly since it's the same op every iteration.

**Fast paths are preserved.** The Python wrappers check shapes before dispatching:
1. Same shape → flat kernel (zero overhead, existing code)
2. Bias broadcast `[...,N] + [N]` → dedicated ADD kernel (tight M×N loop)
3. Everything else → general broadcast kernel

This means the common cases in inference (same-shape residual adds, bias adds, same-shape element-wise ops) hit the fast paths. The general broadcast kernel only fires for patterns like keepdim broadcast (`[B,H,S,S] - [B,H,S,1]`) or exotic shape combinations.

For MATMUL, broadcasting applies to batch dimensions only. The same broadcast-stride concept is used, but the iteration is over batch indices rather than elements, with one sgemm call per batch slice.

## Compiled C Executor

The compiled executor eliminates all Python from the hot path. Python builds the execution plan and "compiles" it into a flat C struct array (`OpNode`) with all buffer pointers and dimensions pre-resolved. A single ctypes call dispatches the entire graph.

**OpNode struct:** Each node in the plan becomes a fixed-size C struct containing the op type, input/output float pointers, output shape, and op-specific extras (e.g., K dimension for matmul). Uses fixed-size arrays (`MAX_INPUTS=8`, `MAX_DIMS=16`) to avoid dynamic allocation in C.

**Compile once, execute many.** `compile(plan)` resolves all tensor names to pointers, precomputes dimensions, and records which struct slots correspond to graph inputs. `run(inputs)` patches just the input pointers and makes one C call. Arena views, constant pointers, and output pointers are all stable between calls.

**Input patching.** Graph input tensors may feed multiple nodes. The compiled plan tracks a mapping of input name → list of (node_index, slot_index) pairs. Before each call, the executor patches those slots with the caller's input pointer. Everything else is unchanged.

## C Build Structure

**Single source of truth for kernels.** `ops.c` contains all kernel implementations. `executor.c` contains only the dispatch table and dispatch functions, forward-declaring the kernel functions it calls. Both files are compiled together into `libexecutor`.

Two shared libraries are built:
- `libruntime` (ops.c only) — loaded by `CBackend` for per-op dispatch
- `libexecutor` (executor.c + ops.c) — loaded by `CompiledExecutor` for compiled dispatch

## Performance Results

Benchmarked on Apple M4 Max, 3-layer MLP (Linear→ReLU→Linear→ReLU→Linear).

### Full ablation (3 trials, median)

| Config    | PyTorch  | Numpy    | C per-op | C exec   | C+fuse   | np/PT | Cop/PT | Cex/PT | fus/PT |
|-----------|----------|----------|----------|----------|----------|-------|--------|--------|--------|
| 1×512     | 20 us    | 37 us    | 71 us    | 13 us    | 13 us    | 1.80  | 3.47   | 0.62   | 0.63   |
| 32×512    | 105 us   | 140 us   | 150 us   | 103 us   | 109 us   | 1.34  | 1.44   | 0.98   | 1.04   |
| 128×512   | 395 us   | 265 us   | 255 us   | 196 us   | 195 us   | 0.67  | 0.65   | 0.50   | 0.49   |
| 1×2048    | 333 us   | 341 us   | 377 us   | 292 us   | 289 us   | 1.02  | 1.13   | 0.88   | 0.87   |
| 32×2048   | 1.07 ms  | 942 us   | 923 us   | 832 us   | 848 us   | 0.88  | 0.86   | 0.78   | 0.79   |

**Columns:** PyTorch = eager baseline. Numpy = per-op executor with NumpyBackend. C per-op = per-op executor with CBackend (Python dispatch loop, C kernels). C exec = compiled C executor without fusion. C+fuse = compiled C executor with fusion. Ratios are vs PyTorch (lower = faster).

### Key takeaways

**Eliminating Python dispatch is the single biggest lever.** Per-op C dispatch is 3.5x slower than PyTorch at batch 1 due to ~7 us of Python overhead per node (for loop, dict lookups, ctypes marshalling). The compiled executor eliminates this entirely: batch 1 swings from 3.5x slower to 1.6x faster.

**Numpy backend is surprisingly competitive.** At large batch sizes, numpy calls the same BLAS as PyTorch but avoids PyTorch's dispatch overhead. At 128×512 it's 0.67x PyTorch — faster than per-op C dispatch because it avoids the ctypes marshalling overhead.

**Element-wise fusion adds <5% on MLP.** Fusing bias+relu into one loop (one memory pass instead of two) is a minor win when matmul dominates. Expected to matter more for attention-heavy architectures with long element-wise chains.

**At dim=2048+, we were losing to PyTorch** because our constant folding pre-transposed weights into a `CblasNoTrans`-friendly layout, which is actually the slow path for BLAS packing at large dimensions. Using `CblasTrans` on the original weight layout (what `nn.Linear` does) is up to 1.5x faster at dim=4096. This is now addressed by the BLAS flag absorption pass.

### Transformer benchmark (3 trials, median)

Single-layer transformer block (LayerNorm → multi-head attention → residual → LayerNorm → FFN → residual). Naive PyTorch uses manual Q/K/V projections with `F.softmax`; SDPA PyTorch uses `F.scaled_dot_product_attention`. "C exec" is compiled C without fusion; "+fusion" adds attention, bias+relu, and matmul+add fusion.

| Config       | PT naive  | PT SDPA   | C exec    | +fusion   | nofus/PT | fused/PT | fused/SDPA |
|--------------|-----------|-----------|-----------|-----------|----------|----------|------------|
| 1×16×64      | 194 us    | 179 us    | 18 us     | 21 us     | 0.09     | 0.11     | 0.12       |
| 4×16×64      | 259 us    | 201 us    | 52 us     | 64 us     | 0.20     | 0.25     | 0.32       |
| 1×64×128     | 342 us    | 251 us    | 125 us    | 122 us    | 0.36     | 0.36     | 0.49       |
| 4×64×128     | 687 us    | 460 us    | 371 us    | 335 us    | 0.54     | 0.49     | 0.73       |
| 1×128×256    | 783 us    | 654 us    | 506 us    | 486 us    | 0.65     | 0.62     | 0.74       |
| 4×128×256    | 2.20 ms   | 1.50 ms   | 1.65 ms   | 1.20 ms   | 0.75     | 0.54     | 0.80       |

**We beat both PyTorch baselines at every config.** At small sizes (1×16×64), dispatch overhead elimination gives us ~10x over naive PyTorch and ~8x over SDPA. At the largest config (4×128×256), fused attention plus the other fusions give us 0.54x naive and 0.80x SDPA.

**Fusion impact is clearest at larger sizes.** At 4×128×256, fusion drops us from 0.75x to 0.54x naive — a 27% improvement from attention + MLP fusion. At tiny sizes, fusion is slightly slower because the fused attention kernel's GCD dispatch and scratch buffer overhead exceeds the memory traffic savings when intermediates fit in cache anyway.

**Beating SDPA validates the project thesis.** PyTorch SDPA uses a highly optimized fused attention kernel, but all the surrounding ops (linear projections, layernorms, reshapes, residual adds) still go through Python dispatch one-at-a-time. Our compiled C executor runs the entire graph in a single C call — the overhead savings on the ~20 non-attention ops compound to overcome SDPA's faster attention kernel.

## LAYERNORM as Compound Op

Like SOFTMAX, LAYERNORM is kept as a single compound op rather than decomposed into primitives (mean, var, rsqrt, mul, add). `torch.export` preserves `aten.layer_norm.default` as a single call with args `(input, normalized_shape, weight, bias, eps?)`. The exporter maps this to a single LAYERNORM node with three inputs (x, gamma, beta) and an `eps` attr.

The C kernel is a two-pass approach (mean first, then variance) — simpler than Welford's online algorithm and sufficient for our purposes since LayerNorm isn't the performance bottleneck. The kernel operates along the last axis: flatten everything else into `rows`, normalize each row of `cols` elements.

**Eps through the extras array.** The compiled C executor passes op-specific parameters via an `int extra[MAX_DIMS]` array. For LAYERNORM, eps is a float, so we bit-cast it into an int on the Python side (`struct.pack('f', eps)` → `struct.unpack('i', ...)`) and back out via `union { int i; float f; }` in C.

## 2D to N-dim: Compiled Executor Hardening

The compiled C executor (`executor.c`) was originally written and tested with 2D MLP tensors. Moving to 3D+ transformer tensors exposed several implicit 2D assumptions in the C dispatch cases. These were all correct in the per-op path (where Python wrappers handle shape decomposition) but broke in the compiled path (where C does it inline).

**Bugs found and fixed:**

1. **BLAS flag absorption too aggressive.** The `absorb_transpose_into_matmul` pass absorbed any TRANSPOSE feeding a MATMUL's B input, including head-reshape permutes that swap non-last dims (e.g., dim0=1, dim1=2 on 4D tensors). Fix: only absorb when `{dim0, dim1} == {ndim-2, ndim-1}`.

2. **TRANSPOSE dispatch only handled 2D.** The C kernel does a simple `[rows, cols]` transpose. For N-dim swapaxes (e.g., the `[B,S,H,dk] → [B,H,S,dk]` permutes in attention), we decompose the tensor into 5 logical regions around the two swapped dims: `[outer, A, middle, B, inner]`. The C code is a 4-nested loop with `memcpy` on the inner dimension.

3. **MATMUL ND×2D: batch pointer overflow.** The batch loop `b + i * b_stride` advanced B's pointer per batch, but for 2D weights there's only one B slice. Fix: detect ND×2D via `extra[2]` flag and flatten A's batch dims into M for a single sgemm.

4. **ADD dispatch assumed bias broadcasting.** `kernel_add` reads `b[j]` (broadcasting the last dim), which is correct for bias adds but wrong for element-wise adds like residual connections (`x + attn_out`). Fix: `extra[0]` flag distinguishes element-wise (flat loop) from bias broadcast.

**The lesson:** compiled executors that bake shapes into structs need to explicitly encode the same dispatch logic that dynamic wrappers get for free. Every Python `if a.ndim > 2` needs a corresponding C dispatch path.

## Attention Fusion Strategy

Users can write attention in many ways — manual softmax decomposition, `F.softmax`, or `F.scaled_dot_product_attention`. All three should optimize down to the same fused ATTENTION op. Tracing all three through `torch.export` (see `explore_attention.py`) reveals:

- **Naive (manual softmax):** 11 ATen ops — matmul, div, max, getitem, sub, exp, sum, div, matmul, plus reshape/permute for head splitting
- **F.softmax:** 7 ATen ops — matmul, div, softmax, matmul, plus reshape/permute
- **SDPA:** 4 ATen ops — scaled_dot_product_attention, plus reshape/permute

Recognition strategy (multi-level pattern matching):

1. **SDPA (implemented):** The exporter maps `aten.scaled_dot_product_attention` directly to our ATTENTION op. No passes needed — it's a 1:1 op mapping like the other ATen handlers.

2. **F.softmax (implemented):** The key insight is that `absorb_into_matmul` folds the scalar DIV (by √d_k) into the first MATMUL's `alpha` parameter. This reduces the 4-node chain `MATMUL → DIV → SOFTMAX → MATMUL` to 3 nodes `MATMUL(alpha=1/√d_k) → SOFTMAX → MATMUL`, which fits the generic `FusionPattern` infrastructure. The validator checks `transpose_b=True` (Q @ K^T pattern) and softmax on the last axis. External inputs collected by `_apply_fusion` are naturally `[Q, K, V]` — exactly what the ATTENTION kernel expects.

3. **Naive (not yet implemented):** The manual softmax decomposition creates a DAG — the DIV output feeds both MAX and SUB, violating the sole-consumer constraint in `_try_match`. The DAG fusion framework could potentially be extended for this, but the subgraph is significantly more complex than GELU (~11 nodes with multiple fan-out/reconverge points).

**N-dim batch support.** The ATTENTION op handles arbitrary leading batch dimensions. Q/K/V can be 3D `[BH, S, D]` or 4D `[B, H, S, D]` — the memory layout is byte-identical, so the C kernel doesn't care.

**ATTENTION is the atomic unit, not SOFTMAX.** We do NOT canonicalize manual softmax into a SOFTMAX op first. The flash attention algorithm interleaves softmax computation with matmuls tile-by-tile — it never computes a full standalone softmax. Canonicalizing to SOFTMAX would lose the structure flash attention needs.

**Multiple kernel implementations behind one op.** The graph-level ATTENTION op is backend-agnostic. The backend can dispatch to:
- A simple fused kernel that materializes the full S×S attention matrix (correct, easy to implement)
- A tiled/online flash attention kernel that never materializes the full matrix (better memory, better perf at long sequences)

Both produce identical results. The graph passes don't care which runs.

## Fused Attention Kernels

Two fused attention kernel implementations in `csrc/ops.c`, sharing the same interface. Both take Q, K, V as `[batch_heads, seq_len, head_dim]`, a scratch buffer, a causal flag, and produce the output. The graph-level ATTENTION op is backend-agnostic — the backend can dispatch to either kernel.

### Standard kernel (`kernel_attention`)

For each (batch, head) slice:
1. `S = Q @ K^T * scale` — single `cblas_sgemm` with `alpha = 1/sqrt(d_k)`, folding the scale into the matmul for free
2. If causal: apply `-INFINITY` mask to above-diagonal entries
3. `P = softmax(S)` — row-wise, in-place on the scratch buffer
4. `O = P @ V` — single `cblas_sgemm`

Scratch requirement: `S × S` floats per concurrent thread. Materializes the full attention matrix but eliminates all intermediate tensor allocations that the unfused graph would need.

### Flash kernel (`kernel_attention_flash`)

Tiled online-softmax attention that never materializes the full S×S matrix. For each (batch, head) slice, processes Q in blocks of B_r=32 rows:

1. For each K/V block of B_c=32 rows:
   - `S_tile = Q_block @ K_block^T * scale` via sgemm (B_r × B_c tile)
   - If causal: mask above-diagonal entries in the tile
   - Online softmax update: find tile-row max, compute correction factor `exp(m_old - m_new)`, rescale accumulated O and running sum, compute `exp(S - m_new)`, accumulate new sum
   - `O_block += P_tile @ V_block` via sgemm with `beta=1.0` to accumulate
2. Final normalization: `O /= l` (running sum per row)

Scratch requirement: `B_r × B_c` floats (1 tile = 4KB vs S² for standard). Tile sizes chosen so the working set (Q/K/V blocks + S tile + O block ≈ 36KB at D=64) fits in L1 cache.

The flash kernel trades compute for memory — the rescaling corrections and many small sgemm calls add overhead, but it avoids writing and reading the S² attention matrix. On CPU, the crossover where flash beats standard is at very long sequences (S > 1024) where S² blows the cache. On GPU (where flash attention was designed), the tradeoff is much more favorable because compute is cheap relative to memory bandwidth.

### SIMD softmax optimization

The standalone `kernel_softmax` and the standard attention kernel's inline softmax were originally scalar loops — three passes per row (`max`, `exp+sum`, `normalize`), each element processed one at a time. Profiling showed softmax took **79% of total attention time** at S=512 and was **4.8x slower** than PyTorch's softmax at the same size.

On macOS, we replaced the scalar loops with Accelerate's SIMD functions:
- `vDSP_maxv` — vectorized max (NEON parallel reduction)
- `vDSP_vsadd` — subtract max from each element (NEON)
- `vvexpf` — vectorized exp (4 floats per NEON cycle vs 1 for scalar `expf`)
- `vDSP_sve` — vectorized sum
- `vDSP_vsdiv` — normalize (vectorized divide)

The key win is `vvexpf`: `expf` is an expensive multi-instruction polynomial approximation, and the scalar version processes one float at a time. The vectorized version computes 4 in parallel using NEON packed registers.

**Results (softmax on 512×512 matrix):** 431us → 300us (1.4x faster). At 256×256: 112us → 36us (3x faster, now beats PyTorch). The improvement is less dramatic at S=512 because 5 separate vDSP function calls per 512-element row have non-trivial call overhead — a single fused SIMD loop would do better, but this is good enough for our purposes.

### GCD threading

Both attention kernels parallelize across (batch, head) slices using Grand Central Dispatch (`dispatch_apply`). Each slice is completely independent — different regions of Q/K/V/output memory. Each GCD thread indexes into its own scratch region (planner-allocated).

GCD is native to macOS (no build dependencies), manages its own thread pool, and handles work-stealing automatically. The `dispatch_apply` call blocks until all slices complete.

**Impact on multi-head attention (bh=8):**

| S | Before (sequential) | After (GCD) | Speedup |
|---|---|---|---|
| 128 | 0.86x PT | **0.42x PT** | 2.0x |
| 256 | 0.80x PT | **0.23x PT** | 3.5x |
| 512 | 1.49x PT | **0.32x PT** | 4.7x |

At S=256 with 8 heads, we went from 519us to 149us — a 3.5x speedup from threading, putting us 4.3x faster than eager PyTorch.

### Performance summary

Benchmarked on Apple M4 Max, fused attention kernels (standard variant).

| Config | numpy | C std | PT eager | PT SDPA | std/PT |
|---|---|---|---|---|---|
| 1b×4h×64s | 55us | 25us | 69us | 47us | **0.37** |
| 1b×4h×256s | 639us | 122us | 302us | 113us | **0.40** |
| 2b×8h×128s | 806us | 174us | 422us | 127us | **0.41** |
| 2b×8h×256s | 3.50ms | 399us | 1.48ms | 284us | **0.27** |

We beat eager PyTorch by 2-4x across all configs. SDPA still wins by ~2x at larger sizes — their heavily optimized C++ implementation has finer-grained SIMD, better tiling, and more mature threading. But we're in the same ballpark.

## Element-wise Fusion: Interpreter vs Bespoke Kernels

The classic argument for element-wise fusion is simple: a chain of N ops does N read→compute→write round-trips through memory. A fused kernel does one read, applies all N operations in registers, and one write. For memory-bound workloads this should be a significant win.

We built a general-purpose interpreter-based element-wise fusion system, benchmarked it against unfused dispatch and a hand-written bespoke kernel, and found that **the interpreter approach is slower than no fusion at all.** The project now uses bespoke pattern-matched kernels instead.

### What we built (and removed)

The `FUSED_ELEMENTWISE` system was a micro-op interpreter. The pass walked the graph looking for chains of element-wise ops (ADD, SUB, MUL, DIV, RELU, EXP), replaced them with a single fused node carrying a list of `(op_code, ext_input_idx, swap)` micro-ops, and dispatched to a C kernel that:

1. Loaded the primary input into a `float acc`
2. For each element, looped through the micro-ops applying each via a `switch`
3. Handled broadcasting via the odometer stride pattern (per-input offsets with coordinate increment/carry)

This is architecturally elegant — one kernel handles arbitrary chains with arbitrary broadcasting. But it introduces per-element overhead that doesn't exist in flat loops.

### Why it's slower: the benchmark

Tested on Apple M4 Max, 3-layer MLP (Linear→ReLU→Linear→ReLU→Linear), compiled C executor. Three configurations:
- **No fusion:** separate ADD and RELU dispatch (tight flat loops)
- **Interpreted:** `FUSED_ELEMENTWISE` interpreter kernel
- **Bespoke:** `FUSED_BIAS_RELU` hand-written kernel (`for (j) out[i*N+j] = max(a[i*N+j] + bias[j], 0)`)

| Config    | No fusion | Interpreted | Bespoke | interp/nf | bespoke/nf |
|-----------|-----------|-------------|---------|-----------|------------|
| 1×512     | 13 us     | 15 us       | 13 us   | 1.15      | 0.98       |
| 32×512    | 107 us    | 154 us      | 96 us   | 1.44      | 0.90       |
| 128×512   | 206 us    | 385 us      | 196 us  | 1.87      | 0.95       |
| 1×2048    | 296 us    | 300 us      | 294 us  | 1.01      | 0.99       |
| 32×2048   | 868 us    | 1044 us     | 856 us  | 1.20      | 0.99       |
| 128×2048  | 1.85 ms   | 2.64 ms     | 1.84 ms | 1.43      | 1.00       |

**The interpreter is 1.15-1.87x slower than no fusion.** The per-element overhead of the micro-op switch, broadcast stride tracking, and coordinate bookkeeping costs more than the memory round-trip it eliminates. This is worst at 128×512 where element-wise ops are a larger fraction of total compute relative to the matmuls.

**The bespoke kernel matches no-fusion performance** (ratios ~1.0) because it has the same tight-loop structure as the unfused kernels — just with two operations per element instead of two separate loops. No interpreter overhead, no broadcast indirection.

### The lesson

General element-wise fusion only pays off if the kernel overhead is negligible compared to memory bandwidth savings. For a micro-op interpreter, the overhead scales linearly with element count — exactly when memory bandwidth savings should be largest. The crossover point where interpreter fusion wins doesn't exist for chains this short (2 ops).

Real runtimes solve this with **codegen**: they JIT-compile a specialized kernel for each unique fused subgraph, getting bespoke-kernel speed with general-fusion flexibility. That's a much larger engineering effort (MLIR, TVM, Triton, XLA all do this) and outside our scope.

**Our approach: pattern-matched bespoke kernels.** We register specific patterns (BIAS+RELU, MATMUL+ADD) in the `FusionPattern` registry, each backed by a hand-written C kernel. This gives us bespoke speed for the patterns that matter, with minimal machinery. The number of patterns in an inference runtime is small enough that this scales fine — ONNX Runtime itself uses this approach for many of its fusions.

### Graph mutation ordering

A note on implementation: `_apply_fusion` must remove original chain nodes *before* adding the fused node. `Graph.remove_node()` pops the producer mapping for the removed node's output. If the fused node is added first (registering itself as producer of the last chain node's output), then removing the last chain node clobbers that entry. The `_apply_elementwise_fusion` code had this ordering correct with an explicit comment; `_apply_fusion` originally had it backwards (undetected because no patterns were registered until FUSED_BIAS_RELU).

## MATMUL + ADD Fusion

The `fuse_matmul_add` pass catches standalone MATMUL → ADD pairs where the ADD wasn't claimed by pattern fusion (e.g., the final linear layer in an MLP, which has no activation after it). It replaces them with a single `MATMUL_ADD` node.

**The sgemm beta trick.** `cblas_sgemm` computes `C = alpha * A @ B + beta * C`. By pre-filling the output buffer with the broadcast bias and setting `beta=1.0`, sgemm adds the bias as part of its accumulation — eliminating the separate ADD kernel and its memory round-trip. The C dispatch broadcasts the 1D bias `[N]` into every row of the `[M, N]` output, then calls `kernel_matmul_beta()` with `beta=1.0`.

**Why this runs after pattern fusion.** Pattern fusion should get first crack at ADD nodes — if there's an ADD+RELU chain, fusing them into FUSED_BIAS_RELU (one post-matmul pass doing both add and relu) is better than absorbing the ADD into sgemm (which would leave RELU as a separate dispatch). The beta trick is only advantageous for isolated bias adds where there's nothing else to chain with.

**Pipeline ordering rationale:** MATMUL → ADD → RELU becomes MATMUL → FUSED_BIAS_RELU via the `bias_relu` pattern. Only MATMUL → ADD (no RELU) becomes MATMUL_ADD via this pass. This gives the best result for both cases: the inner layers get fused bias+relu, and the final layer gets the sgemm beta trick.

**Carries forward attrs.** The fused node inherits the original MATMUL's attributes (notably `transpose_b`), so the CblasTrans optimization for `nn.Linear` weights is preserved through the fusion.

## DAG Fusion Framework

### Motivation

The `FusionPattern` registry handles linear chain patterns (A→B→C) but can't match non-linear subgraphs where an intermediate node has multiple consumers. GELU's tanh approximation (`0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`) is an 8-node DAG where `x` fans out to `pow` and `mul` paths that reconverge. The sole-consumer constraint in `_try_match` rejects this at the first fan-out point.

### Approach

The `fuse_dags` pass walks the graph looking for "seed" nodes that begin known DAG patterns. For GELU, the seed is a POW node with exponent=3. From there, it verifies the full 8-node subgraph structurally: POW → MUL(0.044715) → ADD(x) → MUL(sqrt(2/pi)) → TANH → ADD(1) → MUL(x) → MUL(0.5). At each step, it validates op types, constant values, and input/output relationships.

The matching is more manual than chain fusion — each DAG pattern is a custom verification function rather than a declarative pattern spec. But for the handful of activation functions that need this treatment, the explicitness is a feature: each pattern is self-documenting and easy to debug.

### Extensibility

Adding a new DAG pattern (SiLU, GeGLU, etc.) is ~10-15 lines: a seed detection condition and a subgraph verification function. The `fuse_dags` pass iterates all registered DAG matchers.

## RESHAPE of Graph Inputs in Compiled Dispatch

**Status:** Open — needs design decision before implementing.

### The Problem

When compiling the full HuggingFace GPT-2 model (with embeddings), `torch.export` emits `view(input_ids, [-1, 16])` as the very first op — a RESHAPE of the graph input. This crashes `compile_plan` because RESHAPE binding happens at compile time but graph input buffers are only available at execute time:

```python
# compile_plan, line ~244
for node in plan.order:
    if node.op == OpType.RESHAPE:
        inp_buf = graph.tensors[node.inputs[0]].buffer  # None for graph inputs!
        out_tensor.buffer = inp_buf.reshape(...)         # AttributeError
```

This wasn't exposed before because the old `GPT2Body` test takes float hidden states directly (no RESHAPE of the input). The HF model starts with `view(input_ids, ...)`.

The specific case that triggered this is a **same-shape reshape** — `(1, 16)` → `(1, 16)` — because `torch.export` resolves `-1` to the concrete dimension. But the general problem applies to any RESHAPE whose input chain traces back to a graph input without an intervening non-RESHAPE compute node.

### Context: How RESHAPE Works Today

RESHAPE is zero-copy in both execution paths:
- **Per-op dispatch:** The Python loop creates a numpy view (`input_buf.reshape(shape)`) and assigns it as the output tensor's buffer. Works fine because inputs are bound before execution starts.
- **Compiled dispatch:** `compile_plan` pre-binds RESHAPE outputs as views during compilation so that downstream COpNodes get valid pointers. This is where the crash happens — graph input buffers don't exist yet.

The planner treats RESHAPE outputs as aliases of their input (sharing the same arena offset via `_resolve_alias`). They're excluded from the C execution order since there's no kernel to run.

### Possible Fixes

#### Option A: Eliminate trivial (same-shape) reshapes

Add a pass that removes RESHAPE nodes where `input_shape == output_shape`, rewiring consumers to reference the original tensor directly. This is a standard graph optimization.

**Pros:**
- Simple, clean, general-purpose optimization (not a workaround)
- Reduces graph noise — these no-op reshapes are common from `torch.export`
- No changes to the executor

**Cons:**
- Only fixes the same-shape case. A model with a genuine reshape of its input (e.g., flattening `(B, S)` → `(B*S,)`) would still crash. In practice this may never happen since most models reshape intermediates, not inputs — but it's not a complete fix.

#### Option B: Defer RESHAPE binding to execute time

Track RESHAPE nodes whose inputs aren't bound at compile time. Store them on the `CompiledPlan`. In `execute_compiled`, after patching input pointers, process deferred RESHAPEs to create their views, then patch any COpNode slots that reference those outputs.

**Pros:**
- Handles all cases correctly (same-shape and different-shape)
- Architecturally sound — graph inputs are inherently per-call, so their derived views should be too

**Cons:**
- Adds complexity to CompiledPlan and execute_compiled (new field, new patch-slot tracking, per-call rebinding logic)
- RESHAPE views create new numpy arrays each call, so any COpNode that references a deferred RESHAPE output needs pointer patching per call (similar to how `input_slots` and `slice_patch_slots` already work, but a third variant of the same pattern)
- The executor's compile/execute split is already complex with input_slots and slice_ops; this adds a third deferred-binding mechanism

#### Option C: Bind a dummy buffer at compile time, rebind at execute time

During `compile_plan`, allocate a temporary numpy array of the right shape for RESHAPE-of-input outputs. This lets pointer resolution proceed normally. At execute time, rebind with the actual view and patch pointers.

**Pros:**
- Compile-time code stays simple (no special-casing)
- Handles all cases

**Cons:**
- Wasteful (allocates memory that's immediately discarded)
- Still needs per-call pointer patching (same complexity as Option B)
- Conceptually misleading — the dummy buffer has no meaningful data

#### Option D: Both A and B

Eliminate trivial reshapes as a graph optimization (Option A), and also implement deferred binding (Option B) as a safety net for the general case. The pass eliminates the common case cheaply; the executor handles anything the pass can't remove.

**Pros:**
- Complete solution: graph is cleaner AND executor is robust
- Each piece is independently useful

**Cons:**
- Most implementation work
- Option B's complexity may not be justified if no real model ever hits a non-trivial reshape of a graph input

### Recommendation

Not committing to an approach yet. The immediate question is: **does any real model produce a non-trivial RESHAPE of a graph input?** If the answer is "no, it's always same-shape from torch.export resolving dynamic shapes to concrete ones," then Option A alone is sufficient and cleanest. If there are real cases, Option D is the robust choice.

For now, the 2-layer HF GPT-2 works with per-op dispatch (which handles this correctly) and with compiled dispatch at all tested sequence lengths via the manual `GPT2Inference` wrapper. The full HF model compiled path is blocked only on this issue.

### Resolution (2026-02-15): Two bugs, not one

Running `GPT2LMHeadModel` directly (no wrapper, `use_cache=False`) exposed two bugs. Neither was the RESHAPE dispatch issue we anticipated — the planner's external-alias rejection (line 320-325) correctly gives the RESHAPE output its own arena slot. The bugs were:

**Bug 1: `dispatch_reshape` hardcodes `sizeof(float)`.** The RESHAPE of `input_ids` copies int64 data (8 bytes/element) but `dispatch_reshape` used `total_elements(node) * sizeof(float)` — copying only half the bytes. The EMBEDDING op then read truncated indices and segfaulted on out-of-bounds table lookups.

Fix: added `int elem_size` field to the `OpNode` struct (`runtime.h`). Set from the output tensor's dtype during compilation (`compiled.py:_build_node`). Used in `dispatch_reshape`, the N-dim branch of `dispatch_transpose`, and `kernel_slice`. Compute ops (matmul, add, etc.) continue using float — they're inherently typed. When quantized ops are added (INT8 matmul), they'll be separate op codes with their own dispatch functions, not dtype branches inside existing kernels. Data movement ops are dtype-agnostic and use `elem_size`.

**Bug 2: Planner lifetime tracking for rejected aliases.** This was the actual segfault after fixing bug 1. The RESHAPE output `view` (int64, 64 bytes) and the EMBEDDING output `embedding` (float32, 24KB) were both allocated at arena offset 0 — overlapping. The EMBEDDING kernel read `ids[0]`, wrote 3KB of float output starting at offset 0, and destroyed the remaining indices.

Root cause in `_compute_lifetimes` (planner.py, line ~310):

```python
# OLD (buggy):
root = get_root(_resolve_alias(inp, graph))

# NEW (fixed):
root = get_root(inp)
```

`_resolve_alias` follows the **graph-level** alias chain: `view` → `input_ids`. Then `get_root('input_ids')` = `input_ids` (external). Since externals aren't tracked, `dies_at['view']` was never set, defaulting to `born_at['view']` = step 0. The planner thought `view` was dead immediately, freed offset 0, and allocated the EMBEDDING output there.

The fix uses `get_root(inp)` which follows the **planner's actual sharing decisions**. Since the planner rejected the alias (external input), `view` is not in `memory_root`, so `get_root('view')` = `view` — correctly extending `view`'s own lifetime to the step where EMBEDDING consumes it.

This is correct for both cases:
- **Alias shared** (intermediate input): `memory_root['view'] = root_A`, so `get_root('view')` = `root_A`. Extends root_A's lifetime. Same as before.
- **Alias rejected** (external input): `view` not in `memory_root`, so `get_root('view')` = `view`. Extends view's own lifetime. Previously broken.

**Verification:** The `_resolve_alias` call on line 306 (for consumed counting used by in-place decisions) is still correct — it needs graph-structural relationships for the "is this the last consumer?" check. Only the death-tracking on line 310 was wrong, because it needs to extend the lifetime of whichever tensor actually owns the arena memory, which is a planner decision, not a graph-structure fact.

Full GPT-2 (124M, 12 layers) now runs end-to-end through the compiled C executor directly from HuggingFace with no wrapper. Max logit diff vs PyTorch: 0.000092.

## Validation Framework

### Motivation

The runtime had one validation method: `Graph.validate()`, a monolithic structural check added early in the project and never revisited. It caught reference errors and cycles but nothing else — no semantic checks (are MATMUL input shapes compatible?), no pipeline-stage-specific checks (did constant folding eliminate all fold-only ops?), no memory plan validation (do arena offsets actually avoid overlapping?).

Both executors had runtime guards for fold-only ops (`if node.op.value >= 100: raise RuntimeError`), which meant a compile-time error was caught at execution time. An interviewer would reasonably ask: "Why not catch this earlier?"

### Design space

We considered three approaches:

**Option A: Context bag.** A `ValidationContext` accumulates state as the pipeline runs — each stage adds its output, validators pull what they need. Uniform `(ValidationContext) -> list[str]` signature. Problem: god object with no compile-time guarantee that the data a validator needs has actually been populated. A POST_PLAN check could accidentally reach for something that doesn't exist until PRE_EXECUTE.

**Option B: Phase-typed protocols.** Different validator signatures per phase (`GraphCheck`, `PlanCheck`, `CompileCheck`). Type-safe — a plan validator can't accidentally depend on executor state. But you lose the unified registry; dispatching validators requires knowing which protocol you're using.

**Option C: Components validate their own output.** No validation abstraction at all. The planner validates its plan before returning, the executor validates the plan before compiling. Simplest, closest to how ORT works, but validation logic scatters across components rather than being collected and inspectable.

### What we chose

A hybrid of A and B, avoiding the worst of each. Validators are registered with a `Phase` tag and a check function. The phase tag determines both *when* the validator runs and *what type* it receives:

- Graph phases (POST_EXPORT through POST_RESOLVE_OPTIMIZE): `check(graph: Graph) -> list[ValidationResult]`
- POST_PLAN: `check(plan: MemoryPlan) -> list[ValidationResult]`
- PRE_EXECUTE: `check(plan: ExecutionPlan) -> list[ValidationResult]`

The phase is the contract — it tells you when the validator runs and what's available. No context bag, no god object. The type of `T` is implied by the phase rather than enforced by generics (Python generics don't buy much at runtime anyway).

### MemoryPlan vs ExecutionPlan

This design revealed that the old `ExecutionPlan` was really a memory plan — it's the planner's output containing execution order, arena layout, and offsets. It knows nothing about dispatch targets.

The PRE_EXECUTE phase needs to validate things like "does every op have a C dispatch function?" — which requires knowing the executor type. So we renamed the planner's output to `MemoryPlan` and created a new `ExecutionPlan` that bundles:

```python
@dataclass
class ExecutionPlan:
    graph: Graph
    memory: MemoryPlan
    executor_type: str      # "compiled" or "interpreted"
    backend: str            # "c", "numpy", "c+numpy"
```

The Session constructs this after planning, before compilation. Validators at PRE_EXECUTE can inspect both the memory layout and the dispatch configuration.

### Severity model

Severity lives on the result, not the validator. A single validator might produce both errors and warnings — "fold-only op survived" is an error, "dead node detected" is a warning. Three levels: ERROR (execution will fail), WARNING (suspicious), INFO (diagnostic).

The Session maps a user-facing `validation` parameter to a failure threshold:
- `"strict"`: fail on WARNING or ERROR
- `"normal"`: fail on ERROR only (default)
- `"none"`: skip validation entirely

### Package structure

Core types live in `validation/core.py` to avoid circular imports — the validator submodules (`graph.py`, `plan.py`, `execution.py`) import from `core`, and `__init__.py` re-exports and triggers registration by importing the submodules. The registry is a module-level list populated by `@register_validator` decorators at import time.

### Compiled executor boundary

We considered adding a PRE_DISPATCH phase for validating COpNode structs (raw pointers, op codes, array bounds) before the C handoff. This is qualitatively different from domain-level validation — it's FFI defensive checking where the failure mode is `abort()` instead of a Python exception.

Decision: keep FFI-level checks inside `CompiledExecutor._validate_structs()` rather than in the validation framework. Domain-level pre-dispatch checks (like "all ops have C dispatch support") belong in PRE_EXECUTE. Pointer sanity checks are assertions, not validators.
