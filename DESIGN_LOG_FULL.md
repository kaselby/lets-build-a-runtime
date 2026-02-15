# Design Decisions

Detailed technical decisions and rationale. See `CLAUDE.md` for high-level principles.

## Op Representation

**TRANSPOSE vs PERMUTE are separate ops.** A 2D axis swap has different optimization opportunities (BLAS transpose flags, fusion into matmul) than a general N-dimensional permute (which requires data movement). The exporter inspects the permutation axes and emits the appropriate op.

**addmm decomposition.** `torch.export` emits `aten.addmm(bias, input, weight)` for linear layers. We decompose this into separate MATMUL + ADD nodes so the fusion pass has something to work with.

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

No shared mutable state between handlers. The getitem handler is self-contained — it reads split metadata from the FX graph rather than relying on a stash dict.

## Weight Transpose Handling

`nn.Linear` stores weights as `[out_features, in_features]` and computes `input @ weight.T`. After `torch.export`, the graph contains an explicit TRANSPOSE node followed by MATMUL. Three options for handling this:

1. **Constant folding** — pre-transpose at load time, call `cblas_sgemm` with `CblasNoTrans`. Materializes a contiguous `[in, out]` copy. Seems like the obvious choice, but is actually a **pessimization at large dimensions** (see below).
2. **Strided views** — reinterpret the `[out, in]` buffer with swapped strides. No copy, but C kernels can't handle non-contiguous data without extra logic.
3. **BLAS transpose flags** — fuse TRANSPOSE into MATMUL, keep weights in original `[out, in]` layout, call `cblas_sgemm` with `CblasTrans`. This is what `nn.Linear` does internally and is the fastest option at large dimensions.

**We use option 3.** The fusion pass pattern-matches TRANSPOSE → MATMUL and absorbs the transpose as a `transpose_b` attribute on the MATMUL node. The C kernel passes `CblasTrans` to `cblas_sgemm`.

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

The `absorb_transpose_into_matmul` pass handles TRANSPOSE → MATMUL patterns that the exporter didn't catch at export time. The exporter handles `aten.linear` directly (emitting MATMUL with `transpose_b=True`), but manual `x @ w.T` or `torch.mm(x, w.permute(1,0))` in model code produces explicit TRANSPOSE nodes followed by MATMUL.

The pass walks all MATMUL nodes, checks if input B (index 1) is produced by a TRANSPOSE with a sole consumer, and rewires the MATMUL to read the original pre-transpose tensor with `transpose_b=True`. This keeps the weight in its original `[out, in]` layout for the fast CblasTrans path.

**This is not a fusion.** Traditional fusion combines ops into a single fused kernel. This is flag absorption — telling BLAS to handle a data layout concern internally. The generic FusionPattern machinery doesn't fit because it collects external inputs in chain order, which would scramble the A/B input ordering. A dedicated pass is cleaner.

**Critical ordering constraint.** This pass must run before constant folding. If constant folding runs first, it evaluates TRANSPOSE(constant_weight) eagerly, materializing a pre-transposed copy. The TRANSPOSE node disappears, the absorption pass finds nothing to do, and we end up on the slow CblasNoTrans path.

The `Graph.rewire_input()` method was added to support this pattern — it changes a node's input tensor and updates the consumer index atomically.

## RESHAPE as Zero-Copy Alias

RESHAPE doesn't move data — it reinterprets dimension boundaries on the same contiguous memory. This is fundamentally different from TRANSPOSE/PERMUTE, which rearrange elements into a new physical layout. A `[4, 3]` tensor reshaped to `[3, 4]` has identical memory; the same tensor transposed to `[3, 4]` has elements in completely different positions.

The runtime handles RESHAPE as a zero-copy alias:

1. **Planner:** Identifies RESHAPE nodes and builds an internal alias map (output → root input, following chains). RESHAPE outputs get no arena allocation. The root tensor's lifetime is extended to cover all alias consumers.
2. **Executor:** When it encounters a RESHAPE node during dispatch, it creates a numpy view of the input buffer with the new shape and assigns it to the output tensor. No kernel is dispatched, no data moves.
3. **Constant folding:** If RESHAPE's input is a constant, constant folding handles it naturally — `np.reshape` returns a view, which gets promoted to a constant. The RESHAPE node is then removed by DCE.

This matters for transformer models where RESHAPE is used heavily for head splitting (`[B, S, D] → [B, S, H, D_k]`) and merging (`[B, S, H, D_k] → [B, S, D]`). Six reshapes per attention layer, all zero-cost.

## Memory Planning

The arena is for intermediate activations only. Weights live in their own buffers loaded from the model. Key reason: weights are permanent (alive across all inference calls, never freed), while intermediates are ephemeral (produced and consumed within a single pass). Mixing them would bloat the arena and prevent reuse.

The planner uses greedy first-fit offset assignment: for each tensor in birth order, filter to only temporally-alive allocations, then find the lowest offset that fits in the gaps.

## Scratch Buffers (Kernel Workspace)

Some fused kernels need temporary workspace that isn't an input or output — e.g., fused attention needs an S×S buffer for the attention matrix. This is the first case in our runtime where a kernel needs an intermediate buffer, and it raised a design question: where does that memory come from?

**Design principle: the planner owns all memory.** Every other kernel operates on pre-allocated buffers — inputs, outputs, and arena views. Letting kernels malloc internally would violate this and add per-call allocation overhead. Instead, scratch buffers go through the same arena allocation as regular intermediates.

**Scratch is a planner concern, not a graph concern.** The graph IR describes computation semantics: "compute attention on Q, K, V, produce O." The scratch buffer is an implementation detail of *how* the kernel executes — it depends on the op type, the tensor shapes, and potentially which backend runs it (a flash attention kernel needs much less scratch than the standard kernel). Different backends could have different scratch requirements for the same op. The graph shouldn't know about any of this.

### How it works

**Registry.** A scratch calculator registry maps OpType to a size function:

```python
ScratchCalculator = Callable[[list[tuple[int, ...]], tuple[int, ...]], int]
#                              input_shapes            output_shape     -> bytes
```

The signature takes only the tensor shapes — no graph coupling. Most ops return 0 (no scratch). Attention returns `S * S * sizeof(float)`.

**Planner.** During planning, `_compute_scratch()` queries the registry for each node. If scratch is needed, it creates a `Lifetime` entry with `born=step, dies=step` (single-step — the scratch is only alive during that one kernel call). These lifetimes go through the same `_assign_offsets` first-fit as regular intermediates. The arena is unified; scratch competes for space with regular tensors and benefits from the same overlap analysis.

The resulting offsets are split: regular tensor offsets go into `plan.offsets`, scratch offsets go into `plan.scratch: dict[int, tuple[int, int]]` (node ID → arena offset, size in bytes). The separation is because they need different binding at execution time — regular intermediates are bound by tensor name via the graph's tensor registry, while scratch is bound by node ID and passed as an extra kernel input.

**Executor.** When dispatching a node (per-op or compiled), the executor checks `plan.scratch` for that node. If present, it creates a flat `float32` arena view at the recorded offset and appends it to the kernel's input list. The kernel receives `[Q, K, V, scratch]` and doesn't care where the scratch came from — it's just another float pointer.

For the compiled C path, scratch pointers are baked into the `COpNode.inputs` array at compile time (at index `n_inputs`, then `n_inputs` is incremented). The numpy views are kept alive via a reference list on the `CompiledPlan` to prevent garbage collection.

### What about in-place scratch?

Some kernels can reuse an input buffer as workspace (because they're done reading it before they start writing). In those cases, the kernel just uses the pointer it already has — no planner involvement, no scratch allocation. This is an internal kernel optimization, invisible to the rest of the system. For attention specifically, we can't reuse Q/K/V as scratch since they're the wrong size (S×D vs S×S) and we need all three throughout the computation.

### Scratch vs parallelism tradeoff (open question)

The standard attention kernel can process (batch, head) slices in parallel via GCD `dispatch_apply`. Each concurrent slice needs its own S×S scratch buffer — you can't have two threads writing to the same scratch simultaneously. This creates a tension with the "planner owns all memory" principle.

**The current implementation cheats.** Slice 0 uses the planner-provided scratch buffer; slices 1+ `malloc` their own inside the kernel. This violates the design principle and adds per-call allocation overhead (though the comment claims ~50ns malloc vs ~500us work per slice).

**Option A: Parallel-aware scratch.** The scratch calculator allocates `batch_heads * S * S * sizeof(float)` instead of `S * S`. Each GCD thread offsets into its portion (`scratch + bh * S * S`). No malloc, planner owns everything. But this uses the same total memory as the *unfused* path's intermediate tensors (`scores`, `probs`, etc. are all `[B, H, S, S]`), eliminating the memory benefit of fusion. We've traded memory savings for parallelism — which may be the right tradeoff, but it's worth being explicit about.

**Option B: Sequential standard, parallel flash.** The standard kernel runs slices sequentially with a single S×S scratch buffer. The flash kernel runs in parallel — its scratch is only `B_r × B_c = 32 × 32 = 4KB` per slice, so even `batch_heads` copies is negligible. This preserves the standard kernel's memory advantage (one S×S buffer) while giving parallelism where it's cheap (flash). The downside is the standard kernel leaves performance on the table by not threading.

**Option C: Accept the malloc.** For kernels where the scratch per thread is small relative to the work per thread, the malloc/free cost is genuinely negligible. This pragmatically violates the "planner owns all memory" principle but keeps the code simple. The principle exists to avoid per-call allocation overhead, and if the overhead is demonstrably <0.01% of kernel time, the principle has been satisfied in spirit.

**Current decision: unresolved.** The flash kernel's tiny scratch makes this moot for the primary use case. For the standard kernel, Option B (sequential standard, parallel flash) is the cleanest architecture. Option C (accept the malloc) is the pragmatic fallback if sequential standard is too slow and we haven't wired up flash yet.

## Optimization Pass Pipeline

Passes are plain callables `(Graph) -> bool`. The pipeline is a list, run in order. `run_until_stable()` runs the pipeline repeatedly until no pass reports changes — useful when passes create opportunities for each other.

**Default ordering: MATMUL absorption → constant folding → pattern fusion → dead code elimination.** MATMUL absorption runs first, folding transposes into `transpose_b` flags and scalar MUL/DIV into `alpha` before constant folding can eagerly materialize them (see Weight Transpose Handling above). Pattern-based fusion runs next, claiming specific patterns via the `FusionPattern` registry with priority-based ordering. DCE cleans up dead nodes and unused constants last.

The pipeline is designed to be reconfigurable. The `run_pipeline` / `run_until_stable` infrastructure supports reordering, repeating, or swapping passes without changes.

**Pattern-based fusion** uses a registry of `FusionPattern` objects with priority-based grouping. Within each priority level, patterns are matched greedily longest-first. Patterns include optional validators (structural checks beyond op type matching) and attr builders (for carrying forward flags like transpose dims to the fused node). The sole-consumer constraint is critical for fusion correctness: an intermediate tensor can only be eliminated if exactly one node consumes it.

Currently registered patterns:
- **Priority 0:** `ATTENTION` (MATMUL→SOFTMAX→MATMUL, validated by transpose_b and last-axis softmax) and `BIAS_RELU` (ADD+RELU where ADD is a 1D bias broadcast). These run first to claim their nodes before lower-priority patterns.
- **Priority 1:** `MATMUL_ADD` (catches standalone bias adds not claimed by BIAS_RELU).

**Constant folding and DCE** work as a pair. Folding promotes a node's output to a constant and removes the node, but leaves the input constants in place. DCE then cleans up any input constants that have no remaining consumers. This avoids duplicating weight memory — the original weight is only freed when nothing else references it. Constant folding wraps results in `np.ascontiguousarray()` to guarantee C-compatible memory layout.

## Numpy Evaluators

`passes.py` maintains a registry of numpy evaluator functions (`NUMPY_EVALUATORS`) that compute any op on numpy arrays. Used by constant folding to evaluate ops at build time. These are separate from the numpy backend's in-place kernels — the evaluators return new arrays (what constant folding needs), while the backend kernels write into pre-allocated output buffers via numpy's `out=` parameter (what the executor needs). The two have different memory contracts and should not be conflated.

## Executor and Backends

**Plan vs Executor split.** The `ExecutionPlan` is the static, "compiled" artifact — graph, node order, arena size, offset assignments. The `Executor` is the runtime — it holds backends, allocates memory, binds buffers, and dispatches kernels. Mirrors the ONNX Runtime distinction between `InferenceSession` (runtime) and the optimized graph (static).

**Backend priority dispatch.** Backends are tried in order for each op — first match wins. `[c_backend, numpy_backend]` means C handles what it can, numpy catches the rest. Maps to ONNX Runtime's Execution Provider concept (CPU EP, CUDA EP, etc.).

**Kernel contract: write into pre-allocated output.** Kernels receive input buffers, a pre-allocated output buffer, and an attrs dict. They write the result into the output buffer — no allocations, no return values. This is the natural contract for C kernels and keeps the memory planner in full control.

**Arena reuse across calls.** The executor keeps its arena buffer between inference calls and only reallocates if a larger plan is encountered. Output tensors are copied out of the arena before returning, so the caller isn't holding views into memory that gets overwritten on the next call.

## C Operators

C kernels live in `csrc/ops.c`, compiled to a shared library via the `csrc/Makefile`. MATMUL uses `cblas_sgemm` from Accelerate (macOS) or OpenBLAS (Linux). Element-wise ops (ADD, RELU, DIV, SUB, MUL, EXP), reductions (MAX, SUM), SOFTMAX, TRANSPOSE, and RESHAPE are hand-written loops. Fused attention kernels (standard and flash) combine sgemm + softmax + sgemm into single calls with SIMD softmax and GCD threading on macOS (see Fused Attention Kernels section).

The C backend (`runtime/backends/c_backend.py`) loads the library via ctypes. Each wrapper extracts float pointers from numpy arrays using `.ctypes.data_as()` and passes dimensions as ints. Zero-copy: C operates directly on the same memory backing the numpy arrays.

**N-dim handling in wrappers.** C kernels are flat/2D (they take M, N, K etc.). The Python wrappers handle higher-dimensional inputs. Fast paths for common cases, general broadcast for everything else:
- **MATMUL ND×2D** (e.g., `[B, S, D] @ [D, N]^T` for linear layers): flatten A's batch dims into M, call a single sgemm.
- **MATMUL matching batch dims** (e.g., `[B, H, S, D] @ [B, H, D, S]`): loop over batch slices, one sgemm per slice.
- **MATMUL broadcast batch dims** (e.g., `[B, H, S, D] @ [H, D, S]`): odometer loop over the broadcast batch shape, computing A/B slice offsets via broadcast strides, one sgemm per slice.
- **ADD bias** (`[..., N] + [N]`): dedicated kernel with M×N loop.
- **ADD/SUB/MUL/DIV same shape**: flat kernel, no broadcast overhead.
- **ADD/SUB/MUL/DIV general broadcast**: `kernel_broadcast_binop` in C (see below).

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

**OpNode struct:** Each node in the plan becomes a fixed-size C struct containing the op type, input/output float pointers, output shape, and op-specific extras (e.g., K dimension for matmul). Uses fixed-size arrays (`MAX_INPUTS=8`, `MAX_DIMS=16`) to avoid dynamic allocation in C. The larger limits accommodate fused ops that carry multiple inputs and broadcast strides in the extras array.

**Compile once, execute many.** `compile_plan()` resolves all tensor names to pointers, precomputes dimensions, and records which struct slots correspond to graph inputs. `execute_compiled()` patches just the input pointers and makes one C call. Arena views, constant pointers, and output pointers are all stable between calls.

**Input patching.** Graph input tensors may feed multiple nodes. The compiled plan tracks a mapping of input name → list of (node_index, slot_index) pairs. Before each call, the executor patches those slots with the caller's input pointer. Everything else is unchanged.

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

The compiled C executor (`executor.c`) was originally written and tested with 2D MLP tensors. Moving to 3D+ transformer tensors exposed several implicit 2D assumptions in the C dispatch cases. These were all correct in the per-op path (where Python wrappers handle shape decomposition) but broke in the compiled path (where the C switch statement does it inline).

**Bugs found and fixed:**

1. **BLAS flag absorption too aggressive.** The `absorb_transpose_into_matmul` pass absorbed any TRANSPOSE feeding a MATMUL's B input, including head-reshape permutes that swap non-last dims (e.g., dim0=1, dim1=2 on 4D tensors). Fix: only absorb when `{dim0, dim1} == {ndim-2, ndim-1}`.

2. **TRANSPOSE dispatch only handled 2D.** The C kernel does a simple `[rows, cols]` transpose. For N-dim swapaxes (e.g., the `[B,S,H,dk] → [B,H,S,dk]` permutes in attention), we decompose the tensor into 5 logical regions around the two swapped dims: `[outer, A, middle, B, inner]`. The C code is a 4-nested loop with `memcpy` on the inner dimension.

3. **MATMUL ND×2D: batch pointer overflow.** The batch loop `b + i * b_stride` advanced B's pointer per batch, but for 2D weights there's only one B slice. Fix: detect ND×2D via `extra[2]` flag and flatten A's batch dims into M for a single sgemm.

4. **ADD dispatch assumed bias broadcasting.** `kernel_add` reads `b[j]` (broadcasting the last dim), which is correct for bias adds but wrong for element-wise adds like residual connections (`x + attn_out`). Fix: `extra[0]` flag distinguishes element-wise (flat loop) from bias broadcast.

**The per-op C backend wrappers already handled all these cases** — they check `.ndim`, `.shape == .shape`, etc. The lesson: compiled executors that bake shapes into structs need to explicitly encode the same dispatch logic that dynamic wrappers get for free. Every Python `if a.ndim > 2` needs a corresponding C dispatch path.

## Attention Fusion Strategy

Users can write attention in many ways — manual softmax decomposition, `F.softmax`, or `F.scaled_dot_product_attention`. All three should optimize down to the same fused ATTENTION op. Tracing all three through `torch.export` (see `explore_attention.py`) reveals:

- **Naive (manual softmax):** 11 ATen ops — matmul, div, max, getitem, sub, exp, sum, div, matmul, plus reshape/permute for head splitting
- **F.softmax:** 7 ATen ops — matmul, div, softmax, matmul, plus reshape/permute
- **SDPA:** 4 ATen ops — scaled_dot_product_attention, plus reshape/permute

Recognition strategy (multi-level pattern matching):

1. **SDPA (implemented):** The exporter maps `aten.scaled_dot_product_attention` directly to our ATTENTION op. No passes needed — it's a 1:1 op mapping like the other ATen handlers.

2. **F.softmax (implemented):** The key insight is that `absorb_into_matmul` folds the scalar DIV (by √d_k) into the first MATMUL's `alpha` parameter. This reduces the 4-node chain `MATMUL → DIV → SOFTMAX → MATMUL` to 3 nodes `MATMUL(alpha=1/√d_k) → SOFTMAX → MATMUL`, which fits the generic `FusionPattern` infrastructure. The validator checks `transpose_b=True` (Q @ K^T pattern) and softmax on the last axis. External inputs collected by `_apply_fusion` are naturally `[Q, K, V]` — exactly what the ATTENTION kernel expects.

3. **Naive (not yet implemented):** The manual softmax decomposition creates a DAG — the DIV output feeds both MAX and SUB, violating the sole-consumer constraint in `_try_match`. This needs either a specialized attention recognition pass or an extended pattern matcher that supports DAG patterns.

**N-dim batch support.** The ATTENTION op handles arbitrary leading batch dimensions. Q/K/V can be 3D `[BH, S, D]` or 4D `[B, H, S, D]` — the memory layout is byte-identical, so the C kernel doesn't care. The planner scratch calculator, executor `_fill_extras`, C backend wrapper, and compiled C dispatch all use `shape[-2]` for seq_len and `shape[-1]` for head_dim, with batch_heads derived from the remaining dimensions. This was needed because SDPA and F.softmax attention operate on 4D tensors after head splitting.

**ATTENTION is the atomic unit, not SOFTMAX.** We do NOT canonicalize manual softmax into a SOFTMAX op first. The flash attention algorithm interleaves softmax computation with matmuls tile-by-tile — it never computes a full standalone softmax. Canonicalizing to SOFTMAX would lose the structure flash attention needs.

**Multiple kernel implementations behind one op.** The graph-level ATTENTION op is backend-agnostic. The backend can dispatch to:
- A simple fused kernel that materializes the full S×S attention matrix (correct, easy to implement)
- A tiled/online flash attention kernel that never materializes the full matrix (better memory, better perf at long sequences)

Both produce identical results. The graph passes don't care which runs.

## Fused Attention Kernels

Two fused attention kernel implementations in `csrc/ops.c`, sharing the same interface. Both take Q, K, V as `[batch_heads, seq_len, head_dim]`, a scratch buffer, and produce the output. The graph-level ATTENTION op is backend-agnostic — the backend can dispatch to either kernel.

### Standard kernel (`kernel_attention`)

For each (batch, head) slice:
1. `S = Q @ K^T * scale` — single `cblas_sgemm` with `alpha = 1/sqrt(d_k)`, folding the scale into the matmul for free
2. `P = softmax(S)` — row-wise, in-place on the scratch buffer
3. `O = P @ V` — single `cblas_sgemm`

Scratch requirement: `S × S` floats per concurrent thread. Materializes the full attention matrix but eliminates all intermediate tensor allocations that the unfused graph would need.

### Flash kernel (`kernel_attention_flash`)

Tiled online-softmax attention that never materializes the full S×S matrix. For each (batch, head) slice, processes Q in blocks of B_r=32 rows:

1. For each K/V block of B_c=32 rows:
   - `S_tile = Q_block @ K_block^T * scale` via sgemm (B_r × B_c tile)
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

Both attention kernels parallelize across (batch, head) slices using Grand Central Dispatch (`dispatch_apply`). Each slice is completely independent — different regions of Q/K/V/output memory. The only shared resource is the scratch buffer, solved by having slice 0 reuse the caller-provided buffer and other slices malloc their own (negligible: ~50ns malloc vs ~500us of work per slice).

GCD is native to macOS (no build dependencies), manages its own thread pool, and handles work-stealing automatically. The `dispatch_apply` call blocks until all slices complete.

**Scratch buffer issue.** The current implementation mallocs per-thread scratch buffers inside the kernel, violating the "planner owns all memory" principle. See "Scratch vs parallelism tradeoff" in the Scratch Buffers section for the full analysis and options.

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

**The sgemm beta trick.** `cblas_sgemm` computes `C = alpha * A @ B + beta * C`. By pre-filling the output buffer with the broadcast bias and setting `beta=1.0`, sgemm adds the bias as part of its accumulation — eliminating the separate ADD kernel and its memory round-trip. The C dispatch in `executor.c` broadcasts the 1D bias `[N]` into every row of the `[M, N]` output, then calls `kernel_matmul_beta()` with `beta=1.0`.

**Why this runs after pattern fusion.** Pattern fusion should get first crack at ADD nodes — if there's an ADD+RELU chain, fusing them into FUSED_BIAS_RELU (one post-matmul pass doing both add and relu) is better than absorbing the ADD into sgemm (which would leave RELU as a separate dispatch). The beta trick is only advantageous for isolated bias adds where there's nothing else to chain with.

**Pipeline ordering rationale:** MATMUL → ADD → RELU becomes MATMUL → FUSED_BIAS_RELU via the `bias_relu` pattern. Only MATMUL → ADD (no RELU) becomes MATMUL_ADD via this pass. This gives the best result for both cases: the inner layers get fused bias+relu, and the final layer gets the sgemm beta trick.

**Carries forward attrs.** The fused node inherits the original MATMUL's attributes (notably `transpose_b`), so the CblasTrans optimization for `nn.Linear` weights is preserved through the fusion.

## Attention Scratch Allocation

**Planner-owned, not kernel-allocated.** The attention kernels (both standard and flash) use GCD `dispatch_apply` to parallelize across batch×head slices. Each concurrent thread needs its own scratch buffer for the attention matrix (or tile). Originally each thread called `malloc`/`free` — this violated the "planner owns all memory" principle.

**Fix: planner allocates `batch_heads × per_slice_scratch`.** The scratch calculator registered for `ATTENTION` computes `batch_heads × seq_len × seq_len × 4` bytes. Each GCD thread indexes into its region at `scratch + bh * scratch_per_slice`. Zero synchronization, zero allocation in the kernel.

**Memory tradeoff.** This over-allocates when `batch_heads >> num_cores`, since GCD only runs ~`num_cores` blocks concurrently. The ideal allocation is `min(batch_heads, num_cores)` scratch regions, but mapping iteration indices to scratch slots in GCD requires an atomic pool (GCD doesn't expose thread indices). We chose simplicity over optimal memory — the scratch has a single-step lifetime so the planner reclaims it immediately.

**Flash attention scratch is tiny regardless.** The flash kernel's per-slice scratch is `32×32×4` = 4KB (one tile, not the full S×S matrix), so even at `batch_heads=128` it's only 512KB total.

## C Build Structure

**Single source of truth for kernels.** `ops.c` contains all kernel implementations. `executor.c` contains only the dispatch loop and forward-declares the kernel functions it calls. Both files are compiled together into `libexecutor`. This eliminates the previous pattern of duplicating every kernel as a `static` function in `executor.c`.

Two shared libraries are built:
- `libruntime` (ops.c only) — loaded by `c_backend.py` for per-op dispatch
- `libexecutor` (executor.c + ops.c) — loaded by `executor.py` for compiled dispatch

## Non-Contiguous SLICE Breaks Compiled Execution

**The problem.** The compiled C execution path (`execute_compiled`) is supposed to be a single `execute()` call — the whole point is eliminating Python overhead between ops. But graphs with non-contiguous SLICE ops (notably GPT-2's QKV split) break this into segmented execution: run C nodes → pause → do a SLICE copy in Python → resume C nodes. This is a workaround for a missing C kernel, not a design decision.

**Why it exists.** SLICE ops come in two flavors:

- *Alias SLICEs* — contiguous chunks. Splitting along dim 0 (or using a flat byte_offset) produces chunks that are contiguous in memory. These are treated like RESHAPE: bind a pointer at compile time, skip at execution time. No problem.
- *Non-alias SLICEs* — non-contiguous chunks. Splitting along a non-leading dimension (e.g., GPT-2 projects to `[B, S, 3*D]` then splits along dim=2 to get Q, K, V each `[B, S, D]`). The three chunks are interleaved in memory, so you can't just take a pointer offset — you need a strided copy to produce a contiguous output.

The planner distinguishes these in `_find_reshape_aliases`: `dim is None or dim == 0` → alias, otherwise → non-alias with its own arena allocation. The per-op Python executor handles the strided copy via `np.ascontiguousarray(input_buf[slices])`. But the C executor's `OP_SLICE` dispatch is a no-op `break` — there's no C kernel for the strided copy.

**The workaround.** `compile_plan` pulls non-alias SLICEs out of the C execution order and records their positions. `execute_compiled` then runs C segments between them, executing each SLICE in Python via numpy before resuming. This adds Python re-entry overhead and complicates what should be a clean single-call path.

**The fix.** Implement a strided copy kernel in C (`kernel_slice`) that copies a non-contiguous slice into a contiguous output buffer, and dispatch it from `executor.c` like any other op. The segmented execution code in `compile_plan`/`execute_compiled` can then be removed, restoring the single-call invariant.

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

The planner treats RESHAPE outputs as aliases of their input (sharing the same arena offset via `_find_reshape_aliases`). They're excluded from the C execution order since there's no kernel to run.

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
