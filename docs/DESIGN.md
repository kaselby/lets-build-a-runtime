# Design Notes

Technical decisions and rationale for the inference runtime. See `CLAUDE.md` for high-level principles and project structure.

## Architecture Summary

The runtime is an interpreter-style graph executor. A single graph representation flows through the entire pipeline — no multi-level IR lowering. Optimization passes mutate the graph in-place, the planner assigns memory, and the executor dispatches to C kernels.

```
torch.nn.Module
  → torch.export.export()           Capture fx.Graph with ATen ops
  → exporter.export_model()         Map ATen ops to our IR, load weights
  → passes.run_pipeline()           Optimize the graph in-place
  → planner.plan()                  Lifetime analysis, arena layout, scratch allocation
  → CompiledExecutor.compile(plan)  Build C struct array with resolved pointers
  → CompiledExecutor.run(inputs)    One ctypes call for the entire graph
```

The runtime is organized into subpackages: `exporter/` (ATen→IR mapping), `passes/` (optimization), `executor/` (dispatch), `backends/` (kernel implementations). A `session.py` module wraps the full pipeline into a single `InferenceSession` API mirroring ONNX Runtime's interface.

Two execution modes: `InterpretedExecutor` (Python loop calling backend kernels — useful for debugging and ablations) and `CompiledExecutor` (single ctypes call — the fast path). Both subclass a shared `Executor` ABC and produce identical results.

## Graph IR

**Node-centric, implicit edges.** Each node has an op type, a list of input tensor names, and a single output tensor name. Edges are implicit in these references. Tensor metadata (shape, dtype, buffer pointer) lives in `TensorInfo` objects in the graph's tensor registry.

**Single-output nodes.** Each node produces exactly one output tensor. Ops that conceptually produce multiple results (e.g., `torch.split`) are decomposed into individual SLICE nodes, each producing one output that aliases or copies a chunk of the source. This keeps lifetime analysis, memory planning, and graph traversal simple. See DESIGN_LOG_FULL.md for the full tradeoff analysis.

**Inputs and constants are tensors, not nodes.** Graph inputs and weights are entries in the tensor registry with no producer node. Every node in the graph is a real compute operation. This follows the ONNX convention and simplifies passes.

**Named tensors as connective tissue.** Tensor name strings are how everything connects: nodes reference inputs/outputs by name, the planner maps names to offsets, the executor binds names to buffers. Names are preserved from `torch.export` for debuggability.

**Graph mutation for passes.** `rewire_input()`, `remove_node()`, and `remove_tensor()` support in-place graph transformation. Consumer/producer indices are maintained incrementally. Topological order is cached after validation and invalidated on mutation.

## Op Registry (`ops.py`)

All per-op metadata is centralized in a single `OP_REGISTRY: dict[OpType, OpDef]`. Each `OpDef` bundles:

- **`evaluator`** — numpy implementation for constant folding and fallback execution
- **`scratch`** — scratch buffer size calculator for the planner
- **`alias`** — `bool | Callable[[Node], bool]`: whether the op's output shares input memory. RESHAPE is always an alias (`True`). SLICE is conditionally an alias — contiguous slices (dim=0) are zero-copy, non-contiguous slices (dim>0) dispatch to a C kernel.
- **`inplace`** — whether the kernel can safely write into its first input's buffer (all elementwise ops)
- **`extras`** — packs op-specific parameters into the `COpNode.extra[]` array for compiled C dispatch

Adding a new op is a single registry entry. The planner, executor, constant folder, and compiled dispatch all read from this one source of truth.

## Exporter

Maps `torch.export`'s fx.Graph to our IR via a handler registry (`ATEN_HANDLERS`). Split into `exporter/exporter.py` (core three-phase pipeline: placeholders → compute → outputs) and `exporter/handlers.py` (op handlers + utilities).

**The exporter emits primitives; passes optimize.** Handlers produce fine-grained ops without embedding optimization knowledge. The optimization passes handle the rest:

- **`aten.linear`** → `TRANSPOSE(weight) + MATMUL + ADD`. The absorption pass folds the transpose into `transpose_b=True`.
- **`aten.addmm`** (Conv1D) → `MATMUL + ADD`. The absorption pass pre-transposes constant weights and sets `transpose_b=True`.
- **`aten.permute`** → `TRANSPOSE` if it's a two-axis swap, `PERMUTE` if general.
- **Binary ops** (`add`, `div`, `sub`, `mul`) → scalar stored as an attr when the second arg isn't a tensor.
- **`aten.scaled_dot_product_attention`** → `ATTENTION` directly.
- **`aten.layer_norm`**, **`aten._softmax`** → kept as compound ops for efficient single-kernel implementations.
- **`aten.split.Tensor` + `operator.getitem`** → split is a no-op; each getitem emits a `SLICE` node by self-serving split metadata from the FX graph. No shared mutable state between handlers.
- **`aten.view`/`aten.reshape`** → `RESHAPE` (zero-copy alias).

Handler utilities (`_output_meta`, `_emit`, `_get_axis`) and factories (`_make_simple_handler`, `_make_binary_handler`, `_make_reduction_handler`) eliminate per-handler boilerplate.

## Optimization Passes

Passes are `(Graph) -> bool` callables. The default pipeline:

### 1. BLAS flag absorption (`absorb_into_matmul`)

Decomposed into focused helpers, each handling one absorption pattern:

- **`_absorb_transpose_b`:** If the B input comes from a TRANSPOSE swapping the last two dims, rewire to read the original tensor with `transpose_b=True`.
- **`_pretranspose_constant_b`:** For constant B inputs that aren't transposed (e.g., Conv1D weights in `[in, out]` layout), pre-transpose the weight buffer and set `transpose_b=True` for the fast CblasTrans path.
- **`_absorb_input_scalars` / `_absorb_output_scalar`:** Fold scalar MUL/DIV on inputs or output into MATMUL's `alpha` parameter.

Must run before constant folding — otherwise folding eagerly materializes transposes, destroying absorption patterns.

### 2. Constant folding (`constant_fold`)

Evaluates nodes whose inputs are all constants using the `OP_REGISTRY` evaluator. The result becomes a new constant tensor and the node is removed. Wraps results in `np.ascontiguousarray()` for C-compatible layout. DCE cleans up dead input constants.

Infrastructure ops (CAST, EXPAND, SLICE_TENSOR, CMP_NE, etc.) are "fold-only" — they exist solely for subgraphs like GPT-2's causal mask generation that are fully constant-foldable. Both executors raise `RuntimeError` if a fold-only op reaches execution. The C dispatch table enforces this naturally via `DISPATCH_TABLE_SIZE = 100` (fold-only ops have enum values >= 100).

### 3. Causal mask absorption (`absorb_causal_mask`)

Detects causal attention masks (both bool lower-triangular and float upper-triangular-negative-infinity formats) via a unified `_is_causal_mask` function. When a mask feeding an ATTENTION node is identified as causal, it's replaced with a `causal=True` attribute on the ATTENTION node and the mask input is removed.

### 4. Pattern fusion (`fuse`)

Registry of `FusionPattern` objects for linear chain patterns, matched greedily by priority:

- **Priority 0:** `ATTENTION` (MATMUL→SOFTMAX→MATMUL), `CAUSAL_ATTENTION` (same chain with `causal=True`), and `BIAS_RELU` (ADD+RELU where ADD is a 1D bias broadcast).
- **Priority 1:** `MATMUL_ADD` (standalone bias adds not claimed by BIAS_RELU).

The sole-consumer constraint ensures correctness: an intermediate tensor can only be eliminated if exactly one node consumes it.

### 5. DAG fusion (`fuse_dags`)

Handles non-linear subgraph patterns that the chain-based `FusionPattern` can't match. Currently recognizes the GELU tanh approximation: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` — an 8-node DAG where `x` fans out to multiple paths that reconverge. Replaces the subgraph with a single `GELU` op. Extensible to SiLU, GeGLU, etc.

### 6. Dead code elimination (`eliminate_dead_code`)

Removes nodes whose outputs have no consumers and aren't graph outputs. Repeats until stable. Also cleans up constants that have no remaining consumers.

## Memory Planner

**Weights live outside the arena.** The arena is exclusively for intermediate activations and scratch workspace. Weights are permanent, intermediates are ephemeral. Mixing them would prevent reuse.

**Unified alias and in-place sharing.** A single pass in `_compute_lifetimes` handles both alias ops (RESHAPE, contiguous SLICE) and in-place reuse (elementwise ops writing into a dying input) through the same `memory_root` mechanism. For each node in execution order: alias ops unconditionally share their input's memory; in-place-eligible ops (dying input, same size) conditionally share; everything else gets a new lifetime. `get_root()` follows both chains, so chained in-place (e.g., EXP → RELU) works naturally.

**Memory-aware topological ordering.** The execution order uses a Kahn's algorithm variant with a lazy-rescore max-heap. Priority is `freed_bytes - output_alloc_bytes + inplace_bonus` — a net memory delta heuristic that accounts for both freeing inputs and allocating outputs. Stale scores from multi-consumer tensors are corrected lazily on pop.

**First-fit offset assignment.** Tensors are processed in birth order. For each, only temporally-overlapping allocations are considered, and the lowest non-conflicting offset is chosen.

**Scratch buffers.** `OP_REGISTRY` scratch calculators size per-op workspace. Scratch gets single-step lifetimes and goes through the same first-fit allocation as regular intermediates. The executor passes scratch as an extra kernel input, transparent to the graph IR.

## Executor and Backends

**Executor ABC.** `compile(plan)` does one-time preparation; `run(inputs)` does per-call inference. Shared arena management and buffer binding in the base class.

- **`CompiledExecutor`:** Builds a `COpNode` struct array with all pointers pre-resolved. `run()` patches input pointers and makes one C call. Alias ops (RESHAPE, contiguous SLICE) are bound at compile time and skipped during execution. Non-contiguous SLICEs dispatch to a C kernel like any other op — no segmented execution.
- **`InterpretedExecutor`:** Python dispatch loop with backend chain (C first, numpy fallback). Same interface, useful for debugging and ablation benchmarking.

**Backend priority dispatch.** Backends are tried in order for each op — first match wins. `[CBackend, NumpyBackend]` maps to ONNX Runtime's Execution Provider concept.

**Kernel contract.** `(inputs: list[ndarray], output: ndarray, attrs: dict) -> None`. Kernels write into a pre-allocated output buffer. No allocations, no return values.

**Extras encoding.** The C executor passes op-specific parameters via a fixed-size `int extra[MAX_DIMS]` array. Floats are bit-cast into ints. Per-op `extras` packers on `OpDef` handle this — no monolithic if/elif chain.

## C Layer

Kernels in `csrc/ops.c`, dispatch in `csrc/executor.c`. Two shared libraries:
- `libruntime` (ops.c only) — loaded by `CBackend` for per-op dispatch
- `libexecutor` (executor.c + ops.c) — loaded by `CompiledExecutor` for compiled dispatch

**Function pointer dispatch table.** `executor.c` uses a `dispatch_fn` array with C99 designated initializers instead of a switch statement. Each op is a self-contained `dispatch_xxx(OpNode*)` function. Three inline helpers (`total_elements`, `extra_float`, `leading_dims`) eliminate repeated patterns. Adding a new op: add enum value, write dispatch function, add one table line.

**Range-based enum numbering.** OpType values are grouped: 10-19 unary, 20-29 binary, 30-39 reductions, 40-49 BLAS, 50-59 shape, 60-69 normalization, 70-79 fused, 100+ fold-only. Both Python and C use the same `enum OpType`.

**MATMUL** uses `cblas_sgemm` from Accelerate (macOS) or OpenBLAS (Linux). Supports batched operation, `transpose_b`, and `alpha`/`beta` parameters.

**Element-wise ops** (ADD, RELU, DIV, SUB, MUL, EXP, TANH, GELU) are flat loops with scalar variants for compiled dispatch.

**Reductions** (MAX, SUM) decompose into `[outer, axis_size, inner]` and reduce along the middle dimension.

**SOFTMAX** uses Accelerate SIMD functions on macOS (`vDSP_maxv`, `vvexpf`, `vDSP_sve`, `vDSP_vsdiv`) for ~3-4x speedup over scalar `expf`.

**SLICE** (`kernel_slice`) performs strided copies for non-contiguous slices using `[outer, orig_dim_size, start, slice_len, inner]` parameters from the extras array.

**Broadcasting** for binary ops uses `kernel_broadcast_binop` with coordinate-increment-with-carry (odometer pattern). Fast paths for same-shape and bias-broadcast avoid the general kernel.

**N-dim TRANSPOSE** decomposes into `[outer, A, middle, B, inner]` around the two swapped dimensions with a 4-nested loop and `memcpy` on the inner dimension.

## Fused Attention

Two kernel implementations sharing the same interface (`kernel_attention` and `kernel_attention_flash`). Both take `[Q, K, V, scratch]` as `[batch_heads, seq_len, head_dim]`. Both support a `causal` flag for causal masking (applied as `-INFINITY` to above-diagonal entries in the attention matrix before softmax).

**Standard kernel:** `S = Q @ K^T * scale` → causal mask → `P = softmax(S)` → `O = P @ V`. Materializes the full S×S attention matrix. Scratch: `batch_heads × S × S × sizeof(float)`.

**Flash kernel:** Tiled online-softmax (B_r=B_c=32). Never materializes the full S×S matrix. Scratch: `batch_heads × B_r × B_c × sizeof(float)`. Trades compute for memory — crossover where flash beats standard is at long sequences where S² blows the cache.

**GCD threading** on macOS: `dispatch_apply` across batch×head slices. Each slice gets its own scratch region (planner allocates `batch_heads` regions).

**Attention recognition** works at multiple levels:
1. **SDPA:** Exporter maps `aten.scaled_dot_product_attention` directly to ATTENTION.
2. **F.softmax:** Absorption folds scale into alpha, then fusion matches MATMUL→SOFTMAX→MATMUL.
3. **Naive (manual softmax decomposition):** Not yet implemented — DAG pattern violates the sole-consumer constraint.

## Key Design Decisions

### Weight transpose handling

`nn.Linear` stores weights as `[out, in]` and computes `input @ weight.T`. We keep weights in their original layout and pass `CblasTrans` to sgemm, rather than pre-transposing at load time.

**Why:** At large dimensions, `CblasTrans` on `[N×K]` row-major reads rows during tile packing (stride-1, sequential). `CblasNoTrans` on pre-transposed `[K×N]` reads columns (stride-N, cache-hostile). The effect is 1.4-1.5x at dim=2048-4096 on Apple M4 Max.

### Pattern-matched fusion over interpreter fusion

We tried a general element-wise fusion interpreter (micro-op switch loop with broadcast tracking). It was 1.15-1.87x *slower* than unfused dispatch — the per-element interpreter overhead exceeded the memory bandwidth savings. We use pattern-matched bespoke kernels instead: specific patterns backed by hand-written C kernels. Real runtimes solve the general case with JIT codegen — outside our scope. See DESIGN_LOG_FULL.md for the full benchmark.

### MATMUL+ADD fusion via sgemm beta

`cblas_sgemm` computes `C = alpha * A @ B + beta * C`. By pre-filling the output with the broadcast bias and setting `beta=1.0`, sgemm adds the bias as part of its accumulation — no separate ADD kernel or memory round-trip. Only applies to standalone bias adds; ADD+RELU chains fuse into BIAS_RELU instead.

### Exporter emits primitives, passes optimize

The exporter doesn't embed optimization knowledge. `aten.linear` emits `TRANSPOSE + MATMUL + ADD`, not `MATMUL(transpose_b=True) + ADD`. The absorption pass handles the optimization. This keeps the exporter simple (one handler per ATen op, mechanical mapping) and the optimization logic centralized in passes where it can be tested and reasoned about independently.

## Performance

Benchmarked on Apple M4 Max.

### MLP (3-layer, Linear→ReLU→Linear→ReLU→Linear)

| Config    | PyTorch  | Compiled C | +fusion  | fused/PT |
|-----------|----------|------------|----------|----------|
| 1×512     | 20 us    | 13 us      | 13 us    | 0.63     |
| 32×512    | 105 us   | 103 us     | 109 us   | 1.04     |
| 128×512   | 395 us   | 196 us     | 195 us   | 0.49     |
| 1×2048    | 333 us   | 292 us     | 289 us   | 0.87     |
| 32×2048   | 1.07 ms  | 832 us     | 848 us   | 0.79     |

**The compiled C executor eliminates all Python dispatch overhead.** This is the single biggest lever — swinging batch=1 from 3.5x slower (per-op C) to 1.6x faster (compiled).

### Transformer (single-layer block, naive F.softmax attention)

| Config       | PT naive  | PT SDPA   | +fusion   | fused/PT | fused/SDPA |
|--------------|-----------|-----------|-----------|----------|------------|
| 1×16×64      | 194 us    | 179 us    | 21 us     | 0.11     | 0.12       |
| 4×16×64      | 259 us    | 201 us    | 64 us     | 0.25     | 0.32       |
| 4×64×128     | 687 us    | 460 us    | 335 us    | 0.49     | 0.73       |
| 4×128×256    | 2.20 ms   | 1.50 ms   | 1.20 ms   | 0.54     | 0.80       |

**We beat both PyTorch baselines at every config.** At small sizes, dispatch overhead elimination gives ~10x over eager PyTorch. At the largest config, fused attention compounds with MLP fusion for 0.54x naive. Beating SDPA validates the project thesis: our compiled executor's overhead savings on the ~20 non-attention ops overcome SDPA's faster attention kernel.

## Known Limitations

- **No PERMUTE in compiled C dispatch.** General N-dim permutations are unhandled in `executor.c`. Currently safe because transformer permutations are classified as TRANSPOSE.
- **Flash attention scratch over-allocation.** The scratch calculator always allocates standard-kernel-sized scratch (S² per slice). Flash needs only 32×32 per slice.
- **Naive attention recognition.** Manual softmax decomposition creates a DAG pattern that the linear chain matcher can't handle.
- **RESHAPE of graph inputs in compiled dispatch.** `torch.export` can emit RESHAPE of a graph input (e.g., HuggingFace GPT-2's `view(input_ids, [-1, 16])`). The compiled executor binds RESHAPE outputs at compile time, but graph input buffers aren't available yet. See DESIGN_LOG_FULL.md for the analysis and options.
- **No quantization.** Q/DQ insertion and INT8 weight quantization were planned as stretch goals.
