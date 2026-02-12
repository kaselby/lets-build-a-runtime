# Design Notes

Technical decisions and rationale for the inference runtime. See `CLAUDE.md` for high-level principles and project structure.

## Architecture Summary

The runtime is an interpreter-style graph executor. A single graph representation flows through the entire pipeline — no multi-level IR lowering. Optimization passes mutate the graph in-place, the planner assigns memory, and the executor dispatches to C kernels.

```
torch.nn.Module
  → torch.export.export()         Capture fx.Graph with ATen ops
  → exporter.export_model()       Map ATen ops to our IR, load weights
  → passes.run_pipeline()         Optimize the graph in-place
  → planner.plan()                Lifetime analysis, arena layout, scratch allocation
  → executor.compile_plan()       Build C struct array with resolved pointers
  → executor.execute_compiled()   One ctypes call for the entire graph
```

Two execution modes: per-op dispatch (Python loop calling backend kernels — useful for debugging) and compiled C dispatch (single ctypes call — the fast path). Both produce identical results.

## Graph IR

**Node-centric, implicit edges.** Each node has an op type, a list of input tensor names, and a single output tensor name. Edges are implicit in these references. Tensor metadata (shape, dtype, buffer pointer) lives in `TensorInfo` objects in the graph's tensor registry.

**Inputs and constants are tensors, not nodes.** Graph inputs and weights are entries in the tensor registry with no producer node. Every node in the graph is a real compute operation. This follows the ONNX convention and simplifies passes — no need to special-case "virtual" input nodes.

**Named tensors as connective tissue.** Tensor name strings are how everything connects: nodes reference inputs/outputs by name, the planner maps names to offsets, the executor binds names to buffers. Names are preserved from `torch.export` for debuggability.

**Graph mutation for passes.** `rewire_input()`, `remove_node()`, and `remove_tensor()` support in-place graph transformation. Consumer/producer indices are maintained incrementally. Topological order is cached after validation and invalidated on mutation.

## Exporter

Maps `torch.export`'s fx.Graph to our IR via a handler registry (`ATEN_HANDLERS`). Each ATen op gets a handler function that creates the corresponding node(s). Key mappings:

- **`aten.linear(input, weight, bias?)`** → `MATMUL(input, weight, transpose_b=True)` + `ADD(bias)`. Keeps weight in its original `[out, in]` layout for the fast `CblasTrans` path (see Weight Transpose Handling below).
- **`aten.permute`** → `TRANSPOSE` if it's a two-axis swap, `PERMUTE` if general. Separate ops because two-axis swaps have optimization opportunities (BLAS flags, dedicated C kernel) that general permutations don't.
- **Binary ops** (`add`, `div`, `sub`, `mul`) → scalar stored as an attr when the second arg isn't a tensor. Avoids materializing a full constant tensor for scalar operations.
- **`aten.scaled_dot_product_attention`** → `ATTENTION` directly. No fusion pass needed.
- **`aten.layer_norm`**, **`aten._softmax`** → kept as compound ops rather than decomposed. Enables efficient single-kernel implementations and preserves structure for attention fusion.
- **`aten.view`/`aten.reshape`** → `RESHAPE`. The planner treats these as zero-copy aliases.

## Optimization Passes

Passes are `(Graph) -> bool` callables. The pipeline runs in order; `run_until_stable()` iterates until no pass reports changes. The default pipeline:

### 1. BLAS flag absorption (`absorb_into_matmul`)

For each MATMUL node, absorbs adjacent ops into sgemm parameters:

- **Transpose absorption (backward):** If the B input comes from a TRANSPOSE swapping the last two dims, rewire to read the original tensor with `transpose_b=True`. This keeps `nn.Linear` weights in their original `[out, in]` layout, which is significantly faster at large dimensions due to BLAS packing access patterns (see Weight Transpose Handling).
- **Scalar absorption (backward/forward):** If either input comes from a scalar MUL/DIV, or the sole consumer is a scalar MUL/DIV, fold the scalar into the MATMUL's `alpha` parameter. Matmul is bilinear: `s*(A@B) = (s*A)@B`.

Must run before constant folding — otherwise folding eagerly materializes transposes and scalars, destroying the absorption patterns.

### 2. Constant folding (`constant_fold`)

Evaluates nodes whose inputs are all constants using a numpy evaluator registry. The result becomes a new constant tensor and the node is removed. Wraps results in `np.ascontiguousarray()` to guarantee C-compatible layout. Input constants are left in place; DCE cleans up any that become dead.

### 3. Pattern fusion (`fuse`)

Registry of `FusionPattern` objects matched greedily by priority. Lower priority number = matched first. Within a priority level, longer patterns are tried first.

- **Priority 0:** `ATTENTION` (MATMUL→SOFTMAX→MATMUL, validated by `transpose_b` and last-axis softmax) and `BIAS_RELU` (ADD+RELU where ADD is a 1D bias broadcast).
- **Priority 1:** `MATMUL_ADD` (standalone bias adds not claimed by BIAS_RELU).

The sole-consumer constraint ensures correctness: an intermediate tensor can only be eliminated if exactly one node consumes it.

The attention fusion works because `absorb_into_matmul` already folded the scalar `1/sqrt(d_k)` into the first MATMUL's alpha, reducing the 4-node chain (MATMUL→DIV→SOFTMAX→MATMUL) to 3 nodes that match the pattern.

### 4. Dead code elimination (`eliminate_dead_code`)

Removes nodes whose outputs have no consumers and aren't graph outputs. Repeats until stable. Also cleans up constants that have no remaining consumers.

## Memory Planner

**Weights live outside the arena.** The arena is exclusively for intermediate activations and scratch workspace. Weights are permanent (alive across all inference calls), intermediates are ephemeral. Mixing them would prevent reuse.

**Lifetime analysis.** For each intermediate tensor, `born` = step when its producer runs, `dies` = step when its last consumer runs. Inputs and constants are excluded.

**First-fit offset assignment.** Tensors are processed in birth order. For each, only temporally-overlapping allocations are considered, and the lowest non-conflicting offset is chosen.

**RESHAPE as zero-copy alias.** RESHAPE outputs share their input's arena memory. The planner builds an alias map (following chains), assigns no arena space for aliases, and extends the root tensor's lifetime to cover all alias consumers. Six reshapes per attention layer, all zero-cost.

**Scratch buffers.** A scratch calculator registry maps OpType to a size function. Scratch gets single-step lifetimes (`born=dies=step`) and goes through the same first-fit allocation as regular intermediates. The executor passes scratch as an extra kernel input, transparent to the graph IR. Currently only ATTENTION has scratch (one S×S matrix per batch×head slice for GCD parallel execution).

## Executor and Backends

**Backend priority dispatch.** Backends are tried in order for each op — first match wins. `[CBackend, NumpyBackend]` means C handles what it can, numpy catches the rest. Maps to ONNX Runtime's Execution Provider concept.

**Kernel contract.** `(inputs: list[ndarray], output: ndarray, attrs: dict) -> None`. Kernels write into a pre-allocated output buffer. No allocations, no return values. The planner owns all memory.

**Compiled C executor.** Python builds a flat `OpNode` struct array with all buffer pointers and dimensions pre-resolved. A single `execute()` C call dispatches the whole graph. Arena views, constant pointers, and output pointers are stable between calls — only graph input pointers are patched per inference.

**Input patching.** Graph inputs may feed multiple nodes. The compiled plan tracks input name → list of (node_index, slot_index). Before each call, those slots are patched with the caller's input pointer.

## C Operators

Kernels in `csrc/ops.c`, dispatch loop in `csrc/executor.c`. Two shared libraries:
- `libruntime` (ops.c only) — loaded by `c_backend.py` for per-op dispatch
- `libexecutor` (executor.c + ops.c) — loaded by `executor.py` for compiled dispatch

**MATMUL** uses `cblas_sgemm` from Accelerate (macOS) or OpenBLAS (Linux). Supports batched operation, `transpose_b`, and `alpha`/`beta` parameters. Three entry points: `kernel_matmul` (alpha=1, beta=0), `kernel_matmul_beta` (alpha=1, custom beta), `kernel_matmul_ab` (custom alpha and beta).

**Element-wise ops** (ADD, RELU, DIV, SUB, MUL, EXP) are flat loops. Scalar variants (`kernel_add_scalar`, etc.) handle the scalar-attr case in compiled dispatch.

**Reductions** (MAX, SUM) decompose the tensor into `[outer, axis_size, inner]` and reduce along the middle dimension.

**SOFTMAX** uses Accelerate SIMD functions on macOS (`vDSP_maxv`, `vvexpf`, `vDSP_sve`, `vDSP_vsdiv`) for ~3-4x speedup over scalar `expf`. Falls back to scalar loops on other platforms.

**LAYERNORM** is a two-pass kernel (mean, then variance), operating along the last axis.

**Broadcasting** for binary ops uses `kernel_broadcast_binop` — a single C kernel with coordinate-increment-with-carry (odometer pattern). The Python wrapper computes broadcast strides (0 for broadcast dims, normal otherwise). Fast paths for same-shape and bias-broadcast patterns avoid the general kernel.

**N-dim TRANSPOSE** in compiled dispatch decomposes the tensor into `[outer, A, middle, B, inner]` around the two swapped dimensions and uses a 4-nested loop with `memcpy` on the inner dimension. The per-op C backend falls back to numpy for N-dim transposes.

## Fused Attention

Two kernel implementations sharing the same interface (`kernel_attention` and `kernel_attention_flash`). Both take `[Q, K, V, scratch]` as `[batch_heads, seq_len, head_dim]`.

**Standard kernel:** `S = Q @ K^T * scale` → `P = softmax(S)` → `O = P @ V`. Materializes the full S×S attention matrix. Scratch requirement: `batch_heads × S × S × sizeof(float)`.

**Flash kernel:** Tiled online-softmax (B_r=B_c=32). Never materializes the full S×S matrix. For each query block, iterates over key/value blocks, maintaining running max and sum for numerically stable online softmax. Scratch requirement: `batch_heads × B_r × B_c × sizeof(float)` (much smaller). Trades compute (rescaling corrections, many small sgemms) for memory. Crossover where flash beats standard is at long sequences where S² blows the cache.

**GCD threading** on macOS: `dispatch_apply` across batch×head slices. Each slice gets its own scratch region (planner allocates `batch_heads` regions). The scratch calculator always allocates for the standard kernel's requirements, so the flash kernel over-allocates but works correctly.

**Attention recognition** works at multiple levels:
1. **SDPA:** Exporter maps `aten.scaled_dot_product_attention` directly to ATTENTION.
2. **F.softmax:** `absorb_into_matmul` folds the scale into alpha, then fusion matches the 3-node MATMUL→SOFTMAX→MATMUL pattern.
3. **Naive (manual softmax decomposition):** Not yet implemented — the DAG pattern violates the sole-consumer constraint in the linear chain matcher.

## Key Design Decisions

### Weight transpose handling

`nn.Linear` stores weights as `[out, in]` and computes `input @ weight.T`. We keep weights in their original layout and pass `CblasTrans` to sgemm, rather than pre-transposing at load time.

**Why:** At large dimensions, `CblasTrans` on `[N×K]` row-major reads rows of the stored matrix during tile packing (stride-1, sequential). `CblasNoTrans` on the pre-transposed `[K×N]` reads columns (stride-N, cache-hostile). The effect is 1.4-1.5x at dim=2048-4096 on Apple M4 Max.

### Pattern-matched fusion over interpreter fusion

We tried a general element-wise fusion interpreter (micro-op switch loop with broadcast tracking). It was 1.15-1.87x *slower* than unfused dispatch — the per-element interpreter overhead exceeded the memory bandwidth savings. We use pattern-matched bespoke kernels instead: specific patterns (BIAS_RELU, MATMUL_ADD, ATTENTION) backed by hand-written C kernels. This gives bespoke speed for the patterns that matter with minimal machinery. Real runtimes solve the general case with JIT codegen (MLIR, TVM, Triton) — outside our scope.

### MATMUL+ADD fusion via sgemm beta

`cblas_sgemm` computes `C = alpha * A @ B + beta * C`. By pre-filling the output with the broadcast bias and setting `beta=1.0`, sgemm adds the bias as part of its accumulation — no separate ADD kernel or memory round-trip. This only applies to standalone bias adds; ADD+RELU chains fuse into BIAS_RELU instead (one post-matmul pass doing both add and relu).

### Compiled executor extras encoding

The C executor passes op-specific parameters via a fixed-size `int extra[MAX_DIMS]` array. Floats (alpha, eps, scalar values) are bit-cast into ints via `union { int i; float f; }` / `struct.pack('f', ...)`. Zero in `extra[3]` for MATMUL alpha means "default 1.0" — avoids needing a separate flag.

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

- **No PERMUTE in compiled C dispatch.** General N-dim permutations are unhandled in `executor.c`. Currently safe because transformer permutations are classified as TRANSPOSE, but would silently fail for true N-dim permutations.
- **Flash attention scratch over-allocation.** The scratch calculator always allocates standard-kernel-sized scratch (S² per slice). Flash needs only 32×32 per slice.
- **Naive attention recognition.** Manual softmax decomposition creates a DAG pattern that the linear chain matcher can't handle.
- **No quantization.** Q/DQ insertion and INT8 weight quantization were planned as stretch goals.
