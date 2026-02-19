# TODO

## In Progress

### GQA (Grouped Query Attention)
Support models where Q has more heads than K/V (e.g., Qwen3: 16Q/8KV).

- [x] C kernel: `group_size` parameter, stride K/V by `bh / group_size`
- [x] Dispatcher: read `group_size` from extras
- [x] Extras packer: append `group_size`
- [x] C backend wrapper: pass `group_size`
- [x] Numpy backend: `np.repeat` K/V when `group_size > 1`
- [x] Exporter: detect Q/K shape mismatch in `_handle_sdpa`, infer `group_size`
- [x] Fusion pattern: `[RESHAPE, EXPAND, RESHAPE, ATTENTION] → ATTENTION(group_size=N)` for legacy expand-style GQA
- [x] Tests: GQA attention correctness (both backends, both executors, fusion pattern)

## Qwen3 End-to-End

All individual ops are implemented. Need integration and verification.

- [x] New unary ops: RSQRT, SILU, NEG, COS, SIN (C kernels + backends + handlers)
- [x] CAT op (segmented memcpy kernel + handler)
- [x] `aten.to.dtype` handler
- [x] RMSNorm DAG fusion (pow→sum→div→add→rsqrt→mul→mul → RMSNORM)
- [x] SiLU DAG fusion (neg→exp→add→div → SILU)
- [x] Gated activation fusion (SILU/GELU + MUL → GATED_ACT, with/without bias)
- [x] GQA support (see above)
- [x] Export Qwen3-0.6B body, run full pipeline, verify correctness against PyTorch
- [ ] Benchmark: our runtime vs PyTorch eager vs ORT

## Architectural Improvements

### Pass infrastructure rework
Extract the chain fusion pattern matching engine (`_try_match`, `_apply_fusion`, priority sweep) into reusable infrastructure. Currently only usable via `fuse()`. Goal: any pass at any pipeline stage can define and apply patterns.

Candidates for migration:
- `absorb_into_matmul`: TRANSPOSE→MATMUL (set `transpose_b`), scalar MUL/DIV→MATMUL (set `alpha`)
- Ordering constraint: these run before constant folding, so the engine needs to be invocable at arbitrary pipeline stages, not just in `fuse()`

## Stretch Goals

### GPU / Triton
- [ ] Triton kernel implementations (matmul, elementwise, reductions, softmax, attention)
- [ ] GPU backend class (same Backend protocol)
- [ ] GPU memory model (arena on GPU, weight transfer)
- [ ] Flash attention via Triton
- [ ] Benchmark: GPU vs CPU, vs PyTorch CUDA, vs torch.compile

### Quantization
- [ ] Q/DQ node insertion pass
- [ ] INT8 weight quantization (per-tensor or per-channel)
- [ ] Mixed-precision dtype tracking through planner and executor
- [ ] Quantized C kernels (INT8 matmul)
- [ ] Accuracy vs fp32, performance measurements

### Benchmarks
- [x] GPT-2 end-to-end: our runtime vs PyTorch eager vs torch.compile vs ORT
- [ ] Head-to-head comparison table with analysis

## Cleanup

### Profile broadcast binop path in compiled executor
The new broadcast dispatch path (mode 2 for MUL/SUB/DIV, mode 3 for ADD) uses `kernel_broadcast_binop` with per-element coordinate-increment-with-carry — much slower than the flat kernel. Verify this isn't on the critical path for Qwen3 (RoPE cos/sin multiplies hit it). Consider a specialized fast path for the common "broadcast one dim" case if it shows up in profiles.

### Verify exporter changes don't regress non-dynamic and GPT-2 dynamic exports
The exporter now passes `strict=False` + `prefer_deferred_runtime_asserts_over_guards=True` when `dynamic_dims` is specified, and skips ops with `val is None` in metadata. Verify these changes don't affect: (1) static exports (no dynamic_dims), (2) GPT-2 dynamic shape export/rebind, (3) any edge cases where `val is None` might incorrectly skip a real compute op.

### Audit all `sys.maxsize` / INT64_MAX sentinel guards
Torch uses `sys.maxsize` (9223372036854775807) as "to the end" in slice ops. With `strict=False` + deferred runtime asserts, these can appear as SymInts rather than plain ints, bypassing our guards. Do a full pass through the exporter handlers (`handlers.py`) and shape inference (`ops.py`) to ensure all end-index paths clamp or detect sentinels correctly — not just the full-dimension no-op case.

## Known Limitations

- **No PERMUTE in compiled C dispatch.** `OP_PERMUTE` has no dispatch function — hits abort(). Safe because transformer permutations are TRANSPOSE (two-axis swaps).
