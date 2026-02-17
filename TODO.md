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
- [ ] GQA support (see above)
- [ ] Export Qwen3-0.6B body, run full pipeline, verify correctness against PyTorch
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
- [ ] GPT-2 end-to-end: our runtime vs PyTorch eager vs torch.compile vs ORT
- [ ] Head-to-head comparison table with analysis

## Known Limitations

- **No PERMUTE in compiled C dispatch.** `OP_PERMUTE` has no dispatch function — hits abort(). Safe because transformer permutations are TRANSPOSE (two-axis swaps).
- **Flash attention scratch over-allocation.** Scratch calculator always allocates standard-kernel-sized scratch (S^2 per slice). Flash only needs B_r x B_c per slice.
