# Project Status

## Done

### Core Runtime (Phase 1: Custom Transformer Block)

Graph IR, exporter, optimization passes, memory planner, executor, backends, C kernels, compiled C executor — all working end-to-end for custom transformer blocks (naive attention, F.softmax, SDPA). Beats PyTorch (including SDPA) at every tested config. See DESIGN.md for full architecture details.

<details>
<summary>Detailed completion list</summary>

#### Graph IR (`runtime/ir.py`)
- [x] OpType enum: MATMUL, ADD, RELU, TRANSPOSE, PERMUTE, DIV, SUB, MUL, EXP, MAX, SUM, SOFTMAX, RESHAPE, LAYERNORM, MATMUL_ADD, FUSED_BIAS_RELU, ATTENTION
- [x] TensorInfo with named tensors and numpy buffer
- [x] Node dataclass (op, inputs, output, attrs)
- [x] Graph with builder methods, connectivity indices, toposort, validation
- [x] Graph mutation methods for optimization passes (remove_node, remove_tensor, rewire_input)

#### Exporter (`runtime/exporter.py`)
- [x] torch.export integration (no decomposition — handle high-level ops directly)
- [x] Placeholder classification via graph_signature (inputs vs constants)
- [x] Weight data extraction from state_dict
- [x] Handler registry for ATen op dispatch
- [x] aten.linear → MATMUL(transpose_b=True) + ADD
- [x] aten.mm, aten.matmul, aten.bmm → MATMUL
- [x] aten.permute → TRANSPOSE (2-axis swap) or PERMUTE (general)
- [x] aten.t, aten.numpy_T, aten.transpose.int → TRANSPOSE
- [x] aten.relu → RELU
- [x] Binary ops (add, div, sub, mul) with scalar second arg support
- [x] aten.exp → EXP
- [x] Reduction ops (amax, max.dim, sum.dim_IntList) with getitem handling
- [x] aten._softmax, aten.softmax.int → SOFTMAX
- [x] aten.reshape, aten.view → RESHAPE
- [x] aten.unflatten, aten.unsqueeze, aten.squeeze → RESHAPE (from meta)
- [x] aten.layer_norm.default → LAYERNORM
- [x] aten.scaled_dot_product_attention → ATTENTION (direct mapping)
- [x] aten.contiguous → no-op alias
- [x] aten.mean.dim → decomposed to SUM + scalar DIV

#### Optimization Passes (`runtime/passes.py`)
- [x] Pass pipeline infrastructure (run_pipeline, run_until_stable)
- [x] Numpy evaluator registry (all ops)
- [x] BLAS flag absorption (transpose, scalar MUL/DIV into MATMUL alpha — backward and forward)
- [x] Constant folding (with contiguous array guarantee)
- [x] Fusion pattern registry with priority-based greedy matching
- [x] Dead code elimination (repeating, plus unused constant cleanup)
- [x] ATTENTION fusion: MATMUL→SOFTMAX→MATMUL (after scalar absorption)
- [x] FUSED_BIAS_RELU: ADD+RELU (priority 0)
- [x] MATMUL_ADD: sgemm beta=1.0 trick (priority 1)

#### Memory Planner (`runtime/planner.py`)
- [x] Lifetime analysis, greedy first-fit offset assignment
- [x] Zero-copy RESHAPE aliasing (chain following, lifetime extension)
- [x] Scratch buffer registry (single-step lifetimes, unified arena allocation)

#### Executor (`runtime/executor.py`)
- [x] Per-op dispatch with backend priority (C → numpy fallback)
- [x] Compiled C dispatch (single ctypes call for whole graph)
- [x] Arena allocation/reuse, buffer binding, input patching per call
- [x] Zero-copy RESHAPE handling in both paths
- [x] Scratch buffer passing as extra kernel input

#### C Operators (`csrc/ops.c`) & Compiled Executor (`csrc/executor.c`)
- [x] MATMUL (cblas_sgemm, batched, transpose_b, alpha/beta)
- [x] ADD (bias broadcast + element-wise + scalar), RELU, TRANSPOSE (2D + N-dim swapaxes)
- [x] DIV, SUB, MUL, EXP (element-wise + scalar variants)
- [x] MAX, SUM (reduction via outer/axis_size/inner)
- [x] SOFTMAX (SIMD vDSP/vForce on macOS), LAYERNORM (two-pass)
- [x] Broadcast binary op (odometer pattern with stride-0 broadcast dims)
- [x] FUSED_BIAS_RELU, MATMUL_ADD (sgemm beta), ATTENTION (standard + flash + GCD threading)
- [x] OpNode struct with void* pointers (supports non-float inputs)
- [x] default: abort() for unknown ops

#### Backends (`runtime/backends/`)
- [x] CBackend: all core ops + MATMUL_ADD + scalar C kernels
- [x] NumpyBackend: full coverage as fallback

#### Testing & Benchmarking
- [x] ~153 tests: op correctness, pass invariants, planner properties, end-to-end oracle
- [x] End-to-end: MLP (5 configs), Transformer F.softmax (4 configs), Transformer SDPA (4 configs)
- [x] Performance ablation: MLP and Transformer (PyTorch vs numpy vs C per-op vs compiled ± fusion)
- [x] Compiled C with fusion beats PyTorch SDPA at all tested configs (0.12–0.80x)

</details>

### GPT-2 Ops (Phase 2, in progress)

New ops and exporter handlers for GPT-2's architecture. C kernels, numpy backend, compiled executor dispatch, and tests all implemented. End-to-end GPT-2 pipeline not yet verified.

- [x] SLICE op: zero-copy view with byte_offset aliasing in planner (dim=0 contiguous splits)
- [x] POW op: element-wise x^scalar (C kernel + compiled dispatch)
- [x] TANH op: element-wise tanh (C kernel with vvtanhf SIMD on macOS)
- [x] GELU op: fused tanh approximation kernel (SIMD vvtanhf on macOS)
- [x] EMBEDDING op: table lookup with int64 indices (first non-float-pointer op)
- [x] Exporter: aten.addmm (Conv1D layout — pre-transpose weight at export time)
- [x] Exporter: aten.split.Tensor → deferred SLICE via getitem
- [x] Exporter: aten.dropout → no-op alias (inference mode)
- [x] Exporter: weight lookup fix for registered buffers (causal mask in exported.constants)
- [x] recognize_gelu pass: 8-node GELU tanh approximation → single GELU node
- [x] Causal attention support (upper-triangular -inf mask in evaluator)
- [x] Planner: SLICE aliasing with byte offsets (contiguous dim=0 splits)
- [x] Executor: SLICE handling in per-op and compiled paths
- [x] COpNode inputs/output changed to void* (supports int64 embedding indices)
- [x] Tests for all new ops, SLICE aliasing, GELU recognition

### ORT Benchmark Comparisons (in progress)

Ablation scripts comparing our runtime against ONNX Runtime across attention variants and fusion strategies.

- [x] `ablation/bench_ort_attention.py`: 6 attention variants × 5 configs, ORT graph inspection, Python optimizer
- [x] `ablation/bench_ort_fusion.py`: transformer-specific optimizer investigation
- [x] `ablation/bench_transformer_ablation.py`: our runtime's ablation across execution modes

---

## TODO

### 1. GPT-2 End-to-End
Get GPT-2 (transformer body) running through the full pipeline with correctness and performance verification. Most of the ops are implemented — this is integration and debugging.

- [x] Run GPT-2 body (2-layer) through export → optimize → plan → compiled execute
- [x] Correctness verification against PyTorch (fp32 match)
- [x] Handle any remaining unsupported ATen ops that surface during tracing
- [x] Causal masking in fused C attention kernel
- [ ] End-to-end benchmark: our runtime vs PyTorch eager vs PyTorch with torch.compile
- [ ] End-to-end benchmark: our runtime vs ONNX Runtime (with and without attention fusion)

### 2. ORT Benchmark Comparisons
Finalize the comparison story against ONNX Runtime — scripts exist, needs final runs and analysis.

- [ ] Run bench_ort_attention.py across all configs, collect and document results
- [ ] Run bench_ort_fusion.py, document which ORT optimization paths fire
- [ ] Head-to-head comparison table: our runtime vs ORT (single-thread and multi-thread)
- [ ] Analysis writeup: where we win, where ORT wins, and why

### 3. GPU Support
Triton backend for GPU execution — the major new capability.

- [ ] Triton kernel implementations (MATMUL, element-wise ops, reductions, softmax, layernorm, attention)
- [ ] Triton backend class (same Backend protocol, GPU tensors instead of numpy)
- [ ] GPU memory model (arena on GPU, weight transfer, input/output staging)
- [ ] GPU executor path (separate from C compiled executor — likely Python dispatch with Triton kernels)
- [ ] Flash attention on GPU (Triton implementation or wrap existing library)
- [ ] Benchmark: GPU vs CPU, GPU vs PyTorch CUDA, GPU vs PyTorch with torch.compile
- [ ] Multi-backend support: dispatch some ops to GPU, others to CPU

### 4. Quantization
Q/DQ node insertion, INT8 weight quantization, mixed-precision execution.

- [ ] Q/DQ node insertion pass (graph rewrite: insert quantize/dequantize around weight consumers)
- [ ] INT8 weight quantization (per-tensor or per-channel scale/zero-point)
- [ ] Mixed-precision graph support (dtype tracking through planner and executor)
- [ ] Quantized C kernels (INT8 MATMUL via CBLAS or custom, INT8 element-wise ops)
- [ ] Quantized Triton kernels (if GPU support lands first)
- [ ] Accuracy measurement: quantized vs fp32 output degradation
- [ ] Performance measurement: quantized vs fp32 speedup (memory bandwidth reduction)

### 5. Qwen3 End-to-End
Run Qwen3-0.6B (transformer body) through the pipeline. Requires several new ops beyond GPT-2.

- [ ] RoPE (rotary position embeddings) — likely needs SIN, COS, and element-wise rotation ops
- [ ] GQA (grouped query attention) — K/V have fewer heads than Q, needs broadcast in attention
- [ ] SwiGLU activation — gate * silu(x), needs SILU op or decomposition
- [ ] Trace Qwen3 body, identify full op gap (explore_qwen.py is a start)
- [ ] Implement missing ops (exporter handlers, C kernels, backend wrappers)
- [ ] Correctness verification against PyTorch
- [ ] Benchmark against PyTorch and ORT

---

## Known Limitations

- **No PERMUTE in compiled C dispatch.** `OP_PERMUTE` has no `case` in executor.c — hits `default: abort()`. Safe for now because transformer permutations are classified as TRANSPOSE (two-axis swaps), but any true N-dim permutation would crash.
- **MATMUL_ADD constant folding evaluator uses `.T` (2D-only).** `passes.py:215` uses `ins[1].T` which is a 2D property. Should be `np.swapaxes(ins[1], -2, -1)` for N-dim correctness. Low risk since constant folding rarely hits N-dim matmuls.
- **Flash attention scratch over-allocation.** Scratch calculator always allocates standard-kernel-sized scratch (S² per slice). Flash only needs B_r×B_c per slice.
- **Naive attention recognition not implemented.** Manual softmax decomposition creates a DAG pattern that the linear chain matcher can't handle.
