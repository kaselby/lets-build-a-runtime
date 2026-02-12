# Transformer Ablation Benchmark

Performance ablation for a single-layer transformer block, measuring how each optimization contributes to the runtime's speedup over PyTorch.

## What it measures

For each variant, the benchmark reports wall-clock time broken into three categories:
- **Attention** — the Q@K^T → softmax → @V computation (fused or primitive ops)
- **LayerNorm** — the two LayerNorm ops in the block
- **Other** — everything else (Q/K/V/O projections, FFN matmuls, residual adds, reshapes)

Timing uses `mach_absolute_time` per-op inside the C dispatch loop for nanosecond granularity. PyTorch baselines use `perf_counter_ns` around the relevant ops.

## Variants

### Baselines
| Variant | Description |
|---------|-------------|
| **PT naive** | Eager PyTorch with manual `F.softmax` attention |
| **PT SDPA** | Eager PyTorch with `F.scaled_dot_product_attention` |

### Python executor (small configs only)
| Variant | Description |
|---------|-------------|
| **Numpy per-op** | Python dispatch loop, numpy kernels, all fusion passes |
| **C per-op** | Python dispatch loop, C kernels via ctypes, all fusion passes |

### Incremental fusion (small configs only)
Compiled C executor with fusion passes added one at a time. Shows the marginal contribution of each graph optimization.

| Variant | Passes applied |
|---------|---------------|
| **No passes** | Raw exported graph, no optimization |
| **+ fold/DCE** | Constant folding + dead code elimination |
| **+ BLAS absorb** | + transpose absorption into MATMUL (`transpose_b`), scalar DIV/MUL into `alpha` |
| **+ MATMUL_ADD** | + MATMUL+ADD → fused `sgemm` with `beta=1.0` |
| **+ BIAS_RELU** | + ADD+RELU → single-pass `bias_relu` kernel |

### Attention kernel variants
All non-attention fusion passes active. Isolates the attention kernel implementation.

| Variant | Attention kernel | Graph |
|---------|-----------------|-------|
| **Attn: prim scalar** | Unfused MATMUL→SOFTMAX→MATMUL, scalar softmax | No attention fusion |
| **Attn: fused scalar** | Fused kernel, scalar softmax, sequential | Full fusion |
| **Attn: fused SIMD** | Fused kernel, Accelerate vDSP/vForce softmax, sequential | Full fusion |
| **Attn: fused GCD** | Fused kernel, SIMD softmax, GCD `dispatch_apply` across batch×head | Full fusion |
| **Attn: flash GCD** | Flash attention (tiled online softmax, B_r=B_c=32), SIMD, GCD | Full fusion |

### LayerNorm kernel variants (small configs only)
All fusion passes active, attention at SIMD+GCD.

| Variant | LayerNorm kernel |
|---------|-----------------|
| **LN: scalar** | Pure scalar loops |
| **LN: SIMD** | Accelerate vDSP per row, sequential |
| **LN: SIMD+GCD** | Accelerate vDSP + GCD threading across rows |

### Full optimization
| Variant | Description |
|---------|-------------|
| **Flash + LN opt** | Flash attention + SIMD+GCD LayerNorm — everything maxed |

## Configs

| Name | d_model | n_heads | seq_len | head_dim | Attn scratch |
|------|---------|---------|---------|----------|-------------|
| Toy | 64 | 4 | 32 | 16 | 0.0 MB |
| Small | 256 | 4 | 128 | 64 | 0.3 MB |
| Medium | 512 | 8 | 256 | 64 | 2.1 MB |
| GPT-2 | 768 | 12 | 512 | 64 | 12.6 MB |
| 1B | 2048 | 16 | 512 | 128 | 16.8 MB |
| 3B | 3072 | 24 | 1024 | 128 | 100.7 MB |
| 7B | 4096 | 32 | 1024 | 128 | 134.2 MB |
| 7B-4K | 4096 | 32 | 4096 | 128 | 2.1 GB |
| 1B-8K | 2048 | 16 | 8192 | 128 | 4.3 GB |

All configs use batch=1. Configs with d_model >= 2048 run a reduced variant set (8 instead of 18) to keep runtime manageable.

## Running

```bash
# Full run (all configs)
python ablation/bench_transformer_ablation.py

# Specific configs
python ablation/bench_transformer_ablation.py --configs Toy,Small,GPT-2

# Skip seq >= 4096 configs
python ablation/bench_transformer_ablation.py --skip-large
```

## Building

The ablation C library must be built before running:

```bash
cd ablation && make
```

This compiles `ablation.c` (variant kernels + parametric dispatch) linked with `../csrc/ops.c` (shared kernels).

## Key findings (Apple M4 Max)

**Dispatch overhead is the biggest lever at small sizes.** The compiled C executor with zero optimization passes is already 7-10x faster than PyTorch at Toy/Small sizes — just from eliminating Python dispatch between ops.

**GCD threading dominates attention optimization.** At GPT-2 scale: scalar→SIMD gives 1.3x on attention, but SIMD→GCD gives 5.5x. Parallelizing across batch×head slices is where the real win is.

**Flash attention never wins on M4 Max.** The standard materialized kernel beats flash at every config, even seq=8192 where the S×S matrix is 256MB. M4's large L2 cache and high memory bandwidth make materialization cheaper than flash's extra compute (rescaling corrections, many small tiled sgemms). Flash would likely win on GPU where memory bandwidth is the bottleneck.

**At large sizes, FFN matmuls dominate.** At 7B, attention is ~6% of total time with GCD — the FFN projections (4096→16384→4096) consume 90%+ of the budget. Both runtimes spend most of their time in the same Accelerate BLAS, so our advantage narrows.

**LayerNorm SIMD+GCD matters at medium sizes.** At GPT-2: scalar LN=936us vs SIMD+GCD=144us (6.5x), shifting total time by 15%. At 7B+ it's <1% of total, negligible.
