/*
 * ops.h — Kernel declarations and shared macros for the inference runtime.
 *
 * All kernel prototypes live here. Each ops/ .c file includes this header,
 * as does executor.c. This replaces the ad-hoc forward declarations that
 * were previously at the top of executor.c.
 */

#ifndef OPS_H
#define OPS_H

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#endif

#include <math.h>
#include <float.h>
#include <string.h>
#include <stddef.h>

/* ----------------------------------------------------------------
 * PARALLEL_FOR — GCD-threaded loop with chunking
 *
 * Splits `n` iterations across GCD threads, targeting >=16KB of work
 * per chunk to amortize dispatch overhead. Falls back to a plain
 * for loop on non-Apple platforms or when n is small.
 *
 * Parameters:
 *   n              — total iteration count
 *   bytes_per_iter — bytes of data touched per iteration (for chunking)
 *   i              — loop variable name (declared by the macro)
 *   body           — loop body (can reference `i`)
 * ---------------------------------------------------------------- */
#ifdef __APPLE__
#define PARALLEL_FOR(n, bytes_per_iter, i, body) do {                   \
    int _pf_n = (n);                                                    \
    int _pf_min = (16 * 1024) / (bytes_per_iter);                       \
    if (_pf_min < 1) _pf_min = 1;                                      \
    if (_pf_n > _pf_min) {                                              \
        int _pf_nchunks = (_pf_n + _pf_min - 1) / _pf_min;             \
        dispatch_apply((size_t)_pf_nchunks,                             \
            dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),     \
            ^(size_t _pf_chunk) {                                       \
                int _pf_s = (int)_pf_chunk * _pf_min;                   \
                int _pf_e = _pf_s + _pf_min;                            \
                if (_pf_e > _pf_n) _pf_e = _pf_n;                       \
                for (int i = _pf_s; i < _pf_e; i++) { body }            \
            });                                                         \
    } else {                                                            \
        for (int i = 0; i < _pf_n; i++) { body }                        \
    }                                                                   \
} while(0)
#else
#define PARALLEL_FOR(n, bytes_per_iter, i, body)                        \
    for (int i = 0; i < (n); i++) { body }
#endif

/* ----------------------------------------------------------------
 * Kernel prototypes — grouped by source file
 * ---------------------------------------------------------------- */

/* matmul.c */
void kernel_matmul_ab(const float* a, const float* b, float* out,
                      int M, int N, int K, int batches, int trans_b,
                      float alpha, float beta);
void kernel_matmul_beta(const float* a, const float* b, float* out,
                        int M, int N, int K, int batches, int trans_b,
                        float beta);
void kernel_matmul(const float* a, const float* b, float* out,
                   int M, int N, int K, int batches, int trans_b);

/* elementwise.c */
void kernel_add(const float* a, const float* bias, float* out, int M, int N);
void kernel_add_scalar(const float* a, float s, float* out, int n);
void kernel_sub(const float* a, const float* b, float* out, int n);
void kernel_sub_scalar(const float* a, float s, float* out, int n);
void kernel_mul(const float* a, const float* b, float* out, int n);
void kernel_mul_scalar(const float* a, float s, float* out, int n);
void kernel_div(const float* a, const float* b, float* out, int n);
void kernel_div_scalar(const float* a, float s, float* out, int n);
void kernel_relu(const float* x, float* out, int n);
void kernel_exp(const float* x, float* out, int n);
void kernel_pow_scalar(const float* x, float scalar, float* out, int n);
void kernel_tanh(const float* x, float* out, int n);
void kernel_gelu_tanh(const float* x, float* out, int n);
void kernel_rsqrt(const float* x, float* out, int n);
void kernel_silu(const float* x, float* out, int n);
void kernel_neg(const float* x, float* out, int n);
void kernel_cos(const float* x, float* out, int n);
void kernel_sin(const float* x, float* out, int n);
void kernel_bias_relu(const float* a, const float* bias, float* out, int M, int N);
void kernel_gated_silu(const float* x, const float* up, const float* bias,
                       float* out, int M, int N, int has_bias);
void kernel_gated_gelu(const float* x, const float* up, const float* bias,
                       float* out, int M, int N, int has_bias);

/* reduce.c */
void kernel_max(const float* x, float* out, int outer, int axis_size, int inner);
void kernel_sum(const float* x, float* out, int outer, int axis_size, int inner);
void kernel_softmax(const float* x, float* out, int rows, int cols);

/* norm.c */
void kernel_layernorm(const float* x, const float* gamma, const float* beta,
                      float* out, int rows, int cols, float eps);
void kernel_rmsnorm(const float* x, const float* weight, float* out,
                    int rows, int cols, float eps);

/* attn.c */
void kernel_attention(const float* Q, const float* K, const float* V,
                      float* output, float* scratch, const float* mask,
                      int batch_heads, int seq_len, int head_dim,
                      int causal, int group_size);
void kernel_attention_flash(const float* Q, const float* K, const float* V,
                            float* output, float* scratch,
                            int batch_heads, int seq_len, int head_dim,
                            int causal, int group_size);

/* shape.c */
void kernel_transpose(const float* a, float* out, int rows, int cols);
void kernel_broadcast_binop(const float* a, const float* b, float* out,
                            const int* a_strides, const int* b_strides,
                            const int* out_shape, int ndim, int op);
void kernel_embedding(const long* ids, const float* table, float* out,
                      int seq_len, int embed_dim);
void kernel_slice(const void* x, void* out,
                  int outer, int orig_dim_size, int start,
                  int slice_len, int inner, int elem_size);
void kernel_cat(void* const* inputs, void* out,
                const int* dim_sizes, int n_inputs,
                int outer, int inner, int elem_size);

#endif /* OPS_H */
