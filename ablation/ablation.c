/*
 * ablation.c — Variant kernels and parametric executor for ablation benchmarks.
 *
 * Self-contained: duplicates OpNode struct and dispatch logic from executor.c
 * so we don't touch existing files. Linked with ops.c for shared kernels.
 *
 * Provides:
 *   kernel_attention_scalar  — fused attention, scalar softmax, sequential
 *   kernel_attention_simd    — fused attention, SIMD softmax, sequential (no GCD)
 *   execute_ablation         — parametric executor with variant selection
 *   timed_execute_ablation   — per-op nanosecond timing with variant selection
 *
 * Build: see Makefile (libablation target)
 */

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>
#endif

#include <math.h>
#include <float.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ----------------------------------------------------------------
 * Op constants (must match Python OpType enum auto() values)
 * ---------------------------------------------------------------- */
#define OP_MATMUL          1
#define OP_ADD             2
#define OP_RELU            3
#define OP_TRANSPOSE       4
#define OP_PERMUTE         5
#define OP_DIV             6
#define OP_SUB             7
#define OP_MUL             8
#define OP_EXP             9
#define OP_MAX             10
#define OP_SUM             11
#define OP_SOFTMAX         12
#define OP_RESHAPE         13
#define OP_LAYERNORM       14
#define OP_MATMUL_ADD      15
#define OP_FUSED_BIAS_RELU 16
#define OP_ATTENTION       17

#define MAX_INPUTS 8
#define MAX_DIMS   16

typedef struct {
    int op;
    int n_inputs;
    float* inputs[MAX_INPUTS];
    float* output;
    int out_shape[MAX_DIMS];
    int n_dims;
    int extra[MAX_DIMS];
} OpNode;

/* ----------------------------------------------------------------
 * Forward declarations for ops.c kernels
 * ---------------------------------------------------------------- */
void kernel_matmul_ab(const float* a, const float* b, float* out,
                      int M, int N, int K, int batches, int trans_b,
                      float alpha, float beta);
void kernel_matmul_beta(const float* a, const float* b, float* out,
                        int M, int N, int K, int batches, int trans_b,
                        float beta);
void kernel_add(const float* a, const float* bias, float* out, int M, int N);
void kernel_relu(const float* x, float* out, int n);
void kernel_transpose(const float* a, float* out, int rows, int cols);
void kernel_div(const float* a, const float* b, float* out, int n);
void kernel_sub(const float* a, const float* b, float* out, int n);
void kernel_mul(const float* a, const float* b, float* out, int n);
void kernel_add_scalar(const float* a, float s, float* out, int n);
void kernel_div_scalar(const float* a, float s, float* out, int n);
void kernel_sub_scalar(const float* a, float s, float* out, int n);
void kernel_mul_scalar(const float* a, float s, float* out, int n);
void kernel_exp(const float* x, float* out, int n);
void kernel_max(const float* x, float* out, int outer, int axis_size, int inner);
void kernel_sum(const float* x, float* out, int outer, int axis_size, int inner);
void kernel_softmax(const float* x, float* out, int rows, int cols);
void kernel_layernorm(const float* x, const float* gamma, const float* beta,
                      float* out, int rows, int cols, float eps);
void kernel_bias_relu(const float* a, const float* bias, float* out, int M, int N);
void kernel_attention(const float* Q, const float* K, const float* V,
                      float* output, float* scratch,
                      int batch_heads, int seq_len, int head_dim);
void kernel_attention_flash(const float* Q, const float* K, const float* V,
                            float* output, float* scratch,
                            int batch_heads, int seq_len, int head_dim);


/* ================================================================
 * VARIANT KERNELS
 * ================================================================ */

/* ----------------------------------------------------------------
 * Scalar softmax — pure C loops, no SIMD
 *
 * Same algorithm as kernel_softmax's non-Apple path, but always
 * used regardless of platform (for controlled ablation).
 * ---------------------------------------------------------------- */
static void softmax_scalar(const float* x, float* out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        const float* row_in = x + i * cols;
        float* row_out = out + i * cols;

        float mx = row_in[0];
        for (int j = 1; j < cols; j++)
            if (row_in[j] > mx) mx = row_in[j];

        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            row_out[j] = expf(row_in[j] - mx);
            sum += row_out[j];
        }

        for (int j = 0; j < cols; j++)
            row_out[j] /= sum;
    }
}

/* ----------------------------------------------------------------
 * Fused attention — scalar softmax, sequential (no SIMD, no GCD)
 *
 * Identical algorithm to kernel_attention but:
 *   - softmax uses scalar loops (no vDSP/vForce)
 *   - slices are processed sequentially (no dispatch_apply)
 * ---------------------------------------------------------------- */
void kernel_attention_scalar(const float* Q, const float* K, const float* V,
                             float* output, float* scratch,
                             int batch_heads, int seq_len, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int slice = seq_len * head_dim;
    int scratch_sz = seq_len * seq_len;

    for (int bh = 0; bh < batch_heads; bh++) {
        float* S = scratch + bh * scratch_sz;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, seq_len, head_dim, scale,
                    Q + bh * slice, head_dim,
                    K + bh * slice, head_dim,
                    0.0f, S, seq_len);

        softmax_scalar(S, S, seq_len, seq_len);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, head_dim, seq_len, 1.0f,
                    S, seq_len,
                    V + bh * slice, head_dim,
                    0.0f, output + bh * slice, head_dim);
    }
}

/* ----------------------------------------------------------------
 * Fused attention — SIMD softmax, sequential (no GCD)
 *
 * Uses kernel_softmax from ops.c (Accelerate SIMD on macOS)
 * but processes slices sequentially — isolates SIMD from threading.
 * ---------------------------------------------------------------- */
void kernel_attention_simd(const float* Q, const float* K, const float* V,
                           float* output, float* scratch,
                           int batch_heads, int seq_len, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int slice = seq_len * head_dim;
    int scratch_sz = seq_len * seq_len;

    for (int bh = 0; bh < batch_heads; bh++) {
        float* S = scratch + bh * scratch_sz;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, seq_len, head_dim, scale,
                    Q + bh * slice, head_dim,
                    K + bh * slice, head_dim,
                    0.0f, S, seq_len);

        kernel_softmax(S, S, seq_len, seq_len);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, head_dim, seq_len, 1.0f,
                    S, seq_len,
                    V + bh * slice, head_dim,
                    0.0f, output + bh * slice, head_dim);
    }
}

/* kernel_attention from ops.c is variant 2 (SIMD + GCD) */


/* ----------------------------------------------------------------
 * LayerNorm variants
 *
 * ops.c's kernel_layernorm now uses SIMD (vDSP) + GCD threading.
 * We provide scalar-only and SIMD-only variants for ablation.
 * ---------------------------------------------------------------- */

/* Scalar layernorm row — pure C loops, no SIMD */
static void layernorm_row_scalar(const float* row_in, float* row_out,
                                  const float* gamma, const float* beta,
                                  int cols, float eps) {
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) sum += row_in[j];
    float mean = sum / cols;

    float var = 0.0f;
    for (int j = 0; j < cols; j++) {
        float d = row_in[j] - mean;
        row_out[j] = d;
        var += d * d;
    }
    float inv_std = 1.0f / sqrtf(var / cols + eps);

    for (int j = 0; j < cols; j++) {
        row_out[j] = row_out[j] * inv_std * gamma[j] + beta[j];
    }
}

void kernel_layernorm_scalar(const float* x, const float* gamma, const float* beta,
                              float* out, int rows, int cols, float eps) {
    for (int i = 0; i < rows; i++) {
        layernorm_row_scalar(x + i * cols, out + i * cols,
                              gamma, beta, cols, eps);
    }
}

/* SIMD layernorm row — Accelerate vDSP, no GCD threading */
#ifdef __APPLE__
static void layernorm_row_simd(const float* row_in, float* row_out,
                                const float* gamma, const float* beta,
                                int cols, float eps) {
    vDSP_Length n = (vDSP_Length)cols;

    float sum;
    vDSP_sve(row_in, 1, &sum, n);
    float mean = sum / cols;

    float neg_mean = -mean;
    vDSP_vsadd(row_in, 1, &neg_mean, row_out, 1, n);

    float var_sum;
    vDSP_svesq(row_out, 1, &var_sum, n);
    float inv_std = 1.0f / sqrtf(var_sum / cols + eps);

    vDSP_vsmul(row_out, 1, &inv_std, row_out, 1, n);
    vDSP_vma(row_out, 1, gamma, 1, beta, 1, row_out, 1, n);
}
#endif

void kernel_layernorm_simd(const float* x, const float* gamma, const float* beta,
                            float* out, int rows, int cols, float eps) {
#ifdef __APPLE__
    for (int i = 0; i < rows; i++) {
        layernorm_row_simd(x + i * cols, out + i * cols,
                            gamma, beta, cols, eps);
    }
#else
    /* No Accelerate — fall back to scalar */
    kernel_layernorm_scalar(x, gamma, beta, out, rows, cols, eps);
#endif
}

/* kernel_layernorm from ops.c is variant 2 (SIMD + GCD) */


/* ================================================================
 * PARAMETRIC DISPATCH
 *
 * Same logic as executor.c dispatch(), but selects variant kernels
 * for SOFTMAX, ATTENTION, and LAYERNORM ops based on mode flags.
 *
 * softmax_mode:   0 = SIMD (kernel_softmax), 1 = scalar (softmax_scalar)
 * attn_mode:      0 = scalar, 1 = SIMD, 2 = SIMD+GCD (kernel_attention)
 * layernorm_mode: 0 = scalar, 1 = SIMD, 2 = SIMD+GCD (kernel_layernorm)
 * ================================================================ */

static void dispatch_variant(OpNode* node, int softmax_mode, int attn_mode,
                              int layernorm_mode) {
    switch (node->op) {

    case OP_MATMUL: {
        int N = node->out_shape[node->n_dims - 1];
        int K = node->extra[0];
        int trans_b = node->extra[1];
        int b_is_2d = node->extra[2];
        union { int i; float f; } alpha_u;
        alpha_u.i = node->extra[3];
        float alpha = alpha_u.i ? alpha_u.f : 1.0f;
        if (b_is_2d) {
            int M_total = 1;
            for (int i = 0; i < node->n_dims - 1; i++)
                M_total *= node->out_shape[i];
            kernel_matmul_ab(node->inputs[0], node->inputs[1], node->output,
                             M_total, N, K, 1, trans_b, alpha, 0.0f);
        } else {
            int M = node->out_shape[node->n_dims - 2];
            int batches = 1;
            for (int i = 0; i < node->n_dims - 2; i++)
                batches *= node->out_shape[i];
            kernel_matmul_ab(node->inputs[0], node->inputs[1], node->output,
                             M, N, K, batches, trans_b, alpha, 0.0f);
        }
        break;
    }

    case OP_ADD: {
        int n = 1;
        for (int i = 0; i < node->n_dims; i++) n *= node->out_shape[i];
        if (node->extra[0] == 2) {
            union { int i; float f; } s;
            s.i = node->extra[1];
            kernel_add_scalar(node->inputs[0], s.f, node->output, n);
        } else if (node->extra[0] == 1) {
            kernel_add(node->inputs[0], node->inputs[1], node->output, 1, n);
        } else {
            int N = node->out_shape[node->n_dims - 1];
            kernel_add(node->inputs[0], node->inputs[1], node->output, n / N, N);
        }
        break;
    }

    case OP_RELU: {
        int n = 1;
        for (int i = 0; i < node->n_dims; i++) n *= node->out_shape[i];
        kernel_relu(node->inputs[0], node->output, n);
        break;
    }

    case OP_TRANSPOSE: {
        if (node->n_dims == 2) {
            kernel_transpose(node->inputs[0], node->output,
                             node->extra[0], node->extra[1]);
        } else {
            int outer  = node->extra[0];
            int A      = node->extra[1];
            int middle = node->extra[2];
            int B      = node->extra[3];
            int inner  = node->extra[4];
            const float* x = node->inputs[0];
            float* out = node->output;
            for (int o = 0; o < outer; o++)
              for (int b = 0; b < B; b++)
                for (int m = 0; m < middle; m++)
                  for (int a = 0; a < A; a++) {
                    int in_off  = (((o*A + a)*middle + m)*B + b)*inner;
                    int out_off = (((o*B + b)*middle + m)*A + a)*inner;
                    memcpy(out + out_off, x + in_off, inner * sizeof(float));
                  }
        }
        break;
    }

    case OP_DIV: {
        int n = 1;
        for (int i = 0; i < node->n_dims; i++) n *= node->out_shape[i];
        if (node->extra[0]) {
            union { int i; float f; } s;
            s.i = node->extra[1];
            kernel_div_scalar(node->inputs[0], s.f, node->output, n);
        } else {
            kernel_div(node->inputs[0], node->inputs[1], node->output, n);
        }
        break;
    }

    case OP_SUB: {
        int n = 1;
        for (int i = 0; i < node->n_dims; i++) n *= node->out_shape[i];
        if (node->extra[0]) {
            union { int i; float f; } s;
            s.i = node->extra[1];
            kernel_sub_scalar(node->inputs[0], s.f, node->output, n);
        } else {
            kernel_sub(node->inputs[0], node->inputs[1], node->output, n);
        }
        break;
    }

    case OP_MUL: {
        int n = 1;
        for (int i = 0; i < node->n_dims; i++) n *= node->out_shape[i];
        if (node->extra[0]) {
            union { int i; float f; } s;
            s.i = node->extra[1];
            kernel_mul_scalar(node->inputs[0], s.f, node->output, n);
        } else {
            kernel_mul(node->inputs[0], node->inputs[1], node->output, n);
        }
        break;
    }

    case OP_EXP: {
        int n = 1;
        for (int i = 0; i < node->n_dims; i++) n *= node->out_shape[i];
        kernel_exp(node->inputs[0], node->output, n);
        break;
    }

    case OP_MAX: {
        int axis = node->extra[0];
        int axis_size = node->extra[1];
        int outer = 1, inner = 1;
        for (int i = 0; i < axis; i++) outer *= node->out_shape[i];
        for (int i = axis; i < node->n_dims; i++) inner *= node->out_shape[i];
        kernel_max(node->inputs[0], node->output, outer, axis_size, inner);
        break;
    }

    case OP_SUM: {
        int axis = node->extra[0];
        int axis_size = node->extra[1];
        int outer = 1, inner = 1;
        for (int i = 0; i < axis; i++) outer *= node->out_shape[i];
        for (int i = axis; i < node->n_dims; i++) inner *= node->out_shape[i];
        kernel_sum(node->inputs[0], node->output, outer, axis_size, inner);
        break;
    }

    case OP_SOFTMAX: {
        int cols = node->out_shape[node->n_dims - 1];
        int rows = 1;
        for (int i = 0; i < node->n_dims - 1; i++) rows *= node->out_shape[i];
        if (softmax_mode == 1)
            softmax_scalar(node->inputs[0], node->output, rows, cols);
        else
            kernel_softmax(node->inputs[0], node->output, rows, cols);
        break;
    }

    case OP_RESHAPE:
        /* Zero-copy: handled by Python alias binding */
        break;

    case OP_LAYERNORM: {
        int cols = node->out_shape[node->n_dims - 1];
        int rows = 1;
        for (int i = 0; i < node->n_dims - 1; i++) rows *= node->out_shape[i];
        union { int i; float f; } eps_u;
        eps_u.i = node->extra[0];
        switch (layernorm_mode) {
        case 0:
            kernel_layernorm_scalar(
                node->inputs[0], node->inputs[1], node->inputs[2],
                node->output, rows, cols, eps_u.f);
            break;
        case 1:
            kernel_layernorm_simd(
                node->inputs[0], node->inputs[1], node->inputs[2],
                node->output, rows, cols, eps_u.f);
            break;
        default:
            kernel_layernorm(
                node->inputs[0], node->inputs[1], node->inputs[2],
                node->output, rows, cols, eps_u.f);
            break;
        }
        break;
    }

    case OP_MATMUL_ADD: {
        int N = node->out_shape[node->n_dims - 1];
        int K = node->extra[0];
        int trans_b = node->extra[1];
        int b_is_2d = node->extra[2];
        float* bias = node->inputs[2];
        int rows = 1;
        for (int i = 0; i < node->n_dims - 1; i++)
            rows *= node->out_shape[i];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < N; c++)
                node->output[r * N + c] = bias[c];
        if (b_is_2d) {
            int M_total = 1;
            for (int i = 0; i < node->n_dims - 1; i++)
                M_total *= node->out_shape[i];
            kernel_matmul_beta(node->inputs[0], node->inputs[1], node->output,
                               M_total, N, K, 1, trans_b, 1.0f);
        } else {
            int M = node->out_shape[node->n_dims - 2];
            int batches = 1;
            for (int i = 0; i < node->n_dims - 2; i++)
                batches *= node->out_shape[i];
            kernel_matmul_beta(node->inputs[0], node->inputs[1], node->output,
                               M, N, K, batches, trans_b, 1.0f);
        }
        break;
    }

    case OP_FUSED_BIAS_RELU: {
        int N = node->out_shape[node->n_dims - 1];
        int M = 1;
        for (int i = 0; i < node->n_dims - 1; i++) M *= node->out_shape[i];
        kernel_bias_relu(node->inputs[0], node->inputs[1], node->output, M, N);
        break;
    }

    case OP_ATTENTION: {
        int seq_len = node->extra[0];
        int head_dim = node->extra[1];
        int batch_heads = 1;
        for (int i = 0; i < node->n_dims - 2; i++)
            batch_heads *= node->out_shape[i];
        switch (attn_mode) {
        case 0:
            kernel_attention_scalar(
                node->inputs[0], node->inputs[1], node->inputs[2],
                node->output, node->inputs[3],
                batch_heads, seq_len, head_dim);
            break;
        case 1:
            kernel_attention_simd(
                node->inputs[0], node->inputs[1], node->inputs[2],
                node->output, node->inputs[3],
                batch_heads, seq_len, head_dim);
            break;
        case 3:
            kernel_attention_flash(
                node->inputs[0], node->inputs[1], node->inputs[2],
                node->output, node->inputs[3],
                batch_heads, seq_len, head_dim);
            break;
        default:
            kernel_attention(
                node->inputs[0], node->inputs[1], node->inputs[2],
                node->output, node->inputs[3],
                batch_heads, seq_len, head_dim);
            break;
        }
        break;
    }

    default:
        fprintf(stderr, "ablation: unknown op %d\n", node->op);
        abort();
    }
}


/* ================================================================
 * PUBLIC API
 * ================================================================ */

/*
 * execute_ablation — run a compiled plan with variant selection.
 *
 * softmax_mode:   0 = SIMD (default), 1 = scalar
 * attn_mode:      0 = scalar, 1 = SIMD, 2 = SIMD+GCD (default), 3 = flash
 * layernorm_mode: 0 = scalar, 1 = SIMD, 2 = SIMD+GCD (default)
 */
void execute_ablation(OpNode* nodes, int n_nodes,
                      int softmax_mode, int attn_mode, int layernorm_mode) {
    for (int i = 0; i < n_nodes; i++)
        dispatch_variant(&nodes[i], softmax_mode, attn_mode, layernorm_mode);
}

/*
 * timed_execute_ablation — same as execute_ablation but records
 * per-op timing in nanoseconds.
 *
 * Caller provides a double[n_nodes] array. On return, times_ns[i]
 * contains the wall-clock nanoseconds for op i.
 */
void timed_execute_ablation(OpNode* nodes, int n_nodes, double* times_ns,
                            int softmax_mode, int attn_mode,
                            int layernorm_mode) {
#ifdef __APPLE__
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    double ns_per_tick = (double)tb.numer / (double)tb.denom;

    for (int i = 0; i < n_nodes; i++) {
        uint64_t t0 = mach_absolute_time();
        dispatch_variant(&nodes[i], softmax_mode, attn_mode, layernorm_mode);
        uint64_t t1 = mach_absolute_time();
        times_ns[i] = (double)(t1 - t0) * ns_per_tick;
    }
#else
    /* Fallback: no per-op timing, just execute */
    for (int i = 0; i < n_nodes; i++) {
        dispatch_variant(&nodes[i], softmax_mode, attn_mode, layernorm_mode);
        times_ns[i] = 0.0;
    }
#endif
}
