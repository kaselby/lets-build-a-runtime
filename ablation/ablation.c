/*
 * ablation.c — Variant kernels and parametric executor for ablation benchmarks.
 *
 * Duplicates dispatch logic from executor.c with additional kernel variants
 * for controlled performance ablation. Linked with ops/ .c files for shared kernels.
 *
 * Provides:
 *   kernel_attention_scalar      — standard attention, scalar softmax, sequential
 *   kernel_attention_simd        — standard attention, SIMD softmax, sequential
 *   kernel_attention_flash_param — flash attention with configurable tile sizes
 *   kernel_layernorm_scalar      — layernorm, scalar, sequential
 *   kernel_layernorm_simd        — layernorm, SIMD, sequential
 *   execute_ablation             — parametric executor with variant selection
 *   timed_execute_ablation       — per-op nanosecond timing with variant selection
 *
 * Build: see Makefile (libablation target)
 */

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

#include <stdio.h>
#include <stdlib.h>

#include "runtime.h"
#include "ops.h"


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
                             float* output, float* scratch, const float* mask,
                             int batch_heads, int seq_len, int head_dim,
                             int causal, int group_size) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int slice = seq_len * head_dim;
    int scratch_sz = seq_len * seq_len;

    for (int bh = 0; bh < batch_heads; bh++) {
        int kv_bh = bh / group_size;
        float* S = scratch + bh * scratch_sz;
        const float* mask_bh = mask ? mask + bh * scratch_sz : NULL;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, seq_len, head_dim, scale,
                    Q + bh * slice, head_dim,
                    K + kv_bh * slice, head_dim,
                    0.0f, S, seq_len);

        /* Apply mask or causal */
        if (mask_bh) {
            int ss = seq_len * seq_len;
            for (int i = 0; i < ss; i++)
                S[i] += mask_bh[i];
        } else if (causal) {
            for (int i = 0; i < seq_len; i++)
                for (int j = i + 1; j < seq_len; j++)
                    S[i * seq_len + j] = -INFINITY;
        }

        softmax_scalar(S, S, seq_len, seq_len);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, head_dim, seq_len, 1.0f,
                    S, seq_len,
                    V + kv_bh * slice, head_dim,
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
                           float* output, float* scratch, const float* mask,
                           int batch_heads, int seq_len, int head_dim,
                           int causal, int group_size) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int slice = seq_len * head_dim;
    int scratch_sz = seq_len * seq_len;

    for (int bh = 0; bh < batch_heads; bh++) {
        int kv_bh = bh / group_size;
        float* S = scratch + bh * scratch_sz;
        const float* mask_bh = mask ? mask + bh * scratch_sz : NULL;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, seq_len, head_dim, scale,
                    Q + bh * slice, head_dim,
                    K + kv_bh * slice, head_dim,
                    0.0f, S, seq_len);

        /* Apply mask or causal */
        if (mask_bh) {
            int ss = seq_len * seq_len;
            for (int i = 0; i < ss; i++)
                S[i] += mask_bh[i];
        } else if (causal) {
            for (int i = 0; i < seq_len; i++)
                for (int j = i + 1; j < seq_len; j++)
                    S[i * seq_len + j] = -INFINITY;
        }

        kernel_softmax(S, S, seq_len, seq_len);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, head_dim, seq_len, 1.0f,
                    S, seq_len,
                    V + kv_bh * slice, head_dim,
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


/* ----------------------------------------------------------------
 * Parameterized flash attention — configurable tile sizes
 *
 * Same algorithm as kernel_attention_flash in ops/attn.c, but
 * takes br_max/bc_max as parameters instead of #defines. Enables
 * block size ablation (32x32, 64x128, 128x256, 256x512).
 *
 * Uses C99 VLAs for row state (m[], l[]) — max 256 floats = 1KB.
 * ---------------------------------------------------------------- */
static void attention_flash_slice_param(
        const float* Q_bh, const float* K_bh, const float* V_bh,
        float* O_bh, float* scratch, int seq_len, int head_dim,
        float scale, int causal, int br_max, int bc_max) {

    for (int qi = 0; qi < seq_len; qi += br_max) {
        int br = (qi + br_max <= seq_len) ? br_max : seq_len - qi;

        const float* Q_block = Q_bh + qi * head_dim;
        float*       O_block = O_bh + qi * head_dim;

        float m[br_max];  /* VLA — running row max */
        float l[br_max];  /* VLA — running row sum */
        for (int r = 0; r < br; r++) {
            m[r] = -FLT_MAX;
            l[r] = 0.0f;
        }
        for (int r = 0; r < br; r++)
            for (int d = 0; d < head_dim; d++)
                O_block[r * head_dim + d] = 0.0f;

        for (int kj = 0; kj < seq_len; kj += bc_max) {
            int bc = (kj + bc_max <= seq_len) ? bc_max : seq_len - kj;

            /* Causal: skip fully-masked tiles */
            if (causal && kj > qi + br - 1)
                break;

            const float* K_block = K_bh + kj * head_dim;
            const float* V_block = V_bh + kj * head_dim;

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        br, bc, head_dim,
                        scale,
                        Q_block, head_dim,
                        K_block, head_dim,
                        0.0f,
                        scratch, bc);

            /* Causal mask: set future positions to -inf in partial tiles */
            if (causal && kj + bc > qi) {
                for (int r = 0; r < br; r++) {
                    int last_valid = qi + r - kj;
                    if (last_valid < bc - 1) {
                        int start = (last_valid < 0) ? 0 : last_valid + 1;
                        for (int c = start; c < bc; c++)
                            scratch[r * bc + c] = -INFINITY;
                    }
                }
            }

            for (int r = 0; r < br; r++) {
                float* S_row = scratch + r * bc;
                float* O_row = O_block + r * head_dim;

#ifdef __APPLE__
                float m_tile;
                vDSP_maxv(S_row, 1, &m_tile, (vDSP_Length)bc);
                float m_new = m[r] > m_tile ? m[r] : m_tile;

                float correction = expf(m[r] - m_new);
                l[r] *= correction;
                vDSP_vsmul(O_row, 1, &correction, O_row, 1, (vDSP_Length)head_dim);

                float neg_m = -m_new;
                vDSP_vsadd(S_row, 1, &neg_m, S_row, 1, (vDSP_Length)bc);
                vvexpf(S_row, S_row, &bc);

                float l_new;
                vDSP_sve(S_row, 1, &l_new, (vDSP_Length)bc);
                l[r] += l_new;
                m[r] = m_new;
#else
                float m_tile = S_row[0];
                for (int c = 1; c < bc; c++)
                    if (S_row[c] > m_tile) m_tile = S_row[c];
                float m_new = m[r] > m_tile ? m[r] : m_tile;

                float correction = expf(m[r] - m_new);
                l[r] *= correction;
                for (int d = 0; d < head_dim; d++)
                    O_row[d] *= correction;

                float l_new = 0.0f;
                for (int c = 0; c < bc; c++) {
                    float p = expf(S_row[c] - m_new);
                    S_row[c] = p;
                    l_new += p;
                }
                l[r] += l_new;
                m[r] = m_new;
#endif
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        br, head_dim, bc,
                        1.0f,
                        scratch, bc,
                        V_block, head_dim,
                        1.0f,
                        O_block, head_dim);
        }

        for (int r = 0; r < br; r++) {
            float* O_row = O_block + r * head_dim;
#ifdef __APPLE__
            vDSP_vsdiv(O_row, 1, &l[r], O_row, 1, (vDSP_Length)head_dim);
#else
            float inv_l = 1.0f / l[r];
            for (int d = 0; d < head_dim; d++)
                O_row[d] *= inv_l;
#endif
        }
    }
}

static void kernel_attention_flash_param(
        const float* Q, const float* K, const float* V,
        float* output, float* scratch,
        int batch_heads, int seq_len, int head_dim,
        int causal, int group_size, int br_max, int bc_max) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int slice = seq_len * head_dim;
    int scratch_per_slice = br_max * bc_max;

    PARALLEL_FOR(batch_heads, slice * (int)sizeof(float), bh, {
        int kv_bh = bh / group_size;
        attention_flash_slice_param(
            Q + bh * slice, K + kv_bh * slice,
            V + kv_bh * slice, output + bh * slice,
            scratch + bh * scratch_per_slice,
            seq_len, head_dim, scale, causal, br_max, bc_max);
    });
}


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

/* Helpers matching executor.c */
static inline int total_elements(const OpNode* node) {
    int n = 1;
    for (int i = 0; i < node->n_dims; i++) n *= node->out_shape[i];
    return n;
}

static inline float extra_float(const OpNode* node, int idx) {
    union { int i; float f; } u;
    u.i = node->extra[idx];
    return u.f;
}

static inline int leading_dims(const OpNode* node) {
    int rows = 1;
    for (int i = 0; i < node->n_dims - 1; i++) rows *= node->out_shape[i];
    return rows;
}

static void dispatch_variant(OpNode* node, int softmax_mode, int attn_mode,
                              int layernorm_mode) {
    switch (node->op) {

    case OP_MATMUL: {
        int N = node->out_shape[node->n_dims - 1];
        int K = node->extra[0];
        int trans_b = node->extra[1];
        int b_is_2d = node->extra[2];
        float alpha = node->extra[3] ? extra_float(node, 3) : 1.0f;
        if (b_is_2d) {
            int M_total = leading_dims(node);
            kernel_matmul_ab((const float*)node->inputs[0],
                             (const float*)node->inputs[1],
                             (float*)node->output,
                             M_total, N, K, 1, trans_b, alpha, 0.0f);
        } else {
            int M = node->out_shape[node->n_dims - 2];
            int batches = 1;
            for (int i = 0; i < node->n_dims - 2; i++)
                batches *= node->out_shape[i];
            kernel_matmul_ab((const float*)node->inputs[0],
                             (const float*)node->inputs[1],
                             (float*)node->output,
                             M, N, K, batches, trans_b, alpha, 0.0f);
        }
        break;
    }

    case OP_ADD: {
        int n = total_elements(node);
        if (node->extra[0] == 2) {
            kernel_add_scalar((const float*)node->inputs[0],
                              extra_float(node, 1),
                              (float*)node->output, n);
        } else if (node->extra[0] == 1) {
            kernel_add((const float*)node->inputs[0],
                       (const float*)node->inputs[1],
                       (float*)node->output, 1, n);
        } else {
            int N = node->out_shape[node->n_dims - 1];
            kernel_add((const float*)node->inputs[0],
                       (const float*)node->inputs[1],
                       (float*)node->output, n / N, N);
        }
        break;
    }

    case OP_RELU: {
        kernel_relu((const float*)node->inputs[0],
                    (float*)node->output, total_elements(node));
        break;
    }

    case OP_TRANSPOSE: {
        if (node->n_dims == 2) {
            kernel_transpose((const float*)node->inputs[0],
                             (float*)node->output,
                             node->extra[0], node->extra[1]);
        } else {
            int outer  = node->extra[0];
            int A      = node->extra[1];
            int middle = node->extra[2];
            int B      = node->extra[3];
            int inner  = node->extra[4];
            const float* x = (const float*)node->inputs[0];
            float* out = (float*)node->output;
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
        int n = total_elements(node);
        if (node->extra[0]) {
            kernel_div_scalar((const float*)node->inputs[0],
                              extra_float(node, 1),
                              (float*)node->output, n);
        } else {
            kernel_div((const float*)node->inputs[0],
                       (const float*)node->inputs[1],
                       (float*)node->output, n);
        }
        break;
    }

    case OP_SUB: {
        int n = total_elements(node);
        if (node->extra[0]) {
            kernel_sub_scalar((const float*)node->inputs[0],
                              extra_float(node, 1),
                              (float*)node->output, n);
        } else {
            kernel_sub((const float*)node->inputs[0],
                       (const float*)node->inputs[1],
                       (float*)node->output, n);
        }
        break;
    }

    case OP_MUL: {
        int n = total_elements(node);
        if (node->extra[0]) {
            kernel_mul_scalar((const float*)node->inputs[0],
                              extra_float(node, 1),
                              (float*)node->output, n);
        } else {
            kernel_mul((const float*)node->inputs[0],
                       (const float*)node->inputs[1],
                       (float*)node->output, n);
        }
        break;
    }

    case OP_EXP: {
        kernel_exp((const float*)node->inputs[0],
                   (float*)node->output, total_elements(node));
        break;
    }

    case OP_TANH: {
        kernel_tanh((const float*)node->inputs[0],
                    (float*)node->output, total_elements(node));
        break;
    }

    case OP_POW: {
        kernel_pow_scalar((const float*)node->inputs[0],
                          extra_float(node, 0),
                          (float*)node->output, total_elements(node));
        break;
    }

    case OP_GELU: {
        kernel_gelu_tanh((const float*)node->inputs[0],
                         (float*)node->output, total_elements(node));
        break;
    }

    case OP_MAX: {
        int axis = node->extra[0];
        int axis_size = node->extra[1];
        int outer = 1, inner = 1;
        for (int i = 0; i < axis; i++) outer *= node->out_shape[i];
        for (int i = axis; i < node->n_dims; i++) inner *= node->out_shape[i];
        kernel_max((const float*)node->inputs[0],
                   (float*)node->output, outer, axis_size, inner);
        break;
    }

    case OP_SUM: {
        int axis = node->extra[0];
        int axis_size = node->extra[1];
        int outer = 1, inner = 1;
        for (int i = 0; i < axis; i++) outer *= node->out_shape[i];
        for (int i = axis; i < node->n_dims; i++) inner *= node->out_shape[i];
        kernel_sum((const float*)node->inputs[0],
                   (float*)node->output, outer, axis_size, inner);
        break;
    }

    case OP_SOFTMAX: {
        int cols = node->out_shape[node->n_dims - 1];
        int rows = leading_dims(node);
        if (softmax_mode == 1)
            softmax_scalar((const float*)node->inputs[0],
                           (float*)node->output, rows, cols);
        else
            kernel_softmax((const float*)node->inputs[0],
                           (float*)node->output, rows, cols);
        break;
    }

    case OP_RESHAPE: {
        /* Usually zero-copy alias, but handle copy case */
        if (node->inputs[0] != node->output) {
            memcpy(node->output, node->inputs[0],
                   total_elements(node) * node->elem_size);
        }
        break;
    }

    case OP_SLICE: {
        /* extra = [outer, orig_dim_size, start, slice_len, inner] */
        if (node->extra[0] == 0) break;
        kernel_slice(node->inputs[0], node->output,
                     node->extra[0], node->extra[1], node->extra[2],
                     node->extra[3], node->extra[4], node->elem_size);
        break;
    }

    case OP_EMBEDDING: {
        /* inputs: [ids (int64), table (float)], extra[0] = embed_dim */
        int embed_dim = node->extra[0];
        int seq_len = leading_dims(node);
        kernel_embedding((const long*)node->inputs[0],
                         (const float*)node->inputs[1],
                         (float*)node->output,
                         seq_len, embed_dim);
        break;
    }

    case OP_LAYERNORM: {
        int cols = node->out_shape[node->n_dims - 1];
        int rows = leading_dims(node);
        float eps = extra_float(node, 0);
        switch (layernorm_mode) {
        case 0:
            kernel_layernorm_scalar(
                (const float*)node->inputs[0],
                (const float*)node->inputs[1],
                (const float*)node->inputs[2],
                (float*)node->output, rows, cols, eps);
            break;
        case 1:
            kernel_layernorm_simd(
                (const float*)node->inputs[0],
                (const float*)node->inputs[1],
                (const float*)node->inputs[2],
                (float*)node->output, rows, cols, eps);
            break;
        default:
            kernel_layernorm(
                (const float*)node->inputs[0],
                (const float*)node->inputs[1],
                (const float*)node->inputs[2],
                (float*)node->output, rows, cols, eps);
            break;
        }
        break;
    }

    case OP_MATMUL_ADD: {
        int N = node->out_shape[node->n_dims - 1];
        int K = node->extra[0];
        int trans_b = node->extra[1];
        int b_is_2d = node->extra[2];
        const float* bias = (const float*)node->inputs[2];
        float* out = (float*)node->output;
        int rows = leading_dims(node);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < N; c++)
                out[r * N + c] = bias[c];
        if (b_is_2d) {
            kernel_matmul_beta((const float*)node->inputs[0],
                               (const float*)node->inputs[1],
                               (float*)node->output,
                               rows, N, K, 1, trans_b, 1.0f);
        } else {
            int M = node->out_shape[node->n_dims - 2];
            int batches = 1;
            for (int i = 0; i < node->n_dims - 2; i++)
                batches *= node->out_shape[i];
            kernel_matmul_beta((const float*)node->inputs[0],
                               (const float*)node->inputs[1],
                               (float*)node->output,
                               M, N, K, batches, trans_b, 1.0f);
        }
        break;
    }

    case OP_FUSED_BIAS_RELU: {
        int N = node->out_shape[node->n_dims - 1];
        int M = leading_dims(node);
        kernel_bias_relu((const float*)node->inputs[0],
                         (const float*)node->inputs[1],
                         (float*)node->output, M, N);
        break;
    }

    case OP_RSQRT: {
        kernel_rsqrt((const float*)node->inputs[0],
                     (float*)node->output, total_elements(node));
        break;
    }

    case OP_SILU: {
        kernel_silu((const float*)node->inputs[0],
                    (float*)node->output, total_elements(node));
        break;
    }

    case OP_NEG: {
        kernel_neg((const float*)node->inputs[0],
                   (float*)node->output, total_elements(node));
        break;
    }

    case OP_COS: {
        kernel_cos((const float*)node->inputs[0],
                   (float*)node->output, total_elements(node));
        break;
    }

    case OP_SIN: {
        kernel_sin((const float*)node->inputs[0],
                   (float*)node->output, total_elements(node));
        break;
    }

    case OP_RMSNORM: {
        int cols = node->out_shape[node->n_dims - 1];
        kernel_rmsnorm((const float*)node->inputs[0],
                       (const float*)node->inputs[1],
                       (float*)node->output,
                       leading_dims(node), cols, extra_float(node, 0));
        break;
    }

    case OP_CAT: {
        int dim = node->extra[0];
        int outer = 1, inner = 1;
        for (int i = 0; i < dim; i++) outer *= node->out_shape[i];
        for (int i = dim + 1; i < node->n_dims; i++) inner *= node->out_shape[i];
        kernel_cat(node->inputs, node->output,
                   &node->extra[1], node->n_inputs,
                   outer, inner, node->elem_size);
        break;
    }

    case OP_GATED_ACT: {
        int has_bias = node->extra[0];
        int act_type = node->extra[1];
        int N = node->out_shape[node->n_dims - 1];
        const float* x    = (const float*)node->inputs[0];
        const float* up   = has_bias ? (const float*)node->inputs[2] : (const float*)node->inputs[1];
        const float* bias = has_bias ? (const float*)node->inputs[1] : NULL;
        int M = leading_dims(node);
        if (act_type == 0)
            kernel_gated_silu(x, up, bias, (float*)node->output, M, N, has_bias);
        else
            kernel_gated_gelu(x, up, bias, (float*)node->output, M, N, has_bias);
        break;
    }

    case OP_ATTENTION: {
        /* extras: [seq_len, head_dim, causal, has_mask, group_size]
         * no mask:   inputs = [Q, K, V, scratch]
         * with mask: inputs = [Q, K, V, mask, scratch] */
        int seq_len    = node->extra[0];
        int head_dim   = node->extra[1];
        int causal     = node->extra[2];
        int has_mask   = node->extra[3];
        int group_size = node->extra[4] > 0 ? node->extra[4] : 1;
        int batch_heads = 1;
        for (int i = 0; i < node->n_dims - 2; i++)
            batch_heads *= node->out_shape[i];
        const float* mask = has_mask ? (const float*)node->inputs[3] : NULL;
        float* scratch    = has_mask ? (float*)node->inputs[4] : (float*)node->inputs[3];

        /* Scratch is always >= max(S*S, max_tile) per slice — the Python
         * ablation overrides the planner's scratch calculator. All standard
         * and flash variants can run safely from the same plan. */
        switch (attn_mode) {
        case 0:  /* scalar softmax, sequential */
            kernel_attention_scalar(
                (const float*)node->inputs[0],
                (const float*)node->inputs[1],
                (const float*)node->inputs[2],
                (float*)node->output, scratch, mask,
                batch_heads, seq_len, head_dim,
                causal, group_size);
            break;
        case 1:  /* SIMD softmax, sequential */
            kernel_attention_simd(
                (const float*)node->inputs[0],
                (const float*)node->inputs[1],
                (const float*)node->inputs[2],
                (float*)node->output, scratch, mask,
                batch_heads, seq_len, head_dim,
                causal, group_size);
            break;
        case 3:  /* force flash (ops.c kernel, 128x256, GCD) */
            kernel_attention_flash(
                (const float*)node->inputs[0],
                (const float*)node->inputs[1],
                (const float*)node->inputs[2],
                (float*)node->output, scratch,
                batch_heads, seq_len, head_dim,
                causal, group_size);
            break;
        case 4:  /* force standard (ops.c kernel, GCD) */
            kernel_attention(
                (const float*)node->inputs[0],
                (const float*)node->inputs[1],
                (const float*)node->inputs[2],
                (float*)node->output, scratch, mask,
                batch_heads, seq_len, head_dim,
                causal, group_size);
            break;
        case 5:  /* flash 32x32 */
            kernel_attention_flash_param(
                (const float*)node->inputs[0],
                (const float*)node->inputs[1],
                (const float*)node->inputs[2],
                (float*)node->output, scratch,
                batch_heads, seq_len, head_dim,
                causal, group_size, 32, 32);
            break;
        case 6:  /* flash 64x128 */
            kernel_attention_flash_param(
                (const float*)node->inputs[0],
                (const float*)node->inputs[1],
                (const float*)node->inputs[2],
                (float*)node->output, scratch,
                batch_heads, seq_len, head_dim,
                causal, group_size, 64, 128);
            break;
        case 7:  /* flash 128x256 */
            kernel_attention_flash_param(
                (const float*)node->inputs[0],
                (const float*)node->inputs[1],
                (const float*)node->inputs[2],
                (float*)node->output, scratch,
                batch_heads, seq_len, head_dim,
                causal, group_size, 128, 256);
            break;
        case 8:  /* flash 256x512 */
            kernel_attention_flash_param(
                (const float*)node->inputs[0],
                (const float*)node->inputs[1],
                (const float*)node->inputs[2],
                (float*)node->output, scratch,
                batch_heads, seq_len, head_dim,
                causal, group_size, 256, 512);
            break;
        default:  /* mode 2: adaptive (production behavior) */
            if (!has_mask && seq_len > 256) {
                kernel_attention_flash(
                    (const float*)node->inputs[0],
                    (const float*)node->inputs[1],
                    (const float*)node->inputs[2],
                    (float*)node->output, scratch,
                    batch_heads, seq_len, head_dim,
                    causal, group_size);
            } else {
                kernel_attention(
                    (const float*)node->inputs[0],
                    (const float*)node->inputs[1],
                    (const float*)node->inputs[2],
                    (float*)node->output, scratch, mask,
                    batch_heads, seq_len, head_dim,
                    causal, group_size);
            }
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
 * attn_mode:      0 = scalar, 1 = SIMD, 2 = adaptive (default), 3 = flash,
 *                 4 = standard, 5-8 = flash block size ablation
 *                 (5=32x32, 6=64x128, 7=128x256, 8=256x512)
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
