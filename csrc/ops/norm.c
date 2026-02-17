#include "../ops.h"

/* ----------------------------------------------------------------
 * LAYERNORM: y = (x - mean) / sqrt(var + eps) * gamma + beta
 *   x: [rows x cols]
 *   gamma, beta: [cols]  (learnable scale and shift)
 *   out: [rows x cols]
 *
 *   Two-pass, numerically stable (variance from centered values).
 *
 *   On macOS, each pass uses Accelerate vDSP SIMD functions and
 *   rows are parallelized across cores via GCD dispatch_apply:
 *     vDSP_sve    — vectorized sum (NEON parallel reduction)
 *     vDSP_vsadd  — vectorized scalar add (center: x - mean)
 *     vDSP_svesq  — vectorized sum of squares (variance)
 *     vDSP_vsmul  — vectorized scalar multiply (scale by inv_std)
 *     vDSP_vma    — vectorized multiply-add (apply gamma and beta)
 * ---------------------------------------------------------------- */

/* Per-row layernorm: compute mean/var and normalize one row. */
static void layernorm_row(const float* row_in, float* row_out,
                          const float* gamma, const float* beta,
                          int cols, float eps) {
#ifdef __APPLE__
    vDSP_Length n = (vDSP_Length)cols;

    /* Pass 1: mean via SIMD reduction */
    float sum;
    vDSP_sve(row_in, 1, &sum, n);
    float mean = sum / cols;

    /* Write centered values (x - mean) into output buffer.
     * This serves double duty: scratch for variance computation
     * and the start of the normalization pipeline. */
    float neg_mean = -mean;
    vDSP_vsadd(row_in, 1, &neg_mean, row_out, 1, n);

    /* Variance from centered values (numerically stable) */
    float var_sum;
    vDSP_svesq(row_out, 1, &var_sum, n);
    float inv_std = 1.0f / sqrtf(var_sum / cols + eps);

    /* Pass 2: scale by inv_std, then apply gamma and beta.
     * row_out already holds (x - mean) from above. */
    vDSP_vsmul(row_out, 1, &inv_std, row_out, 1, n);
    vDSP_vma(row_out, 1, gamma, 1, beta, 1, row_out, 1, n);
#else
    /* Scalar fallback */
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) sum += row_in[j];
    float mean = sum / cols;

    float var = 0.0f;
    for (int j = 0; j < cols; j++) {
        float d = row_in[j] - mean;
        row_out[j] = d;      /* store centered value for reuse */
        var += d * d;
    }
    float inv_std = 1.0f / sqrtf(var / cols + eps);

    for (int j = 0; j < cols; j++) {
        row_out[j] = row_out[j] * inv_std * gamma[j] + beta[j];
    }
#endif
}

void kernel_layernorm(const float* x, const float* gamma, const float* beta,
                      float* out, int rows, int cols, float eps) {
    PARALLEL_FOR(rows, cols * (int)sizeof(float), i, {
        layernorm_row(x + i * cols, out + i * cols, gamma, beta, cols, eps);
    });
}

/* ----------------------------------------------------------------
 * RMSNORM: y = x / sqrt(mean(x^2) + eps) * weight
 *   x: [rows x cols]
 *   weight: [cols]  (learnable scale)
 *   out: [rows x cols]
 *
 *   Simpler than LayerNorm: no mean subtraction, no bias.
 *   On macOS, uses Accelerate SIMD + GCD threading (same strategy
 *   as kernel_layernorm).
 * ---------------------------------------------------------------- */

static void rmsnorm_row(const float* row_in, float* row_out,
                        const float* weight, int cols, float eps) {
#ifdef __APPLE__
    vDSP_Length n = (vDSP_Length)cols;

    /* Sum of squares via SIMD */
    float ss;
    vDSP_svesq(row_in, 1, &ss, n);
    float scale = 1.0f / sqrtf(ss / cols + eps);

    /* Scale input by 1/rms */
    vDSP_vsmul(row_in, 1, &scale, row_out, 1, n);

    /* Element-wise multiply by weight */
    vDSP_vmul(row_out, 1, weight, 1, row_out, 1, n);
#else
    float ss = 0.0f;
    for (int j = 0; j < cols; j++)
        ss += row_in[j] * row_in[j];
    float scale = 1.0f / sqrtf(ss / cols + eps);
    for (int j = 0; j < cols; j++)
        row_out[j] = row_in[j] * scale * weight[j];
#endif
}

void kernel_rmsnorm(const float* x, const float* weight, float* out,
                    int rows, int cols, float eps) {
    PARALLEL_FOR(rows, cols * (int)sizeof(float), i, {
        rmsnorm_row(x + i * cols, out + i * cols, weight, cols, eps);
    });
}

