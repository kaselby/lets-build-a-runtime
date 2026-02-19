#include "../ops.h"

/* ----------------------------------------------------------------
 * ADD (bias): out = a + bias
 *   a: [M x N]  (row-major)
 *   bias: [N]   (broadcast across rows)
 *   out: [M x N] (row-major)
 * ---------------------------------------------------------------- */
void kernel_add(const float* a, const float* bias, float* out,
                int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            out[i * N + j] = a[i * N + j] + bias[j];
        }
    }
}

/* ----------------------------------------------------------------
 * RELU: out = max(x, 0)
 *   x: [n] (flat)
 *   out: [n] (flat)
 * ---------------------------------------------------------------- */
void kernel_relu(const float* x, float* out, long n) {
    for (long i = 0; i < n; i++) {
        out[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

/* ----------------------------------------------------------------
 * Element-wise binary ops: tensor × tensor
 * ---------------------------------------------------------------- */
void kernel_div(const float* a, const float* b, float* out, long n) {
    for (long i = 0; i < n; i++) out[i] = a[i] / b[i];
}

void kernel_sub(const float* a, const float* b, float* out, long n) {
    for (long i = 0; i < n; i++) out[i] = a[i] - b[i];
}

void kernel_mul(const float* a, const float* b, float* out, long n) {
    for (long i = 0; i < n; i++) out[i] = a[i] * b[i];
}

/* ----------------------------------------------------------------
 * Element-wise binary ops: tensor × scalar
 * ---------------------------------------------------------------- */
void kernel_add_scalar(const float* a, float s, float* out, long n) {
    for (long i = 0; i < n; i++) out[i] = a[i] + s;
}

void kernel_div_scalar(const float* a, float s, float* out, long n) {
    for (long i = 0; i < n; i++) out[i] = a[i] / s;
}

void kernel_sub_scalar(const float* a, float s, float* out, long n) {
    for (long i = 0; i < n; i++) out[i] = a[i] - s;
}

void kernel_mul_scalar(const float* a, float s, float* out, long n) {
    for (long i = 0; i < n; i++) out[i] = a[i] * s;
}

/* ----------------------------------------------------------------
 * EXP: out = exp(x)  (element-wise)
 * ---------------------------------------------------------------- */
void kernel_exp(const float* x, float* out, long n) {
    for (long i = 0; i < n; i++) {
        out[i] = expf(x[i]);
    }
}

/* ----------------------------------------------------------------
 * BIAS_RELU: out = max(a + bias, 0)
 *   a: [M x N]  (row-major)
 *   bias: [N]   (broadcast across rows)
 *   out: [M x N] (row-major)
 *
 *   Single pass over memory — no intermediate buffer between add and relu.
 *   Bespoke fused kernel for benchmarking against the general
 *   FUSED_ELEMENTWISE interpreter.
 * ---------------------------------------------------------------- */
void kernel_bias_relu(const float* a, const float* bias, float* out,
                      int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float v = a[i * N + j] + bias[j];
            out[i * N + j] = v > 0.0f ? v : 0.0f;
        }
    }
}

/* ----------------------------------------------------------------
 * GATED_SILU: out = silu(x + bias?) * up
 *   x, up: [M, N]  bias: [N] (optional)  out: [M, N]
 * ---------------------------------------------------------------- */
void kernel_gated_silu(const float* x, const float* up, const float* bias,
                       float* out, int M, int N, int has_bias) {
    if (has_bias) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float v = x[i * N + j] + bias[j];
                out[i * N + j] = v / (1.0f + expf(-v)) * up[i * N + j];
            }
        }
    } else {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float v = x[i * N + j];
                out[i * N + j] = v / (1.0f + expf(-v)) * up[i * N + j];
            }
        }
    }
}

/* ----------------------------------------------------------------
 * GATED_GELU: out = gelu_tanh(x + bias?) * up
 *   x, up: [M, N]  bias: [N] (optional)  out: [M, N]
 * ---------------------------------------------------------------- */
void kernel_gated_gelu(const float* x, const float* up, const float* bias,
                       float* out, int M, int N, int has_bias) {
    if (has_bias) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float v = x[i * N + j] + bias[j];
                out[i * N + j] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v))) * up[i * N + j];
            }
        }
    } else {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float v = x[i * N + j];
                out[i * N + j] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v))) * up[i * N + j];
            }
        }
    }
}

/* ----------------------------------------------------------------
 * POW_SCALAR: out = x^scalar  (element-wise)
 *   x: [n] (flat)
 *   out: [n] (flat)
 * ---------------------------------------------------------------- */
void kernel_pow_scalar(const float* x, float scalar, float* out, long n) {
    for (long i = 0; i < n; i++) {
        out[i] = powf(x[i], scalar);
    }
}

/* ----------------------------------------------------------------
 * TANH: out = tanh(x)  (element-wise)
 *   x: [n] (flat)
 *   out: [n] (flat)
 * ---------------------------------------------------------------- */
void kernel_tanh(const float* x, float* out, long n) {
    for (long i = 0; i < n; i++) {
        out[i] = tanhf(x[i]);
    }
}

/* ----------------------------------------------------------------
 * GELU (tanh approximation):
 *   out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 *   On macOS, uses Accelerate's vvtanhf for SIMD vectorization of
 *   the tanh part, same approach as kernel_softmax with vvexpf.
 * ---------------------------------------------------------------- */
void kernel_gelu_tanh(const float* x, float* out, long n) {
#ifdef __APPLE__
    /* Compute the tanh argument: sqrt(2/pi) * (x + 0.044715 * x^3) */
    for (long i = 0; i < n; i++) {
        float xi = x[i];
        out[i] = 0.7978845608f * (xi + 0.044715f * xi * xi * xi);
    }
    /* Vectorized tanh via Accelerate (NEON SIMD).
     * vvtanhf takes int* count — chunk to avoid overflow for n > INT_MAX. */
    for (long off = 0; off < n; ) {
        int chunk = (n - off > INT_MAX) ? INT_MAX : (int)(n - off);
        vvtanhf(out + off, out + off, &chunk);
        off += chunk;
    }
    /* Final: 0.5 * x * (1 + tanh_result) */
    for (long i = 0; i < n; i++) {
        out[i] = 0.5f * x[i] * (1.0f + out[i]);
    }
#else
    for (long i = 0; i < n; i++) {
        float xi = x[i];
        out[i] = 0.5f * xi * (1.0f + tanhf(0.7978845608f * (xi + 0.044715f * xi * xi * xi)));
    }
#endif
}

/* ----------------------------------------------------------------
 * RSQRT: out = 1/sqrt(x)  (element-wise)
 * ---------------------------------------------------------------- */
void kernel_rsqrt(const float* x, float* out, long n) {
    for (long i = 0; i < n; i++) out[i] = 1.0f / sqrtf(x[i]);
}

/* ----------------------------------------------------------------
 * SILU: out = x * sigmoid(x)  (element-wise)
 * ---------------------------------------------------------------- */
void kernel_silu(const float* x, float* out, long n) {
    for (long i = 0; i < n; i++) out[i] = x[i] / (1.0f + expf(-x[i]));
}

/* ----------------------------------------------------------------
 * NEG: out = -x  (element-wise)
 * ---------------------------------------------------------------- */
void kernel_neg(const float* x, float* out, long n) {
    for (long i = 0; i < n; i++) out[i] = -x[i];
}

/* ----------------------------------------------------------------
 * COS: out = cos(x)  (element-wise)
 * ---------------------------------------------------------------- */
void kernel_cos(const float* x, float* out, long n) {
    for (long i = 0; i < n; i++) out[i] = cosf(x[i]);
}

/* ----------------------------------------------------------------
 * SIN: out = sin(x)  (element-wise)
 * ---------------------------------------------------------------- */
void kernel_sin(const float* x, float* out, long n) {
    for (long i = 0; i < n; i++) out[i] = sinf(x[i]);
}
