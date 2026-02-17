#include "../ops.h"

/* ----------------------------------------------------------------
 * MAX: reduce max along middle axis
 *   x: logically [outer x axis_size x inner]
 *   out: [outer x inner]
 * ---------------------------------------------------------------- */
void kernel_max(const float* x, float* out, int outer, int axis_size, int inner) {
    for (int o = 0; o < outer; o++) {
        for (int j = 0; j < inner; j++) {
            float mx = -FLT_MAX;
            for (int a = 0; a < axis_size; a++) {
                float v = x[(o * axis_size + a) * inner + j];
                if (v > mx) mx = v;
            }
            out[o * inner + j] = mx;
        }
    }
}

/* ----------------------------------------------------------------
 * SUM: reduce sum along middle axis
 *   x: logically [outer x axis_size x inner]
 *   out: [outer x inner]
 * ---------------------------------------------------------------- */
void kernel_sum(const float* x, float* out, int outer, int axis_size, int inner) {
    for (int o = 0; o < outer; o++) {
        for (int j = 0; j < inner; j++) {
            float s = 0.0f;
            for (int a = 0; a < axis_size; a++) {
                s += x[(o * axis_size + a) * inner + j];
            }
            out[o * inner + j] = s;
        }
    }
}

/* ----------------------------------------------------------------
 * SOFTMAX: numerically stable softmax along last axis
 *   x: [rows x cols]
 *   out: [rows x cols]
 *   softmax is along cols dimension
 *
 *   On macOS, uses Accelerate's vDSP/vForce SIMD functions:
 *     vDSP_maxv   — vectorized max (NEON parallel reduction)
 *     vDSP_vsadd  — vectorized scalar add (subtract max from each element)
 *     vvexpf      — vectorized exp (NEON, 4 floats per cycle)
 *     vDSP_sve    — vectorized sum
 *     vDSP_vsdiv  — vectorized scalar divide (normalize)
 *   This processes 4 floats per instruction vs 1 for scalar expf,
 *   giving ~4-5x speedup on large rows.
 * ---------------------------------------------------------------- */
void kernel_softmax(const float* x, float* out, int rows, int cols) {
#ifdef __APPLE__
    int n = cols;
    for (int i = 0; i < rows; i++) {
        const float* row_in = x + i * cols;
        float* row_out = out + i * cols;

        /* Max (SIMD parallel reduction) */
        float mx;
        vDSP_maxv(row_in, 1, &mx, (vDSP_Length)cols);

        /* out = x - max (SIMD scalar add) */
        float neg_mx = -mx;
        vDSP_vsadd(row_in, 1, &neg_mx, row_out, 1, (vDSP_Length)cols);

        /* out = exp(out) (SIMD vectorized exp, 4 floats/cycle on NEON) */
        vvexpf(row_out, row_out, &n);

        /* Sum (SIMD reduction) */
        float sum;
        vDSP_sve(row_out, 1, &sum, (vDSP_Length)cols);

        /* out /= sum (SIMD scalar divide) */
        vDSP_vsdiv(row_out, 1, &sum, row_out, 1, (vDSP_Length)cols);
    }
#else
    /* Scalar fallback for non-Apple platforms */
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
#endif
}
