/*
 * ops.c — C operator kernels for the inference runtime.
 *
 * Each kernel operates on raw float pointers and writes directly
 * to a pre-allocated output buffer. No memory allocation happens here —
 * that's the memory planner's job.
 *
 * MATMUL uses CBLAS (Accelerate on macOS, OpenBLAS on Linux).
 * Everything else is hand-written loops.
 *
 * Build: see Makefile
 */

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <stddef.h>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif
#include <string.h>
#include <math.h>
#include <float.h>

/* ----------------------------------------------------------------
 * MATMUL: out = a @ op(b)  (batched)
 *   a: [batches x M x K]  (row-major, batches=1 for 2D)
 *   b: [batches x K x N]  (row-major) if trans_b=0
 *      [batches x N x K]  (row-major) if trans_b=1
 *   out: [batches x M x N] (row-major)
 *
 * Loops over batch slices calling cblas_sgemm for each [M,K]x[K,N] pair.
 * For 2D inputs, batches=1 and the loop runs once — zero overhead.
 *
 * trans_b=1 is the fast path for nn.Linear weights stored as
 * [out_features, in_features]. CblasTrans reads rows of the stored
 * matrix (stride-1, sequential) during tile packing, which is
 * significantly faster than the stride-N column reads of CblasNoTrans
 * at large dimensions.
 * ---------------------------------------------------------------- */
void kernel_matmul_ab(const float* a, const float* b, float* out,
                      int M, int N, int K, int batches, int trans_b,
                      float alpha, float beta) {
    enum CBLAS_TRANSPOSE tb = trans_b ? CblasTrans : CblasNoTrans;
    int ldb = trans_b ? K : N;
    int a_stride = M * K;
    int b_stride = trans_b ? N * K : K * N;
    int out_stride = M * N;
    for (int i = 0; i < batches; i++) {
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            tb,
            M, N, K,
            alpha,
            a + i * a_stride, K,
            b + i * b_stride, ldb,
            beta,
            out + i * out_stride, N
        );
    }
}

void kernel_matmul_beta(const float* a, const float* b, float* out,
                        int M, int N, int K, int batches, int trans_b,
                        float beta) {
    kernel_matmul_ab(a, b, out, M, N, K, batches, trans_b, 1.0f, beta);
}

void kernel_matmul(const float* a, const float* b, float* out,
                   int M, int N, int K, int batches, int trans_b) {
    kernel_matmul_ab(a, b, out, M, N, K, batches, trans_b, 1.0f, 0.0f);
}

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
void kernel_relu(const float* x, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

/* ----------------------------------------------------------------
 * TRANSPOSE: out = a^T
 *   a: [rows x cols]  (row-major)
 *   out: [cols x rows] (row-major)
 * ---------------------------------------------------------------- */
void kernel_transpose(const float* a, float* out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out[j * rows + i] = a[i * cols + j];
        }
    }
}

/* ----------------------------------------------------------------
 * Element-wise binary ops: tensor × tensor
 * ---------------------------------------------------------------- */
void kernel_div(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] / b[i];
}

void kernel_sub(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] - b[i];
}

void kernel_mul(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
}

/* ----------------------------------------------------------------
 * Element-wise binary ops: tensor × scalar
 * ---------------------------------------------------------------- */
void kernel_add_scalar(const float* a, float s, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] + s;
}

void kernel_div_scalar(const float* a, float s, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] / s;
}

void kernel_sub_scalar(const float* a, float s, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] - s;
}

void kernel_mul_scalar(const float* a, float s, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] * s;
}

/* ----------------------------------------------------------------
 * EXP: out = exp(x)  (element-wise)
 * ---------------------------------------------------------------- */
void kernel_exp(const float* x, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = expf(x[i]);
    }
}

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
#ifdef __APPLE__
    if (rows > 1) {
        /* Chunk rows so each GCD block does enough work to amortize
         * dispatch overhead. Target: >=16KB of input per block, with
         * at most 2× the number of cores to avoid over-splitting. */
        int min_rows_per_chunk = (16 * 1024) / (cols * (int)sizeof(float));
        if (min_rows_per_chunk < 1) min_rows_per_chunk = 1;
        int n_chunks = (rows + min_rows_per_chunk - 1) / min_rows_per_chunk;

        dispatch_queue_t queue = dispatch_get_global_queue(
            QOS_CLASS_USER_INITIATED, 0);
        dispatch_apply((size_t)n_chunks, queue, ^(size_t chunk) {
            int start = (int)chunk * min_rows_per_chunk;
            int end = start + min_rows_per_chunk;
            if (end > rows) end = rows;
            for (int i = start; i < end; i++) {
                layernorm_row(x + i * cols, out + i * cols,
                              gamma, beta, cols, eps);
            }
        });
        return;
    }
#endif
    for (int i = 0; i < rows; i++) {
        layernorm_row(x + i * cols, out + i * cols,
                      gamma, beta, cols, eps);
    }
}

/* ----------------------------------------------------------------
 * BROADCAST BINARY OP: out = a op b with general broadcasting
 *
 *   a_strides/b_strides: pre-computed broadcast strides per dimension.
 *   Broadcast dims have stride 0, so the same element is reused.
 *   out_shape: shape of the output (= broadcast result shape).
 *   ndim: number of dimensions.
 *   op: 0=ADD, 1=SUB, 2=MUL, 3=DIV
 *
 *   Iteration uses coordinate increment with carry (odometer pattern).
 *   The inner dimension increments and breaks immediately for most
 *   elements, so the overhead is O(1) amortized.
 * ---------------------------------------------------------------- */
#define BROADCAST_MAX_DIMS 8

void kernel_broadcast_binop(
    const float* a, const float* b, float* out,
    const int* a_strides, const int* b_strides,
    const int* out_shape, int ndim, int op)
{
    int total = 1;
    for (int d = 0; d < ndim; d++) total *= out_shape[d];

    int coords[BROADCAST_MAX_DIMS] = {0};
    int a_off = 0, b_off = 0;

    for (int i = 0; i < total; i++) {
        float va = a[a_off], vb = b[b_off];
        switch (op) {
            case 0: out[i] = va + vb; break;
            case 1: out[i] = va - vb; break;
            case 2: out[i] = va * vb; break;
            case 3: out[i] = va / vb; break;
        }
        /* Odometer: increment coordinates from last dim */
        for (int d = ndim - 1; d >= 0; d--) {
            coords[d]++;
            a_off += a_strides[d];
            b_off += b_strides[d];
            if (coords[d] < out_shape[d]) break;
            a_off -= coords[d] * a_strides[d];
            b_off -= coords[d] * b_strides[d];
            coords[d] = 0;
        }
    }
}

/* ----------------------------------------------------------------
 * ATTENTION (standard): fused multi-head attention
 *
 *   Q, K, V: [batch_heads x seq_len x head_dim]  (contiguous slices)
 *   output:  [batch_heads x seq_len x head_dim]
 *   scratch: [seq_len x seq_len]  (workspace for attention matrix)
 *
 *   For each (batch, head) slice:
 *     S = Q @ K^T * (1/sqrt(head_dim))     via sgemm alpha
 *     P = softmax(S, dim=-1)               row-wise, in-place on scratch
 *     O = P @ V                            via sgemm
 *
 *   On macOS, slices are dispatched in parallel via GCD (Grand Central
 *   Dispatch). Each thread gets its own scratch buffer — the caller's
 *   buffer is reused by slice 0, others are thread-local allocations.
 * ---------------------------------------------------------------- */

/* Per-slice attention: sgemm → softmax → sgemm */
static void attention_slice(const float* Q_bh, const float* K_bh,
                            const float* V_bh, float* O_bh,
                            float* scratch, int seq_len, int head_dim,
                            float scale) {
    /* S = Q @ K^T * scale */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, seq_len, head_dim,
                scale,
                Q_bh, head_dim,
                K_bh, head_dim,
                0.0f,
                scratch, seq_len);

    /* Softmax in-place (SIMD on Apple) */
    kernel_softmax(scratch, scratch, seq_len, seq_len);

    /* O = P @ V */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                seq_len, head_dim, seq_len,
                1.0f,
                scratch, seq_len,
                V_bh, head_dim,
                0.0f,
                O_bh, head_dim);
}

void kernel_attention(const float* Q, const float* K, const float* V,
                      float* output, float* scratch,
                      int batch_heads, int seq_len, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int slice = seq_len * head_dim;
    int scratch_per_slice = seq_len * seq_len;

#ifdef __APPLE__
    if (batch_heads > 1) {
        dispatch_queue_t queue = dispatch_get_global_queue(
            QOS_CLASS_USER_INITIATED, 0);
        dispatch_apply((size_t)batch_heads, queue, ^(size_t bh) {
            attention_slice(Q + bh * slice, K + bh * slice,
                            V + bh * slice, output + bh * slice,
                            scratch + bh * scratch_per_slice,
                            seq_len, head_dim, scale);
        });
        return;
    }
#endif

    /* Single-slice or non-Apple: sequential */
    for (int bh = 0; bh < batch_heads; bh++) {
        attention_slice(Q + bh * slice, K + bh * slice,
                        V + bh * slice, output + bh * slice,
                        scratch + bh * scratch_per_slice,
                        seq_len, head_dim, scale);
    }
}


/* ----------------------------------------------------------------
 * ATTENTION (flash): tiled online-softmax attention
 *
 *   Same interface as kernel_attention — drop-in replacement.
 *   Never materializes the full S×S attention matrix. Instead:
 *
 *   For each (batch, head) slice:
 *     For each query block (B_r rows of Q):
 *       Initialize running state: m=-inf, l=0, O=0
 *       For each key/value block (B_c rows of K, V):
 *         S_tile = Q_block @ K_block^T * scale        [B_r × B_c]
 *         Update running max, rescale old accumulators
 *         P_tile = exp(S_tile - m_new)                [B_r × B_c]
 *         O += P_tile @ V_block
 *         Update running sum
 *       Normalize: O /= l
 *
 *   scratch requirement: B_r × B_c floats (one tile, not S×S).
 *
 *   Tile sizes are chosen so the working set (Q/K/V blocks + S tile +
 *   O block) fits in L1 cache. With B_r=B_c=32 and D=64:
 *   ~36 KB working set, comfortably within 48-64 KB L1.
 * ---------------------------------------------------------------- */
#define FLASH_BR 32
#define FLASH_BC 32

/* Per-slice flash attention */
static void attention_flash_slice(const float* Q_bh, const float* K_bh,
                                  const float* V_bh, float* O_bh,
                                  float* scratch, int seq_len, int head_dim,
                                  float scale) {
    for (int qi = 0; qi < seq_len; qi += FLASH_BR) {
        int br = (qi + FLASH_BR <= seq_len) ? FLASH_BR : seq_len - qi;

        const float* Q_block = Q_bh + qi * head_dim;
        float*       O_block = O_bh + qi * head_dim;

        float m[FLASH_BR];
        float l[FLASH_BR];
        for (int r = 0; r < br; r++) {
            m[r] = -FLT_MAX;
            l[r] = 0.0f;
        }
        for (int r = 0; r < br; r++)
            for (int d = 0; d < head_dim; d++)
                O_block[r * head_dim + d] = 0.0f;

        for (int kj = 0; kj < seq_len; kj += FLASH_BC) {
            int bc = (kj + FLASH_BC <= seq_len) ? FLASH_BC : seq_len - kj;

            const float* K_block = K_bh + kj * head_dim;
            const float* V_block = V_bh + kj * head_dim;

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        br, bc, head_dim,
                        scale,
                        Q_block, head_dim,
                        K_block, head_dim,
                        0.0f,
                        scratch, bc);

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

void kernel_attention_flash(const float* Q, const float* K, const float* V,
                            float* output, float* scratch,
                            int batch_heads, int seq_len, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int slice = seq_len * head_dim;
    int scratch_per_slice = FLASH_BR * FLASH_BC;

#ifdef __APPLE__
    if (batch_heads > 1) {
        dispatch_queue_t queue = dispatch_get_global_queue(
            QOS_CLASS_USER_INITIATED, 0);
        dispatch_apply((size_t)batch_heads, queue, ^(size_t bh) {
            attention_flash_slice(Q + bh * slice, K + bh * slice,
                                  V + bh * slice, output + bh * slice,
                                  scratch + bh * scratch_per_slice,
                                  seq_len, head_dim, scale);
        });
        return;
    }
#endif

    for (int bh = 0; bh < batch_heads; bh++) {
        attention_flash_slice(Q + bh * slice, K + bh * slice,
                              V + bh * slice, output + bh * slice,
                              scratch + bh * scratch_per_slice,
                              seq_len, head_dim, scale);
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

