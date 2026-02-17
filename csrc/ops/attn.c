#include "../ops.h"

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

/* Per-slice attention: sgemm → [mask] → softmax → sgemm
 *
 * mask_bh: NULL (no mask), or pointer to [seq_len x seq_len] additive mask.
 *          For causal masking the caller passes NULL and sets causal=1;
 *          the kernel generates the upper-triangular -inf pattern on the fly.
 */
static void attention_slice(const float* Q_bh, const float* K_bh,
                            const float* V_bh, float* O_bh,
                            float* scratch, const float* mask_bh,
                            int seq_len, int head_dim,
                            float scale, int causal) {
    /* S = Q @ K^T * scale */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, seq_len, head_dim,
                scale,
                Q_bh, head_dim,
                K_bh, head_dim,
                0.0f,
                scratch, seq_len);

    /* Apply mask: either custom additive mask or causal (never both) */
    if (mask_bh) {
        int ss = seq_len * seq_len;
        for (int i = 0; i < ss; i++)
            scratch[i] += mask_bh[i];
    } else if (causal) {
        for (int i = 0; i < seq_len; i++)
            for (int j = i + 1; j < seq_len; j++)
                scratch[i * seq_len + j] = -INFINITY;
    }

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
                      float* output, float* scratch, const float* mask,
                      int batch_heads, int seq_len, int head_dim,
                      int causal, int group_size) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int slice = seq_len * head_dim;
    int scratch_per_slice = seq_len * seq_len;

    PARALLEL_FOR(batch_heads, slice * (int)sizeof(float), bh, {
        int kv_bh = bh / group_size;
        const float* mask_bh = mask ? mask + bh * scratch_per_slice : NULL;
        attention_slice(Q + bh * slice, K + kv_bh * slice,
                        V + kv_bh * slice, output + bh * slice,
                        scratch + bh * scratch_per_slice, mask_bh,
                        seq_len, head_dim, scale, causal);
    });
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
#define FLASH_BR 128
#define FLASH_BC 256

/* Adaptive dispatch: use flash for seq_len > this threshold.
 * Must match FLASH_SEQ_THRESHOLD in executor.c, ops.py, c_backend.py. */
#define FLASH_SEQ_THRESHOLD 256

/* Per-slice flash attention (with optional causal masking).
 *
 * When causal=1, for query row qi+r and key column kj+c we require
 * kj+c <= qi+r (lower-triangular). Three cases per KV tile:
 *   - kj >= qi+br:  entire tile masked → skip (don't even sgemm)
 *   - kj+bc <= qi:  entire tile unmasked → process normally
 *   - otherwise:    partial → apply causal mask after sgemm
 */
static void attention_flash_slice(const float* Q_bh, const float* K_bh,
                                  const float* V_bh, float* O_bh,
                                  float* scratch, int seq_len, int head_dim,
                                  float scale, int causal) {
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

        /* KV tile limit: for causal, stop when entire tile is masked */
        int kv_end = causal ? seq_len : seq_len;
        for (int kj = 0; kj < kv_end; kj += FLASH_BC) {
            int bc = (kj + FLASH_BC <= seq_len) ? FLASH_BC : seq_len - kj;

            /* Causal: skip fully-masked tiles (all keys after all queries) */
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
                    int last_valid = qi + r - kj;  /* last unmasked column */
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

void kernel_attention_flash(const float* Q, const float* K, const float* V,
                            float* output, float* scratch,
                            int batch_heads, int seq_len, int head_dim,
                            int causal, int group_size) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int slice = seq_len * head_dim;
    int scratch_per_slice = FLASH_BR * FLASH_BC;

    PARALLEL_FOR(batch_heads, slice * (int)sizeof(float), bh, {
        int kv_bh = bh / group_size;
        attention_flash_slice(Q + bh * slice, K + kv_bh * slice,
                              V + kv_bh * slice, output + bh * slice,
                              scratch + bh * scratch_per_slice,
                              seq_len, head_dim, scale, causal);
    });
}

