#include "../ops.h"

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
 * MATMUL (M=1): out = a @ op(b)  via sgemv
 *   a: [batches x K]   (one row per batch)
 *   b: [batches x K x N] or [batches x N x K] if trans_b
 *   out: [batches x N]
 *
 * sgemv skips sgemm's tiling/blocking setup, which is pure overhead
 * when there's only one output row. The transpose mapping flips:
 *   trans_b=1: b stored as N×K → sgemv(NoTrans) reads rows directly
 *   trans_b=0: b stored as K×N → sgemv(Trans) transposes on the fly
 * ---------------------------------------------------------------- */
void kernel_matmul_gemv(const float* a, const float* b, float* out,
                        int N, int K, int batches, int trans_b,
                        float alpha, float beta) {
    enum CBLAS_TRANSPOSE tv = trans_b ? CblasNoTrans : CblasTrans;
    int mv_rows = trans_b ? N : K;
    int mv_cols = trans_b ? K : N;
    int ldb = trans_b ? K : N;
    int b_stride = K * N;
    for (int i = 0; i < batches; i++) {
        cblas_sgemv(CblasRowMajor, tv, mv_rows, mv_cols, alpha,
                    b + i * b_stride, ldb,
                    a + i * K, 1,
                    beta,
                    out + i * N, 1);
    }
}
