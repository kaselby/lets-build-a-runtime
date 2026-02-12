/*
 * executor.c — C dispatch loop for the inference runtime.
 *
 * Receives a pre-compiled array of OpNode structs from Python and
 * executes them sequentially. No memory allocation, no name resolution —
 * just a tight loop dispatching to kernel functions defined in ops.c.
 *
 * The Python side builds the OpNode array once ("compilation"), then
 * patches input pointers and calls execute() per inference.
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* Must match the Python-side OpType enum values */
#define OP_MATMUL    1
#define OP_ADD       2
#define OP_RELU      3
#define OP_TRANSPOSE 4
#define OP_PERMUTE   5
#define OP_DIV       6
#define OP_SUB       7
#define OP_MUL       8
#define OP_EXP       9
#define OP_MAX       10
#define OP_SUM       11
#define OP_SOFTMAX   12
#define OP_RESHAPE   13
#define OP_LAYERNORM 14
#define OP_MATMUL_ADD 15
#define OP_FUSED_BIAS_RELU 16
#define OP_ATTENTION 17
#define OP_SLICE     18
#define OP_POW       19
#define OP_TANH      20
#define OP_GELU      21
#define OP_EMBEDDING 22

#define MAX_INPUTS 8
#define MAX_DIMS   16

typedef struct {
    int op;
    int n_inputs;
    void* inputs[MAX_INPUTS];
    void* output;
    int out_shape[MAX_DIMS];
    int n_dims;
    int extra[MAX_DIMS];  /* op-specific: axes, flags, etc. */
} OpNode;

/* ----------------------------------------------------------------
 * Forward declarations for kernels in ops.c
 * ---------------------------------------------------------------- */

void kernel_matmul(const float* a, const float* b, float* out,
                   int M, int N, int K, int batches, int trans_b);
void kernel_matmul_ab(const float* a, const float* b, float* out,
                      int M, int N, int K, int batches, int trans_b,
                      float alpha, float beta);
void kernel_matmul_beta(const float* a, const float* b, float* out,
                        int M, int N, int K, int batches, int trans_b,
                        float beta);
void kernel_add(const float* a, const float* bias, float* out,
                int M, int N);
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
void kernel_bias_relu(const float* a, const float* bias, float* out,
                      int M, int N);
void kernel_attention(const float* Q, const float* K, const float* V,
                      float* output, float* scratch,
                      int batch_heads, int seq_len, int head_dim);
void kernel_pow_scalar(const float* x, float scalar, float* out, int n);
void kernel_tanh(const float* x, float* out, int n);
void kernel_gelu_tanh(const float* x, float* out, int n);
void kernel_embedding(const long* ids, const float* table, float* out,
                      int seq_len, int embed_dim);

/* ----------------------------------------------------------------
 * Dispatch
 * ---------------------------------------------------------------- */

static void dispatch(OpNode* node) {
    switch (node->op) {
        case OP_MATMUL: {
            /* out_shape = [..., M, N], extra[0] = K, extra[1] = trans_b,
             * extra[2] = b_is_2d (ND×2D: flatten batch into M),
             * extra[3] = alpha as float bits (0 means 1.0) */
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
            /* extra[0]: 0 = bias broadcast, 1 = element-wise, 2 = scalar */
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
                int rows = node->extra[0];
                int cols = node->extra[1];
                kernel_transpose(node->inputs[0], node->output, rows, cols);
            } else {
                /* General N-dim swapaxes: decompose into [outer, A, mid, B, inner]
                 * extra = [outer, A, middle, B, inner] */
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
            /* extra[0]: 0 = two-tensor, 1 = scalar (extra[1] = float bits) */
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
            int outer = 1, inner = 1;
            int axis_size = node->extra[1];
            for (int i = 0; i < axis; i++) outer *= node->out_shape[i];
            for (int i = axis; i < node->n_dims; i++) inner *= node->out_shape[i];
            kernel_max(node->inputs[0], node->output, outer, axis_size, inner);
            break;
        }
        case OP_SUM: {
            int axis = node->extra[0];
            int outer = 1, inner = 1;
            int axis_size = node->extra[1];
            for (int i = 0; i < axis; i++) outer *= node->out_shape[i];
            for (int i = axis; i < node->n_dims; i++) inner *= node->out_shape[i];
            kernel_sum(node->inputs[0], node->output, outer, axis_size, inner);
            break;
        }
        case OP_SOFTMAX: {
            int cols = node->out_shape[node->n_dims - 1];
            int rows = 1;
            for (int i = 0; i < node->n_dims - 1; i++) rows *= node->out_shape[i];
            kernel_softmax(node->inputs[0], node->output, rows, cols);
            break;
        }
        case OP_RESHAPE:
            /* Zero-copy: handled by Python alias binding, never dispatched */
            break;
        case OP_LAYERNORM: {
            int cols = node->out_shape[node->n_dims - 1];
            int rows = 1;
            for (int i = 0; i < node->n_dims - 1; i++) rows *= node->out_shape[i];
            union { int i; float f; } eps_u;
            eps_u.i = node->extra[0];
            kernel_layernorm(node->inputs[0], node->inputs[1], node->inputs[2],
                             node->output, rows, cols, eps_u.f);
            break;
        }
        case OP_MATMUL_ADD: {
            /* inputs: [A, B, bias], extras: [K, trans_b, b_is_2d]
             * Pre-fill output with broadcast bias, then sgemm with beta=1.0. */
            int N = node->out_shape[node->n_dims - 1];
            int K = node->extra[0];
            int trans_b = node->extra[1];
            int b_is_2d = node->extra[2];
            const float* bias = (const float*)node->inputs[2];
            float* out = (float*)node->output;

            int rows = 1;
            for (int i = 0; i < node->n_dims - 1; i++)
                rows *= node->out_shape[i];
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < N; c++)
                    out[r * N + c] = bias[c];

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
            /* inputs: [Q, K, V, scratch], extras: [seq_len, head_dim]
             * out_shape = [..., seq_len, head_dim] (leading dims are batch) */
            int seq_len = node->extra[0];
            int head_dim = node->extra[1];
            int batch_heads = 1;
            for (int i = 0; i < node->n_dims - 2; i++)
                batch_heads *= node->out_shape[i];
            kernel_attention(node->inputs[0], node->inputs[1], node->inputs[2],
                             node->output, node->inputs[3],
                             batch_heads, seq_len, head_dim);
            break;
        }
        case OP_SLICE:
            /* Zero-copy: handled by Python alias binding, like RESHAPE */
            break;
        case OP_POW: {
            /* extra[0] = scalar exponent as float bits */
            int n = 1;
            for (int i = 0; i < node->n_dims; i++) n *= node->out_shape[i];
            union { int i; float f; } s;
            s.i = node->extra[0];
            kernel_pow_scalar(node->inputs[0], s.f, node->output, n);
            break;
        }
        case OP_TANH: {
            int n = 1;
            for (int i = 0; i < node->n_dims; i++) n *= node->out_shape[i];
            kernel_tanh(node->inputs[0], node->output, n);
            break;
        }
        case OP_GELU: {
            int n = 1;
            for (int i = 0; i < node->n_dims; i++) n *= node->out_shape[i];
            kernel_gelu_tanh(node->inputs[0], node->output, n);
            break;
        }
        case OP_EMBEDDING: {
            /* inputs: [ids, table], extra[0] = embed_dim
             * ids are long* (int64), table is float* */
            int embed_dim = node->extra[0];
            int seq_len = 1;
            for (int i = 0; i < node->n_dims - 1; i++)
                seq_len *= node->out_shape[i];
            kernel_embedding((const long*)node->inputs[0],
                             (const float*)node->inputs[1],
                             (float*)node->output,
                             seq_len, embed_dim);
            break;
        }
        default:
            fprintf(stderr, "executor: unknown op %d\n", node->op);
            abort();
    }
}

/* ----------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------- */

void execute(OpNode* nodes, int n_nodes) {
    for (int i = 0; i < n_nodes; i++) {
        dispatch(&nodes[i]);
    }
}
