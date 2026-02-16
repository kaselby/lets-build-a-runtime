/*
 * executor.c — C dispatch loop for the inference runtime.
 *
 * Receives a pre-compiled array of OpNode structs from Python and
 * executes them sequentially. No memory allocation, no name resolution —
 * just a tight loop dispatching via function pointer table.
 *
 * The Python side builds the OpNode array once ("compilation"), then
 * patches input pointers and calls execute() per inference.
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "runtime.h"

/* Table size: must exceed the highest op code (currently 72). */
#define DISPATCH_TABLE_SIZE 100

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
                      float* output, float* scratch, const float* mask,
                      int batch_heads, int seq_len, int head_dim,
                      int causal);
void kernel_pow_scalar(const float* x, float scalar, float* out, int n);
void kernel_tanh(const float* x, float* out, int n);
void kernel_gelu_tanh(const float* x, float* out, int n);
void kernel_embedding(const long* ids, const float* table, float* out,
                      int seq_len, int embed_dim);
void kernel_slice(const void* x, void* out,
                  int outer, int orig_dim_size, int start,
                  int slice_len, int inner, int elem_size);

/* ----------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------- */

/* Product of all output shape dimensions. */
static inline int total_elements(const OpNode* node) {
    int n = 1;
    for (int i = 0; i < node->n_dims; i++) n *= node->out_shape[i];
    return n;
}

/* Reinterpret extra[idx] as a float (bit-cast). */
static inline float extra_float(const OpNode* node, int idx) {
    union { int i; float f; } u;
    u.i = node->extra[idx];
    return u.f;
}

/* Product of out_shape[0..n_dims-2] (everything except the last dim). */
static inline int leading_dims(const OpNode* node) {
    int rows = 1;
    for (int i = 0; i < node->n_dims - 1; i++) rows *= node->out_shape[i];
    return rows;
}

/* ----------------------------------------------------------------
 * Dispatch functions — one per op
 * ---------------------------------------------------------------- */

/* --- Element-wise unary --- */

static void dispatch_relu(OpNode* node) {
    kernel_relu(node->inputs[0], node->output, total_elements(node));
}

static void dispatch_exp(OpNode* node) {
    kernel_exp(node->inputs[0], node->output, total_elements(node));
}

static void dispatch_tanh(OpNode* node) {
    kernel_tanh(node->inputs[0], node->output, total_elements(node));
}

static void dispatch_pow(OpNode* node) {
    /* extra[0] = scalar exponent as float bits */
    kernel_pow_scalar(node->inputs[0], extra_float(node, 0),
                      node->output, total_elements(node));
}

static void dispatch_gelu(OpNode* node) {
    kernel_gelu_tanh(node->inputs[0], node->output, total_elements(node));
}

/* --- Element-wise binary --- */

static void dispatch_add(OpNode* node) {
    /* extra[0]: 0 = bias broadcast, 1 = element-wise, 2 = scalar */
    int n = total_elements(node);
    if (node->extra[0] == 2) {
        kernel_add_scalar(node->inputs[0], extra_float(node, 1),
                          node->output, n);
    } else if (node->extra[0] == 1) {
        kernel_add(node->inputs[0], node->inputs[1], node->output, 1, n);
    } else {
        int N = node->out_shape[node->n_dims - 1];
        kernel_add(node->inputs[0], node->inputs[1], node->output, n / N, N);
    }
}

static void dispatch_sub(OpNode* node) {
    /* extra[0]: 0 = two-tensor, 1 = scalar (extra[1] = float bits) */
    int n = total_elements(node);
    if (node->extra[0]) {
        kernel_sub_scalar(node->inputs[0], extra_float(node, 1),
                          node->output, n);
    } else {
        kernel_sub(node->inputs[0], node->inputs[1], node->output, n);
    }
}

static void dispatch_mul(OpNode* node) {
    int n = total_elements(node);
    if (node->extra[0]) {
        kernel_mul_scalar(node->inputs[0], extra_float(node, 1),
                          node->output, n);
    } else {
        kernel_mul(node->inputs[0], node->inputs[1], node->output, n);
    }
}

static void dispatch_div(OpNode* node) {
    int n = total_elements(node);
    if (node->extra[0]) {
        kernel_div_scalar(node->inputs[0], extra_float(node, 1),
                          node->output, n);
    } else {
        kernel_div(node->inputs[0], node->inputs[1], node->output, n);
    }
}

/* --- Reductions --- */

static void dispatch_max(OpNode* node) {
    int axis = node->extra[0];
    int axis_size = node->extra[1];
    int outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= node->out_shape[i];
    for (int i = axis; i < node->n_dims; i++) inner *= node->out_shape[i];
    kernel_max(node->inputs[0], node->output, outer, axis_size, inner);
}

static void dispatch_sum(OpNode* node) {
    int axis = node->extra[0];
    int axis_size = node->extra[1];
    int outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= node->out_shape[i];
    for (int i = axis; i < node->n_dims; i++) inner *= node->out_shape[i];
    kernel_sum(node->inputs[0], node->output, outer, axis_size, inner);
}

static void dispatch_softmax(OpNode* node) {
    int cols = node->out_shape[node->n_dims - 1];
    kernel_softmax(node->inputs[0], node->output, leading_dims(node), cols);
}

/* --- MatMul / BLAS --- */

static void dispatch_matmul(OpNode* node) {
    /* extra: [K, trans_b, b_is_2d, alpha_bits]
     * alpha_bits == 0 means alpha = 1.0 */
    int N = node->out_shape[node->n_dims - 1];
    int K = node->extra[0];
    int trans_b = node->extra[1];
    int b_is_2d = node->extra[2];
    float alpha = node->extra[3] ? extra_float(node, 3) : 1.0f;

    if (b_is_2d) {
        int M_total = leading_dims(node);
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
}

/* --- Shape / data movement --- */

static void dispatch_reshape(OpNode* node) {
    /* Usually zero-copy (alias), but dispatched when the input is an
     * external buffer (graph input) that the planner can't alias. */
    if (node->inputs[0] != node->output) {
        memcpy(node->output, node->inputs[0],
               total_elements(node) * node->elem_size);
    }
}

static void dispatch_transpose(OpNode* node) {
    if (node->n_dims == 2) {
        kernel_transpose(node->inputs[0], node->output,
                         node->extra[0], node->extra[1]);
    } else {
        /* General N-dim swapaxes: decompose into [outer, A, mid, B, inner] */
        int outer  = node->extra[0];
        int A      = node->extra[1];
        int middle = node->extra[2];
        int B      = node->extra[3];
        int inner  = node->extra[4];
        int es     = node->elem_size;
        const char* x = node->inputs[0];
        char* out = node->output;
        int chunk = inner * es;
        for (int o = 0; o < outer; o++)
          for (int b = 0; b < B; b++)
            for (int m = 0; m < middle; m++)
              for (int a = 0; a < A; a++) {
                int in_off  = (((o*A + a)*middle + m)*B + b)*inner;
                int out_off = (((o*B + b)*middle + m)*A + a)*inner;
                memcpy(out + out_off * es, x + in_off * es, chunk);
              }
    }
}

static void dispatch_slice(OpNode* node) {
    /* extra = [outer, orig_dim_size, start, slice_len, inner]
     * Contiguous slices (dim=0) are aliases filtered by Python —
     * guard here too in case one leaks through. */
    if (node->extra[0] == 0) return;
    kernel_slice(node->inputs[0], node->output,
                 node->extra[0], node->extra[1], node->extra[2],
                 node->extra[3], node->extra[4], node->elem_size);
}

static void dispatch_embedding(OpNode* node) {
    /* inputs: [ids (int64), table (float)], extra[0] = embed_dim */
    int embed_dim = node->extra[0];
    int seq_len = leading_dims(node);
    kernel_embedding((const long*)node->inputs[0],
                     (const float*)node->inputs[1],
                     (float*)node->output,
                     seq_len, embed_dim);
}

/* --- Normalization / compound --- */

static void dispatch_layernorm(OpNode* node) {
    int cols = node->out_shape[node->n_dims - 1];
    kernel_layernorm(node->inputs[0], node->inputs[1], node->inputs[2],
                     node->output, leading_dims(node), cols,
                     extra_float(node, 0));
}

/* --- Fused ops --- */

static void dispatch_matmul_add(OpNode* node) {
    /* inputs: [A, B, bias], extras: [K, trans_b, b_is_2d]
     * Pre-fill output with broadcast bias, then sgemm with beta=1.0. */
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
        kernel_matmul_beta(node->inputs[0], node->inputs[1], node->output,
                           rows, N, K, 1, trans_b, 1.0f);
    } else {
        int M = node->out_shape[node->n_dims - 2];
        int batches = 1;
        for (int i = 0; i < node->n_dims - 2; i++)
            batches *= node->out_shape[i];
        kernel_matmul_beta(node->inputs[0], node->inputs[1], node->output,
                           M, N, K, batches, trans_b, 1.0f);
    }
}

static void dispatch_fused_bias_relu(OpNode* node) {
    int N = node->out_shape[node->n_dims - 1];
    kernel_bias_relu(node->inputs[0], node->inputs[1], node->output,
                     leading_dims(node), N);
}

static void dispatch_attention(OpNode* node) {
    /* extras: [seq_len, head_dim, causal, has_mask]
     * no mask:   inputs = [Q, K, V, scratch]
     * with mask: inputs = [Q, K, V, mask, scratch]  */
    int seq_len  = node->extra[0];
    int head_dim = node->extra[1];
    int causal   = node->extra[2];
    int has_mask = node->extra[3];
    int batch_heads = 1;
    for (int i = 0; i < node->n_dims - 2; i++)
        batch_heads *= node->out_shape[i];
    const float* mask = has_mask ? node->inputs[3] : NULL;
    float* scratch    = has_mask ? node->inputs[4] : node->inputs[3];
    kernel_attention(node->inputs[0], node->inputs[1], node->inputs[2],
                     node->output, scratch, mask,
                     batch_heads, seq_len, head_dim, causal);
}

/* ----------------------------------------------------------------
 * Dispatch table
 *
 * Indexed by op code. Unregistered slots are NULL — dispatch()
 * checks and aborts with a clear error. Adding a new op is:
 *   1. Add the enum value
 *   2. Write dispatch_xxx()
 *   3. Add one line here
 * ---------------------------------------------------------------- */

static dispatch_fn dispatch_table[DISPATCH_TABLE_SIZE] = {
    /* Element-wise unary */
    [OP_RELU]      = dispatch_relu,
    [OP_EXP]       = dispatch_exp,
    [OP_TANH]      = dispatch_tanh,
    [OP_POW]       = dispatch_pow,
    [OP_GELU]      = dispatch_gelu,

    /* Element-wise binary */
    [OP_ADD]       = dispatch_add,
    [OP_SUB]       = dispatch_sub,
    [OP_MUL]       = dispatch_mul,
    [OP_DIV]       = dispatch_div,

    /* Reductions */
    [OP_MAX]       = dispatch_max,
    [OP_SUM]       = dispatch_sum,
    [OP_SOFTMAX]   = dispatch_softmax,

    /* MatMul / BLAS */
    [OP_MATMUL]    = dispatch_matmul,

    /* Shape / data movement */
    [OP_RESHAPE]   = dispatch_reshape,
    [OP_TRANSPOSE] = dispatch_transpose,
    [OP_SLICE]     = dispatch_slice,
    [OP_EMBEDDING] = dispatch_embedding,

    /* Normalization / compound */
    [OP_LAYERNORM] = dispatch_layernorm,

    /* Fused ops */
    [OP_MATMUL_ADD]      = dispatch_matmul_add,
    [OP_FUSED_BIAS_RELU] = dispatch_fused_bias_relu,
    [OP_ATTENTION]       = dispatch_attention,
};

/* ----------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------- */

void execute(OpNode* nodes, int n_nodes) {
    for (int i = 0; i < n_nodes; i++) {
        int op = nodes[i].op;
        if (op < 0 || op >= DISPATCH_TABLE_SIZE || !dispatch_table[op]) {
            fprintf(stderr, "executor: unknown op %d\n", op);
            abort();
        }
        dispatch_table[op](&nodes[i]);
    }
}
