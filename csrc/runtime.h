/*
 * runtime.h — Shared definitions for the inference runtime C layer.
 *
 * Must match Python OpType enum (runtime_edited/ir.py).
 * Range-based numbering: related ops cluster together.
 */

#ifndef RUNTIME_H
#define RUNTIME_H

enum OpType {
    /* Element-wise unary (10-19) */
    OP_RELU      = 10,
    OP_EXP       = 11,
    OP_TANH      = 12,
    OP_POW       = 13,
    OP_GELU      = 14,

    /* Element-wise binary (20-29) */
    OP_ADD       = 20,
    OP_SUB       = 21,
    OP_MUL       = 22,
    OP_DIV       = 23,

    /* Reductions (30-39) */
    OP_MAX       = 30,
    OP_SUM       = 31,
    OP_SOFTMAX   = 32,

    /* MatMul / BLAS (40-49) */
    OP_MATMUL    = 40,

    /* Shape / data movement (50-59) */
    OP_RESHAPE   = 50,
    OP_TRANSPOSE = 51,
    OP_PERMUTE   = 52,
    OP_SLICE     = 53,
    OP_EMBEDDING = 54,

    /* Normalization / compound (60-69) */
    OP_LAYERNORM = 60,

    /* Fused ops (70-79) */
    OP_MATMUL_ADD      = 70,
    OP_FUSED_BIAS_RELU = 71,
    OP_ATTENTION       = 72,
};

#define MAX_INPUTS 8
#define MAX_DIMS   16

typedef struct {
    int op;
    int n_inputs;
    void* inputs[MAX_INPUTS];
    void* output;
    int out_shape[MAX_DIMS];
    int n_dims;
    int elem_size;        /* bytes per element (4 for float, 8 for int64) */
    int extra[MAX_DIMS];  /* op-specific: axes, flags, etc. */
} OpNode;

/* Dispatch function signature -- every op handler has this form. */
typedef void (*dispatch_fn)(OpNode*);

#endif /* RUNTIME_H */
