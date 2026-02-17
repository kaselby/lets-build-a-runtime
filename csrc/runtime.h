/*
 * runtime.h — Shared definitions for the inference runtime C layer.
 *
 * Must match Python OpType enum (runtime_edited/ir.py).
 * Range-based numbering: related ops cluster together.
 */

#ifndef RUNTIME_H
#define RUNTIME_H

enum OpType {
    /* Element-wise unary (100-199) */
    OP_RELU      = 100,
    OP_EXP       = 101,
    OP_TANH      = 102,
    OP_POW       = 103,
    OP_GELU      = 104,
    OP_RSQRT     = 105,
    OP_SILU      = 106,
    OP_NEG       = 107,
    OP_COS       = 108,
    OP_SIN       = 109,

    /* Element-wise binary (200-299) */
    OP_ADD       = 200,
    OP_SUB       = 201,
    OP_MUL       = 202,
    OP_DIV       = 203,

    /* Reductions (300-399) */
    OP_MAX       = 300,
    OP_SUM       = 301,
    OP_SOFTMAX   = 302,

    /* MatMul / BLAS (400-499) */
    OP_MATMUL    = 400,

    /* Shape / data movement (500-599) */
    OP_RESHAPE   = 500,
    OP_TRANSPOSE = 501,
    OP_PERMUTE   = 502,
    OP_SLICE     = 503,
    OP_EMBEDDING = 504,
    OP_CAT       = 505,

    /* Normalization / compound (600-699) */
    OP_LAYERNORM = 600,
    OP_RMSNORM   = 601,

    /* Fused ops (1000-1499) */
    OP_MATMUL_ADD      = 1000,
    OP_FUSED_BIAS_RELU = 1001,
    OP_ATTENTION       = 1002,
    OP_GATED_ACT       = 1003,
};

#define MAX_INPUTS 12
#define MAX_DIMS   24

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
