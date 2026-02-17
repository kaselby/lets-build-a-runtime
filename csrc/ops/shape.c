#include "../ops.h"

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
#define BROADCAST_MAX_DIMS 24

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
 * EMBEDDING: table lookup
 *   ids: [seq_len] (integer token IDs, stored as long)
 *   table: [vocab_size x embed_dim] (row-major)
 *   out: [seq_len x embed_dim]
 *
 *   Copies one row of the embedding table per token via memcpy.
 * ---------------------------------------------------------------- */
void kernel_embedding(const long* ids, const float* table, float* out,
                      int seq_len, int embed_dim) {
    for (int i = 0; i < seq_len; i++) {
        memcpy(out + i * embed_dim,
               table + ids[i] * embed_dim,
               embed_dim * sizeof(float));
    }
}

/* ----------------------------------------------------------------
 * SLICE: strided copy along a non-contiguous dimension
 *   x: logically [outer x orig_dim_size x inner]
 *   out: [outer x slice_len x inner]
 *
 *   For each outer slice, copies slice_len contiguous chunks of
 *   `inner` elements starting at offset `start` in the sliced dim.
 *   Contiguous slices (dim=0) are zero-copy aliases handled by
 *   Python — this kernel only runs for dim>0.
 *   elem_size: bytes per element (4 for float, 8 for int64, etc.)
 * ---------------------------------------------------------------- */
void kernel_slice(const void* x, void* out,
                  int outer, int orig_dim_size, int start,
                  int slice_len, int inner, int elem_size) {
    int src_stride = orig_dim_size * inner * elem_size;
    int dst_stride = slice_len * inner * elem_size;
    int copy_bytes = slice_len * inner * elem_size;
    const char* src = x;
    char* dst = out;
    int start_off = start * inner * elem_size;
    for (int o = 0; o < outer; o++) {
        memcpy(dst + o * dst_stride,
               src + o * src_stride + start_off,
               copy_bytes);
    }
}

/* ----------------------------------------------------------------
 * CAT: concatenate tensors along a dimension (reverse of SLICE)
 *   inputs: array of n_inputs pointers to source tensors
 *   out: output buffer
 *   dim_sizes: per-input size along the concat dimension
 *   n_inputs: number of tensors to concatenate
 *   outer: product of dims before concat dim
 *   inner: product of dims after concat dim
 *   elem_size: bytes per element
 *
 *   For each outer slice, copies each input's contribution as a
 *   contiguous chunk of dim_sizes[k] * inner elements.
 * ---------------------------------------------------------------- */
void kernel_cat(void* const* inputs, void* out,
                const int* dim_sizes, int n_inputs,
                int outer, int inner, int elem_size) {
    int total_dim = 0;
    for (int k = 0; k < n_inputs; k++) total_dim += dim_sizes[k];

    for (int o = 0; o < outer; o++) {
        int dst_off = o * total_dim * inner;
        for (int k = 0; k < n_inputs; k++) {
            int chunk = dim_sizes[k] * inner;
            int src_off = o * dim_sizes[k] * inner;
            memcpy((char*)out + dst_off * elem_size,
                   (const char*)inputs[k] + src_off * elem_size,
                   chunk * elem_size);
            dst_off += chunk;
        }
    }
}

