"""Op definitions: per-op metadata unified in one place.

Each OpDef describes everything the runtime needs to know about an op
on the Python side: how to evaluate it (numpy), how much scratch it
needs, whether it's a memory alias, etc. This replaces the scattered
registries that were previously in folding.py, planner.py, and
hardcoded alias checks in the planner/executor.

Adding a new op: define an OpDef and add it to OP_REGISTRY.
"""

import struct
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .ir import Graph, Node, OpType


# Numpy evaluator: (inputs, attrs) -> output array
NumpyEvaluator = Callable[[list[np.ndarray], dict[str, Any]], np.ndarray]

# Scratch calculator: (input_shapes, output_shape, attrs) -> bytes needed
ScratchCalculator = Callable[[list[tuple[int, ...]], tuple[int, ...], dict], int]

# Extras packer: (node, graph) -> list of ints for COpNode.extra[]
ExtrasPacker = Callable[[Node, Graph], list[int]]

# Shape inference: (input_shapes, attrs) -> output_shape
ShapeInfer = Callable[[list[tuple[int, ...]], dict[str, Any]], tuple[int, ...]]


def _float_bits(f: float) -> int:
    """Bit-cast a float32 to int32 for packing into COpNode.extra[].

    The C side recovers the float via: union { int i; float f; } u; u.i = extra[k];
    """
    return struct.unpack('i', struct.pack('f', f))[0]


# Alias predicate: True if this node's output shares its input's memory.
# Can be a plain bool (same for all instances) or a callable that inspects
# the node's attrs (e.g., SLICE is alias only when dim=0).
AliasPredicate = bool | Callable[[Node], bool]

# Alias offset: byte offset of alias output relative to the input buffer.
# RESHAPE is always 0 (same start), SLICE returns byte_offset into the input.
AliasOffset = Callable[[Node], int]


@dataclass
class OpDef:
    """Complete Python-side definition of an op type.

    Fields:
        evaluator: Numpy implementation. Used for constant folding and as a
            fallback execution path. None = op can't be evaluated in numpy.
        scratch: Computes scratch buffer size in bytes given tensor shapes.
            None = op doesn't need scratch workspace.
        alias: Output shares input's memory (no computation, just reinterpretation).
            RESHAPE changes shape metadata, contiguous SLICE adds a byte offset.
            The planner skips arena allocation for alias outputs and extends
            the root tensor's lifetime instead. Can be a bool or a callable
            (node) -> bool for ops where it depends on attrs.
        alias_offset: Byte offset of the alias output relative to the input.
            For RESHAPE this is 0 (same start). For SLICE this is the
            byte_offset into the input tensor. None = 0. The planner adds
            this to the root's arena offset when propagating offsets.
        inplace: Safe to write output into the first input's buffer. The kernel
            must not read any input element after that position has been written.
            The planner can assign the output to the same arena offset as a
            dying first input when this is True.
        extras: Packs op-specific parameters into COpNode.extra[] for the
            compiled C executor. Returns a list of ints. Floats must be
            bit-cast via _float_bits(). None = no extras needed.
        shape: Computes output shape from input shapes and attrs. Used by
            infer_shapes() to propagate shapes through the graph without
            re-exporting. None = output shape equals first input's shape
            (correct for element-wise, softmax, layernorm, etc.).
    """
    evaluator: NumpyEvaluator | None = None
    scratch: ScratchCalculator | None = None
    alias: AliasPredicate = False
    alias_offset: AliasOffset | None = None
    inplace: bool = False
    extras: ExtrasPacker | None = None
    shape: ShapeInfer | None = None

    def is_alias(self, node: Node) -> bool:
        """Check whether this node is an alias (zero-copy) op."""
        if callable(self.alias):
            return self.alias(node)
        return self.alias


# ---------------------------------------------------------------------------
# Evaluator functions (too complex for inline lambdas)
# ---------------------------------------------------------------------------

def _eval_softmax(ins: list[np.ndarray], attrs: dict[str, Any]) -> np.ndarray:
    x = ins[0]
    axis = attrs["axis"]
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def _eval_layernorm(ins: list[np.ndarray], attrs: dict[str, Any]) -> np.ndarray:
    x, gamma, beta = ins[0], ins[1], ins[2]
    eps = attrs.get("eps", 1e-5)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def _eval_attention(ins: list[np.ndarray], attrs: dict[str, Any]) -> np.ndarray:
    Q, K, V = ins[0], ins[1], ins[2]
    scale = 1.0 / np.sqrt(Q.shape[-1])
    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) * scale
    if len(ins) > 3:
        scores = scores + ins[3]
    if attrs.get("causal"):
        seq_len = Q.shape[-2]
        mask = np.triu(np.full((seq_len, seq_len), -np.inf, dtype=np.float32), k=1)
        scores = scores + mask
    scores -= np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores)
    weights = e / np.sum(e, axis=-1, keepdims=True)
    return np.matmul(weights, V)


# ---------------------------------------------------------------------------
# Extras packers (for COpNode.extra[] in compiled executor)
# ---------------------------------------------------------------------------

def _matmul_extras(node: Node, graph: Graph) -> list[int]:
    """extra = [K, trans_b, b_is_2d, alpha_bits]"""
    a_shape = graph.tensors[node.inputs[0]].shape
    b_shape = graph.tensors[node.inputs[1]].shape
    K = a_shape[-1]
    trans_b = 1 if node.attrs.get("transpose_b") else 0
    b_is_2d = 1 if len(b_shape) == 2 and len(a_shape) > 2 else 0
    alpha = node.attrs.get("alpha", 1.0)
    alpha_bits = _float_bits(alpha) if alpha != 1.0 else 0
    return [K, trans_b, b_is_2d, alpha_bits]


def _transpose_extras(node: Node, graph: Graph) -> list[int]:
    """2D: extra = [rows, cols]. N-dim: extra = [outer, A, middle, B, inner]."""
    in_shape = graph.tensors[node.inputs[0]].shape
    if len(in_shape) == 2:
        return [in_shape[0], in_shape[1]]

    dim0 = node.attrs.get("dim0", 0)
    dim1 = node.attrs.get("dim1", 1)
    if dim0 > dim1:
        dim0, dim1 = dim1, dim0

    outer = 1
    for d in range(dim0):
        outer *= in_shape[d]
    A = in_shape[dim0]
    middle = 1
    for d in range(dim0 + 1, dim1):
        middle *= in_shape[d]
    B = in_shape[dim1]
    inner = 1
    for d in range(dim1 + 1, len(in_shape)):
        inner *= in_shape[d]
    return [outer, A, middle, B, inner]


def _broadcast_strides(shape: tuple, out_shape: tuple) -> list[int]:
    """Compute element strides for broadcasting shape against out_shape.

    Dims where the input has size 1 (or is absent) get stride 0 — the kernel
    re-reads the same element, implementing broadcast.
    """
    ndim = len(out_shape)
    offset = ndim - len(shape)
    strides = []
    for d in range(ndim):
        if d < offset or shape[d - offset] == 1:
            strides.append(0)
        else:
            stride = 1
            for k in range(d - offset + 1, len(shape)):
                stride *= shape[k]
            strides.append(stride)
    return strides


def _pack_broadcast(a_shape: tuple, b_shape: tuple,
                    out_shape: tuple) -> list[int]:
    """Pack broadcast strides into extras: [ndim, a_strides..., b_strides..., out_shape...]."""
    ndim = len(out_shape)
    a_strides = _broadcast_strides(a_shape, out_shape)
    b_strides = _broadcast_strides(b_shape, out_shape)
    return [ndim] + a_strides + b_strides + list(out_shape)


def _add_extras(node: Node, graph: Graph) -> list[int]:
    """extra[0] = mode: 0=bias broadcast, 1=element-wise, 2=scalar, 3=general broadcast.
    Scalar:    extra[1] = float bits.
    Broadcast: extra[1] = ndim, then a_strides, b_strides, out_shape.
    """
    if "scalar" in node.attrs:
        return [2, _float_bits(node.attrs["scalar"])]
    a_shape = graph.tensors[node.inputs[0]].shape
    b_shape = graph.tensors[node.inputs[1]].shape
    if a_shape == b_shape:
        return [1]
    # Check for simple bias broadcast: b is 1D, last dim matches
    if len(b_shape) == 1 and b_shape[-1] == a_shape[-1]:
        return [0]
    # General broadcast
    out_shape = node.attrs.get("_out_shape") or graph.tensors[node.output].shape
    return [3] + _pack_broadcast(a_shape, b_shape, out_shape)


def _scalar_binop_extras(node: Node, graph: Graph) -> list[int]:
    """For MUL, SUB, DIV: extra[0] = mode: 0=element-wise, 1=scalar, 2=broadcast.
    Scalar:    extra[1] = float bits.
    Broadcast: extra[1] = ndim, then a_strides, b_strides, out_shape.
    """
    if "scalar" in node.attrs:
        return [1, _float_bits(node.attrs["scalar"])]
    a_shape = graph.tensors[node.inputs[0]].shape
    b_shape = graph.tensors[node.inputs[1]].shape
    if a_shape == b_shape:
        return [0]
    # General broadcast
    out_shape = graph.tensors[node.output].shape
    return [2] + _pack_broadcast(a_shape, b_shape, out_shape)


def _reduction_extras(node: Node, graph: Graph) -> list[int]:
    """extra = [axis, axis_size]"""
    axis = node.attrs.get("axis", -1)
    in_shape = graph.tensors[node.inputs[0]].shape
    return [axis, in_shape[axis]]


def _layernorm_extras(node: Node, graph: Graph) -> list[int]:
    """extra = [eps_bits]"""
    eps = node.attrs.get("eps", 1e-5)
    return [_float_bits(eps)]


def _rmsnorm_extras(node: Node, graph: Graph) -> list[int]:
    """extra = [eps_bits]"""
    eps = node.attrs.get("eps", 1e-5)
    return [_float_bits(eps)]


def _attention_extras(node: Node, graph: Graph) -> list[int]:
    """extra = [seq_len, head_dim, causal, has_mask, group_size]"""
    q_shape = graph.tensors[node.inputs[0]].shape
    has_mask = 1 if len(node.inputs) > 3 else 0
    group_size = node.attrs.get("group_size", 1)
    return [q_shape[-2], q_shape[-1], 1 if node.attrs.get("causal") else 0, has_mask, group_size]


def _pow_extras(node: Node, graph: Graph) -> list[int]:
    """extra = [scalar_bits]"""
    return [_float_bits(node.attrs.get("scalar", 2.0))]


def _embedding_extras(node: Node, graph: Graph) -> list[int]:
    """extra = [embed_dim]"""
    return [graph.tensors[node.inputs[1]].shape[-1]]


def _cat_extras(node: Node, graph: Graph) -> list[int]:
    """extra = [dim, dim_size_0, dim_size_1, ...]"""
    dim = node.attrs["dim"]
    sizes = [graph.tensors[inp].shape[dim] for inp in node.inputs]
    return [dim] + sizes


def _gated_act_extras(node: Node, graph: Graph) -> list[int]:
    """extra = [has_bias, act_type (0=silu, 1=gelu)]"""
    has_bias = 1 if node.attrs.get("has_bias") else 0
    act_type = 0 if node.attrs.get("act", "silu") == "silu" else 1
    return [has_bias, act_type]


def _eval_gated_act(ins: list[np.ndarray], attrs: dict[str, Any]) -> np.ndarray:
    has_bias = attrs.get("has_bias", False)
    act = attrs.get("act", "silu")
    if has_bias:
        x, bias, up = ins[0], ins[1], ins[2]
        v = x + bias
    else:
        x, up = ins[0], ins[1]
        v = x
    if act == "silu":
        activated = v / (1.0 + np.exp(-v))
    else:  # gelu
        activated = 0.5 * v * (1 + np.tanh(0.7978845608 * (v + 0.044715 * v**3)))
    return activated * up


def _slice_alias(node: Node) -> bool:
    """Contiguous SLICE (dim=0) is zero-copy; non-contiguous needs a kernel."""
    return node.attrs.get("dim", 0) == 0


def _slice_extras(node: Node, graph: Graph) -> list[int]:
    """extra = [outer, orig_dim_size, slice_start, slice_len, inner]

    Pre-computes the strided copy layout so the C kernel doesn't need
    the full input shape. Only used for non-contiguous slices (dim>0);
    contiguous slices are aliases and never reach the C executor.
    """
    in_shape = graph.tensors[node.inputs[0]].shape
    dim = node.attrs["dim"]
    start = node.attrs["start"]
    end = min(node.attrs["end"], in_shape[dim])

    outer = 1
    for d in range(dim):
        outer *= in_shape[d]
    inner = 1
    for d in range(dim + 1, len(in_shape)):
        inner *= in_shape[d]

    return [outer, in_shape[dim], start, end - start, inner]


# ---------------------------------------------------------------------------
# Shape inference functions
# ---------------------------------------------------------------------------
# Only needed for ops whose output shape differs from the first input.
# Ops that preserve the first input's shape (element-wise, softmax,
# layernorm, etc.) use the None default.

def _shape_broadcast_binary(in_shapes, attrs):
    """Binary ops with broadcasting (ADD, SUB, MUL, DIV)."""
    if "scalar" in attrs:
        return in_shapes[0]
    return tuple(np.broadcast_shapes(in_shapes[0], in_shapes[1]))


def _shape_matmul(in_shapes, attrs):
    """MATMUL: (..., M, K) × (..., K, N) -> (..., M, N)."""
    a = in_shapes[0]
    b = in_shapes[1]
    N = b[-2] if attrs.get("transpose_b") else b[-1]
    return (*a[:-1], N)


def _shape_reshape(in_shapes, attrs):
    target = list(attrs["shape"])
    # If any dimension is still a string (unresolved symbol), return as-is
    if any(isinstance(d, str) for d in target):
        return tuple(target)
    # Resolve -1 from input element count
    in_total = 1
    for d in in_shapes[0]:
        in_total *= d
    neg_idx = None
    known = 1
    for i, d in enumerate(target):
        if d == -1:
            neg_idx = i
        else:
            known *= d
    if neg_idx is not None:
        target[neg_idx] = in_total // known
    return tuple(target)


def _shape_transpose(in_shapes, attrs):
    shape = list(in_shapes[0])
    d0, d1 = attrs["dim0"], attrs["dim1"]
    shape[d0], shape[d1] = shape[d1], shape[d0]
    return tuple(shape)


def _shape_permute(in_shapes, attrs):
    shape = in_shapes[0]
    return tuple(shape[i] for i in attrs["axes"])


def _shape_slice(in_shapes, attrs):
    shape = list(in_shapes[0])
    dim = attrs["dim"]
    end = min(attrs["end"], shape[dim])  # clamp sentinel (sys.maxsize → dim size)
    shape[dim] = end - attrs.get("start", 0)
    return tuple(shape)


def _shape_reduce(in_shapes, attrs):
    shape = list(in_shapes[0])
    axis = attrs["axis"]
    if attrs.get("keepdim", False):
        shape[axis] = 1
    else:
        shape.pop(axis)
    return tuple(shape)


def _shape_attention(in_shapes, attrs):
    """ATTENTION: Q shape with last dim from V."""
    return (*in_shapes[0][:-1], in_shapes[2][-1])


def _shape_embedding(in_shapes, attrs):
    """EMBEDDING: (*indices_shape, embed_dim)."""
    return (*in_shapes[0], in_shapes[1][-1])


def _shape_cat(in_shapes, attrs):
    """CAT: sum the concat dim across all inputs."""
    dim = attrs["dim"]
    result = list(in_shapes[0])
    result[dim] = sum(s[dim] for s in in_shapes)
    return tuple(result)


# ---------------------------------------------------------------------------
# Scratch calculators
# ---------------------------------------------------------------------------

# Adaptive attention dispatch: must match FLASH_SEQ_THRESHOLD in
# csrc/ops/attn.c and csrc/executor.c.
FLASH_SEQ_THRESHOLD = 256
FLASH_BR = 128
FLASH_BC = 256


def _use_flash_attention(seq_len: int, attrs: dict) -> bool:
    """Whether to use flash attention for the given config.

    Flash supports causal natively (skips ~half the tiles).
    Only custom masks require the standard path.
    """
    return seq_len > FLASH_SEQ_THRESHOLD


def _attention_scratch(input_shapes, output_shape, attrs) -> int:
    """Scratch for fused attention: one score matrix per batch*head slice.

    Standard kernel needs S*S per slice (full attention matrix).
    Flash kernel needs B_r*B_c per slice (one tile).
    Adaptive dispatch chooses based on seq_len and causal/mask flags.
    """
    q_shape = input_shapes[0]
    batch_heads = 1
    for d in q_shape[:-2]:
        batch_heads *= d
    seq_len = q_shape[-2]
    if _use_flash_attention(seq_len, attrs):
        scratch_per_slice = FLASH_BR * FLASH_BC
    else:
        scratch_per_slice = seq_len * seq_len
    return batch_heads * scratch_per_slice * 4  # float32


# ---------------------------------------------------------------------------
# Fold-only evaluators (too complex for inline lambdas)
# ---------------------------------------------------------------------------

def _eval_slice_tensor(ins: list[np.ndarray], attrs: dict[str, Any]) -> np.ndarray:
    x, dim = ins[0], attrs["dim"]
    start = attrs.get("start", 0)
    end = attrs.get("end", x.shape[dim])
    slices = tuple(slice(start, end) if d == dim else slice(None) for d in range(x.ndim))
    return x[slices]


def _eval_diff(ins: list[np.ndarray], attrs: dict[str, Any]) -> np.ndarray:
    prepend = ins[1] if len(ins) > 1 else None
    return np.diff(ins[0], n=attrs.get("n", 1), axis=attrs.get("dim", -1), prepend=prepend)


def _eval_index(ins: list[np.ndarray], attrs: dict[str, Any]) -> np.ndarray:
    tensor = ins[0]
    n_indices = attrs.get("n_indices", len(ins) - 1)
    none_positions = set(attrs.get("none_positions", []))
    indices = []
    tensor_idx = 1
    for i in range(n_indices):
        if i in none_positions:
            indices.append(slice(None))
        else:
            indices.append(ins[tensor_idx])
            tensor_idx += 1
    return tensor[tuple(indices)]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

OP_REGISTRY: dict[OpType, OpDef] = {
    # --- Element-wise (all safe for in-place on first input) ---
    # shape=None → output shape equals first input's shape.
    OpType.ADD:     OpDef(inplace=True, extras=_add_extras, shape=_shape_broadcast_binary,
                          evaluator=lambda ins, a: ins[0] + (a["scalar"] if "scalar" in a else ins[1])),
    OpType.SUB:     OpDef(inplace=True, extras=_scalar_binop_extras, shape=_shape_broadcast_binary,
                          evaluator=lambda ins, a: ins[0] - (a["scalar"] if "scalar" in a else ins[1])),
    OpType.MUL:     OpDef(inplace=True, extras=_scalar_binop_extras, shape=_shape_broadcast_binary,
                          evaluator=lambda ins, a: ins[0] * (a["scalar"] if "scalar" in a else ins[1])),
    OpType.DIV:     OpDef(inplace=True, extras=_scalar_binop_extras, shape=_shape_broadcast_binary,
                          evaluator=lambda ins, a: ins[0] / (a["scalar"] if "scalar" in a else ins[1])),
    OpType.RELU:    OpDef(inplace=True,
                          evaluator=lambda ins, a: np.maximum(ins[0], 0)),
    OpType.EXP:     OpDef(inplace=True,
                          evaluator=lambda ins, a: np.exp(ins[0])),
    OpType.POW:     OpDef(inplace=True, extras=_pow_extras,
                          evaluator=lambda ins, a: np.power(ins[0], a["scalar"])),
    OpType.TANH:    OpDef(inplace=True,
                          evaluator=lambda ins, a: np.tanh(ins[0])),
    OpType.GELU:    OpDef(inplace=False,  # C kernel not in-place safe (macOS vvtanhf path)
                          evaluator=lambda ins, a: 0.5 * ins[0] * (1 + np.tanh(0.7978845608 * (ins[0] + 0.044715 * ins[0]**3)))),
    OpType.RSQRT:   OpDef(inplace=True,
                          evaluator=lambda ins, a: 1.0 / np.sqrt(ins[0])),
    OpType.SILU:    OpDef(inplace=True,
                          evaluator=lambda ins, a: ins[0] / (1.0 + np.exp(-ins[0]))),
    OpType.NEG:     OpDef(inplace=True,
                          evaluator=lambda ins, a: -ins[0]),
    OpType.COS:     OpDef(inplace=True,
                          evaluator=lambda ins, a: np.cos(ins[0])),
    OpType.SIN:     OpDef(inplace=True,
                          evaluator=lambda ins, a: np.sin(ins[0])),

    # --- Reductions ---
    OpType.MAX:     OpDef(extras=_reduction_extras, shape=_shape_reduce,
                          evaluator=lambda ins, a: np.max(ins[0], axis=a["axis"], keepdims=a.get("keepdim", False))),
    OpType.SUM:     OpDef(extras=_reduction_extras, shape=_shape_reduce,
                          evaluator=lambda ins, a: np.sum(ins[0], axis=a["axis"], keepdims=a.get("keepdim", False))),
    OpType.SOFTMAX: OpDef(evaluator=_eval_softmax),

    # --- Shape ops ---
    OpType.RESHAPE:     OpDef(alias=True, shape=_shape_reshape,
                              evaluator=lambda ins, a: ins[0].reshape(a["shape"])),
    OpType.TRANSPOSE:   OpDef(extras=_transpose_extras, shape=_shape_transpose,
                              evaluator=lambda ins, a: np.swapaxes(ins[0], a["dim0"], a["dim1"])),
    OpType.PERMUTE:     OpDef(evaluator=lambda ins, a: np.transpose(ins[0], a["axes"]),
                              shape=_shape_permute),
    OpType.SLICE:       OpDef(alias=_slice_alias, shape=_shape_slice,
                              alias_offset=lambda n: n.attrs.get("byte_offset", 0),
                              extras=_slice_extras,
                              evaluator=_eval_slice_tensor),

    # --- Compound ops ---
    OpType.LAYERNORM:   OpDef(extras=_layernorm_extras, evaluator=_eval_layernorm),
    OpType.RMSNORM:     OpDef(extras=_rmsnorm_extras,
                              evaluator=lambda ins, a: ins[0] / np.sqrt(np.mean(ins[0]**2, axis=-1, keepdims=True) + a.get("eps", 1e-5)) * ins[1]),
    OpType.MATMUL:      OpDef(extras=_matmul_extras, shape=_shape_matmul,
                              evaluator=lambda ins, a: ins[0] @ (np.swapaxes(ins[1], -2, -1) if a.get("transpose_b") else ins[1]) * a.get("alpha", 1.0)),
    OpType.EMBEDDING:   OpDef(extras=_embedding_extras, shape=_shape_embedding,
                              evaluator=lambda ins, a: ins[1][ins[0].astype(int)]),
    OpType.CAT:         OpDef(extras=_cat_extras, shape=_shape_cat,
                              evaluator=lambda ins, a: np.concatenate(ins, axis=a["dim"])),

    # --- Fused ops ---
    OpType.MATMUL_ADD:      OpDef(extras=_matmul_extras, shape=_shape_matmul,
                                  evaluator=lambda ins, a: ins[0] @ (np.swapaxes(ins[1], -2, -1) if a.get("transpose_b") else ins[1]) * a.get("alpha", 1.0) + ins[2]),
    OpType.FUSED_BIAS_RELU: OpDef(inplace=True, evaluator=lambda ins, a: np.maximum(ins[0] + ins[1], 0)),
    OpType.ATTENTION:       OpDef(extras=_attention_extras, scratch=_attention_scratch,
                                  shape=_shape_attention, evaluator=_eval_attention),
    OpType.GATED_ACT:       OpDef(extras=_gated_act_extras, evaluator=_eval_gated_act),

    # --- Fold-only ops (5000+) ---
    # Infrastructure ops for mask generation, type conversion, etc.
    # Must be eliminated by constant folding before execution — neither
    # executor supports them. Evaluators here are used by constant_fold().
    OpType.ARANGE:       OpDef(evaluator=lambda ins, a: np.arange(a["start"], a["end"], dtype=np.dtype(a["dtype"])),
                              shape=lambda in_shapes, a: (a["end"] - a["start"],)),
    OpType.CAST:         OpDef(evaluator=lambda ins, a: ins[0].astype(a["target_dtype"])),
    OpType.EXPAND:       OpDef(evaluator=lambda ins, a: np.broadcast_to(ins[0], a["shape"])),
    OpType.DIFF:         OpDef(evaluator=_eval_diff),
    OpType.CMP_NE:       OpDef(evaluator=lambda ins, a: ins[0] != (a["scalar"] if "scalar" in a else ins[1])),
    OpType.CMP_LE:       OpDef(evaluator=lambda ins, a: ins[0] <= ins[1]),
    OpType.CMP_EQ:       OpDef(evaluator=lambda ins, a: ins[0] == ins[1]),
    OpType.CUMSUM:       OpDef(evaluator=lambda ins, a: np.cumsum(ins[0], axis=a["dim"])),
    OpType.BITWISE_AND:  OpDef(evaluator=lambda ins, a: ins[0] & ins[1]),
    OpType.INDEX:        OpDef(evaluator=_eval_index),
}


# ---------------------------------------------------------------------------
# Shape propagation
# ---------------------------------------------------------------------------

def resolve_graph(graph: Graph, bindings: dict[str, int]) -> Graph:
    """Create a concrete copy of a symbolic graph.

    Returns a new Graph with all symbolic dimensions resolved to concrete
    values. The original graph is not modified — its attrs keep symbol
    strings, so it can be re-resolved with different bindings.

    Weight buffers are shared (not copied). Only tensor metadata, node
    attrs, and connectivity indices are copied and resolved.

    Args:
        graph: An optimized graph, possibly containing symbolic strings
            in node attrs (e.g., RESHAPE shape=(1, "L", 768)).
        bindings: Maps symbol names to concrete values, e.g., {'L': 50}.

    Returns:
        A new Graph with all symbols resolved, shapes propagated, and
        ready for planning and compilation.
    """
    resolved = Graph()

    # --- Copy tensors (share weight buffers) ---
    for name, t in graph.tensors.items():
        info = resolved.add_tensor(name, t.shape, t.dtype)
        info.buffer = t.buffer  # share, don't copy

    # --- Copy nodes with resolved attrs ---
    def _resolve(v):
        if isinstance(v, str) and v in bindings:
            return bindings[v]
        if isinstance(v, tuple):
            return tuple(_resolve(x) for x in v)
        if isinstance(v, list):
            return [_resolve(x) for x in v]
        return v

    for node in graph:
        resolved_attrs = {k: _resolve(v) for k, v in node.attrs.items()}
        resolved.add_node(node.op, list(node.inputs), node.output, resolved_attrs)

    # --- Copy graph roles ---
    resolved.inputs = list(graph.inputs)
    resolved.outputs = list(graph.outputs)
    resolved.constants = list(graph.constants)
    resolved.dynamic_dims = graph.dynamic_dims

    # --- Resolve input shapes from bindings ---
    input_shapes = {}
    for name in resolved.inputs:
        shape = list(resolved.tensors[name].shape)
        for sym_name, specs in graph.dynamic_dims.items():
            if sym_name in bindings:
                for tensor_name, dim_idx in specs:
                    if tensor_name == name:
                        shape[dim_idx] = bindings[sym_name]
        input_shapes[name] = tuple(shape)

    # --- Propagate shapes ---
    infer_shapes(resolved, input_shapes)

    # --- Fix RESHAPE attrs to match inferred output shapes ---
    # Resolves -1 in view/reshape args to concrete values
    for node in resolved:
        if node.op == OpType.RESHAPE:
            node.attrs["shape"] = resolved.tensors[node.output].shape

    # --- Post-resolution passes ---
    # Fold ops that depended on dynamic dims (e.g., ARANGE with end=seq_len)
    # and clean up any dead subgraphs.
    from .passes import POST_RESOLUTION_PIPELINE, run_pipeline
    run_pipeline(resolved, POST_RESOLUTION_PIPELINE)

    return resolved


def infer_shapes(graph: Graph, input_shapes: dict[str, tuple[int, ...]]) -> None:
    """Propagate shapes through the graph given new input dimensions.

    Sets input tensor shapes from input_shapes, then walks the graph in
    topological order applying per-op shape rules to derive all
    intermediate shapes. Constants keep their existing shapes.

    This allows re-planning and re-compiling for a new input shape
    (e.g., different sequence length) without re-exporting.

    Args:
        graph: An already-optimized graph (passes should be run once before
            this; shape inference replaces re-export, not re-optimization).
        input_shapes: Map of graph input names to new concrete shapes.
    """
    for name, shape in input_shapes.items():
        graph.tensors[name].shape = shape

    for node in graph:
        op_def = OP_REGISTRY.get(node.op)
        in_shapes = [graph.tensors[inp].shape for inp in node.inputs]
        if op_def is not None and op_def.shape is not None:
            graph.tensors[node.output].shape = op_def.shape(in_shapes, node.attrs)
        else:
            # Default: preserve first input's shape
            graph.tensors[node.output].shape = in_shapes[0]
