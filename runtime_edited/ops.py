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


def _float_bits(f: float) -> int:
    """Bit-cast a float32 to int32 for packing into COpNode.extra[].

    The C side recovers the float via: union { int i; float f; } u; u.i = extra[k];
    """
    return struct.unpack('i', struct.pack('f', f))[0]


# Alias predicate: True if this node's output shares its input's memory.
# Can be a plain bool (same for all instances) or a callable that inspects
# the node's attrs (e.g., SLICE is alias only when dim=0).
AliasPredicate = bool | Callable[[Node], bool]


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
        inplace: Safe to write output into the first input's buffer. The kernel
            must not read any input element after that position has been written.
            The planner can assign the output to the same arena offset as a
            dying first input when this is True.
        extras: Packs op-specific parameters into COpNode.extra[] for the
            compiled C executor. Returns a list of ints. Floats must be
            bit-cast via _float_bits(). None = no extras needed.
    """
    evaluator: NumpyEvaluator | None = None
    scratch: ScratchCalculator | None = None
    alias: AliasPredicate = False
    inplace: bool = False
    extras: ExtrasPacker | None = None

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


def _add_extras(node: Node, graph: Graph) -> list[int]:
    """extra[0] = mode: 0=bias broadcast, 1=element-wise, 2=scalar.
    For scalar mode: extra[1] = scalar value as float bits.
    """
    if "scalar" in node.attrs:
        return [2, _float_bits(node.attrs["scalar"])]
    a_shape = graph.tensors[node.inputs[0]].shape
    b_shape = graph.tensors[node.inputs[1]].shape
    mode = 1 if a_shape == b_shape else 0
    return [mode]


def _scalar_binop_extras(node: Node, graph: Graph) -> list[int]:
    """For DIV, SUB, MUL: extra = [is_scalar, scalar_bits] when scalar."""
    if "scalar" in node.attrs:
        return [1, _float_bits(node.attrs["scalar"])]
    return []


def _reduction_extras(node: Node, graph: Graph) -> list[int]:
    """extra = [axis, axis_size]"""
    axis = node.attrs.get("axis", -1)
    in_shape = graph.tensors[node.inputs[0]].shape
    return [axis, in_shape[axis]]


def _layernorm_extras(node: Node, graph: Graph) -> list[int]:
    """extra = [eps_bits]"""
    eps = node.attrs.get("eps", 1e-5)
    return [_float_bits(eps)]


def _attention_extras(node: Node, graph: Graph) -> list[int]:
    """extra = [seq_len, head_dim, causal]"""
    q_shape = graph.tensors[node.inputs[0]].shape
    return [q_shape[-2], q_shape[-1], 1 if node.attrs.get("causal") else 0]


def _pow_extras(node: Node, graph: Graph) -> list[int]:
    """extra = [scalar_bits]"""
    return [_float_bits(node.attrs.get("scalar", 2.0))]


def _embedding_extras(node: Node, graph: Graph) -> list[int]:
    """extra = [embed_dim]"""
    return [graph.tensors[node.inputs[1]].shape[-1]]


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
    end = node.attrs["end"]

    outer = 1
    for d in range(dim):
        outer *= in_shape[d]
    inner = 1
    for d in range(dim + 1, len(in_shape)):
        inner *= in_shape[d]

    return [outer, in_shape[dim], start, end - start, inner]


# ---------------------------------------------------------------------------
# Scratch calculators
# ---------------------------------------------------------------------------

def _attention_scratch(input_shapes, output_shape, attrs) -> int:
    """Scratch for fused attention: one score matrix per batch*head slice.

    Standard kernel needs S*S per slice (full attention matrix).
    Flash kernel needs B_r*B_c per slice (one tile).
    """
    q_shape = input_shapes[0]
    batch_heads = 1
    for d in q_shape[:-2]:
        batch_heads *= d
    seq_len = q_shape[-2]
    if attrs.get("flash"):
        scratch_per_slice = 32 * 32  # FLASH_BR * FLASH_BC
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
    OpType.ADD:     OpDef(inplace=True, extras=_add_extras,
                          evaluator=lambda ins, a: ins[0] + (a["scalar"] if "scalar" in a else ins[1])),
    OpType.SUB:     OpDef(inplace=True, extras=_scalar_binop_extras,
                          evaluator=lambda ins, a: ins[0] - (a["scalar"] if "scalar" in a else ins[1])),
    OpType.MUL:     OpDef(inplace=True, extras=_scalar_binop_extras,
                          evaluator=lambda ins, a: ins[0] * (a["scalar"] if "scalar" in a else ins[1])),
    OpType.DIV:     OpDef(inplace=True, extras=_scalar_binop_extras,
                          evaluator=lambda ins, a: ins[0] / (a["scalar"] if "scalar" in a else ins[1])),
    OpType.RELU:    OpDef(inplace=True,
                          evaluator=lambda ins, a: np.maximum(ins[0], 0)),
    OpType.EXP:     OpDef(inplace=True,
                          evaluator=lambda ins, a: np.exp(ins[0])),
    OpType.POW:     OpDef(inplace=True, extras=_pow_extras,
                          evaluator=lambda ins, a: np.power(ins[0], a["scalar"])),
    OpType.TANH:    OpDef(inplace=True,
                          evaluator=lambda ins, a: np.tanh(ins[0])),
    OpType.GELU:    OpDef(inplace=True,
                          evaluator=lambda ins, a: 0.5 * ins[0] * (1 + np.tanh(0.7978845608 * (ins[0] + 0.044715 * ins[0]**3)))),

    # --- Reductions ---
    OpType.MAX:     OpDef(extras=_reduction_extras,
                          evaluator=lambda ins, a: np.max(ins[0], axis=a["axis"], keepdims=a.get("keepdim", False))),
    OpType.SUM:     OpDef(extras=_reduction_extras,
                          evaluator=lambda ins, a: np.sum(ins[0], axis=a["axis"], keepdims=a.get("keepdim", False))),
    OpType.SOFTMAX: OpDef(evaluator=_eval_softmax),

    # --- Shape ops ---
    OpType.RESHAPE:     OpDef(alias=True,
                              evaluator=lambda ins, a: ins[0].reshape(a["shape"])),
    OpType.TRANSPOSE:   OpDef(extras=_transpose_extras,
                              evaluator=lambda ins, a: np.swapaxes(ins[0], a["dim0"], a["dim1"])),
    OpType.PERMUTE:     OpDef(evaluator=lambda ins, a: np.transpose(ins[0], a["axes"])),
    OpType.SLICE:       OpDef(alias=_slice_alias, extras=_slice_extras),

    # --- Compound ops ---
    OpType.LAYERNORM:   OpDef(extras=_layernorm_extras, evaluator=_eval_layernorm),
    OpType.MATMUL:      OpDef(extras=_matmul_extras,
                              evaluator=lambda ins, a: ins[0] @ (np.swapaxes(ins[1], -2, -1) if a.get("transpose_b") else ins[1]) * a.get("alpha", 1.0)),
    OpType.EMBEDDING:   OpDef(extras=_embedding_extras,
                              evaluator=lambda ins, a: ins[1][ins[0].astype(int)]),

    # --- Fused ops ---
    OpType.MATMUL_ADD:      OpDef(extras=_matmul_extras,
                                  evaluator=lambda ins, a: ins[0] @ (ins[1].T if a.get("transpose_b") else ins[1]) * a.get("alpha", 1.0) + ins[2]),
    OpType.FUSED_BIAS_RELU: OpDef(evaluator=lambda ins, a: np.maximum(ins[0] + ins[1], 0)),
    OpType.ATTENTION:       OpDef(extras=_attention_extras, scratch=_attention_scratch,
                                  evaluator=_eval_attention),

    # --- Fold-only ops (100+) ---
    # Infrastructure ops for mask generation, type conversion, etc.
    # Must be eliminated by constant folding before execution — neither
    # executor supports them. Evaluators here are used by constant_fold().
    OpType.CAST:         OpDef(evaluator=lambda ins, a: ins[0].astype(a["target_dtype"])),
    OpType.EXPAND:       OpDef(evaluator=lambda ins, a: np.broadcast_to(ins[0], a["shape"])),
    OpType.SLICE_TENSOR: OpDef(evaluator=_eval_slice_tensor),
    OpType.DIFF:         OpDef(evaluator=_eval_diff),
    OpType.CMP_NE:       OpDef(evaluator=lambda ins, a: ins[0] != (a["scalar"] if "scalar" in a else ins[1])),
    OpType.CMP_LE:       OpDef(evaluator=lambda ins, a: ins[0] <= ins[1]),
    OpType.CMP_EQ:       OpDef(evaluator=lambda ins, a: ins[0] == ins[1]),
    OpType.CUMSUM:       OpDef(evaluator=lambda ins, a: np.cumsum(ins[0], axis=a["dim"])),
    OpType.BITWISE_AND:  OpDef(evaluator=lambda ins, a: ins[0] & ins[1]),
    OpType.INDEX:        OpDef(evaluator=_eval_index),
}
