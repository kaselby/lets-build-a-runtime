"""ATen op handlers for the exporter.

Each handler maps one or more ATen ops to our graph IR. Handlers are
registered into ATEN_HANDLERS, which is imported by exporter.py.
"""

import operator
from typing import Callable

import numpy as np
import torch

from ..ir import Graph, OpType

# Handler signature: (fx_node, graph, node_map) -> None
OpHandler = Callable[[object, Graph, dict[str, str]], None]


# --- Common utilities ---

def _output_meta(fx_node) -> tuple[tuple[int, ...], str]:
    """Extract output shape and dtype from fx node tracing metadata."""
    val = fx_node.meta["val"]
    return tuple(val.shape), str(val.dtype).replace("torch.", "")


def _emit(fx_node, graph: Graph, node_map: dict[str, str],
          op: OpType, inputs: list[str], attrs: dict | None = None) -> None:
    """Register output tensor, add compute node, update node_map."""
    shape, dtype = _output_meta(fx_node)
    tensor_name = fx_node.name
    graph.add_tensor(tensor_name, shape, dtype)
    graph.add_node(op, inputs, tensor_name, attrs)
    node_map[fx_node.name] = tensor_name


def _get_axis(fx_node, arg_index: int = 1, default: int = -1) -> int:
    """Extract and normalize an axis argument from an fx node.

    Handles negative dims, single-element lists, and kwargs fallback.
    """
    ndim = len(fx_node.args[0].meta["val"].shape)
    axis = fx_node.args[arg_index] if len(fx_node.args) > arg_index else fx_node.kwargs.get("dim", default)
    if isinstance(axis, (list, tuple)) and len(axis) == 1:
        axis = axis[0]
    if isinstance(axis, int):
        axis = axis % ndim
    return axis


def _is_transpose(axes: list[int]) -> bool:
    """Check if a permutation is a simple two-axis swap (transpose).

    A transpose swaps exactly two dimensions and leaves all others in place.
    Examples: [1, 0] on 2D, [0, 2, 1] on 3D, [0, 1, 3, 2] on 4D.
    """
    swapped = sum(1 for i, a in enumerate(axes) if a != i)
    return swapped == 2


# --- Handler factories ---

def _make_binary_handler(op: OpType) -> OpHandler:
    """Create a handler for binary ops where the second arg may be a scalar.

    ATen binary ops like add.Tensor, mul.Tensor always have the tensor as
    the first arg. The second arg is either another tensor (fx node) or a
    Python literal (int/float) that was evaluated during tracing.
    """
    def handler(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
        inputs = [node_map[fx_node.args[0].name]]
        attrs = {}

        second_arg = fx_node.args[1]
        if hasattr(second_arg, "name"):
            inputs.append(node_map[second_arg.name])
        else:
            attrs["scalar"] = float(second_arg)

        _emit(fx_node, graph, node_map, op, inputs, attrs or None)
    return handler


def _make_simple_handler(op: OpType) -> OpHandler:
    """Create a handler for a 1:1 ATen op -> our op mapping.

    All args that look like fx nodes become inputs. No attrs.
    Works for unary ops (exp, tanh, relu) and multi-input ops
    where every arg is a tensor (le, eq, bitwise_and).
    """
    def handler(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
        inputs = [node_map[arg.name] for arg in fx_node.args if hasattr(arg, "name")]
        _emit(fx_node, graph, node_map, op, inputs)
    return handler


# --- Reduction and axis-based handlers ---

def _make_reduction_handler(op: OpType) -> OpHandler:
    """Create a handler for reduction ops with axis and keepdim attrs."""
    def handler(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
        inputs = [node_map[fx_node.args[0].name]]
        axis = _get_axis(fx_node)
        keepdim = fx_node.args[2] if len(fx_node.args) > 2 else fx_node.kwargs.get("keepdim", False)
        _emit(fx_node, graph, node_map, op, inputs, {"axis": axis, "keepdim": keepdim})
    return handler


def _handle_max_dim(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten.max.dim — returns (values, indices) tuple.

    Emits a MAX node for the values. The indices are not supported;
    getitem with index 0 aliases to the values tensor downstream.
    """
    # meta["val"] is a tuple (values_tensor, indices_tensor) — read shape from values
    val = fx_node.meta["val"][0]
    shape = tuple(val.shape)
    dtype = str(val.dtype).replace("torch.", "")

    inputs = [node_map[fx_node.args[0].name]]
    axis = _get_axis(fx_node)
    keepdim = fx_node.args[2] if len(fx_node.args) > 2 else fx_node.kwargs.get("keepdim", False)

    tensor_name = fx_node.name
    graph.add_tensor(tensor_name, shape, dtype)
    graph.add_node(OpType.MAX, inputs, tensor_name, {"axis": axis, "keepdim": keepdim})
    node_map[fx_node.name] = tensor_name


def _handle_softmax(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten._softmax / aten.softmax.int — normalize along axis."""
    inputs = [node_map[fx_node.args[0].name]]
    axis = _get_axis(fx_node)
    _emit(fx_node, graph, node_map, OpType.SOFTMAX, inputs, {"axis": axis})


def _handle_mean(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten.mean.dim — decompose into SUM + DIV by element count."""
    shape, dtype = _output_meta(fx_node)
    inputs = [node_map[fx_node.args[0].name]]
    axis = _get_axis(fx_node)
    keepdim = fx_node.args[2] if len(fx_node.args) > 2 else fx_node.kwargs.get("keepdim", False)

    # SUM along axis -> intermediate
    sum_name = f"{fx_node.name}_sum"
    graph.add_tensor(sum_name, shape, dtype)
    graph.add_node(OpType.SUM, inputs, sum_name, {"axis": axis, "keepdim": keepdim})

    # DIV by element count -> final output
    count = int(fx_node.args[0].meta["val"].shape[axis])
    tensor_name = fx_node.name
    graph.add_tensor(tensor_name, shape, dtype)
    graph.add_node(OpType.DIV, [sum_name], tensor_name, {"scalar": float(count)})
    node_map[fx_node.name] = tensor_name


# --- Shape ops: transpose ---

def _handle_transpose_2d(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten.t / aten.numpy_T — always a 2D dim0/dim1 swap."""
    inputs = [node_map[fx_node.args[0].name]]
    _emit(fx_node, graph, node_map, OpType.TRANSPOSE, inputs, {"dim0": 0, "dim1": 1})


def _handle_transpose_int(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten.transpose.int(tensor, dim0, dim1) with negative dim normalization."""
    inputs = [node_map[fx_node.args[0].name]]
    ndim = len(fx_node.args[0].meta["val"].shape)
    dim0 = fx_node.args[1] % ndim
    dim1 = fx_node.args[2] % ndim
    _emit(fx_node, graph, node_map, OpType.TRANSPOSE, inputs, {"dim0": dim0, "dim1": dim1})


# --- Shape ops: permute ---

def _handle_permute(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten.permute — TRANSPOSE if it's a 2-axis swap, PERMUTE otherwise.

    The distinction matters because TRANSPOSE can be absorbed into matmul
    flags by the BLAS absorption pass, while general PERMUTE cannot.
    """
    inputs = [node_map[fx_node.args[0].name]]
    axes = list(fx_node.args[1])

    if _is_transpose(axes):
        dim0, dim1 = [i for i, a in enumerate(axes) if a != i]
        _emit(fx_node, graph, node_map, OpType.TRANSPOSE, inputs, {"dim0": dim0, "dim1": dim1})
    else:
        _emit(fx_node, graph, node_map, OpType.PERMUTE, inputs, {"axes": axes})


# --- Shape ops: reshape ---

def _handle_reshape(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten.view, aten.reshape, aten.unflatten, aten.unsqueeze, aten.squeeze.

    Target shape is always read from output metadata rather than args,
    since args may contain unresolved -1 dims while meta is fully concrete.
    """
    inputs = [node_map[fx_node.args[0].name]]
    shape, _ = _output_meta(fx_node)
    _emit(fx_node, graph, node_map, OpType.RESHAPE, inputs, {"shape": shape})


# --- Multi-node handlers: linear, addmm ---

def _handle_linear(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten.linear(input, weight, bias?).

    Weight is [out_features, in_features]. We emit TRANSPOSE + MATMUL
    (the BLAS absorption pass folds the TRANSPOSE into transpose_b=True).
    If bias is present, emits a separate ADD node after the matmul.
    """
    input_name = node_map[fx_node.args[0].name]
    weight_name = node_map[fx_node.args[1].name]
    bias_arg = fx_node.args[2] if len(fx_node.args) > 2 else None

    shape, dtype = _output_meta(fx_node)

    # TRANSPOSE(weight[N,K]) -> weight_t[K,N]
    weight_t_name = f"{fx_node.name}_weight_t"
    weight_shape = graph.tensors[weight_name].shape
    graph.add_tensor(weight_t_name, (weight_shape[1], weight_shape[0]), dtype)
    graph.add_node(OpType.TRANSPOSE, [weight_name], weight_t_name, {"dim0": 0, "dim1": 1})

    if bias_arg is not None:
        # MATMUL -> intermediate, then ADD bias -> final output
        mm_name = f"{fx_node.name}_mm"
        graph.add_tensor(mm_name, shape, dtype)
        graph.add_node(OpType.MATMUL, [input_name, weight_t_name], mm_name)

        tensor_name = fx_node.name
        graph.add_tensor(tensor_name, shape, dtype)
        graph.add_node(OpType.ADD, [mm_name, node_map[bias_arg.name]], tensor_name)
        node_map[fx_node.name] = tensor_name
    else:
        _emit(fx_node, graph, node_map, OpType.MATMUL, [input_name, weight_t_name])


def _handle_addmm(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten.addmm(bias, input, weight) — Conv1D layout.

    Weight is [K, N] (already the right shape for input @ weight).
    Emits MATMUL + ADD. The BLAS absorption pass will pre-transpose
    the constant weight to [N, K] and set transpose_b=True for performance.
    """
    bias_name = node_map[fx_node.args[0].name]
    input_name = node_map[fx_node.args[1].name]
    weight_name = node_map[fx_node.args[2].name]

    shape, dtype = _output_meta(fx_node)

    # MATMUL(input, weight[K,N]) -> intermediate
    mm_name = f"{fx_node.name}_mm"
    graph.add_tensor(mm_name, shape, dtype)
    graph.add_node(OpType.MATMUL, [input_name, weight_name], mm_name)

    # ADD(mm_out, bias) -> final output
    tensor_name = fx_node.name
    graph.add_tensor(tensor_name, shape, dtype)
    graph.add_node(OpType.ADD, [mm_name, bias_name], tensor_name)
    node_map[fx_node.name] = tensor_name


# --- Single-node handlers with attrs ---

def _handle_layer_norm(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten.layer_norm.default(input, normalized_shape, weight, bias, eps?)."""
    inputs = [
        node_map[fx_node.args[0].name],  # x
        node_map[fx_node.args[2].name],  # weight (gamma)
        node_map[fx_node.args[3].name],  # bias (beta)
    ]
    eps = fx_node.args[4] if len(fx_node.args) > 4 else 1e-5
    _emit(fx_node, graph, node_map, OpType.LAYERNORM, inputs, {"eps": eps})


def _handle_pow(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten.pow.Tensor_Scalar — unary with scalar exponent."""
    inputs = [node_map[fx_node.args[0].name]]
    _emit(fx_node, graph, node_map, OpType.POW, inputs, {"scalar": float(fx_node.args[1])})


def _handle_ne_scalar(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten.ne.Scalar — not-equal comparison with scalar."""
    inputs = [node_map[fx_node.args[0].name]]
    _emit(fx_node, graph, node_map, OpType.CMP_NE, inputs, {"scalar": fx_node.args[1]})


def _handle_embedding(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten.embedding — table lookup.

    ATen order is (weight_table, indices), but our kernel expects
    (indices, weight_table) so indices come first.
    """
    inputs = [
        node_map[fx_node.args[1].name],  # indices
        node_map[fx_node.args[0].name],  # weight table
    ]
    _emit(fx_node, graph, node_map, OpType.EMBEDDING, inputs)


# --- No-op handlers ---

def _handle_noop(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle ops that are identity aliases at inference time (contiguous, dropout, alias)."""
    node_map[fx_node.name] = node_map[fx_node.args[0].name]


def _handle_assert_metadata(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten._assert_tensor_metadata — pure no-op, skip entirely."""
    pass


# --- Handler registry ---

ATEN_HANDLERS: dict[object, OpHandler] = {
    # Simple unary ops (no attrs, single tensor input)
    torch.ops.aten.relu.default:        _make_simple_handler(OpType.RELU),
    torch.ops.aten.exp.default:         _make_simple_handler(OpType.EXP),
    torch.ops.aten.tanh.default:        _make_simple_handler(OpType.TANH),

    # Simple multi-input ops (all args are tensors, no attrs)
    torch.ops.aten.mm.default:          _make_simple_handler(OpType.MATMUL),
    torch.ops.aten.matmul.default:      _make_simple_handler(OpType.MATMUL),
    torch.ops.aten.bmm.default:         _make_simple_handler(OpType.MATMUL),
    torch.ops.aten.le.Tensor:           _make_simple_handler(OpType.CMP_LE),
    torch.ops.aten.eq.Tensor:           _make_simple_handler(OpType.CMP_EQ),
    torch.ops.aten.__and__.Tensor:      _make_simple_handler(OpType.BITWISE_AND),

    # Binary ops (second arg may be tensor or scalar)
    torch.ops.aten.add.Tensor:          _make_binary_handler(OpType.ADD),
    torch.ops.aten.sub.Tensor:          _make_binary_handler(OpType.SUB),
    torch.ops.aten.mul.Tensor:          _make_binary_handler(OpType.MUL),
    torch.ops.aten.div.Tensor:          _make_binary_handler(OpType.DIV),

    # Reductions (axis + keepdim)
    torch.ops.aten.amax.default:        _make_reduction_handler(OpType.MAX),
    torch.ops.aten.sum.dim_IntList:     _make_reduction_handler(OpType.SUM),
    torch.ops.aten.max.dim:             _handle_max_dim,
    torch.ops.aten.mean.dim:            _handle_mean,

    # Axis-based (no keepdim)
    torch.ops.aten._softmax.default:    _handle_softmax,
    torch.ops.aten.softmax.int:         _handle_softmax,

    # Transpose
    torch.ops.aten.t.default:           _handle_transpose_2d,
    torch.ops.aten.numpy_T.default:     _handle_transpose_2d,
    torch.ops.aten.transpose.int:       _handle_transpose_int,

    # Permute (classified as TRANSPOSE or PERMUTE)
    torch.ops.aten.permute.default:     _handle_permute,

    # Reshape (all variants read target shape from output metadata)
    torch.ops.aten.view.default:        _handle_reshape,
    torch.ops.aten.reshape.default:     _handle_reshape,
    torch.ops.aten.unflatten.int:       _handle_reshape,
    torch.ops.aten.unsqueeze.default:   _handle_reshape,
    torch.ops.aten.squeeze.dim:         _handle_reshape,

    # Linear and Conv1D (decomposed into primitive ops)
    torch.ops.aten.linear.default:      _handle_linear,
    torch.ops.aten.addmm.default:       _handle_addmm,

    # Single-node ops with attrs
    torch.ops.aten.layer_norm.default:  _handle_layer_norm,
    torch.ops.aten.pow.Tensor_Scalar:   _handle_pow,
    torch.ops.aten.ne.Scalar:           _handle_ne_scalar,
    torch.ops.aten.embedding.default:   _handle_embedding,

    # No-ops (alias or inference-mode identity)
    torch.ops.aten.contiguous.default:                      _handle_noop,
    torch.ops.aten.dropout.default:                         _handle_noop,
    torch.ops.aten.alias.default:                           _handle_noop,
    torch.ops.aten._assert_tensor_metadata.default:         _handle_assert_metadata,
}
