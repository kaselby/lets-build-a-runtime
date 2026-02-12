"""Export a PyTorch model to our graph IR.

Takes a torch.nn.Module, runs it through torch.export (without
decompositions) and maps the resulting fx.Graph into our
Graph/Node/TensorInfo representation.

We handle high-level ops like aten.linear directly rather than
decomposing them into primitives. This preserves useful information
(e.g., linear's weight layout) and avoids a decompose-then-fuse
round-trip.

Key mapping decisions:
  - aten.linear(input, weight, bias?) -> MATMUL(input, weight, transpose_b=True) + ADD
  - aten.permute -> TRANSPOSE (2-axis swap) or PERMUTE (general)
  - aten.relu -> RELU
  - placeholder nodes -> graph.inputs (via graph_signature) or graph.constants (for params)
  - output node -> graph.outputs
"""

import operator

import numpy as np
import torch
import torch.nn as nn
from torch.export.graph_signature import InputKind, OutputKind

from .ir import Graph, OpType


from typing import Callable

# Handler signature: (fx_node, graph, node_map) -> None
OpHandler = Callable[[object, Graph, dict[str, str]], None]

# Populated after handler functions are defined (see bottom of file)
ATEN_HANDLERS: dict[object, OpHandler] = {}


def export_model(model: nn.Module, example_inputs: tuple) -> Graph:
    """Export a PyTorch model to our Graph IR.

    Args:
        model: The PyTorch model to export.
        example_inputs: Tuple of example input tensors for tracing.

    Returns:
        A populated Graph ready for optimization passes.
    """
    # Step 1: torch.export (no decomposition — we handle high-level ops directly)
    exported = torch.export.export(model, example_inputs)

    graph = Graph()

    # fx node name -> our tensor name. Lets compute nodes look up their
    # inputs by the fx node names that appear in args.
    node_map: dict[str, str] = {}

    # Step 2: Walk placeholders — register tensors, classify as input vs constant
    _process_placeholders(exported, graph, node_map)

    # Step 3: Walk call_function nodes — map ATen ops to our IR
    _process_compute_nodes(exported, graph, node_map)

    # Step 4: Walk output node — register graph outputs
    _process_outputs(exported, graph, node_map)

    return graph


def _process_placeholders(exported, graph: Graph, node_map: dict[str, str]) -> None:
    """Register placeholder nodes as input or constant tensors.

    Uses graph_signature.input_specs to reliably classify each placeholder
    as a user input or a model parameter (weight/bias).
    """
    # Build lookups from graph_signature
    input_kinds = {}          # placeholder name -> InputKind
    param_targets = {}        # placeholder name -> state_dict key (e.g. "fc1.weight")
    for spec in exported.graph_signature.input_specs:
        input_kinds[spec.arg.name] = spec.kind
        if spec.target is not None:
            param_targets[spec.arg.name] = spec.target

    for fx_node in exported.graph_module.graph.nodes:
        if fx_node.op != "placeholder":
            continue

        # Extract shape/dtype from tracing metadata
        val = fx_node.meta["val"]
        shape = tuple(val.shape)
        dtype = str(val.dtype).replace("torch.", "")  # "torch.float32" -> "float32"

        # Use the fx node name as our tensor name (readable, unique)
        tensor_name = fx_node.name
        tensor_info = graph.add_tensor(tensor_name, shape, dtype)
        node_map[fx_node.name] = tensor_name

        # Classify based on graph_signature
        kind = input_kinds[fx_node.name]
        if kind == InputKind.USER_INPUT:
            graph.inputs.append(tensor_name)
        elif kind in (InputKind.PARAMETER, InputKind.BUFFER):
            graph.constants.append(tensor_name)
            # Load actual weight data from the state dict
            state_key = param_targets[fx_node.name]
            tensor_info.buffer = exported.state_dict[state_key].numpy(force=True)


def _process_compute_nodes(exported, graph: Graph, node_map: dict[str, str]) -> None:
    """Map ATen compute ops to our graph nodes via the handler registry."""
    for fx_node in exported.graph_module.graph.nodes:
        if fx_node.op != "call_function":
            continue

        handler = ATEN_HANDLERS.get(fx_node.target)
        if handler is None:
            raise ValueError(f"Unsupported ATen op: {fx_node.target}")
        handler(fx_node, graph, node_map)


def _make_simple_handler(op: OpType) -> OpHandler:
    """Create a handler for a 1:1 ATen op -> our op mapping."""
    def handler(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
        val = fx_node.meta["val"]
        shape = tuple(val.shape)
        dtype = str(val.dtype).replace("torch.", "")
        input_names = [node_map[arg.name] for arg in fx_node.args if hasattr(arg, "name")]
        tensor_name = fx_node.name
        graph.add_tensor(tensor_name, shape, dtype)
        graph.add_node(op, input_names, tensor_name)
        node_map[fx_node.name] = tensor_name
    return handler


def _is_transpose(axes: list[int]) -> bool:
    """Check if a permutation is a simple two-axis swap (transpose).

    A transpose swaps exactly two dimensions and leaves all others in place.
    Examples: [1, 0] on 2D, [0, 2, 1] on 3D, [0, 1, 3, 2] on 4D.
    """
    swapped = sum(1 for i, a in enumerate(axes) if a != i)
    return swapped == 2


def _handle_permute(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Map aten.permute to TRANSPOSE (2-axis swap) or PERMUTE (general)."""
    val = fx_node.meta["val"]
    shape = tuple(val.shape)
    dtype = str(val.dtype).replace("torch.", "")

    input_names = [node_map[fx_node.args[0].name]]
    axes = list(fx_node.args[1])

    if _is_transpose(axes):
        # Find the two swapped dims
        dim0, dim1 = [i for i, a in enumerate(axes) if a != i]
        op = OpType.TRANSPOSE
        attrs = {"dim0": dim0, "dim1": dim1}
    else:
        op = OpType.PERMUTE
        attrs = {"axes": axes}

    tensor_name = fx_node.name
    graph.add_tensor(tensor_name, shape, dtype)
    graph.add_node(op, input_names, tensor_name, attrs)
    node_map[fx_node.name] = tensor_name


def _handle_linear(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten.linear(input, weight, bias?).

    Weight is [out_features, in_features]. We emit MATMUL with transpose_b=True
    so the C kernel passes CblasTrans to sgemm — keeping the weight in its
    original layout for better BLAS packing performance at large dimensions.

    If bias is present, emits a separate ADD node after the matmul.
    """
    input_arg = fx_node.args[0]
    weight_arg = fx_node.args[1]
    bias_arg = fx_node.args[2] if len(fx_node.args) > 2 else None

    val = fx_node.meta["val"]
    out_shape = tuple(val.shape)
    out_dtype = str(val.dtype).replace("torch.", "")

    if bias_arg is not None:
        # MATMUL(input, weight, transpose_b=True) -> intermediate
        mm_name = f"{fx_node.name}_mm"
        mm_inputs = [node_map[input_arg.name], node_map[weight_arg.name]]
        graph.add_tensor(mm_name, out_shape, out_dtype)
        graph.add_node(OpType.MATMUL, mm_inputs, mm_name, {"transpose_b": True})

        # ADD(mm_out, bias) -> final output
        add_name = fx_node.name
        add_inputs = [mm_name, node_map[bias_arg.name]]
        graph.add_tensor(add_name, out_shape, out_dtype)
        graph.add_node(OpType.ADD, add_inputs, add_name)
        node_map[fx_node.name] = add_name
    else:
        # No bias — just MATMUL with transpose_b
        tensor_name = fx_node.name
        mm_inputs = [node_map[input_arg.name], node_map[weight_arg.name]]
        graph.add_tensor(tensor_name, out_shape, out_dtype)
        graph.add_node(OpType.MATMUL, mm_inputs, tensor_name, {"transpose_b": True})
        node_map[fx_node.name] = tensor_name


def _process_outputs(exported, graph: Graph, node_map: dict[str, str]) -> None:
    """Register graph output tensors."""
    for fx_node in exported.graph_module.graph.nodes:
        if fx_node.op != "output":
            continue
        # output node's args is a tuple of (tuple of output nodes,)
        for arg in fx_node.args[0]:
            if hasattr(arg, "name"):
                graph.outputs.append(node_map[arg.name])


def _handle_reduction(op: OpType) -> OpHandler:
    """Create a handler for reduction ops (amax, sum) with axis and keepdim attrs."""
    def handler(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
        val = fx_node.meta["val"]
        # Some reduction ops return a tuple (values, indices) — e.g. amax vs max.
        # We handle the single-tensor case here.
        if isinstance(val, (tuple, list)):
            val = val[0]
        shape = tuple(val.shape)
        dtype = str(val.dtype).replace("torch.", "")
        ndim = len(fx_node.args[0].meta["val"].shape)

        input_names = [node_map[fx_node.args[0].name]]
        # axis can be an int or list of ints; normalize to int if single-element list
        axis = fx_node.args[1] if len(fx_node.args) > 1 else fx_node.kwargs.get("dim", -1)
        if isinstance(axis, (list, tuple)) and len(axis) == 1:
            axis = axis[0]
        # Normalize negative axis
        if isinstance(axis, int) and axis < 0:
            axis += ndim
        keepdim = fx_node.args[2] if len(fx_node.args) > 2 else fx_node.kwargs.get("keepdim", False)

        tensor_name = fx_node.name
        graph.add_tensor(tensor_name, shape, dtype)
        graph.add_node(op, input_names, tensor_name, {"axis": axis, "keepdim": keepdim})
        node_map[fx_node.name] = tensor_name
    return handler


def _handle_mean(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten.mean.dim — decompose into SUM + DIV by element count.

    Reuses existing SUM and DIV ops rather than adding a new MEAN op type.
    """
    val = fx_node.meta["val"]
    shape = tuple(val.shape)
    dtype = str(val.dtype).replace("torch.", "")
    input_val = fx_node.args[0].meta["val"]
    ndim = len(input_val.shape)

    input_names = [node_map[fx_node.args[0].name]]
    axis = fx_node.args[1] if len(fx_node.args) > 1 else fx_node.kwargs.get("dim", -1)
    if isinstance(axis, (list, tuple)) and len(axis) == 1:
        axis = axis[0]
    if isinstance(axis, int) and axis < 0:
        axis += ndim
    keepdim = fx_node.args[2] if len(fx_node.args) > 2 else fx_node.kwargs.get("keepdim", False)

    # SUM along axis
    sum_name = f"{fx_node.name}_sum"
    graph.add_tensor(sum_name, shape, dtype)
    graph.add_node(OpType.SUM, input_names, sum_name, {"axis": axis, "keepdim": keepdim})

    # DIV by element count along the reduced axis
    count = int(input_val.shape[axis])

    tensor_name = fx_node.name
    graph.add_tensor(tensor_name, shape, dtype)
    graph.add_node(OpType.DIV, [sum_name], tensor_name, {"scalar": float(count)})
    node_map[fx_node.name] = tensor_name


def _handle_softmax(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten._softmax and aten.softmax.int (input, dim, ...)."""
    val = fx_node.meta["val"]
    shape = tuple(val.shape)
    dtype = str(val.dtype).replace("torch.", "")
    ndim = len(fx_node.args[0].meta["val"].shape)

    input_names = [node_map[fx_node.args[0].name]]
    axis = fx_node.args[1] if len(fx_node.args) > 1 else fx_node.kwargs.get("dim", -1)
    if isinstance(axis, int) and axis < 0:
        axis += ndim

    tensor_name = fx_node.name
    graph.add_tensor(tensor_name, shape, dtype)
    graph.add_node(OpType.SOFTMAX, input_names, tensor_name, {"axis": axis})
    node_map[fx_node.name] = tensor_name


def _handle_exp(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten.exp — unary, single input."""
    val = fx_node.meta["val"]
    shape = tuple(val.shape)
    dtype = str(val.dtype).replace("torch.", "")

    input_names = [node_map[fx_node.args[0].name]]
    tensor_name = fx_node.name
    graph.add_tensor(tensor_name, shape, dtype)
    graph.add_node(OpType.EXP, input_names, tensor_name)
    node_map[fx_node.name] = tensor_name


def _handle_transpose_int(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten.transpose.int(tensor, dim0, dim1) with negative dim normalization."""
    val = fx_node.meta["val"]
    shape = tuple(val.shape)
    dtype = str(val.dtype).replace("torch.", "")
    ndim = len(fx_node.args[0].meta["val"].shape)

    input_names = [node_map[fx_node.args[0].name]]
    dim0 = fx_node.args[1]
    dim1 = fx_node.args[2]
    # Normalize negative dims
    if dim0 < 0:
        dim0 += ndim
    if dim1 < 0:
        dim1 += ndim

    tensor_name = fx_node.name
    graph.add_tensor(tensor_name, shape, dtype)
    graph.add_node(OpType.TRANSPOSE, input_names, tensor_name, {"dim0": dim0, "dim1": dim1})
    node_map[fx_node.name] = tensor_name


def _make_binary_handler(op: OpType) -> OpHandler:
    """Create a handler for binary ops where the second arg may be a scalar.

    If the second arg is a tensor, creates a 2-input node as usual.
    If it's a scalar, stores it as attrs["scalar"] — no constant tensor
    allocated, and the node has only 1 tensor input.
    """
    def handler(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
        val = fx_node.meta["val"]
        shape = tuple(val.shape)
        dtype = str(val.dtype).replace("torch.", "")

        input_names = [node_map[fx_node.args[0].name]]
        attrs = {}

        second_arg = fx_node.args[1]
        if hasattr(second_arg, "name"):
            input_names.append(node_map[second_arg.name])
        else:
            attrs["scalar"] = float(second_arg)

        tensor_name = fx_node.name
        graph.add_tensor(tensor_name, shape, dtype)
        graph.add_node(op, input_names, tensor_name, attrs)
        node_map[fx_node.name] = tensor_name
    return handler


def _handle_max_dim(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten.max.dim — returns (values, indices) tuple.

    Emits a MAX node for the values. The indices are not supported;
    getitem with index 0 aliases to the values tensor.
    """
    val = fx_node.meta["val"]
    # meta["val"] is a tuple (values_tensor, indices_tensor)
    values_val = val[0]
    shape = tuple(values_val.shape)
    dtype = str(values_val.dtype).replace("torch.", "")
    ndim = len(fx_node.args[0].meta["val"].shape)

    input_names = [node_map[fx_node.args[0].name]]
    dim = fx_node.args[1] if len(fx_node.args) > 1 else fx_node.kwargs.get("dim", -1)
    if isinstance(dim, int) and dim < 0:
        dim += ndim
    keepdim = fx_node.args[2] if len(fx_node.args) > 2 else fx_node.kwargs.get("keepdim", False)

    tensor_name = fx_node.name
    graph.add_tensor(tensor_name, shape, dtype)
    graph.add_node(OpType.MAX, input_names, tensor_name, {"axis": dim, "keepdim": keepdim})
    node_map[fx_node.name] = tensor_name


def _handle_getitem(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle operator.getitem — unpacks tuples from ops like max.dim.

    Index 0 (values) aliases to the producer's output tensor.
    Other indices (e.g., indices from max) are ignored.
    """
    source_name = fx_node.args[0].name
    index = fx_node.args[1]
    if index == 0 and source_name in node_map:
        node_map[fx_node.name] = node_map[source_name]


def _handle_transpose_2d(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten.t and aten.numpy_T — always a 2D dim0/dim1 swap."""
    val = fx_node.meta["val"]
    shape = tuple(val.shape)
    dtype = str(val.dtype).replace("torch.", "")

    input_names = [node_map[fx_node.args[0].name]]
    tensor_name = fx_node.name
    graph.add_tensor(tensor_name, shape, dtype)
    graph.add_node(OpType.TRANSPOSE, input_names, tensor_name, {"dim0": 0, "dim1": 1})
    node_map[fx_node.name] = tensor_name


def _handle_layer_norm(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten.layer_norm.default(input, normalized_shape, weight, bias, eps?).

    Emits a single LAYERNORM node with three inputs (x, weight, bias).
    The eps value is stored as an attr (default 1e-5).
    """
    val = fx_node.meta["val"]
    shape = tuple(val.shape)
    dtype = str(val.dtype).replace("torch.", "")

    input_names = [
        node_map[fx_node.args[0].name],  # x
        node_map[fx_node.args[2].name],  # weight (gamma)
        node_map[fx_node.args[3].name],  # bias (beta)
    ]
    eps = fx_node.args[4] if len(fx_node.args) > 4 else 1e-5

    tensor_name = fx_node.name
    graph.add_tensor(tensor_name, shape, dtype)
    graph.add_node(OpType.LAYERNORM, input_names, tensor_name, {"eps": eps})
    node_map[fx_node.name] = tensor_name


def _handle_reshape(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten.view / aten.reshape — emit RESHAPE with target shape."""
    val = fx_node.meta["val"]
    shape = tuple(val.shape)
    dtype = str(val.dtype).replace("torch.", "")

    input_names = [node_map[fx_node.args[0].name]]
    target_shape = tuple(fx_node.args[1])

    tensor_name = fx_node.name
    graph.add_tensor(tensor_name, shape, dtype)
    graph.add_node(OpType.RESHAPE, input_names, tensor_name, {"shape": target_shape})
    node_map[fx_node.name] = tensor_name


def _handle_reshape_from_meta(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle ops that reshape by inferring target shape from output metadata.

    Used for unflatten, unsqueeze, squeeze — ops where the target shape isn't
    passed as a flat list arg but is best read from the traced output shape.
    """
    val = fx_node.meta["val"]
    shape = tuple(val.shape)
    dtype = str(val.dtype).replace("torch.", "")

    input_names = [node_map[fx_node.args[0].name]]

    tensor_name = fx_node.name
    graph.add_tensor(tensor_name, shape, dtype)
    graph.add_node(OpType.RESHAPE, input_names, tensor_name, {"shape": shape})
    node_map[fx_node.name] = tensor_name


def _handle_contiguous(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten.contiguous — no-op (our runtime always writes contiguous buffers)."""
    node_map[fx_node.name] = node_map[fx_node.args[0].name]


def _handle_sdpa(fx_node, graph: Graph, node_map: dict[str, str]) -> None:
    """Handle aten.scaled_dot_product_attention → ATTENTION directly.

    Takes Q, K, V as the first three args (all tensors). Maps straight
    to our fused ATTENTION op — no fusion pass needed.
    """
    val = fx_node.meta["val"]
    shape = tuple(val.shape)
    dtype = str(val.dtype).replace("torch.", "")

    input_names = [
        node_map[fx_node.args[0].name],  # Q
        node_map[fx_node.args[1].name],  # K
        node_map[fx_node.args[2].name],  # V
    ]

    tensor_name = fx_node.name
    graph.add_tensor(tensor_name, shape, dtype)
    graph.add_node(OpType.ATTENTION, input_names, tensor_name)
    node_map[fx_node.name] = tensor_name


# --- Handler registry ---
# Adding a new ATen op: either use _make_simple_handler for 1:1 mappings,
# or write a custom handler function with signature (fx_node, graph, node_map) -> None.

ATEN_HANDLERS.update({
    torch.ops.aten.linear.default:      _handle_linear,
    torch.ops.aten.relu.default:        _make_simple_handler(OpType.RELU),
    torch.ops.aten.mm.default:          _make_simple_handler(OpType.MATMUL),
    torch.ops.aten.matmul.default:      _make_simple_handler(OpType.MATMUL),
    torch.ops.aten.add.Tensor:          _make_binary_handler(OpType.ADD),
    torch.ops.aten.permute.default:     _handle_permute,
    torch.ops.aten.t.default:           _handle_transpose_2d,
    torch.ops.aten.numpy_T.default:     _handle_transpose_2d,
    # Element-wise binary ops (scalar-aware: scalar stored as attr, not tensor)
    torch.ops.aten.div.Tensor:          _make_binary_handler(OpType.DIV),
    torch.ops.aten.sub.Tensor:          _make_binary_handler(OpType.SUB),
    torch.ops.aten.mul.Tensor:          _make_binary_handler(OpType.MUL),
    torch.ops.aten.exp.default:         _handle_exp,
    # Transpose with explicit dims
    torch.ops.aten.transpose.int:       _handle_transpose_int,
    # Reduction ops
    torch.ops.aten.amax.default:        _handle_reduction(OpType.MAX),
    torch.ops.aten.max.dim:             _handle_max_dim,
    torch.ops.aten.sum.dim_IntList:     _handle_reduction(OpType.SUM),
    # Compound ops
    torch.ops.aten._softmax.default:    _handle_softmax,
    torch.ops.aten.softmax.int:         _handle_softmax,
    torch.ops.aten.layer_norm.default:  _handle_layer_norm,
    # Tuple unpacking
    operator.getitem:                   _handle_getitem,
    # Shape ops
    torch.ops.aten.view.default:        _handle_reshape,
    torch.ops.aten.reshape.default:     _handle_reshape,
    torch.ops.aten.unflatten.int:       _handle_reshape_from_meta,
    torch.ops.aten.unsqueeze.default:   _handle_reshape_from_meta,
    torch.ops.aten.squeeze.dim:         _handle_reshape_from_meta,
    # Batched matmul
    torch.ops.aten.bmm.default:         _make_simple_handler(OpType.MATMUL),
    # No-op (contiguous is always satisfied in our runtime)
    torch.ops.aten.contiguous.default:  _handle_contiguous,
    # Reduction (decomposed into SUM + DIV)
    torch.ops.aten.mean.dim:            _handle_mean,
    # Fused attention (SDPA maps directly to ATTENTION — no fusion pass needed)
    torch.ops.aten.scaled_dot_product_attention.default: _handle_sdpa,
})
