"""PyTorch exporter: torch.nn.Module -> our graph IR.

Uses torch.export to trace the model, then maps the resulting
ATen ops to our IR via the handler registry in handlers.py.

We handle high-level ops like aten.linear directly rather than
decomposing them into primitives. This preserves useful information
(e.g., linear's weight layout) and avoids a decompose-then-fuse
round-trip.

Key mapping decisions:
  - aten.linear(input, weight, bias?) -> TRANSPOSE(weight) + MATMUL + ADD
  - aten.addmm(bias, input, weight) -> MATMUL + ADD (Conv1D layout)
  - aten.permute -> TRANSPOSE (2-axis swap) or PERMUTE (general)
  - placeholder nodes -> graph.inputs (via graph_signature) or graph.constants (for params)
  - output node -> graph.outputs
"""

import numpy as np
import torch
import torch.nn as nn
from torch.export.graph_signature import InputKind

from ...ir import Graph
from .handlers import ATEN_HANDLERS


def export_pytorch(
    model: nn.Module,
    example_inputs: tuple,
    dynamic_dims: dict[str, list[tuple[str, int]]] | None = None,
    example_kwargs: dict | None = None,
) -> Graph:
    """Export a PyTorch model to our Graph IR.

    Sets eval mode and disables gradients for clean tracing,
    then runs torch.export and maps ATen ops to our IR.

    Args:
        model: The PyTorch model to export.
        example_inputs: Tuple of example input tensors for tracing.
        dynamic_dims: Symbol definitions for dynamic dimensions.
            Maps symbol name to list of (input_name, dim_index) tuples.
            e.g., {'L': [('input_ids', 1), ('position_ids', 1)]}
            means dim 1 of both inputs is dynamic, called "L".
        example_kwargs: Keyword arguments for the model forward pass.
            e.g., {'position_ids': position_ids_tensor}

    Returns:
        A populated Graph ready for optimization passes.
    """
    model.eval()
    with torch.no_grad():
        return _export_core(model, example_inputs, dynamic_dims, example_kwargs)


def _export_core(
    model: nn.Module,
    example_inputs: tuple,
    dynamic_dims: dict[str, list[tuple[str, int]]] | None = None,
    example_kwargs: dict | None = None,
) -> Graph:
    """Core export logic — torch.export + ATen handler dispatch."""
    # Step 1: Build torch dynamic_shapes and run torch.export
    torch_dynamic_shapes = None
    if dynamic_dims:
        dims = {}  # symbol name -> Dim object
        torch_dynamic_shapes = {}
        for sym_name, specs in dynamic_dims.items():
            dim_obj = torch.export.Dim(sym_name, min=1)
            dims[sym_name] = dim_obj
            for input_name, dim_idx in specs:
                if input_name not in torch_dynamic_shapes:
                    torch_dynamic_shapes[input_name] = {}
                torch_dynamic_shapes[input_name][dim_idx] = dim_obj

    exported = torch.export.export(
        model, example_inputs,
        kwargs=example_kwargs or {},
        dynamic_shapes=torch_dynamic_shapes,
    )

    # Step 2: Build symbol map (SymInt string repr -> user symbol name)
    symbol_map: dict[str, str] = {}
    if dynamic_dims:
        for fx_node in exported.graph_module.graph.nodes:
            if fx_node.op != "placeholder":
                continue
            val = fx_node.meta["val"]
            for sym_name, specs in dynamic_dims.items():
                for input_name, dim_idx in specs:
                    if fx_node.name == input_name:
                        d = val.shape[dim_idx]
                        if not isinstance(d, int):
                            symbol_map[str(d)] = sym_name

    graph = Graph()

    # fx node name -> our tensor name. Lets compute nodes look up their
    # inputs by the fx node names that appear in args.
    node_map: dict[str, str] = {}

    # Step 3: Walk placeholders — register tensors, classify as input vs constant
    _process_placeholders(exported, graph, node_map)

    # Step 4: Walk call_function nodes — map ATen ops to our IR
    _process_compute_nodes(exported, graph, node_map, symbol_map)

    # Step 5: Walk output node — register graph outputs
    _process_outputs(exported, graph, node_map)

    # Step 6: Store dynamic_dims on graph for downstream resolution
    if dynamic_dims:
        graph.dynamic_dims = dynamic_dims

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
        # Concretize SymInts to plain ints (TensorInfo.shape is always concrete)
        val = fx_node.meta["val"]
        shape = tuple(
            d if isinstance(d, int) else int(d.node.hint)
            for d in val.shape
        )
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
            # Load actual weight data — check state_dict first, then constants
            # (registered buffers like causal masks may be in constants instead)
            state_key = param_targets[fx_node.name]
            if state_key in exported.state_dict:
                tensor_info.buffer = exported.state_dict[state_key].numpy(force=True)
            elif hasattr(exported, 'constants') and state_key in exported.constants:
                tensor_info.buffer = exported.constants[state_key].numpy(force=True)
            else:
                raise ValueError(f"Weight/buffer '{state_key}' not found in state_dict or constants")


def _process_compute_nodes(exported, graph: Graph, node_map: dict[str, str],
                           symbol_map: dict[str, str]) -> None:
    """Map ATen compute ops to our graph nodes via the handler registry."""
    for fx_node in exported.graph_module.graph.nodes:
        if fx_node.op != "call_function":
            continue

        # Skip scalar-producing ops (sym_size, arithmetic on SymInts, etc.)
        # These appear with dynamic shapes; their values are consumed via metadata.
        val = fx_node.meta.get("val")
        if val is not None and not isinstance(val, torch.Tensor):
            if not (isinstance(val, tuple) and val and isinstance(val[0], torch.Tensor)):
                continue

        handler = ATEN_HANDLERS.get(fx_node.target)
        if handler is None:
            raise ValueError(f"Unsupported ATen op: {fx_node.target}")
        handler(fx_node, graph, node_map, symbol_map)


def _process_outputs(exported, graph: Graph, node_map: dict[str, str]) -> None:
    """Register graph output tensors."""
    for fx_node in exported.graph_module.graph.nodes:
        if fx_node.op != "output":
            continue
        # output node's args is a tuple of (tuple of output nodes,)
        for arg in fx_node.args[0]:
            if hasattr(arg, "name"):
                graph.outputs.append(node_map[arg.name])


