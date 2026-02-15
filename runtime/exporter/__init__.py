"""Exporter package: convert models from any framework to our graph IR.

Framework-specific export functions live in subpackages (torch/, etc.).
This module re-exports them and provides shared utilities like summary().

    from runtime.exporter import export_pytorch, summary

    graph = export_pytorch(model, (example_input,))
    print(summary(graph))
"""

from collections import Counter

from ..ir import Graph
from .torch import export_pytorch

# Backward-compatible alias used by tests
export_model = export_pytorch


def summary(graph: Graph) -> str:
    """Human-readable summary of an exported graph.

    Shows inputs, outputs, op distribution, and weight statistics.
    Useful for verifying an export looks sane before running passes.
    """
    lines = []

    # Inputs
    lines.append(f"Inputs ({len(graph.inputs)}):")
    for name in graph.inputs:
        t = graph.tensors[name]
        lines.append(f"  {name}: {t.shape} {t.dtype}")

    # Outputs
    lines.append(f"Outputs ({len(graph.outputs)}):")
    for name in graph.outputs:
        t = graph.tensors[name]
        lines.append(f"  {name}: {t.shape} {t.dtype}")

    # Constants / weights
    total_params = 0
    for name in graph.constants:
        t = graph.tensors[name]
        total_params += int(_prod(t.shape))
    lines.append(f"Constants: {len(graph.constants)} ({total_params:,} parameters)")

    # Op distribution
    op_counts = Counter(node.op.name for node in graph.nodes.values())
    lines.append(f"Nodes: {len(graph.nodes)}")
    for op_name, count in op_counts.most_common():
        lines.append(f"  {op_name}: {count}")

    return "\n".join(lines)


def _prod(shape: tuple[int, ...]) -> int:
    result = 1
    for d in shape:
        result *= d
    return result
