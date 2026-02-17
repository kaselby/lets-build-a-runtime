"""Memory plan validators.

Covers POST_PLAN phase. Validators receive a MemoryPlan
(the planner's output: execution order, arena layout, offsets).
"""

import numpy as np

from ..ir import Graph
from ..planner import MemoryPlan
from .core import Phase, Severity, ValidationResult, register_validator


def _tensor_bytes(graph: Graph, name: str) -> int:
    """Compute the byte size of a tensor from its shape and dtype."""
    tensor = graph.tensors[name]
    return int(np.prod(tensor.shape)) * np.dtype(tensor.dtype).itemsize


@register_validator("memory_plan_integrity", Phase.POST_PLAN)
def validate_memory_plan(plan: MemoryPlan) -> list[ValidationResult]:
    """Verify the memory plan's internal consistency.

    Checks that every intermediate tensor has an arena offset, the arena
    is correctly sized, scratch allocations reference valid nodes, and
    no independently-allocated co-live tensors partially overlap.
    """
    NAME = "memory_plan_integrity"
    results = []
    graph = plan.graph
    external = set(graph.inputs) | set(graph.constants)

    # --- Every non-external intermediate has an offset ---
    for node in plan.order:
        name = node.output
        if name not in external and name not in plan.offsets:
            results.append(ValidationResult(NAME, Severity.ERROR,
                f"Intermediate tensor '{name}' (node {node.id}, "
                f"{node.op.name}) has no arena offset"))

    # --- Arena sizing ---
    # Compute the tightest possible arena from actual allocations.
    max_extent = 0
    for name, offset in plan.offsets.items():
        if name in external:
            continue
        tensor = graph.tensors.get(name)
        if tensor is None:
            continue
        max_extent = max(max_extent, offset + _tensor_bytes(graph, name))

    for _, (offset, size) in plan.scratch.items():
        max_extent = max(max_extent, offset + size)

    if plan.arena_size < max_extent:
        results.append(ValidationResult(NAME, Severity.ERROR,
            f"Arena undersized: {plan.arena_size} bytes "
            f"< required {max_extent} bytes"))
    elif plan.arena_size > max_extent:
        results.append(ValidationResult(NAME, Severity.WARNING,
            f"Arena oversized: {plan.arena_size} bytes "
            f"> minimum {max_extent} bytes "
            f"({plan.arena_size - max_extent} bytes wasted)"))

    # --- Scratch allocations reference valid nodes ---
    order_ids = {node.id for node in plan.order}
    for node_id in plan.scratch:
        if node_id not in order_ids:
            results.append(ValidationResult(NAME, Severity.ERROR,
                f"Scratch allocation for node {node_id} "
                f"references a node not in the execution order"))

    # --- No partial overlaps among co-live allocations ---
    # Build lifetime info from the execution order: each tensor is born
    # when its producer runs and dies when its last consumer runs.
    step_of: dict[str, int] = {}
    for step, node in enumerate(plan.order):
        step_of[node.output] = step

    last_use: dict[str, int] = {}
    for step, node in enumerate(plan.order):
        for inp in node.inputs:
            if inp not in external:
                last_use[inp] = step

    # Collect all arena allocations with their lifetimes
    allocs: list[tuple[str, int, int, int, int]] = []  # (name, offset, size, born, dies)
    for name, offset in plan.offsets.items():
        if name in external or name not in graph.tensors:
            continue
        size = _tensor_bytes(graph, name)
        born = step_of.get(name)
        if born is None or size == 0:
            continue
        dies = last_use.get(name, born)
        allocs.append((name, offset, size, born, dies))

    # Include scratch (single-step lifetimes)
    step_by_id = {node.id: i for i, node in enumerate(plan.order)}
    for node_id, (offset, size) in plan.scratch.items():
        if size == 0:
            continue
        step = step_by_id.get(node_id)
        if step is not None:
            allocs.append((f"__scratch_{node_id}", offset, size, step, step))

    # Pairwise check for partial overlaps among co-live tensors.
    # "Partial overlap" means the memory ranges intersect but neither
    # contains the other — this is always a bug. Containment (one range
    # inside the other) is intentional: alias ops (RESHAPE, SLICE) share
    # their root's memory region.
    for i in range(len(allocs)):
        name_i, off_i, size_i, born_i, dies_i = allocs[i]
        end_i = off_i + size_i
        for j in range(i + 1, len(allocs)):
            name_j, off_j, size_j, born_j, dies_j = allocs[j]

            # Not co-live — lifetimes don't overlap
            if born_i > dies_j or born_j > dies_i:
                continue

            end_j = off_j + size_j

            # Not overlapping in memory
            if end_i <= off_j or end_j <= off_i:
                continue

            # Overlapping and co-live. Allow containment (intentional sharing).
            if (off_i <= off_j and end_i >= end_j) or \
               (off_j <= off_i and end_j >= end_i):
                continue

            results.append(ValidationResult(NAME, Severity.ERROR,
                f"Partial overlap between co-live tensors: "
                f"'{name_i}' [{off_i}:{end_i}) (steps {born_i}-{dies_i}) "
                f"and '{name_j}' [{off_j}:{end_j}) (steps {born_j}-{dies_j})"))

    return results
