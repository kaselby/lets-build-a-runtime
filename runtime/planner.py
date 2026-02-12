"""Memory planner: lifetime analysis and arena offset assignment.

Analyzes a Graph to determine when each intermediate tensor is live,
then packs non-overlapping lifetimes into a shared arena to minimize
memory usage. Produces an ExecutionPlan that the executor consumes.

Scratch buffers: some kernels (e.g., fused attention) need temporary
workspace that isn't an input or output. The planner handles this via
a scratch calculator registry — ops register how much scratch they need
as a function of their tensor shapes, and the planner creates single-step
arena allocations for them. The executor passes scratch as an extra
kernel input, transparent to the graph IR.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .ir import Graph, Node, OpType


@dataclass
class ExecutionPlan:
    """Everything needed to execute a graph: execution order + memory layout."""
    graph: Graph
    order: list[Node]
    arena_size: int                     # total bytes
    offsets: dict[str, int] = field(default_factory=dict)  # intermediate tensor name -> byte offset
    # Scratch workspace: node ID -> (arena byte offset, size in bytes)
    # Allocated by the planner for ops that need temporary workspace.
    scratch: dict[int, tuple[int, int]] = field(default_factory=dict)

    def allocate_arena(self) -> np.ndarray:
        """Allocate the arena buffer."""
        return np.zeros(self.arena_size, dtype=np.uint8)


@dataclass
class Lifetime:
    """When a tensor is born and when it dies, in execution order indices."""
    tensor_name: str
    size_bytes: int
    born: int       # index in execution order where producer runs
    dies: int       # index in execution order where last consumer runs


# ---------------------------------------------------------------------------
# Scratch buffer registry
# ---------------------------------------------------------------------------

# Calculator signature: (input_shapes, output_shape, attrs) -> bytes needed.
# Return 0 for no scratch. The planner calls this for every node whose
# op type has a registered calculator.
ScratchCalculator = Callable[[list[tuple[int, ...]], tuple[int, ...], dict], int]
SCRATCH_CALCULATORS: dict[OpType, ScratchCalculator] = {}


def register_scratch(op: OpType, calc: ScratchCalculator) -> None:
    """Register a scratch size calculator for an op type."""
    SCRATCH_CALCULATORS[op] = calc


FLASH_BR = 32
FLASH_BC = 32


def _attention_scratch(input_shapes: list[tuple[int, ...]], output_shape: tuple[int, ...],
                       attrs: dict) -> int:
    """Scratch for fused attention: one score matrix per batch×head slice.

    Inputs are [Q, K, V] each shaped [..., seq_len, head_dim] where leading
    dims are batch (e.g. [BH, S, D] or [B, H, S, D] — memory layout is identical).
    Each GCD thread gets its own scratch region so they can run in parallel
    without synchronization.

    Standard kernel needs S×S per slice (full attention matrix).
    Flash kernel needs B_r×B_c per slice (one tile).
    """
    q_shape = input_shapes[0]
    batch_heads = 1
    for d in q_shape[:-2]:
        batch_heads *= d
    seq_len = q_shape[-2]
    if attrs.get("flash"):
        scratch_per_slice = FLASH_BR * FLASH_BC
    else:
        scratch_per_slice = seq_len * seq_len
    return batch_heads * scratch_per_slice * 4  # float32

register_scratch(OpType.ATTENTION, _attention_scratch)


def _compute_scratch(order: list[Node], graph: Graph) -> tuple[list[Lifetime], dict[int, tuple[str, int]]]:
    """Compute scratch buffer requirements for nodes that need workspace.

    Returns:
        scratch_lifetimes: Lifetime entries (single-step) for first-fit allocation.
        scratch_info: node_id -> (scratch_name, size_bytes) for the executor.
    """
    scratch_lifetimes: list[Lifetime] = []
    scratch_info: dict[int, tuple[str, int]] = {}

    for step, node in enumerate(order):
        calc = SCRATCH_CALCULATORS.get(node.op)
        if calc is None:
            continue

        input_shapes = [graph.tensors[inp].shape for inp in node.inputs]
        output_shape = graph.tensors[node.output].shape
        size_bytes = calc(input_shapes, output_shape, node.attrs)

        if size_bytes <= 0:
            continue

        scratch_name = f"__scratch_{node.id}"
        scratch_lifetimes.append(Lifetime(
            tensor_name=scratch_name,
            size_bytes=size_bytes,
            born=step,
            dies=step,
        ))
        scratch_info[node.id] = (scratch_name, size_bytes)

    return scratch_lifetimes, scratch_info


def plan(graph: Graph) -> ExecutionPlan:
    """Analyze a graph and produce an execution plan with memory layout.

    Computes tensor lifetimes from the topological order, then assigns
    arena offsets using a greedy first-fit algorithm. RESHAPE nodes are
    treated as zero-copy aliases — their outputs share the input's arena
    region and RESHAPE is stripped from the execution order.
    """
    errors = graph.validate()
    if errors:
        raise ValueError(f"Cannot plan invalid graph: {errors}")

    order = list(graph)  # topological order (uses cached order from validate)

    # Identify RESHAPE aliases — outputs share their input's arena memory
    aliases = _find_reshape_aliases(order)

    lifetimes = _compute_lifetimes(graph, order, aliases)

    # Compute scratch requirements and add to lifetimes for unified first-fit
    scratch_lifetimes, scratch_info = _compute_scratch(order, graph)
    all_lifetimes = lifetimes + scratch_lifetimes

    all_offsets, arena_size = _assign_offsets(all_lifetimes)

    # Separate scratch offsets from regular tensor offsets
    scratch_names = {name for name, _ in scratch_info.values()}
    offsets = {k: v for k, v in all_offsets.items() if k not in scratch_names}
    scratch = {
        node_id: (all_offsets[scratch_name], size_bytes)
        for node_id, (scratch_name, size_bytes) in scratch_info.items()
    }

    return ExecutionPlan(
        graph=graph,
        order=order,
        arena_size=arena_size,
        offsets=offsets,
        scratch=scratch,
    )


def _find_reshape_aliases(order: list[Node]) -> dict[str, str]:
    """Build a map of RESHAPE output tensors to their root (non-RESHAPE) inputs.

    Follows chains: if A → RESHAPE → B → RESHAPE → C, both B and C
    map to A. This ensures all aliases point to the tensor that actually
    needs arena space.
    """
    aliases: dict[str, str] = {}
    for node in order:
        if node.op == OpType.RESHAPE:
            input_name = node.inputs[0]
            # Follow the chain to the root
            root = aliases.get(input_name, input_name)
            aliases[node.output] = root
    return aliases


def _compute_lifetimes(graph: Graph, order: list[Node],
                       aliases: dict[str, str] | None = None) -> list[Lifetime]:
    """Compute the lifetime of each intermediate tensor.

    Inputs and constants are excluded — they live outside the arena.
    Aliases (from RESHAPE) are resolved to their root tensor, extending
    the root's lifetime to cover the alias's consumers.
    """
    aliases = aliases or {}
    external = set(graph.inputs) | set(graph.constants)

    # Map tensor name -> index in execution order where it's produced
    born_at: dict[str, int] = {}
    # Map tensor name -> index of last consumer
    dies_at: dict[str, int] = {}

    alias_set = set(aliases.keys())

    for step, node in enumerate(order):
        # RESHAPE outputs don't get arena space — skip them.
        # Non-alias outputs are born at this step.
        if node.output not in external and node.output not in alias_set:
            born_at[node.output] = step

        # Each input tensor's death is extended to at least this step.
        # Resolve aliases so consuming a reshaped tensor extends the
        # root tensor's lifetime.
        for inp in node.inputs:
            resolved = aliases.get(inp, inp)
            if resolved not in external:
                dies_at[resolved] = step

    # Build Lifetime objects for all intermediate tensors
    lifetimes = []
    for name in born_at:
        tensor = graph.tensors[name]
        dtype = np.dtype(tensor.dtype)
        size_bytes = int(np.prod(tensor.shape)) * dtype.itemsize
        lifetimes.append(Lifetime(
            tensor_name=name,
            size_bytes=size_bytes,
            born=born_at[name],
            dies=dies_at.get(name, born_at[name]),  # if never consumed, dies at birth
        ))

    return lifetimes


def _assign_offsets(lifetimes: list[Lifetime]) -> tuple[dict[str, int], int]:
    """Assign arena offsets using greedy first-fit.

    Process tensors in birth order. For each new tensor, filter existing
    allocations to only those whose lifetimes overlap (i.e., are alive at
    the same time). Then find the lowest offset where the new tensor fits
    in the gaps between those live allocations.

    Returns (offsets dict, total arena size in bytes).
    """
    lifetimes.sort(key=lambda lt: lt.born)

    allocations: list[tuple[int, int, Lifetime]] = []  # (offset, size, lifetime)
    offsets: dict[str, int] = {}
    arena_size = 0

    for lt in lifetimes:
        # Only consider allocations whose lifetimes overlap with this tensor
        alive = [
            (offset, size) for offset, size, alt in allocations
            if lt.born <= alt.dies and alt.born <= lt.dies
        ]
        alive.sort()  # sort by offset

        offset = _first_fit(alive, lt.size_bytes)

        offsets[lt.tensor_name] = offset
        allocations.append((offset, lt.size_bytes, lt))
        arena_size = max(arena_size, offset + lt.size_bytes)

    return offsets, arena_size


def _first_fit(alive_regions: list[tuple[int, int]], size: int) -> int:
    """Find the lowest offset where `size` bytes fit without overlapping alive regions."""
    candidate = 0
    for region_offset, region_size in alive_regions:
        if candidate + size <= region_offset:
            return candidate  # fits in the gap before this region
        candidate = max(candidate, region_offset + region_size)
    return candidate  # fits at the end
