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

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
import heapq

import numpy as np

from .ir import Graph, Node, OpType
from .ops import OP_REGISTRY


# ---------------------------------------------------------------------------
# Planner configuration
# ---------------------------------------------------------------------------

class OrderStrategy(Enum):
    """Topological ordering heuristic for execution scheduling."""
    NAIVE = auto()            # Plain Kahn's, FIFO — no memory awareness
    MEMORY_AWARE_V1 = auto()  # Freed-bytes scored at push time (default)
    MEMORY_AWARE_V2 = auto()  # Lazy re-score on pop, net pressure scoring
    MEMORY_AWARE_V3 = auto()  # Event-driven with version counters


class FitStrategy(Enum):
    """Arena gap-selection strategy for offset assignment."""
    FIRST_FIT = auto()   # Lowest offset that fits (default)
    BEST_FIT = auto()    # Smallest gap that fits (tightest packing)


@dataclass(frozen=True)
class PlannerConfig:
    """Configuration for the memory planner's strategy choices.

    All defaults match the current behavior so existing code is unaffected.
    """
    order: OrderStrategy = OrderStrategy.MEMORY_AWARE_V1
    fit: FitStrategy = FitStrategy.FIRST_FIT
    enable_inplace: bool = True
    enable_aliases: bool = True


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
# Memory-aware topological ordering
# ---------------------------------------------------------------------------

def _tensor_size(graph: Graph, name: str) -> int:
    """Compute the size in bytes of a tensor."""
    tensor = graph.tensors[name]
    return int(np.prod(tensor.shape)) * np.dtype(tensor.dtype).itemsize


def _memory_aware_order(graph: Graph) -> list[Node]:
    """Topological sort that prefers scheduling nodes which free the most memory.

    Standard Kahn's algorithm, but when multiple nodes are ready (in-degree 0),
    we pick the one whose scheduling frees the most input memory. "Frees" means
    this node is the last remaining consumer of an input tensor — once scheduled,
    that tensor's memory can be reused.

    This tends to complete one computation chain before starting another,
    reducing peak memory compared to arbitrary topological orders.
    """
    # Compute in-degrees (only count edges from other nodes, not external tensors)
    in_degree: dict[int, int] = {}
    for node in graph.nodes.values():
        in_degree[node.id] = sum(1 for t in node.inputs if t in graph._producer)

    # Track remaining consumers for each tensor (how many nodes still need to read it)
    remaining_consumers: dict[str, int] = defaultdict(int)
    for node in graph.nodes.values():
        for inp in node.inputs:
            remaining_consumers[inp] += 1

    external = set(graph.inputs) | set(graph.constants)

    def _freed_bytes(node: Node) -> int:
        """Estimate bytes freed by scheduling this node."""
        freed = 0
        for inp in node.inputs:
            if inp not in external and remaining_consumers[inp] == 1:
                freed += _tensor_size(graph, inp)
        return freed

    # Seed the heap with zero-in-degree nodes
    # Heap entries: (-freed_bytes, node_id) — negate for max-heap behavior
    ready = []
    for nid, deg in in_degree.items():
        if deg == 0:
            node = graph.nodes[nid]
            heapq.heappush(ready, (-_freed_bytes(node), nid))

    order: list[Node] = []
    while ready:
        _, nid = heapq.heappop(ready)
        node = graph.nodes[nid]
        order.append(node)

        # Update remaining consumer counts for this node's inputs
        for inp in node.inputs:
            remaining_consumers[inp] -= 1

        # Decrement in-degree for consumers of this node's output
        for consumer_id in graph._consumers.get(node.output, []):
            in_degree[consumer_id] -= 1
            if in_degree[consumer_id] == 0:
                consumer = graph.nodes[consumer_id]
                heapq.heappush(ready, (-_freed_bytes(consumer), consumer_id))

    return order


def _naive_toposort(graph: Graph) -> list[Node]:
    """Plain Kahn's algorithm with FIFO ordering — no memory awareness.

    Baseline for measuring how much memory-aware ordering saves.
    """
    in_degree: dict[int, int] = {}
    for node in graph.nodes.values():
        in_degree[node.id] = sum(1 for t in node.inputs if t in graph._producer)

    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    order: list[Node] = []

    while queue:
        nid = queue.popleft()
        node = graph.nodes[nid]
        order.append(node)
        for consumer_id in graph._consumers.get(node.output, []):
            in_degree[consumer_id] -= 1
            if in_degree[consumer_id] == 0:
                queue.append(consumer_id)

    return order


def _get_order(graph: Graph, strategy: OrderStrategy) -> list[Node]:
    """Dispatch to the requested ordering strategy."""
    if strategy == OrderStrategy.NAIVE:
        return _naive_toposort(graph)
    if strategy == OrderStrategy.MEMORY_AWARE_V1:
        return _memory_aware_order(graph)
    if strategy == OrderStrategy.MEMORY_AWARE_V2:
        from .order import memory_aware_order_v2
        return memory_aware_order_v2(graph)
    if strategy == OrderStrategy.MEMORY_AWARE_V3:
        from .order import memory_aware_order_event_driven
        return memory_aware_order_event_driven(graph)
    raise ValueError(f"Unknown ordering strategy: {strategy}")


# ---------------------------------------------------------------------------
# Alias resolution (graph-level only — follows alias ops in producer chain)
# ---------------------------------------------------------------------------

def _resolve_alias(name: str, graph: Graph) -> str:
    """Walk alias chain in the graph to the non-alias producer.

    Only follows graph-level aliases (RESHAPE, contiguous SLICE) — not
    planner-level in-place decisions. Used for consumer counting, where
    we need graph-structural relationships regardless of in-place choices.
    """
    while True:
        producer = graph.producer(name)
        if producer is None:
            break
        op_def = OP_REGISTRY.get(producer.op)
        if op_def is None or not op_def.is_alias(producer):
            break
        name = producer.inputs[0]
    return name


# ---------------------------------------------------------------------------
# Lifetime analysis (unified alias + in-place sharing)
# ---------------------------------------------------------------------------

def _compute_lifetimes(
    graph: Graph, order: list[Node], config: PlannerConfig | None = None,
) -> tuple[dict[str, Lifetime], dict[str, str]]:
    """Compute lifetimes with unified alias and in-place memory sharing.

    A single pass over the execution order handles both:
      - Alias ops (RESHAPE): unconditionally share input's memory
      - In-place ops (elementwise): share if input is dying and same byte size

    Both result in the same thing: the output doesn't get its own arena space,
    and the root tensor's lifetime extends to cover the output's consumers.

    Returns:
        lifetimes: tensor name -> Lifetime for arena-owning tensors only.
        memory_root: tensor name -> arena-owning root for shared tensors.
    """
    if config is None:
        config = PlannerConfig()
    external = set(graph.inputs) | set(graph.constants)

    # memory_root[name] = the tensor that owns this name's arena memory.
    # Absent = owns its own memory.
    memory_root: dict[str, str] = {}

    def get_root(name: str) -> str:
        """Follow the full sharing chain (alias + in-place) to the arena owner."""
        while name in memory_root:
            name = memory_root[name]
        return name

    # Precompute total consumers per graph-level tensor (alias-resolved only).
    # In-place decisions depend on knowing when a tensor's last graph consumer
    # runs, so this must be computed before the main pass.
    total_consumers: dict[str, int] = defaultdict(int)
    for node in order:
        for inp in node.inputs:
            total_consumers[_resolve_alias(inp, graph)] += 1

    consumed: dict[str, int] = defaultdict(int)
    born_at: dict[str, int] = {}
    dies_at: dict[str, int] = {}

    for step, node in enumerate(order):
        # Update graph-level consumed counts (alias-resolved, not in-place-resolved)
        for inp in node.inputs:
            consumed[_resolve_alias(inp, graph)] += 1

        # Extend memory root's death to this step for all inputs
        for inp in node.inputs:
            root = get_root(_resolve_alias(inp, graph))
            if root not in external:
                dies_at[root] = step

        if node.output in external:
            continue

        op_def = OP_REGISTRY.get(node.op)

        # Alias: share input's memory (but not external buffers — they aren't ours)
        if config.enable_aliases and op_def is not None and op_def.is_alias(node):
            alias_input = _resolve_alias(node.inputs[0], graph)
            root = get_root(alias_input)
            if root not in external:
                memory_root[node.output] = root
                continue

        # In-place: share if input is dying and same byte size
        if config.enable_inplace and op_def is not None and op_def.inplace and node.inputs:
            alias_input = _resolve_alias(node.inputs[0], graph)
            if (alias_input not in external
                    and consumed[alias_input] == total_consumers[alias_input]
                    and _tensor_size(graph, alias_input) == _tensor_size(graph, node.output)):
                memory_root[node.output] = get_root(alias_input)
                continue

        # This output gets its own arena allocation
        born_at[node.output] = step

    # Build Lifetime objects for arena-owning tensors
    lifetimes: dict[str, Lifetime] = {}
    for name in born_at:
        lifetimes[name] = Lifetime(
            tensor_name=name,
            size_bytes=_tensor_size(graph, name),
            born=born_at[name],
            dies=dies_at.get(name, born_at[name]),
        )

    return lifetimes, memory_root


# ---------------------------------------------------------------------------
# Scratch buffers
# ---------------------------------------------------------------------------

def _compute_scratch(order: list[Node], graph: Graph) -> tuple[dict[str, Lifetime], dict[int, tuple[str, int]]]:
    """Compute scratch buffer requirements for nodes that need workspace.

    Scratch buffers are single-step lifetimes (born and die at the same step)
    so they participate in the same first-fit allocation as regular tensors.

    Returns:
        scratch_lifetimes: Lifetime dict for first-fit allocation.
        scratch_info: node_id -> (scratch_name, size_bytes) for the executor.
    """
    scratch_lifetimes: dict[str, Lifetime] = {}
    scratch_info: dict[int, tuple[str, int]] = {}

    for step, node in enumerate(order):
        op_def = OP_REGISTRY.get(node.op)
        if op_def is None or op_def.scratch is None:
            continue

        input_shapes = [graph.tensors[inp].shape for inp in node.inputs]
        output_shape = graph.tensors[node.output].shape
        size_bytes = op_def.scratch(input_shapes, output_shape, node.attrs)

        if size_bytes <= 0:
            continue

        scratch_name = f"__scratch_{node.id}"
        scratch_lifetimes[scratch_name] = Lifetime(
            tensor_name=scratch_name,
            size_bytes=size_bytes,
            born=step,
            dies=step,
        )
        scratch_info[node.id] = (scratch_name, size_bytes)

    return scratch_lifetimes, scratch_info


# ---------------------------------------------------------------------------
# Arena offset assignment
# ---------------------------------------------------------------------------

def _assign_offsets(
    lifetimes: dict[str, Lifetime], n_steps: int,
    fit: FitStrategy = FitStrategy.FIRST_FIT,
) -> tuple[dict[str, int], int]:
    """Assign arena offsets using a greedy gap-selection strategy.

    Walks execution steps in order, maintaining only the currently-alive
    allocations. Tensors are added to the active set at birth and evicted
    after their last consumer runs.

    Returns (offsets dict, total arena size in bytes).
    """
    fit_fn = _best_fit if fit == FitStrategy.BEST_FIT else _first_fit

    # Group by birth step
    born_at: dict[int, list[str]] = defaultdict(list)
    for name, lt in lifetimes.items():
        born_at[lt.born].append(name)

    # Group by eviction step (one past death — alive through dies, freed after)
    evict_at: dict[int, list[str]] = defaultdict(list)
    for name, lt in lifetimes.items():
        evict_at[lt.dies + 1].append(name)

    active: dict[str, tuple[int, int]] = {}  # name -> (offset, size)
    offsets: dict[str, int] = {}
    arena_size = 0

    for step in range(n_steps):
        # Evict tensors whose lifetime ended before this step
        for name in evict_at.get(step, []):
            active.pop(name, None)

        # Allocate tensors born at this step
        for name in born_at.get(step, []):
            lt = lifetimes[name]
            regions = sorted(active.values())
            offset = fit_fn(regions, lt.size_bytes)
            offsets[name] = offset
            active[name] = (offset, lt.size_bytes)
            arena_size = max(arena_size, offset + lt.size_bytes)

    return offsets, arena_size


def _first_fit(alive_regions: list[tuple[int, int]], size: int) -> int:
    """Find the lowest offset where `size` bytes fit without overlapping alive regions."""
    candidate = 0
    for region_offset, region_size in alive_regions:
        if candidate + size <= region_offset:
            return candidate
        candidate = max(candidate, region_offset + region_size)
    return candidate


def _best_fit(alive_regions: list[tuple[int, int]], size: int) -> int:
    """Find the smallest gap where `size` bytes fit without overlapping alive regions."""
    best_offset = -1
    best_waste = float('inf')

    candidate = 0
    for region_offset, region_size in alive_regions:
        gap = region_offset - candidate
        if gap >= size and gap - size < best_waste:
            best_offset = candidate
            best_waste = gap - size
        candidate = max(candidate, region_offset + region_size)

    return candidate if best_offset < 0 else best_offset


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

def plan(graph: Graph, config: PlannerConfig | None = None) -> ExecutionPlan:
    """Analyze a graph and produce an execution plan with memory layout.

    Computes tensor lifetimes from the topological order, then assigns
    arena offsets using a gap-selection strategy. Alias ops (RESHAPE)
    and in-place ops (elementwise) share their input's arena memory
    unless disabled via config.
    """
    if config is None:
        config = PlannerConfig()

    errors = graph.validate()
    if errors:
        raise ValueError(f"Cannot plan invalid graph: {errors}")

    order = _get_order(graph, config.order)

    lifetimes, memory_root = _compute_lifetimes(graph, order, config)

    scratch_lifetimes, scratch_info = _compute_scratch(order, graph)
    all_lifetimes = {**lifetimes, **scratch_lifetimes}

    all_offsets, arena_size = _assign_offsets(all_lifetimes, n_steps=len(order), fit=config.fit)

    # Copy offsets for shared tensors (alias + in-place)
    for name, root in memory_root.items():
        # Follow chain to the arena-owning root
        while root in memory_root:
            root = memory_root[root]
        if root in all_offsets:
            base = all_offsets[root]
            # Apply alias byte offset (e.g., SLICE into the middle of a buffer)
            producer = graph.producer(name)
            if producer is not None:
                op_def = OP_REGISTRY.get(producer.op)
                if op_def is not None and op_def.alias_offset is not None:
                    base += op_def.alias_offset(producer)
            all_offsets[name] = base

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
