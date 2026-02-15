


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


def _memory_aware_order_v2(graph: Graph) -> list[Node]:
    """Topological sort with alias-aware, lazily re-scored memory heuristic.

    Improvements over `_memory_aware_order`:
      1. Scores freed bytes at alias roots (not raw tensor names).
      2. Uses lazy re-scoring on heap pop so priorities stay current as
         consumer counts change.
      3. Uses a simple "memory pressure" score:
           freed_now - output_allocation + small inplace-opportunity bonus.

    This function is intentionally separate so both versions can be compared
    during planner tuning.
    """
    # Compute in-degrees (only count edges from other nodes, not external tensors)
    in_degree: dict[int, int] = {}
    for node in graph.nodes.values():
        in_degree[node.id] = sum(1 for t in node.inputs if t in graph._producer)

    external = set(graph.inputs) | set(graph.constants)

    # Cache tensor sizes once for cheap score recomputation.
    tensor_bytes: dict[str, int] = {
        name: _tensor_size(graph, name) for name in graph.tensors
    }

    # Cache alias-root resolution to avoid repeated chain walks in scoring.
    alias_root_cache: dict[str, str] = {}

    def _root(name: str) -> str:
        cached = alias_root_cache.get(name)
        if cached is not None:
            return cached
        resolved = _resolve_alias(name, graph)
        alias_root_cache[name] = resolved
        return resolved

    # Track remaining consumer edges per alias root.
    remaining_consumers: dict[str, int] = defaultdict(int)
    for node in graph.nodes.values():
        for inp in node.inputs:
            remaining_consumers[_root(inp)] += 1

    def _freed_bytes(node: Node) -> int:
        """Bytes that would become free immediately after scheduling `node`."""
        freed = 0
        seen_roots: set[str] = set()
        for inp in node.inputs:
            root = _root(inp)
            if root in external or root in seen_roots:
                continue
            seen_roots.add(root)
            if remaining_consumers[root] == 1:
                freed += tensor_bytes[root]
        return freed

    def _output_alloc_bytes(node: Node) -> int:
        """Approximate bytes newly allocated by this node's output."""
        if node.output in external:
            return 0
        op_def = OP_REGISTRY.get(node.op)
        if op_def is not None and op_def.alias:
            return 0
        return tensor_bytes[node.output]

    def _inplace_bonus(node: Node) -> int:
        """Small bonus when this output is likely to be consumed in-place next."""
        out_size = tensor_bytes[node.output]
        for consumer_id in graph._consumers.get(node.output, []):
            consumer = graph.nodes[consumer_id]
            op_def = OP_REGISTRY.get(consumer.op)
            if op_def is None or not op_def.inplace:
                continue
            if not consumer.inputs or consumer.inputs[0] != node.output:
                continue
            if tensor_bytes[consumer.output] == out_size:
                # Keep bonus small to avoid dominating freed-vs-alloc pressure.
                return out_size // 8
        return 0

    def _score(node: Node) -> int:
        # Higher is better. Freeing memory is good; creating new live memory is bad.
        return _freed_bytes(node) - _output_alloc_bytes(node) + _inplace_bonus(node)

    # Heap entries: (-score, node_id). Negate for max-heap behavior.
    ready: list[tuple[int, int]] = []
    for nid, deg in in_degree.items():
        if deg == 0:
            node = graph.nodes[nid]
            heapq.heappush(ready, (-_score(node), nid))

    order: list[Node] = []
    while ready:
        neg_score, nid = heapq.heappop(ready)
        node = graph.nodes[nid]

        # Lazy re-score: heap priorities go stale as remaining consumers change.
        current_score = _score(node)
        if -neg_score != current_score:
            heapq.heappush(ready, (-current_score, nid))
            continue

        order.append(node)

        # Consume this node's inputs (edge counts are root-resolved).
        for inp in node.inputs:
            remaining_consumers[_root(inp)] -= 1

        # Decrement in-degree for consumers of this node's output.
        for consumer_id in graph._consumers.get(node.output, []):
            in_degree[consumer_id] -= 1
            if in_degree[consumer_id] == 0:
                consumer = graph.nodes[consumer_id]
                heapq.heappush(ready, (-_score(consumer), consumer_id))

    return order

def _memory_aware_order_event_driven(graph: Graph) -> list[Node]:
    """Topological sort with event-driven priority updates.

    Addresses two weaknesses of `_memory_aware_order`:
      1. Push-time score staleness: ready-node priorities are refreshed when an
         input's remaining consumer count drops to 1.
      2. Freed-only objective: score uses net memory delta
         (freed_input_bytes - estimated_output_allocation_bytes).

    The implementation keeps stale heap entries and skips them on pop via a
    per-node version counter.
    """
    # Compute in-degrees (only edges produced by graph nodes, not external tensors)
    in_degree: dict[int, int] = {}
    for node in graph.nodes.values():
        in_degree[node.id] = sum(1 for t in node.inputs if t in graph._producer)

    external = set(graph.inputs) | set(graph.constants)

    # For each node, count how many times it consumes each alias-resolved root.
    # We count input edges (not unique consumer nodes) so repeated inputs are
    # modeled correctly (e.g., add(x, x)).
    node_root_uses: dict[int, dict[str, int]] = {}

    # Remaining unresolved input edges per root tensor.
    remaining_consumers: dict[str, int] = defaultdict(int)

    # Remaining unresolved input edges per root tensor per consumer node.
    # root -> {node_id: remaining_edge_count_from_this_root}
    remaining_by_root_node: dict[str, dict[int, int]] = defaultdict(dict)

    for node in graph.nodes.values():
        uses: dict[str, int] = defaultdict(int)
        for inp in node.inputs:
            root = _resolve_alias(inp, graph)
            if root in external:
                continue
            uses[root] += 1
            remaining_consumers[root] += 1
        node_root_uses[node.id] = dict(uses)
        for root, count in uses.items():
            remaining_by_root_node[root][node.id] = count

    # Ready heap with stale-entry skipping via versioning.
    # Entry: (-score, node_id, version)
    ready: list[tuple[int, int, int]] = []
    ready_ids: set[int] = set()
    scheduled: set[int] = set()
    version: dict[int, int] = defaultdict(int)

    def _output_allocation_bytes(node: Node) -> int:
        """Estimate bytes newly allocated by this node's output."""
        if node.output in external:
            return 0

        op_def = OP_REGISTRY.get(node.op)
        if op_def is not None and op_def.alias:
            return 0

        out_bytes = _tensor_size(graph, node.output)

        # Approximate planner in-place rule in the score model: if first input's
        # root would die at this node and sizes match, treat output as zero-allocation.
        if op_def is not None and op_def.inplace and node.inputs:
            first_root = _resolve_alias(node.inputs[0], graph)
            first_use_count = node_root_uses[node.id].get(first_root, 0)
            if (
                first_root not in external
                and first_use_count > 0
                and remaining_consumers[first_root] == first_use_count
                and _tensor_size(graph, first_root) == out_bytes
            ):
                return 0

        return out_bytes

    def _freed_bytes(node: Node) -> int:
        """Bytes freed if this node is scheduled next."""
        freed = 0
        for root, use_count in node_root_uses[node.id].items():
            if remaining_consumers[root] == use_count:
                freed += _tensor_size(graph, root)
        return freed

    def _score(node: Node) -> int:
        return _freed_bytes(node) - _output_allocation_bytes(node)

    def _push(node_id: int) -> None:
        """Push a fresh priority entry for a ready node."""
        if node_id in scheduled or node_id not in ready_ids:
            return
        version[node_id] += 1
        heapq.heappush(ready, (-_score(graph.nodes[node_id]), node_id, version[node_id]))

    # Seed ready set
    for node_id, deg in in_degree.items():
        if deg == 0:
            ready_ids.add(node_id)
            _push(node_id)

    order: list[Node] = []
    while ready:
        _neg_score, node_id, v = heapq.heappop(ready)
        if node_id in scheduled:
            continue
        if version[node_id] != v:
            continue  # stale entry

        node = graph.nodes[node_id]
        scheduled.add(node_id)
        ready_ids.discard(node_id)
        order.append(node)

        # Consume this node's input edges (grouped by alias root).
        for root, used_count in node_root_uses[node_id].items():
            remaining_consumers[root] -= used_count

            per_node = remaining_by_root_node[root]
            left_for_this_node = per_node.get(node_id, 0) - used_count
            if left_for_this_node <= 0:
                per_node.pop(node_id, None)
            else:
                per_node[node_id] = left_for_this_node

            # Event-driven update: when only one edge remains for a root,
            # promote its final consumer immediately if already ready.
            if remaining_consumers[root] == 1 and per_node:
                last_consumer = next(iter(per_node))
                if last_consumer in ready_ids and last_consumer not in scheduled:
                    _push(last_consumer)

        # Standard Kahn unlock step.
        for consumer_id in graph._consumers.get(node.output, []):
            in_degree[consumer_id] -= 1
            if in_degree[consumer_id] == 0:
                ready_ids.add(consumer_id)
                _push(consumer_id)

    return order


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
        if op_def is None or not op_def.alias:
            break
        name = producer.inputs[0]
    return name
