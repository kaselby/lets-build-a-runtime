"""Memory-aware topological ordering variants for comparison.

Two alternative implementations beyond the default v1 in planner.py:
  v2: lazy re-score (recommended)
  v3: event-driven (most complex)
"""

from collections import defaultdict
import heapq

import numpy as np

from .ir import Graph, Node
from .ops import OP_REGISTRY


def _tensor_size(graph: Graph, name: str) -> int:
    """Compute the size in bytes of a tensor."""
    tensor = graph.tensors[name]
    return int(np.prod(tensor.shape)) * np.dtype(tensor.dtype).itemsize


def _resolve_alias(name: str, graph: Graph) -> str:
    """Walk alias chain in the graph to the non-alias producer."""
    while True:
        producer = graph.producer(name)
        if producer is None:
            break
        op_def = OP_REGISTRY.get(producer.op)
        if op_def is None or not op_def.is_alias(producer):
            break
        name = producer.inputs[0]
    return name


def memory_aware_order_v2(graph: Graph) -> list[Node]:
    """Topological sort with alias-aware, lazily re-scored memory heuristic.

    Improvements over v1:
      1. Scores freed bytes at alias roots (not raw tensor names).
      2. Uses lazy re-scoring on heap pop so priorities stay current as
         consumer counts change.
      3. Uses a "memory pressure" score:
           freed_now - output_allocation + small inplace-opportunity bonus.
    """
    in_degree: dict[int, int] = {}
    for node in graph.nodes.values():
        in_degree[node.id] = sum(1 for t in node.inputs if t in graph._producer)

    external = set(graph.inputs) | set(graph.constants)

    tensor_bytes: dict[str, int] = {
        name: _tensor_size(graph, name) for name in graph.tensors
    }

    alias_root_cache: dict[str, str] = {}

    def _root(name: str) -> str:
        cached = alias_root_cache.get(name)
        if cached is not None:
            return cached
        resolved = _resolve_alias(name, graph)
        alias_root_cache[name] = resolved
        return resolved

    remaining_consumers: dict[str, int] = defaultdict(int)
    for node in graph.nodes.values():
        for inp in node.inputs:
            remaining_consumers[_root(inp)] += 1

    def _freed_bytes(node: Node) -> int:
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
        if node.output in external:
            return 0
        op_def = OP_REGISTRY.get(node.op)
        if op_def is not None and op_def.is_alias(node):
            return 0
        return tensor_bytes[node.output]

    def _inplace_bonus(node: Node) -> int:
        out_size = tensor_bytes[node.output]
        for consumer_id in graph._consumers.get(node.output, []):
            consumer = graph.nodes[consumer_id]
            op_def = OP_REGISTRY.get(consumer.op)
            if op_def is None or not op_def.inplace:
                continue
            if not consumer.inputs or consumer.inputs[0] != node.output:
                continue
            if tensor_bytes[consumer.output] == out_size:
                return out_size // 8
        return 0

    def _score(node: Node) -> int:
        return _freed_bytes(node) - _output_alloc_bytes(node) + _inplace_bonus(node)

    ready: list[tuple[int, int]] = []
    for nid, deg in in_degree.items():
        if deg == 0:
            node = graph.nodes[nid]
            heapq.heappush(ready, (-_score(node), nid))

    order: list[Node] = []
    while ready:
        neg_score, nid = heapq.heappop(ready)
        node = graph.nodes[nid]

        current_score = _score(node)
        if -neg_score != current_score:
            heapq.heappush(ready, (-current_score, nid))
            continue

        order.append(node)

        for inp in node.inputs:
            remaining_consumers[_root(inp)] -= 1

        for consumer_id in graph._consumers.get(node.output, []):
            in_degree[consumer_id] -= 1
            if in_degree[consumer_id] == 0:
                consumer = graph.nodes[consumer_id]
                heapq.heappush(ready, (-_score(consumer), consumer_id))

    return order


def memory_aware_order_event_driven(graph: Graph) -> list[Node]:
    """Topological sort with event-driven priority updates.

    Addresses two weaknesses of v1:
      1. Push-time score staleness: ready-node priorities are refreshed when an
         input's remaining consumer count drops to 1.
      2. Freed-only objective: score uses net memory delta
         (freed_input_bytes - estimated_output_allocation_bytes).
    """
    in_degree: dict[int, int] = {}
    for node in graph.nodes.values():
        in_degree[node.id] = sum(1 for t in node.inputs if t in graph._producer)

    external = set(graph.inputs) | set(graph.constants)

    node_root_uses: dict[int, dict[str, int]] = {}
    remaining_consumers: dict[str, int] = defaultdict(int)
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

    ready: list[tuple[int, int, int]] = []
    ready_ids: set[int] = set()
    scheduled: set[int] = set()
    version: dict[int, int] = defaultdict(int)

    def _output_allocation_bytes(node: Node) -> int:
        if node.output in external:
            return 0
        op_def = OP_REGISTRY.get(node.op)
        if op_def is not None and op_def.is_alias(node):
            return 0
        out_bytes = _tensor_size(graph, node.output)
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
        freed = 0
        for root, use_count in node_root_uses[node.id].items():
            if remaining_consumers[root] == use_count:
                freed += _tensor_size(graph, root)
        return freed

    def _score(node: Node) -> int:
        return _freed_bytes(node) - _output_allocation_bytes(node)

    def _push(node_id: int) -> None:
        if node_id in scheduled or node_id not in ready_ids:
            return
        version[node_id] += 1
        heapq.heappush(ready, (-_score(graph.nodes[node_id]), node_id, version[node_id]))

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
            continue

        node = graph.nodes[node_id]
        scheduled.add(node_id)
        ready_ids.discard(node_id)
        order.append(node)

        for root, used_count in node_root_uses[node_id].items():
            remaining_consumers[root] -= used_count
            per_node = remaining_by_root_node[root]
            left_for_this_node = per_node.get(node_id, 0) - used_count
            if left_for_this_node <= 0:
                per_node.pop(node_id, None)
            else:
                per_node[node_id] = left_for_this_node
            if remaining_consumers[root] == 1 and per_node:
                last_consumer = next(iter(per_node))
                if last_consumer in ready_ids and last_consumer not in scheduled:
                    _push(last_consumer)

        for consumer_id in graph._consumers.get(node.output, []):
            in_degree[consumer_id] -= 1
            if in_degree[consumer_id] == 0:
                ready_ids.add(consumer_id)
                _push(consumer_id)

    return order
