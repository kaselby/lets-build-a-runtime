"""Graph IR for the inference runtime.

Node-centric design: edges are implicit in each node's input list.
Each node produces exactly one output. Tensor metadata is tracked
separately from nodes — nodes are about computation, tensors are about data.

Inputs and constants (weights) are tensors without producer nodes, not
virtual nodes. Every node in the graph is a real compute operation.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Iterator

import numpy as np


class OpType(Enum):
    """Operator types supported by the runtime.

    Values use range-based numbering so related ops cluster together:
      10–19  Element-wise unary
      20–29  Element-wise binary
      30–39  Reductions
      40–49  MatMul / BLAS
      50–59  Shape / data movement
      60–69  Normalization / compound
      70–79  Fused ops
      100+   Fold-only (constant-folded, never dispatched to C)

    C code (executor.c) uses a matching enum — values must stay in sync.
    """
    # --- Element-wise unary (10–19) ---
    RELU = 10
    EXP  = 11
    TANH = 12
    POW  = 13            # attrs["scalar"]
    GELU = 14            # tanh approximation

    # --- Element-wise binary (20–29) ---
    ADD = 20
    SUB = 21
    MUL = 22
    DIV = 23

    # --- Reductions (30–39) ---
    MAX     = 30         # attrs["axis", "keepdim"]
    SUM     = 31         # attrs["axis", "keepdim"]
    SOFTMAX = 32         # attrs["axis"]

    # --- MatMul / BLAS (40–49) ---
    MATMUL = 40

    # --- Shape / data movement (50–59) ---
    RESHAPE   = 50       # attrs["shape"]
    TRANSPOSE = 51       # 2D swap or N-dim swapaxes
    PERMUTE   = 52       # attrs["axes"]
    SLICE     = 53       # attrs["byte_offset"]
    EMBEDDING = 54       # inputs: [indices, weight_table]

    # --- Normalization / compound (60–69) ---
    LAYERNORM = 60       # attrs["eps"], inputs: [x, weight, bias]

    # --- Fused ops (70–79) ---
    MATMUL_ADD      = 70 # inputs: [A, B, bias]
    FUSED_BIAS_RELU = 71 # inputs: [x, bias]
    ATTENTION       = 72 # inputs: [Q, K, V], scratch: [BH × S × S]

    # --- Fold-only (100+, constant-folded away, never dispatched to C) ---
    CAST         = 100
    EXPAND       = 101
    SLICE_TENSOR = 102   # attrs["dim", "start", "end"]
    DIFF         = 103   # attrs["n", "dim"]
    CMP_NE       = 104
    CMP_LE       = 105
    CMP_EQ       = 106
    CUMSUM       = 107   # attrs["dim"]
    BITWISE_AND  = 108
    INDEX        = 109


@dataclass
class TensorInfo:
    """Metadata for a named tensor in the graph.

    Every node output and graph input is a named tensor. The name is how
    nodes reference data flowing between them. Shape and dtype are known
    statically from tracing.

    The `buffer` field is set before execution — it's a numpy array pointing
    to the actual data. For intermediates, this is a view into the shared
    arena (zero-copy). For weights, it's the array loaded from disk. For
    graph inputs, it's whatever the caller provides.
    """
    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"

    # Set before execution: numpy array backed by arena view, weight storage, or user input
    buffer: np.ndarray | None = None


@dataclass
class Node:
    """A single operation in the computation graph.

    Nodes are purely about computation — what op to run and what tensors
    it reads and writes. Tensor metadata lives in TensorInfo objects,
    looked up by name in the Graph's tensor registry.
    """
    id: int
    op: OpType
    inputs: list[str]       # Names of tensors this node reads
    output: str             # Name of the tensor this node produces

    # Op-specific configuration (e.g., transpose axes, conv padding)
    attrs: dict[str, Any] = field(default_factory=dict)


class Graph:
    """The full computation graph: nodes, tensor metadata, and connectivity.

    Nodes are stored by ID, tensors by name. The graph tracks which tensors
    are external inputs (provided by caller), constants (weights loaded from
    disk), and outputs (returned after inference).

    Connectivity lookups (producer/consumer mappings) are maintained
    incrementally as nodes are added. Topological order is cached after
    validation.
    """

    def __init__(self) -> None:
        self.nodes: dict[int, Node] = {}
        self.tensors: dict[str, TensorInfo] = {}

        # Tensor roles — ordered lists of tensor names
        self.inputs: list[str] = []     # Fed by caller at inference time
        self.outputs: list[str] = []    # Returned to caller after inference
        self.constants: list[str] = []  # Weights/biases, loaded from disk

        # Connectivity indices (maintained by add_node)
        self._producer: dict[str, int] = {}    # tensor name -> node ID that produces it
        self._consumers: dict[str, list[int]] = {}  # tensor name -> node IDs that consume it

        self._next_id: int = 0
        self._order: list[Node] | None = None  # Cached topological order

    # --- Builder methods ---

    def add_tensor(self, name: str, shape: tuple[int, ...], dtype: str = "float32") -> TensorInfo:
        """Register a tensor in the graph. Returns the created TensorInfo."""
        if name in self.tensors:
            raise ValueError(f"Duplicate tensor name: {name}")
        info = TensorInfo(name=name, shape=shape, dtype=dtype)
        self.tensors[name] = info
        self._consumers[name] = []
        return info

    def add_node(self, op: OpType, inputs: list[str], output: str,
                 attrs: dict[str, Any] | None = None) -> Node:
        """Add a compute node to the graph. The output tensor must already be registered.

        Returns the created Node with an auto-assigned ID.
        """
        for inp in inputs:
            if inp not in self.tensors:
                raise ValueError(f"Input tensor '{inp}' not registered")
        if output not in self.tensors:
            raise ValueError(f"Output tensor '{output}' not registered")

        node_id = self._next_id
        self._next_id += 1

        node = Node(
            id=node_id,
            op=op,
            inputs=inputs,
            output=output,
            attrs=attrs or {},
        )
        self.nodes[node_id] = node

        # Update connectivity indices
        self._producer[output] = node_id
        for inp in inputs:
            self._consumers[inp].append(node_id)

        # Invalidate cached order
        self._order = None

        return node

    # --- Connectivity lookups ---

    def producer(self, tensor_name: str) -> Node | None:
        """Return the node that produces this tensor, or None for inputs/constants."""
        node_id = self._producer.get(tensor_name)
        return self.nodes[node_id] if node_id is not None else None

    def consumers(self, tensor_name: str) -> list[Node]:
        """Return all nodes that consume this tensor."""
        return [self.nodes[nid] for nid in self._consumers.get(tensor_name, [])]

    # --- Topological ordering ---

    def _toposort(self) -> list[Node]:
        """Compute topological order via Kahn's algorithm.

        Nodes whose inputs are all external (inputs/constants with no producer)
        have in-degree 0 and go first. Returns the full sorted node list.
        """
        in_degree: dict[int, int] = {}
        for node in self.nodes.values():
            in_degree[node.id] = sum(1 for t in node.inputs if t in self._producer)

        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        order: list[Node] = []

        while queue:
            nid = queue.popleft()
            node = self.nodes[nid]
            order.append(node)

            for consumer_id in self._consumers.get(node.output, []):
                in_degree[consumer_id] -= 1
                if in_degree[consumer_id] == 0:
                    queue.append(consumer_id)

        return order

    def __iter__(self) -> Iterator[Node]:
        """Iterate over nodes in topological order.

        Uses cached order from validate() if available, otherwise computes fresh.
        """
        if self._order is not None:
            yield from self._order
        else:
            order = self._toposort()
            if len(order) != len(self.nodes):
                raise ValueError("Graph has a cycle")
            yield from order

    # --- Validation ---

    def validate(self) -> list[str]:
        """Validate graph structure and cache topological order.

        Returns a list of error strings (empty = valid). On success,
        caches the topological order for iteration and execution.
        """
        errors = []

        # All tensors referenced by nodes must exist in the registry
        for node in self.nodes.values():
            for t in node.inputs:
                if t not in self.tensors:
                    errors.append(f"Node {node.id} references input tensor '{t}' not in registry")
            if node.output not in self.tensors:
                errors.append(f"Node {node.id} output tensor '{node.output}' not in registry")

        # Declared inputs/constants/outputs must be registered tensors
        for name in self.inputs:
            if name not in self.tensors:
                errors.append(f"Graph input '{name}' not in tensor registry")
        for name in self.constants:
            if name not in self.tensors:
                errors.append(f"Graph constant '{name}' not in tensor registry")
        for name in self.outputs:
            if name not in self.tensors:
                errors.append(f"Graph output '{name}' not in tensor registry")

        # Inputs and constants should NOT have a producer node (they come from outside)
        for name in self.inputs:
            if name in self._producer:
                errors.append(f"Graph input '{name}' is produced by node {self._producer[name]}")
        for name in self.constants:
            if name in self._producer:
                errors.append(f"Graph constant '{name}' is produced by node {self._producer[name]}")

        # Outputs should have a producer (they must be computed by something)
        for name in self.outputs:
            if name not in self._producer:
                errors.append(f"Graph output '{name}' has no producer node")

        # Every non-input, non-constant tensor consumed by a node should have a producer
        external = set(self.inputs) | set(self.constants)
        for node in self.nodes.values():
            for t in node.inputs:
                if t not in external and t not in self._producer:
                    errors.append(f"Node {node.id} consumes '{t}' which has no producer and is not an input/constant")

        # Check for dead nodes (output not consumed and not a graph output)
        all_consumed = set()
        for node in self.nodes.values():
            all_consumed.update(node.inputs)
        output_set = set(self.outputs)
        for node in self.nodes.values():
            if node.output not in all_consumed and node.output not in output_set:
                errors.append(f"Node {node.id} output '{node.output}' is never consumed and not a graph output")

        # Cycle detection via toposort
        order = self._toposort()
        if len(order) != len(self.nodes):
            visited = {n.id for n in order}
            stuck = [nid for nid in self.nodes if nid not in visited]
            errors.append(f"Graph has cycles involving nodes: {stuck}")
        elif not errors:
            self._order = order

        return errors

    # --- Mutation (for optimization passes) ---

    def rewire_input(self, node_id: int, old_tensor: str, new_tensor: str) -> None:
        """Change a node's input from one tensor to another.

        Replaces all occurrences of old_tensor in the node's input list
        and updates the consumer index accordingly.
        """
        node = self.nodes[node_id]
        for i, inp in enumerate(node.inputs):
            if inp == old_tensor:
                node.inputs[i] = new_tensor
                self._consumers[old_tensor].remove(node_id)
                self._consumers[new_tensor].append(node_id)
        self._order = None

    def remove_node(self, node_id: int) -> None:
        """Remove a node and clean up connectivity indices.

        Does NOT remove the output tensor — call remove_tensor separately
        if needed.
        """
        node = self.nodes.pop(node_id)
        self._producer.pop(node.output, None)
        for inp in node.inputs:
            consumers = self._consumers.get(inp, [])
            if node_id in consumers:
                consumers.remove(node_id)
        self._order = None

    def remove_tensor(self, name: str) -> None:
        """Remove a tensor from the registry."""
        self.tensors.pop(name, None)
        self._consumers.pop(name, None)
        self._producer.pop(name, None)
        self._order = None
