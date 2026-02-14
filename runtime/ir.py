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

    Values are auto-assigned starting at 1. C code (executor.c) must use
    matching #define values — see the comment block there.
    """
    # --- Existing ops (values 1-5) ---
    MATMUL = auto()      # Matrix multiply (supports batched via batch dims)
    ADD = auto()         # Element-wise add
    RELU = auto()        # Element-wise ReLU
    TRANSPOSE = auto()   # 2D matrix transpose (swap rows/cols)
    PERMUTE = auto()     # General N-dimensional axis reordering; attrs["axes"]

    # --- New element-wise ops (values 6-9) ---
    DIV = auto()         # Element-wise divide
    SUB = auto()         # Element-wise subtract
    MUL = auto()         # Element-wise multiply
    EXP = auto()         # Element-wise exp

    # --- Reduction ops (values 10-11) ---
    MAX = auto()         # Reduce along axis; attrs["axis"], attrs["keepdim"]
    SUM = auto()         # Reduce along axis; attrs["axis"], attrs["keepdim"]

    # --- Compound ops (values 12, 14) ---
    SOFTMAX = auto()     # Softmax along axis; attrs["axis"]

    # --- Shape ops (value 13) ---
    RESHAPE = auto()     # Reshape (view); attrs["shape"]

    LAYERNORM = auto()   # LayerNorm; attrs["eps"], inputs: [x, weight, bias]

    # --- Fused ops (values 15+) ---
    MATMUL_ADD = auto()         # Fused matmul + bias add; inputs: [A, B, bias]
    FUSED_BIAS_RELU = auto()    # Fused bias add + relu; inputs: [x, bias]
    ATTENTION = auto()          # Fused multi-head attention; inputs: [Q, K, V], scratch: [BH × S × S]

    # --- GPT-2 ops (values 18-22) ---
    SLICE = auto()           # Zero-copy slice; attrs["byte_offset"]
    POW = auto()             # Element-wise power; attrs["scalar"]
    TANH = auto()            # Element-wise tanh
    GELU = auto()            # GELU activation (tanh approximation)
    EMBEDDING = auto()       # Table lookup; inputs: [indices, weight_table]

    # --- Mask/infrastructure ops (values 23+, constant-folded away) ---
    CAST = auto()           # Type cast; attrs["target_dtype"]
    EXPAND = auto()         # Broadcast expand; attrs["shape"]
    SLICE_TENSOR = auto()   # Direct tensor slice; attrs["dim", "start", "end"]
    DIFF = auto()           # torch.diff; attrs["n", "dim"], optional prepend input
    CMP_NE = auto()         # Not-equal comparison; attrs["scalar"] or 2-input
    CMP_LE = auto()         # Less-than-or-equal; 2-input
    CMP_EQ = auto()         # Equality comparison; 2-input
    CUMSUM = auto()         # Cumulative sum; attrs["dim"]
    BITWISE_AND = auto()    # Bitwise AND; 2-input
    INDEX = auto()          # Advanced indexing; inputs: [tensor, idx1, idx2, ...]


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
