"""Memory planner tests: arena correctness invariants.

The critical property: no two tensors with overlapping lifetimes should
occupy overlapping regions of the arena. This is checked mechanically
regardless of model architecture.
"""

import numpy as np
import pytest
import torch

from runtime.exporter import export_model
from runtime.ir import Graph, OpType
from runtime.passes import run_pipeline
from runtime.planner import (
    plan, _compute_lifetimes, _find_reshape_aliases,
    SCRATCH_CALCULATORS, register_scratch,
)

from conftest import SimpleMLP


def _check_no_overlap(ep):
    """Verify that no two simultaneously-live tensors share arena memory.

    For every pair of tensors with overlapping lifetimes, their
    [offset, offset+size) ranges in the arena must not intersect.
    """
    order = ep.order
    graph = ep.graph
    offsets = ep.offsets

    aliases = _find_reshape_aliases(order)
    lifetimes = _compute_lifetimes(graph, order, aliases)

    # Build (offset, size, lifetime) for each allocated tensor
    allocs = []
    for lt in lifetimes:
        if lt.tensor_name not in offsets:
            continue
        tensor = graph.tensors[lt.tensor_name]
        size = int(np.prod(tensor.shape)) * np.dtype(tensor.dtype).itemsize
        allocs.append((offsets[lt.tensor_name], size, lt))

    # Check all pairs
    for i, (off_a, sz_a, lt_a) in enumerate(allocs):
        for j, (off_b, sz_b, lt_b) in enumerate(allocs):
            if i >= j:
                continue
            # Check temporal overlap (both alive at the same time)
            if lt_a.born <= lt_b.dies and lt_b.born <= lt_a.dies:
                # Check spatial overlap
                end_a = off_a + sz_a
                end_b = off_b + sz_b
                overlaps = off_a < end_b and off_b < end_a
                assert not overlaps, (
                    f"Arena overlap: {lt_a.tensor_name}[{off_a}:{end_a}) "
                    f"and {lt_b.tensor_name}[{off_b}:{end_b}) "
                    f"are both live during steps {max(lt_a.born, lt_b.born)}-"
                    f"{min(lt_a.dies, lt_b.dies)}"
                )


@pytest.mark.parametrize("batch,dim", [(1, 64), (4, 128), (32, 256)])
def test_no_arena_overlap(batch, dim):
    """No two simultaneously-live tensors should share arena memory."""
    model = SimpleMLP(dim)
    model.eval()
    graph = export_model(model, (torch.randn(batch, dim),))
    run_pipeline(graph)
    ep = plan(graph)
    _check_no_overlap(ep)


@pytest.mark.parametrize("batch,dim", [(1, 64), (4, 128)])
def test_arena_reuse_saves_memory(batch, dim):
    """Arena should be smaller than naive allocation (no reuse)."""
    model = SimpleMLP(dim)
    model.eval()
    graph = export_model(model, (torch.randn(batch, dim),))
    run_pipeline(graph)
    ep = plan(graph)

    # Naive: sum of all intermediate tensor sizes
    external = set(graph.inputs) | set(graph.constants)
    naive_size = 0
    for name in ep.offsets:
        tensor = graph.tensors[name]
        naive_size += int(np.prod(tensor.shape)) * np.dtype(tensor.dtype).itemsize

    assert ep.arena_size <= naive_size, (
        f"Arena ({ep.arena_size}) should be <= naive ({naive_size})"
    )


def test_reshape_not_allocated():
    """RESHAPE output tensors should not get arena allocations."""
    # Build a model that uses reshape
    class ReshapeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(64, 64)

        def forward(self, x):
            y = self.fc(x)
            return y.reshape(2, 32)

    model = ReshapeModel()
    model.eval()
    graph = export_model(model, (torch.randn(1, 64),))
    run_pipeline(graph)

    # Find RESHAPE output tensor names
    reshape_outputs = {n.output for n in graph if n.op == OpType.RESHAPE}

    ep = plan(graph)

    for name in reshape_outputs:
        assert name not in ep.offsets, (
            f"RESHAPE output '{name}' should not have an arena allocation"
        )


# ---------------------------------------------------------------------------
# Scratch buffer tests
# ---------------------------------------------------------------------------

def _build_simple_graph() -> Graph:
    """Build a minimal graph: input -> MATMUL(input, weight) -> output.

    Returns the graph with weight data loaded so it passes validation.
    """
    graph = Graph()
    graph.add_tensor("x", (4, 8))
    graph.inputs.append("x")

    w = graph.add_tensor("w", (8, 8))
    w.buffer = np.ones((8, 8), dtype=np.float32)
    graph.constants.append("w")

    graph.add_tensor("y", (4, 8))
    graph.outputs.append("y")
    graph.add_node(OpType.MATMUL, ["x", "w"], "y")
    return graph


def test_scratch_allocation_exists():
    """Registering a scratch calculator creates an arena allocation in the plan."""
    # Register scratch for MATMUL (just for testing — real MATMUL doesn't need scratch)
    scratch_size = 256  # 64 floats
    old = SCRATCH_CALCULATORS.get(OpType.MATMUL)
    try:
        register_scratch(OpType.MATMUL, lambda in_shapes, out_shape: scratch_size)
        graph = _build_simple_graph()
        ep = plan(graph)

        # The MATMUL node should have a scratch entry
        matmul_node = [n for n in ep.order if n.op == OpType.MATMUL][0]
        assert matmul_node.id in ep.scratch
        offset, size = ep.scratch[matmul_node.id]
        assert size == scratch_size
        assert offset >= 0
        # Arena must be large enough to hold the scratch
        assert ep.arena_size >= offset + size
    finally:
        # Clean up
        if old is not None:
            SCRATCH_CALCULATORS[OpType.MATMUL] = old
        else:
            SCRATCH_CALCULATORS.pop(OpType.MATMUL, None)


def test_scratch_not_in_regular_offsets():
    """Scratch entries should not appear in the regular offsets dict."""
    scratch_size = 128
    old = SCRATCH_CALCULATORS.get(OpType.MATMUL)
    try:
        register_scratch(OpType.MATMUL, lambda in_shapes, out_shape: scratch_size)
        graph = _build_simple_graph()
        ep = plan(graph)

        for name in ep.offsets:
            assert not name.startswith("__scratch_"), (
                f"Scratch entry '{name}' leaked into regular offsets"
            )
    finally:
        if old is not None:
            SCRATCH_CALCULATORS[OpType.MATMUL] = old
        else:
            SCRATCH_CALCULATORS.pop(OpType.MATMUL, None)


def test_scratch_no_overlap_with_intermediates():
    """Scratch arena region must not overlap with any simultaneously-live tensor."""
    # Build a graph with an intermediate that's alive during the MATMUL:
    # input -> ADD(input, bias) -> MATMUL(add_out, weight) -> output
    graph = Graph()
    graph.add_tensor("x", (4, 8))
    graph.inputs.append("x")

    bias = graph.add_tensor("bias", (8,))
    bias.buffer = np.zeros(8, dtype=np.float32)
    graph.constants.append("bias")

    w = graph.add_tensor("w", (8, 8))
    w.buffer = np.ones((8, 8), dtype=np.float32)
    graph.constants.append("w")

    graph.add_tensor("added", (4, 8))
    graph.add_node(OpType.ADD, ["x", "bias"], "added")

    graph.add_tensor("y", (4, 8))
    graph.outputs.append("y")
    graph.add_node(OpType.MATMUL, ["added", "w"], "y")

    scratch_size = 512
    old = SCRATCH_CALCULATORS.get(OpType.MATMUL)
    try:
        register_scratch(OpType.MATMUL, lambda in_shapes, out_shape: scratch_size)
        ep = plan(graph)

        matmul_node = [n for n in ep.order if n.op == OpType.MATMUL][0]
        scratch_offset, scratch_sz = ep.scratch[matmul_node.id]
        scratch_end = scratch_offset + scratch_sz

        # "added" is alive during the MATMUL step — its arena region must not overlap
        assert "added" in ep.offsets
        added_offset = ep.offsets["added"]
        added_size = 4 * 8 * 4  # 4×8 float32
        added_end = added_offset + added_size

        overlaps = scratch_offset < added_end and added_offset < scratch_end
        assert not overlaps, (
            f"Scratch [{scratch_offset}:{scratch_end}) overlaps with "
            f"'added' [{added_offset}:{added_end})"
        )
    finally:
        if old is not None:
            SCRATCH_CALCULATORS[OpType.MATMUL] = old
        else:
            SCRATCH_CALCULATORS.pop(OpType.MATMUL, None)


def test_scratch_zero_not_allocated():
    """A calculator returning 0 should not create a scratch entry."""
    old = SCRATCH_CALCULATORS.get(OpType.MATMUL)
    try:
        register_scratch(OpType.MATMUL, lambda in_shapes, out_shape: 0)
        graph = _build_simple_graph()
        ep = plan(graph)

        matmul_node = [n for n in ep.order if n.op == OpType.MATMUL][0]
        assert matmul_node.id not in ep.scratch
    finally:
        if old is not None:
            SCRATCH_CALCULATORS[OpType.MATMUL] = old
        else:
            SCRATCH_CALCULATORS.pop(OpType.MATMUL, None)


def test_scratch_passed_to_kernel():
    """Executor should append scratch buffer to kernel inputs."""
    received_inputs = []

    class MockBackend:
        name = "mock"
        def get_kernel(self, op):
            if op == OpType.MATMUL:
                def kernel(inputs, output, attrs):
                    received_inputs.append(len(inputs))
                    # Just write zeros so execution completes
                    output[:] = 0
                return kernel
            if op == OpType.ADD:
                def kernel(inputs, output, attrs):
                    output[:] = inputs[0] + inputs[1]
                return kernel
            return None

    from runtime.executor import Executor

    scratch_size = 256
    old = SCRATCH_CALCULATORS.get(OpType.MATMUL)
    try:
        register_scratch(OpType.MATMUL, lambda in_shapes, out_shape: scratch_size)
        graph = _build_simple_graph()
        ep = plan(graph)
        executor = Executor(backends=[MockBackend()])
        x = np.ones((4, 8), dtype=np.float32)
        executor.execute(ep, {"x": x})

        # Kernel should have received 3 inputs: [x, w, scratch]
        assert received_inputs == [3], (
            f"Expected MATMUL kernel to receive 3 inputs (2 + scratch), "
            f"got {received_inputs}"
        )
    finally:
        if old is not None:
            SCRATCH_CALCULATORS[OpType.MATMUL] = old
        else:
            SCRATCH_CALCULATORS.pop(OpType.MATMUL, None)
