"""Tests for dynamic shape support: resolve_graph and Session rebind.

Verifies that exporting once with symbolic dimensions and resolving to
different concrete sizes produces the same results as fresh exports at
each size.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from runtime.exporter import export_pytorch
from runtime.ir import OpType
from runtime.ops import resolve_graph
from runtime.passes import run_pipeline
from runtime.session import Session

from conftest import SDPATransformerBlock


D_MODEL = 64
N_HEADS = 4


@pytest.fixture
def transformer():
    model = SDPATransformerBlock(d_model=D_MODEL, n_heads=N_HEADS)
    model.eval()
    return model


@pytest.fixture
def dynamic_graph(transformer):
    """Export transformer with dynamic seq_len, optimize once."""
    example = torch.randn(1, 8, D_MODEL)
    graph = export_pytorch(
        transformer, (example,),
        dynamic_dims={'S': [('x', 1)]},
    )
    run_pipeline(graph)
    return graph


# ---------------------------------------------------------------------------
# resolve_graph unit tests
# ---------------------------------------------------------------------------

class TestResolveGraph:

    def test_preserves_original(self, dynamic_graph):
        """Original graph attrs keep symbol strings after resolution."""
        # Find a RESHAPE node — its shape attr should contain "S"
        reshape_nodes = [
            n for n in dynamic_graph.nodes.values()
            if n.op == OpType.RESHAPE
        ]
        assert reshape_nodes, "Expected RESHAPE nodes in transformer graph"

        # Capture original symbolic attrs
        original_attrs = {n.id: dict(n.attrs) for n in reshape_nodes}

        # Resolve to a concrete size
        resolved = resolve_graph(dynamic_graph, {'S': 32})

        # Original graph's RESHAPE attrs should be unchanged
        for node in reshape_nodes:
            assert node.attrs == original_attrs[node.id], (
                f"Node {node.id} attrs were mutated by resolve_graph"
            )

        # At least one RESHAPE should have a symbolic "S" in its shape attr
        has_symbol = any(
            isinstance(d, str) and d == 'S'
            for n in reshape_nodes
            for d in n.attrs.get('shape', ())
        )
        assert has_symbol, "Expected at least one RESHAPE with symbolic 'S' dim"

    def test_resolved_shapes_concrete(self, dynamic_graph):
        """Resolved graph has fully concrete shapes (no strings)."""
        resolved = resolve_graph(dynamic_graph, {'S': 16})

        for name, tensor in resolved.tensors.items():
            for d in tensor.shape:
                assert isinstance(d, int), (
                    f"Tensor {name} has non-int dim {d!r} in shape {tensor.shape}"
                )

    def test_shapes_match_fresh_export(self, transformer):
        """Resolved shapes match a fresh export at the same seq_len."""
        # Export with dynamic shapes, optimize
        example = torch.randn(1, 8, D_MODEL)
        dynamic = export_pytorch(
            transformer, (example,),
            dynamic_dims={'S': [('x', 1)]},
        )
        run_pipeline(dynamic)

        for seq_len in [4, 16, 32]:
            # Fresh export at this exact seq_len
            fresh_input = torch.randn(1, seq_len, D_MODEL)
            fresh = export_pytorch(transformer, (fresh_input,))
            run_pipeline(fresh)

            # Resolve dynamic graph to same seq_len
            resolved = resolve_graph(dynamic, {'S': seq_len})

            # Compare all tensor shapes
            for name in resolved.tensors:
                if name not in fresh.tensors:
                    continue
                assert resolved.tensors[name].shape == fresh.tensors[name].shape, (
                    f"Shape mismatch for '{name}' at S={seq_len}: "
                    f"resolved={resolved.tensors[name].shape} vs "
                    f"fresh={fresh.tensors[name].shape}"
                )

    def test_multiple_resolves_independent(self, dynamic_graph):
        """Multiple resolves from same graph produce independent copies."""
        r16 = resolve_graph(dynamic_graph, {'S': 16})
        r32 = resolve_graph(dynamic_graph, {'S': 32})

        # They should have different shapes for the output tensor
        out_name = r16.outputs[0]
        assert r16.tensors[out_name].shape != r32.tensors[out_name].shape

        # The seq_len dimension should match the binding
        assert r16.tensors[out_name].shape[1] == 16
        assert r32.tensors[out_name].shape[1] == 32

    def test_weight_buffers_shared(self, dynamic_graph):
        """Resolved copies share weight buffer data with the original."""
        resolved = resolve_graph(dynamic_graph, {'S': 16})

        for name in dynamic_graph.constants:
            orig_buf = dynamic_graph.tensors[name].buffer
            res_buf = resolved.tensors[name].buffer
            if orig_buf is not None:
                assert res_buf is orig_buf, (
                    f"Weight '{name}' buffer was copied instead of shared"
                )


# ---------------------------------------------------------------------------
# Session integration tests
# ---------------------------------------------------------------------------

class TestSessionDynamic:

    def test_create_with_bindings(self, transformer):
        """Session.create with bindings produces correct output."""
        seq_len = 16
        example = torch.randn(1, 8, D_MODEL)
        graph = export_pytorch(
            transformer, (example,),
            dynamic_dims={'S': [('x', 1)]},
        )

        session = Session(graph)
        session.create(bindings={'S': seq_len})

        x = torch.randn(1, seq_len, D_MODEL)
        with torch.no_grad():
            expected = transformer(x).numpy()

        result = session.run({graph.inputs[0]: x.numpy().copy()})
        output = result[graph.outputs[0]]

        np.testing.assert_allclose(output, expected, atol=1e-4)

    def test_rebind_correctness(self, transformer):
        """Rebinding to a new seq_len produces correct output."""
        example = torch.randn(1, 8, D_MODEL)
        graph = export_pytorch(
            transformer, (example,),
            dynamic_dims={'S': [('x', 1)]},
        )

        session = Session(graph)
        session.create(bindings={'S': 8})

        for seq_len in [4, 16, 32]:
            session.rebind({'S': seq_len})

            x = torch.randn(1, seq_len, D_MODEL)
            with torch.no_grad():
                expected = transformer(x).numpy()

            result = session.run({graph.inputs[0]: x.numpy().copy()})
            output = result[graph.outputs[0]]

            np.testing.assert_allclose(output, expected, atol=1e-4,
                                       err_msg=f"Failed at S={seq_len}")

    def test_rebind_matches_fresh_session(self, transformer):
        """Rebind gives same result as a fresh (non-dynamic) session."""
        seq_len = 24
        x = torch.randn(1, seq_len, D_MODEL)

        # Fresh session (no dynamic shapes)
        fresh_graph = export_pytorch(transformer, (x,))
        fresh_session = Session(fresh_graph)
        fresh_session.create()
        fresh_result = fresh_session.run(
            {fresh_graph.inputs[0]: x.numpy().copy()}
        )

        # Dynamic session with rebind
        example = torch.randn(1, 8, D_MODEL)
        dyn_graph = export_pytorch(
            transformer, (example,),
            dynamic_dims={'S': [('x', 1)]},
        )
        dyn_session = Session(dyn_graph)
        dyn_session.create(bindings={'S': 8})
        dyn_session.rebind({'S': seq_len})
        dyn_result = dyn_session.run(
            {dyn_graph.inputs[0]: x.numpy().copy()}
        )

        np.testing.assert_allclose(
            dyn_result[dyn_graph.outputs[0]],
            fresh_result[fresh_graph.outputs[0]],
            atol=1e-4,
        )
