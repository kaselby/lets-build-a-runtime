"""Session API tests: the user-facing InferenceSession-style interface.

Tests the Session class which wraps export → optimize → plan → compile → run
behind a simple create/run API.
"""

import numpy as np
import pytest
import torch

from runtime.exporter import export_model
from runtime.session import Session

from conftest import SimpleMLP, NaiveTransformerBlock


class TestSessionBasic:

    def test_session_mlp_default(self):
        """Session with default executor (compiled) should match PyTorch."""
        model = SimpleMLP(64)
        model.eval()
        x = torch.randn(4, 64)

        with torch.no_grad():
            expected = model(x).numpy()

        graph = export_model(model, (x,))
        session = Session(graph)
        session.create()
        result = session.run({graph.inputs[0]: x.numpy().copy()})
        output = result[graph.outputs[0]]

        np.testing.assert_allclose(output, expected, atol=1e-4)

    def test_session_transformer_default(self):
        """Session with default executor (compiled) on a transformer."""
        model = NaiveTransformerBlock(64, 4)
        model.eval()
        x = torch.randn(2, 16, 64)

        with torch.no_grad():
            expected = model(x).numpy()

        graph = export_model(model, (x,))
        session = Session(graph)
        session.create()
        result = session.run({graph.inputs[0]: x.numpy().copy()})
        output = result[graph.outputs[0]]

        np.testing.assert_allclose(output, expected, atol=1e-4)

    def test_session_with_interpreted_executor(self):
        """Session should work with interpreted execution mode."""
        model = SimpleMLP(64)
        model.eval()
        x = torch.randn(4, 64)

        with torch.no_grad():
            expected = model(x).numpy()

        graph = export_model(model, (x,))
        session = Session(graph)
        session.create(execution_mode="interpreted", backend="numpy")
        result = session.run({graph.inputs[0]: x.numpy().copy()})
        output = result[graph.outputs[0]]

        np.testing.assert_allclose(output, expected, atol=1e-4)

    def test_session_reuse_across_inputs(self):
        """Session should produce correct results across multiple inputs."""
        model = SimpleMLP(64)
        model.eval()

        graph = export_model(model, (torch.randn(4, 64),))
        session = Session(graph)
        session.create()

        for _ in range(5):
            x = torch.randn(4, 64)
            with torch.no_grad():
                expected = model(x).numpy()
            result = session.run({graph.inputs[0]: x.numpy().copy()})
            np.testing.assert_allclose(result[graph.outputs[0]], expected, atol=1e-4)

    def test_session_run_before_create_raises(self):
        """Calling run() before create() should raise RuntimeError."""
        model = SimpleMLP(64)
        model.eval()
        graph = export_model(model, (torch.randn(4, 64),))
        session = Session(graph)

        with pytest.raises(RuntimeError, match="not created"):
            session.run({"x": np.zeros((4, 64), dtype=np.float32)})

    def test_session_custom_pipeline(self):
        """Session should accept a custom pass pipeline."""
        from runtime.passes import absorb_into_matmul, constant_fold, eliminate_dead_code

        model = SimpleMLP(64)
        model.eval()
        x = torch.randn(4, 64)

        with torch.no_grad():
            expected = model(x).numpy()

        graph = export_model(model, (x,))
        # Minimal pipeline: no fusion
        session = Session(graph)
        session.create(pipeline=[absorb_into_matmul, constant_fold, eliminate_dead_code])
        result = session.run({graph.inputs[0]: x.numpy().copy()})
        output = result[graph.outputs[0]]

        np.testing.assert_allclose(output, expected, atol=1e-4)
