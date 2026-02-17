"""Tests for the validation framework.

Verifies that validators catch real bugs at each pipeline phase.
Each test constructs a deliberately broken artifact and asserts
the appropriate validator catches it with the right severity.
"""

import numpy as np
import pytest
import torch

from runtime.ir import Graph, OpType
from runtime.ops import OP_REGISTRY
from runtime.planner import MemoryPlan, ExecutionPlan, plan
from runtime.validation import (
    Phase, Severity, ValidationResult, ValidationError,
    run_validators,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _errors(results: list[ValidationResult]) -> list[ValidationResult]:
    return [r for r in results if r.severity == Severity.ERROR]


def _warnings(results: list[ValidationResult]) -> list[ValidationResult]:
    return [r for r in results if r.severity == Severity.WARNING]


def _simple_graph() -> Graph:
    """A minimal valid graph: x -> RELU -> y -> RELU -> z."""
    g = Graph()
    g.add_tensor("x", (4, 64), "float32")
    g.add_tensor("y", (4, 64), "float32")
    g.add_tensor("z", (4, 64), "float32")
    g.inputs = ["x"]
    g.outputs = ["z"]
    g.add_node(OpType.RELU, ["x"], "y")
    g.add_node(OpType.RELU, ["y"], "z")
    return g


def _export_mlp():
    """Export a simple MLP for integration tests."""
    from runtime.exporter.torch import export_pytorch

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(64, 128)
            self.fc2 = torch.nn.Linear(128, 64)
        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model = MLP()
    x = torch.randn(4, 64)
    return export_pytorch(model, (x,))


# ===========================================================================
# POST_EXPORT / POST_OPTIMIZE — Structural integrity
# ===========================================================================

class TestStructuralIntegrity:
    """Validators that check graph topology and tensor registry consistency."""

    def test_valid_graph_passes(self):
        g = _simple_graph()
        results = run_validators(Phase.POST_EXPORT, g, fail_on=None)
        assert not _errors(results)

    def test_missing_input_tensor(self):
        """Node references an input tensor not in the registry."""
        g = _simple_graph()
        del g.tensors["x"]
        results = run_validators(Phase.POST_EXPORT, g, fail_on=None)
        errors = _errors(results)
        assert any("unknown input tensor 'x'" in r.message for r in errors)

    def test_missing_output_tensor(self):
        """Node output tensor not in the registry."""
        g = _simple_graph()
        del g.tensors["y"]
        results = run_validators(Phase.POST_EXPORT, g, fail_on=None)
        errors = _errors(results)
        assert any("not in registry" in r.message for r in errors)

    def test_input_has_producer(self):
        """Graph input tensor has a producer node — should be external."""
        g = Graph()
        g.add_tensor("x", (4,), "float32")
        g.add_tensor("y", (4,), "float32")
        g.add_tensor("z", (4,), "float32")
        g.inputs = ["x", "y"]  # y is listed as input...
        g.outputs = ["z"]
        g.add_node(OpType.RELU, ["x"], "y")  # ...but also produced by a node
        g.add_node(OpType.RELU, ["y"], "z")
        results = run_validators(Phase.POST_EXPORT, g, fail_on=None)
        errors = _errors(results)
        assert any("has a producer node" in r.message for r in errors)

    def test_constant_has_producer(self):
        """Graph constant has a producer node — should be external."""
        g = Graph()
        g.add_tensor("x", (4,), "float32")
        g.add_tensor("w", (4,), "float32")
        g.add_tensor("y", (4,), "float32")
        g.constants = ["w"]  # w is listed as constant...
        g.inputs = ["x"]
        g.outputs = ["y"]
        g.add_node(OpType.RELU, ["x"], "w")  # ...but also produced
        g.add_node(OpType.RELU, ["w"], "y")
        results = run_validators(Phase.POST_EXPORT, g, fail_on=None)
        errors = _errors(results)
        assert any("has a producer node" in r.message for r in errors)

    def test_output_no_producer(self):
        """Graph output has no producer node."""
        g = Graph()
        g.add_tensor("x", (4,), "float32")
        g.add_tensor("y", (4,), "float32")
        g.inputs = ["x"]
        g.outputs = ["y"]  # y is output but nobody produces it
        results = run_validators(Phase.POST_EXPORT, g, fail_on=None)
        errors = _errors(results)
        assert any("has no producer node" in r.message for r in errors)

    def test_consumed_tensor_no_source(self):
        """Non-external tensor consumed by a node has no producer."""
        g = Graph()
        g.add_tensor("x", (4,), "float32")
        g.add_tensor("ghost", (4,), "float32")
        g.add_tensor("y", (4,), "float32")
        g.inputs = ["x"]
        g.outputs = ["y"]
        g.add_node(OpType.ADD, ["x", "ghost"], "y")  # ghost has no producer
        results = run_validators(Phase.POST_EXPORT, g, fail_on=None)
        errors = _errors(results)
        assert any("has no producer" in r.message for r in errors)

    def test_dead_node_warning(self):
        """Node whose output is never consumed should produce a warning."""
        g = _simple_graph()
        # Add a dead branch
        g.add_tensor("dead", (4, 64), "float32")
        g.add_node(OpType.RELU, ["x"], "dead")
        results = run_validators(Phase.POST_EXPORT, g, fail_on=None)
        warnings = _warnings(results)
        assert any("Dead node" in r.message and "dead" in r.message for r in warnings)

    def test_role_tensor_missing(self):
        """Graph input/constant/output name not in tensor registry."""
        g = _simple_graph()
        g.inputs.append("nonexistent")
        results = run_validators(Phase.POST_EXPORT, g, fail_on=None)
        errors = _errors(results)
        assert any("nonexistent" in r.message for r in errors)

    def test_post_optimize_same_checks(self):
        """POST_OPTIMIZE runs the same structural checks."""
        g = _simple_graph()
        del g.tensors["x"]
        results = run_validators(Phase.POST_OPTIMIZE, g, fail_on=None)
        assert _errors(results)  # Should catch the same issue


# ===========================================================================
# POST_RESOLVE — Resolution completeness
# ===========================================================================

class TestResolutionCompleteness:
    """Validators that verify all symbolic dimensions are resolved."""

    def test_concrete_graph_passes(self):
        g = _simple_graph()
        results = run_validators(Phase.POST_RESOLVE, g, fail_on=None)
        assert not _errors(results)

    def test_symbol_in_shape(self):
        """Non-integer dimension in tensor shape after resolution."""
        g = Graph()
        g.add_tensor("x", (4, 64), "float32")
        g.add_tensor("y", (4, "L"), "float32")
        g.inputs = ["x"]
        g.outputs = ["y"]
        g.dynamic_dims = {"L": [("x", 1)]}
        g.add_node(OpType.RESHAPE, ["x"], "y", {"shape": (4, "L")})
        results = run_validators(Phase.POST_RESOLVE, g, fail_on=None)
        errors = _errors(results)
        assert any("non-integer dimension" in r.message for r in errors)

    def test_symbol_in_attrs(self):
        """Unresolved symbol string in node attrs after resolution."""
        g = Graph()
        g.add_tensor("x", (4, 64), "float32")
        g.add_tensor("y", (4, 64), "float32")
        g.inputs = ["x"]
        g.outputs = ["y"]
        g.dynamic_dims = {"L": [("x", 1)]}
        g.add_node(OpType.RESHAPE, ["x"], "y", {"shape": (4, "L")})
        results = run_validators(Phase.POST_RESOLVE, g, fail_on=None)
        errors = _errors(results)
        assert any("unresolved symbol" in r.message for r in errors)

    def test_symbol_in_nested_attr(self):
        """Symbol hidden inside a nested tuple in attrs."""
        g = Graph()
        g.add_tensor("x", (2, 3, 4), "float32")
        g.add_tensor("y", (2, 3, 4), "float32")
        g.inputs = ["x"]
        g.outputs = ["y"]
        g.dynamic_dims = {"B": [("x", 0)]}
        g.add_node(OpType.EXPAND, ["x"], "y", {"shape": ("B", 3, 4)})
        results = run_validators(Phase.POST_RESOLVE, g, fail_on=None)
        errors = _errors(results)
        assert any("unresolved symbol" in r.message for r in errors)


# ===========================================================================
# POST_RESOLVE_OPTIMIZE — Execution readiness
# ===========================================================================

class TestExecutionReadiness:
    """Validators that verify the graph is ready for planning and execution."""

    def test_clean_graph_passes(self):
        g = _simple_graph()
        results = run_validators(Phase.POST_RESOLVE_OPTIMIZE, g, fail_on=None)
        assert not _errors(results)

    def test_fold_only_op_survives(self):
        """Fold-only op (value >= 100) not eliminated by constant folding."""
        g = Graph()
        g.add_tensor("x", (4, 64), "float32")
        g.add_tensor("y", (4, 64), "float32")
        g.add_tensor("z", (4, 64), "float32")
        g.inputs = ["x"]
        g.outputs = ["z"]
        g.add_node(OpType.RELU, ["x"], "y")
        g.add_node(OpType.CAST, ["y"], "z", {"target_dtype": "float32"})
        results = run_validators(Phase.POST_RESOLVE_OPTIMIZE, g, fail_on=None)
        errors = _errors(results)
        assert any("Fold-only op CAST" in r.message for r in errors)

    def test_multiple_fold_only_ops(self):
        """Multiple surviving fold-only ops all reported."""
        g = Graph()
        g.add_tensor("x", (4,), "float32")
        g.add_tensor("a", (4,), "float32")
        g.add_tensor("b", (4,), "float32")
        g.add_tensor("c", (4,), "float32")
        g.inputs = ["x"]
        g.outputs = ["c"]
        g.add_node(OpType.CAST, ["x"], "a", {"target_dtype": "float32"})
        g.add_node(OpType.EXPAND, ["a"], "b", {"shape": (4,)})
        g.add_node(OpType.RELU, ["b"], "c")
        results = run_validators(Phase.POST_RESOLVE_OPTIMIZE, g, fail_on=None)
        errors = _errors(results)
        fold_errors = [r for r in errors if "Fold-only op" in r.message]
        assert len(fold_errors) == 2

    def test_alias_chain_integrity(self):
        """Alias ops must have exactly one input."""
        g = Graph()
        g.add_tensor("x", (4, 4), "float32")
        g.add_tensor("y", (16,), "float32")
        g.inputs = ["x"]
        g.outputs = ["y"]
        node = g.add_node(OpType.RESHAPE, ["x"], "y", {"shape": (16,)})
        # Break it: alias op with extra input
        node.inputs.append("x")
        g._consumers["x"].append(node.id)
        results = run_validators(Phase.POST_RESOLVE_OPTIMIZE, g, fail_on=None)
        errors = _errors(results)
        assert any("2 inputs, expected 1" in r.message for r in errors)

    def test_negative_slice_offset(self):
        """SLICE with negative byte_offset is invalid."""
        g = Graph()
        g.add_tensor("x", (8, 4), "float32")
        g.add_tensor("y", (4, 4), "float32")
        g.inputs = ["x"]
        g.outputs = ["y"]
        g.add_node(OpType.SLICE, ["x"], "y",
                   {"dim": 0, "start": 0, "end": 4, "byte_offset": -16})
        results = run_validators(Phase.POST_RESOLVE_OPTIMIZE, g, fail_on=None)
        errors = _errors(results)
        assert any("negative byte_offset" in r.message for r in errors)


# ===========================================================================
# POST_PLAN — Memory plan integrity
# ===========================================================================

class TestMemoryPlanIntegrity:
    """Validators that verify the memory plan's internal consistency."""

    def test_valid_plan_passes(self):
        """A correctly-planned graph should pass all checks."""
        g = _simple_graph()
        memory_plan = plan(g)
        results = run_validators(Phase.POST_PLAN, memory_plan, fail_on=None)
        assert not _errors(results)

    def test_missing_offset(self):
        """Intermediate tensor without an arena offset."""
        g = _simple_graph()
        n1, n2 = list(g.nodes.values())
        memory_plan = MemoryPlan(
            graph=g, order=[n1, n2],
            arena_size=2048,
            offsets={"y": 0},  # z is missing
        )
        results = run_validators(Phase.POST_PLAN, memory_plan, fail_on=None)
        errors = _errors(results)
        assert any("has no arena offset" in r.message for r in errors)

    def test_arena_undersized(self):
        """Arena too small for the allocations."""
        g = _simple_graph()
        n1, n2 = list(g.nodes.values())
        # 4*64*4 = 1024 bytes per tensor
        memory_plan = MemoryPlan(
            graph=g, order=[n1, n2],
            arena_size=512,  # way too small
            offsets={"y": 0, "z": 1024},
        )
        results = run_validators(Phase.POST_PLAN, memory_plan, fail_on=None)
        errors = _errors(results)
        assert any("Arena undersized" in r.message for r in errors)

    def test_arena_oversized_warning(self):
        """Arena larger than needed should produce a warning."""
        g = _simple_graph()
        n1, n2 = list(g.nodes.values())
        memory_plan = MemoryPlan(
            graph=g, order=[n1, n2],
            arena_size=99999,
            offsets={"y": 0, "z": 1024},
        )
        results = run_validators(Phase.POST_PLAN, memory_plan, fail_on=None)
        warnings = _warnings(results)
        assert any("Arena oversized" in r.message for r in warnings)

    def test_partial_overlap_detected(self):
        """Co-live tensors with partial memory overlap."""
        g = _simple_graph()
        n1, n2 = list(g.nodes.values())
        # y is alive when z is born (y is consumed by node producing z)
        # y: 1024 bytes at offset 0, z: 1024 bytes at offset 128 -> overlap!
        memory_plan = MemoryPlan(
            graph=g, order=[n1, n2],
            arena_size=2048,
            offsets={"y": 0, "z": 128},
        )
        results = run_validators(Phase.POST_PLAN, memory_plan, fail_on=None)
        errors = _errors(results)
        assert any("Partial overlap" in r.message for r in errors)

    def test_containment_allowed(self):
        """Alias containment (one inside the other) is intentional, not an error."""
        g = Graph()
        g.add_tensor("x", (4, 64), "float32")
        g.add_tensor("y", (4, 64), "float32")
        g.add_tensor("v", (256,), "float32")  # RESHAPE alias of y
        g.add_tensor("z", (4, 64), "float32")
        g.inputs = ["x"]
        g.outputs = ["z"]
        n1 = g.add_node(OpType.RELU, ["x"], "y")
        n2 = g.add_node(OpType.RESHAPE, ["y"], "v", {"shape": (256,)})
        n3 = g.add_node(OpType.RELU, ["v"], "z")
        # v shares y's memory (containment) — same offset, same size
        memory_plan = MemoryPlan(
            graph=g, order=[n1, n2, n3],
            arena_size=2048,
            offsets={"y": 0, "v": 0, "z": 1024},
        )
        results = run_validators(Phase.POST_PLAN, memory_plan, fail_on=None)
        errors = _errors(results)
        overlap_errors = [r for r in errors if "Partial overlap" in r.message]
        assert not overlap_errors

    def test_invalid_scratch_reference(self):
        """Scratch allocation referencing a non-existent node."""
        g = _simple_graph()
        n1, n2 = list(g.nodes.values())
        memory_plan = MemoryPlan(
            graph=g, order=[n1, n2],
            arena_size=2048,
            offsets={"y": 0, "z": 1024},
            scratch={999: (2000, 128)},  # node 999 doesn't exist
        )
        results = run_validators(Phase.POST_PLAN, memory_plan, fail_on=None)
        errors = _errors(results)
        assert any("not in the execution order" in r.message for r in errors)


# ===========================================================================
# PRE_EXECUTE — Dispatch coverage
# ===========================================================================

class TestDispatchCoverage:
    """Validators that verify every op can be dispatched by the target executor."""

    def test_compiled_valid_graph(self):
        """All standard ops should pass compiled dispatch validation."""
        g = _simple_graph()
        memory_plan = plan(g)
        exec_plan = ExecutionPlan(
            graph=g, memory=memory_plan,
            executor_type="compiled", backend="c",
        )
        results = run_validators(Phase.PRE_EXECUTE, exec_plan, fail_on=None)
        assert not _errors(results)

    def test_compiled_rejects_permute(self):
        """PERMUTE has no C dispatch entry — should error in compiled mode."""
        g = Graph()
        g.add_tensor("x", (2, 3, 4), "float32")
        g.add_tensor("y", (3, 4, 2), "float32")
        g.inputs = ["x"]
        g.outputs = ["y"]
        g.add_node(OpType.PERMUTE, ["x"], "y", {"axes": [1, 2, 0]})
        memory_plan = plan(g)
        exec_plan = ExecutionPlan(
            graph=g, memory=memory_plan,
            executor_type="compiled", backend="c",
        )
        results = run_validators(Phase.PRE_EXECUTE, exec_plan, fail_on=None)
        errors = _errors(results)
        assert any("PERMUTE" in r.message and "no C dispatch" in r.message for r in errors)

    def test_interpreted_numpy_handles_permute(self):
        """PERMUTE is supported by numpy backend — no error in interpreted mode."""
        g = Graph()
        g.add_tensor("x", (2, 3, 4), "float32")
        g.add_tensor("y", (3, 4, 2), "float32")
        g.inputs = ["x"]
        g.outputs = ["y"]
        g.add_node(OpType.PERMUTE, ["x"], "y", {"axes": [1, 2, 0]})
        memory_plan = plan(g)
        exec_plan = ExecutionPlan(
            graph=g, memory=memory_plan,
            executor_type="interpreted", backend="numpy",
        )
        results = run_validators(Phase.PRE_EXECUTE, exec_plan, fail_on=None)
        assert not _errors(results)

    def test_compiled_rejects_fold_only(self):
        """Fold-only ops should be caught even at PRE_EXECUTE (defense in depth)."""
        g = Graph()
        g.add_tensor("x", (4,), "float32")
        g.add_tensor("y", (4,), "float32")
        g.inputs = ["x"]
        g.outputs = ["y"]
        g.add_node(OpType.CAST, ["x"], "y", {"target_dtype": "float32"})
        memory_plan = plan(g)
        exec_plan = ExecutionPlan(
            graph=g, memory=memory_plan,
            executor_type="compiled", backend="c",
        )
        results = run_validators(Phase.PRE_EXECUTE, exec_plan, fail_on=None)
        errors = _errors(results)
        assert any("cannot be compiled" in r.message for r in errors)

    def test_alias_ops_skipped(self):
        """Alias ops (RESHAPE of non-external) shouldn't trigger dispatch checks."""
        g = Graph()
        g.add_tensor("x", (4, 4), "float32")
        g.add_tensor("y", (4, 4), "float32")
        g.add_tensor("v", (16,), "float32")
        g.add_tensor("z", (16,), "float32")
        g.inputs = ["x"]
        g.outputs = ["z"]
        g.add_node(OpType.RELU, ["x"], "y")
        g.add_node(OpType.RESHAPE, ["y"], "v", {"shape": (16,)})
        g.add_node(OpType.RELU, ["v"], "z")
        memory_plan = plan(g)
        exec_plan = ExecutionPlan(
            graph=g, memory=memory_plan,
            executor_type="compiled", backend="c",
        )
        results = run_validators(Phase.PRE_EXECUTE, exec_plan, fail_on=None)
        assert not _errors(results)

    def test_skips_wrong_executor_type(self):
        """Compiled validator doesn't fire for interpreted plans and vice versa."""
        g = _simple_graph()
        memory_plan = plan(g)

        exec_plan = ExecutionPlan(
            graph=g, memory=memory_plan,
            executor_type="interpreted", backend="c+numpy",
        )
        results = run_validators(Phase.PRE_EXECUTE, exec_plan, fail_on=None)
        compiled_results = [r for r in results if r.validator == "compiled_dispatch"]
        assert not compiled_results


# ===========================================================================
# Integration — Session with validation
# ===========================================================================

class TestSessionValidation:
    """End-to-end validation through the Session API."""

    def test_session_with_validation(self):
        """Session.create() with normal validation succeeds on a valid graph."""
        from runtime.session import Session
        graph = _export_mlp()
        session = Session(graph)
        session.create(validation="normal")
        result = session.run({"x": np.random.randn(4, 64).astype(np.float32)})
        assert list(result.values())[0].shape == (4, 64)

    def test_session_strict_mode(self):
        """Session.create() with strict validation (fails on warnings too)."""
        from runtime.session import Session
        graph = _export_mlp()
        session = Session(graph)
        # Strict mode should still pass on a well-formed graph
        session.create(validation="strict")

    def test_session_validation_none(self):
        """Session.create() with validation='none' skips all checks."""
        from runtime.session import Session
        graph = _export_mlp()
        session = Session(graph)
        session.create(validation="none")

    def test_validation_error_raised(self):
        """run_validators raises ValidationError on fatal findings."""
        g = Graph()
        g.add_tensor("x", (4,), "float32")
        g.add_tensor("y", (4,), "float32")
        g.inputs = ["x"]
        g.outputs = ["y"]
        g.add_node(OpType.CAST, ["x"], "y", {"target_dtype": "float32"})

        with pytest.raises(ValidationError) as exc_info:
            run_validators(Phase.POST_RESOLVE_OPTIMIZE, g, fail_on=Severity.ERROR)

        assert exc_info.value.phase == Phase.POST_RESOLVE_OPTIMIZE
        assert len(exc_info.value.results) >= 1


# ===========================================================================
# Edge cases
# ===========================================================================

class TestValidationEdgeCases:
    """Edge cases and boundary conditions for validators."""

    def test_empty_graph(self):
        """Empty graph (no nodes, no inputs/outputs) doesn't crash validators."""
        g = Graph()
        results = run_validators(Phase.POST_EXPORT, g, fail_on=None)
        # No errors — empty graph is structurally valid (if useless)
        assert not _errors(results)

    def test_real_pipeline_all_phases(self):
        """Run a real graph through all graph-level phases."""
        graph = _export_mlp()

        # POST_EXPORT
        results = run_validators(Phase.POST_EXPORT, graph, fail_on=None)
        assert not _errors(results)

        # Optimize
        from runtime.passes.passes import run_pipeline
        run_pipeline(graph)

        # POST_OPTIMIZE
        results = run_validators(Phase.POST_OPTIMIZE, graph, fail_on=None)
        assert not _errors(results)

        # POST_RESOLVE (no dynamic shapes, graph passes through)
        results = run_validators(Phase.POST_RESOLVE, graph, fail_on=None)
        assert not _errors(results)

        # POST_RESOLVE_OPTIMIZE
        results = run_validators(Phase.POST_RESOLVE_OPTIMIZE, graph, fail_on=None)
        assert not _errors(results)

        # POST_PLAN
        memory_plan = plan(graph)
        results = run_validators(Phase.POST_PLAN, memory_plan, fail_on=None)
        assert not _errors(results)

        # PRE_EXECUTE
        exec_plan = ExecutionPlan(
            graph=graph, memory=memory_plan,
            executor_type="compiled", backend="c",
        )
        results = run_validators(Phase.PRE_EXECUTE, exec_plan, fail_on=None)
        assert not _errors(results)
