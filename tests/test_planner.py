"""Memory planner tests for the refactored planner.

Tests cover:
- Lifetime computation (intermediates vs externals, born/dies correctness)
- Unified alias + in-place memory sharing
- Chained sharing (alias→in-place, in-place→in-place)
- Memory-aware topological ordering
- Arena no-overlap invariant
- Arena reuse (smaller than naive allocation)
- Scratch buffer allocation
- First-fit offset assignment
"""

import numpy as np
import pytest

from runtime.ir import Graph, OpType
from runtime.ops import OP_REGISTRY
from runtime.planner import (
    ExecutionPlan,
    FitStrategy,
    Lifetime,
    OrderStrategy,
    PlannerConfig,
    _assign_offsets,
    _compute_lifetimes,
    _compute_scratch,
    _memory_aware_order,
    _resolve_alias,
    _tensor_size,
    plan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_input(g: Graph, name: str, shape: tuple[int, ...]) -> None:
    g.add_tensor(name, shape)
    g.inputs.append(name)


def _add_constant(g: Graph, name: str, shape: tuple[int, ...]) -> None:
    t = g.add_tensor(name, shape)
    t.buffer = np.zeros(shape, dtype=np.float32)
    g.constants.append(name)


def _check_no_overlap(ep: ExecutionPlan) -> None:
    """Verify that no two simultaneously-live tensors share arena memory.

    For every pair of arena-owning tensors with overlapping lifetimes,
    their [offset, offset+size) ranges must not intersect. This is the
    fundamental correctness invariant of the memory planner.
    """
    lifetimes, memory_root = _compute_lifetimes(ep.graph, ep.order)

    allocs = []
    for name, lt in lifetimes.items():
        if name not in ep.offsets:
            continue
        size = _tensor_size(ep.graph, name)
        allocs.append((ep.offsets[name], size, lt))

    for i, (off_a, sz_a, lt_a) in enumerate(allocs):
        for j, (off_b, sz_b, lt_b) in enumerate(allocs):
            if i >= j:
                continue
            # Temporal overlap?
            if lt_a.born <= lt_b.dies and lt_b.born <= lt_a.dies:
                # Spatial overlap?
                overlaps = off_a < off_b + sz_b and off_b < off_a + sz_a
                assert not overlaps, (
                    f"Arena overlap: {lt_a.tensor_name}[{off_a}:{off_a + sz_a}) "
                    f"and {lt_b.tensor_name}[{off_b}:{off_b + sz_b}) "
                    f"both live during steps {max(lt_a.born, lt_b.born)}-"
                    f"{min(lt_a.dies, lt_b.dies)}"
                )


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def _build_linear_chain() -> Graph:
    """x -> ADD(x, bias) -> EXP -> RELU -> out

    Three elementwise ops in a row, all same shape. Tests in-place chaining.
    """
    g = Graph()
    _add_input(g, "x", (2, 8))
    _add_constant(g, "bias", (8,))

    g.add_tensor("a", (2, 8))
    g.add_node(OpType.ADD, ["x", "bias"], "a")

    g.add_tensor("e", (2, 8))
    g.add_node(OpType.EXP, ["a"], "e")

    g.add_tensor("r", (2, 8))
    g.add_node(OpType.RELU, ["e"], "r")
    g.outputs.append("r")
    return g


def _build_branch() -> Graph:
    """x -> ADD -> a, then a consumed by both EXP and RELU (both are outputs).

    The first consumer of 'a' cannot do in-place (a is still alive for the
    second consumer). The second consumer can.
    """
    g = Graph()
    _add_input(g, "x", (2, 8))
    _add_constant(g, "bias", (8,))

    g.add_tensor("a", (2, 8))
    g.add_node(OpType.ADD, ["x", "bias"], "a")

    g.add_tensor("e", (2, 8))
    g.add_node(OpType.EXP, ["a"], "e")

    g.add_tensor("r", (2, 8))
    g.add_node(OpType.RELU, ["a"], "r")

    g.outputs.extend(["e", "r"])
    return g


def _build_reshape_then_inplace() -> Graph:
    """x -> ADD -> a -> RESHAPE -> v -> TANH -> t (output).

    Tests that alias (RESHAPE) and in-place (TANH) chain together:
    v aliases a, then t shares a's memory via in-place on v.
    """
    g = Graph()
    _add_input(g, "x", (2, 8))
    _add_constant(g, "bias", (8,))

    g.add_tensor("a", (2, 8))
    g.add_node(OpType.ADD, ["x", "bias"], "a")

    g.add_tensor("v", (4, 4))
    g.add_node(OpType.RESHAPE, ["a"], "v", {"shape": (4, 4)})

    g.add_tensor("t", (4, 4))
    g.add_node(OpType.TANH, ["v"], "t")
    g.outputs.append("t")
    return g


def _build_size_mismatch() -> Graph:
    """x -> MATMUL(x, w) -> y -> RELU -> r (output).

    MATMUL output y is (4, 16), RELU output r is (4, 16) — same size, so
    in-place works. But we also build a variant where sizes differ.
    """
    g = Graph()
    _add_input(g, "x", (4, 8))
    _add_constant(g, "w", (8, 16))

    g.add_tensor("y", (4, 16))
    g.add_node(OpType.MATMUL, ["x", "w"], "y")

    g.add_tensor("r", (4, 16))
    g.add_node(OpType.RELU, ["y"], "r")
    g.outputs.append("r")
    return g


def _build_reduction_no_inplace() -> Graph:
    """x -> EXP -> e -> SUM(axis=1) -> s (output).

    EXP(2,8) -> SUM -> (2,1). Different sizes, so in-place is not possible
    even though the input is dying.
    """
    g = Graph()
    _add_input(g, "x", (2, 8))

    g.add_tensor("e", (2, 8))
    g.add_node(OpType.EXP, ["x"], "e")

    g.add_tensor("s", (2, 1))
    g.add_node(OpType.SUM, ["e"], "s", {"axis": 1, "keepdim": True})
    g.outputs.append("s")
    return g


def _build_two_independent_chains() -> Graph:
    """Two independent chains of different sizes.

    Chain A (big):   x1 -> ADD(x1, b1) -> big -> RELU -> big_done
    Chain B (small): x2 -> ADD(x2, b2) -> small -> RELU -> small_done

    Memory-aware ordering should complete the big chain first (freeing
    more memory sooner).
    """
    g = Graph()
    _add_input(g, "x1", (1, 1024))
    _add_input(g, "x2", (1, 8))
    _add_constant(g, "b1", (1024,))
    _add_constant(g, "b2", (8,))

    g.add_tensor("big", (1, 1024))
    g.add_node(OpType.ADD, ["x1", "b1"], "big")

    g.add_tensor("small", (1, 8))
    g.add_node(OpType.ADD, ["x2", "b2"], "small")

    g.add_tensor("big_done", (1, 1024))
    g.add_node(OpType.RELU, ["big"], "big_done")

    g.add_tensor("small_done", (1, 8))
    g.add_node(OpType.RELU, ["small"], "small_done")

    g.outputs.extend(["big_done", "small_done"])
    return g


def _build_with_attention() -> Graph:
    """Q, K, V -> ATTENTION -> out. Tests scratch allocation."""
    g = Graph()
    _add_input(g, "Q", (2, 4, 16, 32))  # [B, H, S, D]
    _add_input(g, "K", (2, 4, 16, 32))
    _add_input(g, "V", (2, 4, 16, 32))

    g.add_tensor("attn", (2, 4, 16, 32))
    g.add_node(OpType.ATTENTION, ["Q", "K", "V"], "attn")
    g.outputs.append("attn")
    return g


# ---------------------------------------------------------------------------
# Lifetime computation tests
# ---------------------------------------------------------------------------

class TestLifetimes:

    def test_externals_excluded(self):
        """Graph inputs and constants should not appear in lifetimes."""
        g = _build_linear_chain()
        order = _memory_aware_order(g)
        lifetimes, _ = _compute_lifetimes(g, order)

        assert "x" not in lifetimes
        assert "bias" not in lifetimes

    def test_intermediates_have_lifetimes(self):
        """Non-shared intermediate tensors should appear in lifetimes."""
        g = _build_linear_chain()
        order = _memory_aware_order(g)
        lifetimes, memory_root = _compute_lifetimes(g, order)

        # In a chain where everything is in-place, only the first
        # intermediate ("a") should own arena memory
        assert "a" in lifetimes
        # e and r should be shared (in memory_root), not in lifetimes
        assert "e" not in lifetimes
        assert "r" not in lifetimes

    def test_lifetime_born_before_dies(self):
        """Every lifetime's born step should be <= its dies step."""
        g = _build_branch()
        order = _memory_aware_order(g)
        lifetimes, _ = _compute_lifetimes(g, order)

        for name, lt in lifetimes.items():
            assert lt.born <= lt.dies, (
                f"{name}: born={lt.born} > dies={lt.dies}"
            )

    def test_lifetime_size_correct(self):
        """Lifetime size_bytes should match tensor shape * dtype."""
        g = _build_linear_chain()
        order = _memory_aware_order(g)
        lifetimes, _ = _compute_lifetimes(g, order)

        for name, lt in lifetimes.items():
            expected = _tensor_size(g, name)
            assert lt.size_bytes == expected, (
                f"{name}: expected {expected} bytes, got {lt.size_bytes}"
            )


# ---------------------------------------------------------------------------
# Alias sharing tests
# ---------------------------------------------------------------------------

class TestAliasSharing:

    def test_reshape_is_alias(self):
        """RESHAPE output should map to its input in memory_root."""
        g = _build_reshape_then_inplace()
        order = _memory_aware_order(g)
        _, memory_root = _compute_lifetimes(g, order)

        assert "v" in memory_root
        assert memory_root["v"] == "a"

    def test_reshape_not_in_lifetimes(self):
        """RESHAPE output should not have its own lifetime."""
        g = _build_reshape_then_inplace()
        order = _memory_aware_order(g)
        lifetimes, _ = _compute_lifetimes(g, order)

        assert "v" not in lifetimes

    def test_reshape_extends_root_lifetime(self):
        """Consuming a RESHAPE output should extend the root's lifetime."""
        g = Graph()
        _add_input(g, "x", (4, 8))

        g.add_tensor("e", (4, 8))
        g.add_node(OpType.EXP, ["x"], "e")

        g.add_tensor("v", (2, 16))
        g.add_node(OpType.RESHAPE, ["e"], "v", {"shape": (2, 16)})

        # Matmul consumes v (alias of e) at a later step
        _add_constant(g, "w", (16, 4))
        g.add_tensor("y", (2, 4))
        g.add_node(OpType.MATMUL, ["v", "w"], "y")
        g.outputs.append("y")

        order = _memory_aware_order(g)
        lifetimes, memory_root = _compute_lifetimes(g, order)

        # v is alias of e, so consuming v extends e's lifetime
        assert "v" in memory_root
        step_of = {node.output: i for i, node in enumerate(order)}
        assert lifetimes["e"].dies == step_of["y"]

    def test_resolve_alias_follows_chain(self):
        """_resolve_alias should follow RESHAPE chains to the root."""
        g = Graph()
        _add_input(g, "x", (4, 8))

        g.add_tensor("e", (4, 8))
        g.add_node(OpType.EXP, ["x"], "e")

        g.add_tensor("v1", (2, 16))
        g.add_node(OpType.RESHAPE, ["e"], "v1", {"shape": (2, 16)})

        g.add_tensor("v2", (32,))
        g.add_node(OpType.RESHAPE, ["v1"], "v2", {"shape": (32,)})
        g.outputs.append("v2")

        assert _resolve_alias("v2", g) == "e"
        assert _resolve_alias("v1", g) == "e"
        assert _resolve_alias("e", g) == "e"


# ---------------------------------------------------------------------------
# In-place sharing tests
# ---------------------------------------------------------------------------

class TestInplaceSharing:

    def test_inplace_chain(self):
        """ADD -> EXP -> RELU: all three should share one buffer."""
        g = _build_linear_chain()
        ep = plan(g)

        assert ep.offsets["a"] == ep.offsets["e"] == ep.offsets["r"]

    def test_inplace_chain_minimal_arena(self):
        """Chain of in-place ops should need only one buffer's worth of arena."""
        g = _build_linear_chain()
        ep = plan(g)

        one_buffer = 2 * 8 * 4  # (2, 8) float32
        assert ep.arena_size == one_buffer

    def test_inplace_blocked_by_live_consumer(self):
        """First consumer can't do in-place when input has other consumers."""
        g = _build_branch()
        order = _memory_aware_order(g)
        _, memory_root = _compute_lifetimes(g, order)

        # Find which consumer runs first
        step_of = {node.output: i for i, node in enumerate(order)}

        # The first consumer of "a" should NOT be in memory_root (can't in-place)
        first = "e" if step_of["e"] < step_of["r"] else "r"
        second = "r" if first == "e" else "e"

        assert first not in memory_root, (
            f"First consumer '{first}' should not be in-place (input still alive)"
        )
        # The second consumer CAN in-place (a is now dying)
        assert memory_root[second] == "a"

    def test_inplace_requires_same_size(self):
        """In-place should not happen when input and output differ in size."""
        g = _build_reduction_no_inplace()
        order = _memory_aware_order(g)
        _, memory_root = _compute_lifetimes(g, order)

        # SUM reduces (2,8) to (2,1) — can't in-place despite input dying
        assert "s" not in memory_root

    def test_inplace_not_on_external(self):
        """In-place should not reuse a graph input's buffer."""
        g = Graph()
        _add_input(g, "x", (2, 8))

        g.add_tensor("r", (2, 8))
        g.add_node(OpType.RELU, ["x"], "r")
        g.outputs.append("r")

        order = _memory_aware_order(g)
        _, memory_root = _compute_lifetimes(g, order)

        assert "r" not in memory_root


# ---------------------------------------------------------------------------
# Mixed alias + in-place tests
# ---------------------------------------------------------------------------

class TestMixedSharing:

    def test_alias_then_inplace(self):
        """ADD -> RESHAPE -> TANH: all share one root."""
        g = _build_reshape_then_inplace()
        order = _memory_aware_order(g)
        _, memory_root = _compute_lifetimes(g, order)

        assert memory_root["v"] == "a"  # alias
        assert memory_root["t"] == "a"  # in-place via alias chain

    def test_alias_then_inplace_same_offset(self):
        """ADD -> RESHAPE -> TANH should all have the same arena offset."""
        g = _build_reshape_then_inplace()
        ep = plan(g)

        assert ep.offsets["a"] == ep.offsets["v"] == ep.offsets["t"]


# ---------------------------------------------------------------------------
# Memory-aware ordering tests
# ---------------------------------------------------------------------------

class TestMemoryAwareOrder:

    def test_prefers_freeing_large_tensors(self):
        """Scheduler should finish the big chain before starting the small one."""
        g = _build_two_independent_chains()
        order = _memory_aware_order(g)
        step_of = {node.output: i for i, node in enumerate(order)}

        # After "big" is produced, RELU(big) frees 4096 bytes.
        # That should be preferred over starting the small chain.
        assert step_of["big_done"] < step_of["small"]

    def test_reduces_peak_vs_naive(self):
        """Memory-aware ordering should yield smaller peak arena than naive."""
        g = _build_two_independent_chains()
        g.validate()

        naive_order = list(g)
        naive_lifetimes, _ = _compute_lifetimes(g, naive_order)
        _, naive_arena = _assign_offsets(naive_lifetimes, n_steps=len(naive_order))

        smart_order = _memory_aware_order(g)
        smart_lifetimes, _ = _compute_lifetimes(g, smart_order)
        _, smart_arena = _assign_offsets(smart_lifetimes, n_steps=len(smart_order))

        assert smart_arena <= naive_arena

    def test_valid_topological_order(self):
        """Memory-aware order must still be a valid topological sort."""
        g = _build_linear_chain()
        order = _memory_aware_order(g)
        step_of = {node.output: i for i, node in enumerate(order)}

        # Every node's inputs must be produced before the node runs
        for node in order:
            for inp in node.inputs:
                producer = g.producer(inp)
                if producer is not None:
                    assert step_of[producer.output] < step_of[node.output], (
                        f"Node producing '{node.output}' at step {step_of[node.output]} "
                        f"depends on '{inp}' produced at step {step_of[producer.output]}"
                    )

    def test_all_nodes_included(self):
        """Memory-aware order must include all nodes in the graph."""
        g = _build_two_independent_chains()
        order = _memory_aware_order(g)

        assert len(order) == len(g.nodes)
        order_ids = {n.id for n in order}
        graph_ids = set(g.nodes.keys())
        assert order_ids == graph_ids


# ---------------------------------------------------------------------------
# Arena invariants
# ---------------------------------------------------------------------------

class TestArenaInvariants:

    def test_no_overlap_linear_chain(self):
        g = _build_linear_chain()
        _check_no_overlap(plan(g))

    def test_no_overlap_branch(self):
        g = _build_branch()
        _check_no_overlap(plan(g))

    def test_no_overlap_reshape_chain(self):
        g = _build_reshape_then_inplace()
        _check_no_overlap(plan(g))

    def test_no_overlap_two_chains(self):
        g = _build_two_independent_chains()
        _check_no_overlap(plan(g))

    def test_arena_smaller_than_naive(self):
        """Arena should be <= sum of all intermediate tensor sizes."""
        g = _build_two_independent_chains()
        ep = plan(g)

        external = set(g.inputs) | set(g.constants)
        naive = sum(
            _tensor_size(g, name)
            for name in ep.offsets
            if name not in external
        )
        assert ep.arena_size <= naive


# ---------------------------------------------------------------------------
# Scratch buffer tests
# ---------------------------------------------------------------------------

class TestScratch:

    def test_attention_gets_scratch(self):
        """ATTENTION nodes should have scratch buffer entries in the plan."""
        g = _build_with_attention()
        ep = plan(g)

        attn_node = [n for n in ep.order if n.op == OpType.ATTENTION][0]
        assert attn_node.id in ep.scratch

    def test_scratch_size_correct(self):
        """ATTENTION scratch = batch_heads * seq_len^2 * 4 bytes."""
        g = _build_with_attention()
        ep = plan(g)

        attn_node = [n for n in ep.order if n.op == OpType.ATTENTION][0]
        offset, size = ep.scratch[attn_node.id]

        # [2, 4, 16, 32] -> batch_heads=8, seq_len=16
        expected = 8 * 16 * 16 * 4  # 8 slices * S*S * float32
        assert size == expected

    def test_scratch_not_in_offsets(self):
        """Scratch entries should not appear in the regular offsets dict."""
        g = _build_with_attention()
        ep = plan(g)

        for name in ep.offsets:
            assert not name.startswith("__scratch_")

    def test_scratch_fits_in_arena(self):
        """Scratch offset + size should fit within the arena."""
        g = _build_with_attention()
        ep = plan(g)

        for node_id, (offset, size) in ep.scratch.items():
            assert offset + size <= ep.arena_size

    def test_no_scratch_when_not_needed(self):
        """Ops without scratch calculators should not have scratch entries."""
        g = _build_linear_chain()
        ep = plan(g)

        assert len(ep.scratch) == 0


# ---------------------------------------------------------------------------
# Offset assignment tests
# ---------------------------------------------------------------------------

class TestOffsetAssignment:

    def test_all_intermediates_have_offsets(self):
        """Every intermediate (including shared) should have an offset."""
        g = _build_reshape_then_inplace()
        ep = plan(g)

        external = set(g.inputs) | set(g.constants)
        for node in ep.order:
            if node.output not in external:
                assert node.output in ep.offsets, (
                    f"Missing offset for '{node.output}'"
                )

    def test_shared_tensors_same_offset(self):
        """Tensors sharing memory should have the same arena offset."""
        g = _build_linear_chain()
        ep = plan(g)

        assert ep.offsets["a"] == ep.offsets["e"]
        assert ep.offsets["e"] == ep.offsets["r"]

    def test_offsets_non_negative(self):
        """All offsets should be >= 0."""
        g = _build_two_independent_chains()
        ep = plan(g)

        for name, offset in ep.offsets.items():
            assert offset >= 0, f"Negative offset for '{name}': {offset}"


# ---------------------------------------------------------------------------
# Full plan() integration
# ---------------------------------------------------------------------------

class TestPlan:

    def test_invalid_graph_raises(self):
        """plan() should raise on a graph with structural errors."""
        g = Graph()
        g.add_tensor("x", (4, 8))
        g.inputs.append("x")
        g.add_tensor("y", (4, 8))
        g.add_node(OpType.RELU, ["x"], "y")
        # Declare an output that has no producer
        g.add_tensor("ghost", (4, 8))
        g.outputs.append("ghost")
        with pytest.raises(ValueError, match="Cannot plan invalid graph"):
            plan(g)

    def test_plan_returns_execution_plan(self):
        g = _build_linear_chain()
        ep = plan(g)

        assert isinstance(ep, ExecutionPlan)
        assert ep.arena_size > 0
        assert len(ep.order) == len(g.nodes)
        assert ep.graph is g

    def test_allocate_arena(self):
        """allocate_arena() should return a buffer of the right size."""
        g = _build_linear_chain()
        ep = plan(g)

        arena = ep.allocate_arena()
        assert arena.dtype == np.uint8
        assert len(arena) == ep.arena_size


# ---------------------------------------------------------------------------
# PlannerConfig tests
# ---------------------------------------------------------------------------

class TestPlannerConfig:

    def test_default_config_unchanged(self):
        """plan(g) and plan(g, PlannerConfig()) should produce identical results."""
        g = _build_linear_chain()

        ep_default = plan(g)
        ep_explicit = plan(g, PlannerConfig())

        assert ep_default.arena_size == ep_explicit.arena_size
        assert ep_default.offsets == ep_explicit.offsets

    def test_no_inplace_increases_arena(self):
        """enable_inplace=False should use >= default arena size for in-place chains."""
        g = _build_linear_chain()

        ep_with_inplace = plan(g)
        ep_no_inplace = plan(g, PlannerConfig(enable_inplace=False))

        # Without in-place, three separate buffers are needed instead of one
        assert ep_no_inplace.arena_size >= ep_with_inplace.arena_size
        # For this specific chain, they should be different
        assert ep_no_inplace.arena_size > ep_with_inplace.arena_size

    def test_no_aliases_increases_arena(self):
        """enable_aliases=False should use >= default arena size for graphs with reshapes."""
        g = _build_reshape_then_inplace()

        ep_with_aliases = plan(g)
        ep_no_aliases = plan(g, PlannerConfig(enable_aliases=False))

        # Without aliases, RESHAPE output gets its own allocation
        assert ep_no_aliases.arena_size >= ep_with_aliases.arena_size

    def test_all_strategies_no_overlap(self):
        """All OrderStrategy × FitStrategy combinations should produce valid allocations."""
        g = _build_two_independent_chains()

        for order_strategy in [
            OrderStrategy.NAIVE,
            OrderStrategy.MEMORY_AWARE_V1,
            OrderStrategy.MEMORY_AWARE_V2,
            OrderStrategy.MEMORY_AWARE_V3,
        ]:
            for fit_strategy in [FitStrategy.FIRST_FIT, FitStrategy.BEST_FIT]:
                config = PlannerConfig(order=order_strategy, fit=fit_strategy)
                ep = plan(g, config)
                _check_no_overlap(ep)

    def test_naive_order_valid(self):
        """OrderStrategy.NAIVE should produce a valid topological execution order."""
        g = _build_linear_chain()
        config = PlannerConfig(order=OrderStrategy.NAIVE)
        ep = plan(g, config)

        # Verify it's a valid topological order
        step_of = {node.output: i for i, node in enumerate(ep.order)}
        for node in ep.order:
            for inp in node.inputs:
                producer = g.producer(inp)
                if producer is not None:
                    assert step_of[producer.output] < step_of[node.output]

    def test_best_fit_valid(self):
        """FitStrategy.BEST_FIT should produce valid non-overlapping allocations."""
        g = _build_two_independent_chains()
        config = PlannerConfig(fit=FitStrategy.BEST_FIT)
        ep = plan(g, config)

        _check_no_overlap(ep)

    def test_best_fit_tighter_packing(self):
        """FitStrategy.BEST_FIT should produce <= arena size compared to FIRST_FIT."""
        g = _build_two_independent_chains()

        ep_first = plan(g, PlannerConfig(fit=FitStrategy.FIRST_FIT))
        ep_best = plan(g, PlannerConfig(fit=FitStrategy.BEST_FIT))

        # BEST_FIT should never use more arena than FIRST_FIT
        assert ep_best.arena_size <= ep_first.arena_size

    def test_memory_aware_v1_v2_valid(self):
        """MEMORY_AWARE_V1 and MEMORY_AWARE_V2 should both produce valid orders."""
        g = _build_two_independent_chains()

        for strategy in [OrderStrategy.MEMORY_AWARE_V1, OrderStrategy.MEMORY_AWARE_V2]:
            config = PlannerConfig(order=strategy)
            ep = plan(g, config)
            _check_no_overlap(ep)

    def test_config_immutable(self):
        """PlannerConfig should be frozen (immutable)."""
        config = PlannerConfig()
        with pytest.raises(AttributeError):
            config.enable_inplace = False

    def test_config_defaults_match_current_behavior(self):
        """All config defaults should match the implicit behavior before config existed."""
        config = PlannerConfig()
        assert config.order == OrderStrategy.MEMORY_AWARE_V1
        assert config.fit == FitStrategy.FIRST_FIT
        assert config.enable_inplace is True
        assert config.enable_aliases is True

    def test_disabling_both_shared_memory_features(self):
        """Disabling both in-place and aliases should still produce valid allocations."""
        g = _build_reshape_then_inplace()
        config = PlannerConfig(enable_inplace=False, enable_aliases=False)
        ep = plan(g, config)

        _check_no_overlap(ep)
