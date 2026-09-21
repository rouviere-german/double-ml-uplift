"""d-separation must agree with brute-force path enumeration on every DAG we can generate."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from double_ml_uplift.causal.graph import (
    CausalGraph,
    CycleError,
    GraphError,
    UnknownNodeError,
)

pytestmark = pytest.mark.unit


def chain() -> CausalGraph:
    """``x -> m -> y`` with a separate common cause ``u -> x``, ``u -> y``."""
    return CausalGraph.from_edges(
        ["u", "x", "m", "y"], [("x", "m"), ("m", "y"), ("u", "x"), ("u", "y")]
    )


# --- structure ---------------------------------------------------------------------------


def test_from_edges_rejects_unknown_endpoints_self_loops_and_cycles() -> None:
    with pytest.raises(UnknownNodeError):
        CausalGraph.from_edges(["a"], [("a", "b")])
    with pytest.raises(CycleError, match="self-loop"):
        CausalGraph.from_edges(["a"], [("a", "a")])
    with pytest.raises(CycleError, match="cycle among"):
        CausalGraph.from_edges(["a", "b", "c"], [("a", "b"), ("b", "c"), ("c", "a")])
    with pytest.raises(GraphError, match="at least one node"):
        CausalGraph.from_edges([], [])


def test_duplicate_edges_are_collapsed_and_node_order_is_preserved() -> None:
    graph = CausalGraph.from_edges(["b", "a", "b"], [("a", "b"), ("a", "b")])
    assert graph.nodes == ("b", "a")
    assert graph.edges == (("a", "b"),)
    assert graph.parents("b") == frozenset({"a"})
    assert graph.children("a") == frozenset({"b"})


def test_ancestors_and_descendants_are_inclusive_by_default() -> None:
    graph = chain()
    assert graph.descendants(["x"]) == frozenset({"x", "m", "y"})
    assert graph.descendants(["x"], inclusive=False) == frozenset({"m", "y"})
    assert graph.ancestors(["y"]) == frozenset({"y", "m", "x", "u"})
    assert graph.ancestors(["y"], inclusive=False) == frozenset({"m", "x", "u"})


def test_without_outgoing_edges_keeps_nodes_and_incoming_edges() -> None:
    mutilated = chain().without_outgoing_edges("x")
    assert set(mutilated.nodes) == {"u", "x", "m", "y"}
    assert mutilated.children("x") == frozenset()
    assert mutilated.parents("x") == frozenset({"u"})


def test_queries_about_unknown_nodes_are_refused() -> None:
    graph = chain()
    with pytest.raises(UnknownNodeError, match="zzz"):
        graph.parents("zzz")
    with pytest.raises(UnknownNodeError):
        graph.is_d_separated(["x"], ["zzz"])
    with pytest.raises(UnknownNodeError):
        graph.without_outgoing_edges("zzz")


# --- d-separation semantics --------------------------------------------------------------


def test_chain_is_blocked_by_the_mediator_and_fork_by_the_common_cause() -> None:
    graph = CausalGraph.from_edges(["x", "m", "y"], [("x", "m"), ("m", "y")])
    assert not graph.is_d_separated(["x"], ["y"])
    assert graph.is_d_separated(["x"], ["y"], ["m"])

    fork = CausalGraph.from_edges(["u", "x", "y"], [("u", "x"), ("u", "y")])
    assert not fork.is_d_separated(["x"], ["y"])
    assert fork.is_d_separated(["x"], ["y"], ["u"])


def test_collider_is_closed_until_it_or_a_descendant_is_conditioned_on() -> None:
    graph = CausalGraph.from_edges(["x", "y", "c", "d"], [("x", "c"), ("y", "c"), ("c", "d")])
    assert graph.is_d_separated(["x"], ["y"])
    assert not graph.is_d_separated(["x"], ["y"], ["c"])
    assert not graph.is_d_separated(["x"], ["y"], ["d"])


def test_conditioning_on_a_queried_node_or_overlapping_sets_is_refused() -> None:
    graph = chain()
    with pytest.raises(GraphError, match="cannot condition on a queried node"):
        graph.is_d_separated(["x"], ["y"], ["x"])
    with pytest.raises(GraphError, match="node sets overlap"):
        graph.is_d_separated(["x", "y"], ["y"])


def test_backdoor_paths_only_count_trails_entering_the_treatment() -> None:
    graph = chain()
    paths = graph.backdoor_paths("x", "y")
    assert paths == [["x", "u", "y"]]
    # The directed path x -> m -> y is a causal path, not a backdoor path.
    assert ["x", "m", "y"] in list(graph.simple_paths("x", "y"))


def test_open_paths_lists_exactly_the_trails_left_open() -> None:
    graph = chain()
    assert graph.open_paths("x", "y", []) == [["x", "m", "y"], ["x", "u", "y"]]
    assert graph.open_paths("x", "y", ["u", "m"]) == []


def test_simple_paths_stops_at_the_exploration_limit() -> None:
    graph = CausalGraph.from_edges(
        ["a", "b", "c", "d"], [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
    )
    assert list(graph.simple_paths("a", "d", limit=1)) == []
    assert len(list(graph.simple_paths("a", "d"))) == 2


def test_is_collider_reads_the_position_in_the_trail() -> None:
    graph = CausalGraph.from_edges(["x", "y", "c"], [("x", "c"), ("y", "c")])
    assert graph.is_collider(["x", "c", "y"], 1)
    assert not chain().is_collider(["x", "m", "y"], 1)


# --- the property that matters -----------------------------------------------------------


@st.composite
def dags(draw: st.DrawFn) -> CausalGraph:
    """A random DAG. Edges only run from a lower to a higher index, so acyclicity is free."""
    n_nodes = draw(st.integers(min_value=3, max_value=7))
    names = [f"v{index}" for index in range(n_nodes)]
    candidates = [(names[i], names[j]) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    edges = draw(st.lists(st.sampled_from(candidates), max_size=len(candidates), unique=True))
    return CausalGraph.from_edges(names, edges)


@st.composite
def dag_queries(draw: st.DrawFn) -> tuple[CausalGraph, str, str, list[str]]:
    graph = draw(dags())
    source, target = draw(
        st.lists(st.sampled_from(graph.nodes), min_size=2, max_size=2, unique=True)
    )
    rest = [name for name in graph.nodes if name not in {source, target}]
    conditioned = draw(st.lists(st.sampled_from(rest), unique=True)) if rest else []
    return graph, source, target, conditioned


def brute_force_d_separated(
    graph: CausalGraph, source: str, target: str, conditioned: list[str]
) -> bool:
    """The textbook definition: separated iff every simple trail is blocked."""
    return all(
        graph.path_is_blocked(path, conditioned) for path in graph.simple_paths(source, target)
    )


@settings(max_examples=250, suppress_health_check=[HealthCheck.too_slow])
@given(query=dag_queries())
def test_d_separation_matches_brute_force_path_enumeration(
    query: tuple[CausalGraph, str, str, list[str]],
) -> None:
    graph, source, target, conditioned = query
    assert graph.is_d_separated([source], [target], conditioned) == brute_force_d_separated(
        graph, source, target, conditioned
    )


@given(query=dag_queries())
def test_d_separation_is_symmetric_in_its_two_node_sets(
    query: tuple[CausalGraph, str, str, list[str]],
) -> None:
    graph, source, target, conditioned = query
    assert graph.is_d_separated([source], [target], conditioned) == graph.is_d_separated(
        [target], [source], conditioned
    )


@given(query=dag_queries())
def test_adjacent_nodes_are_never_d_separated(
    query: tuple[CausalGraph, str, str, list[str]],
) -> None:
    graph, source, target, conditioned = query
    assume(target in graph.undirected_neighbours(source))
    assert not graph.is_d_separated([source], [target], conditioned)


@given(graph=dags())
def test_every_generated_graph_is_acyclic_and_topologically_consistent(
    graph: CausalGraph,
) -> None:
    for source, target in graph.edges:
        assert source in graph.ancestors([target], inclusive=False)
        assert target not in graph.ancestors([source])
