"""Backdoor identification: sound, locally minimal, complete, and closed against leakage."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from double_ml_uplift.causal.graph import CausalGraph
from double_ml_uplift.causal.identification import (
    LeakageError,
    NotIdentifiedError,
    format_path,
    identify_backdoor,
    leakage_guard,
    verify_adjustment_set,
)
from double_ml_uplift.causal.spec import CausalSpec, NodeSpec, bundled_spec

pytestmark = pytest.mark.unit

Edge = tuple[str, str]


# --- a programmatic spec builder, so a fixture DAG is three lines ------------------------


def make_spec(
    nodes: Sequence[str],
    edges: Sequence[Edge],
    *,
    treatment: str = "t",
    outcome: str = "y",
    unobserved: Iterable[str] = (),
    forbidden: Iterable[str] = (),
    name: str = "fixture",
) -> CausalSpec:
    """Build a valid :class:`CausalSpec` from a bare DAG, inferring the declared roles."""
    graph = CausalGraph.from_edges(nodes, edges)
    downstream = (graph.descendants([treatment]) | graph.descendants([outcome])) - {
        treatment,
        outcome,
    }
    hidden = set(unobserved)
    node_specs: dict[str, NodeSpec] = {}
    for node in nodes:
        if node == treatment:
            node_specs[node] = NodeSpec(kind="binary", role="treatment")
        elif node == outcome:
            node_specs[node] = NodeSpec(kind="continuous", role="outcome")
        else:
            node_specs[node] = NodeSpec(
                kind="continuous",
                role="post_treatment" if node in downstream else "confounder",
                observed=node not in hidden,
            )
    return CausalSpec(
        version=1,
        name=name,
        unit_id="row_id",
        treatment=treatment,
        outcome=outcome,
        nodes=node_specs,
        edges=tuple(edges),
        forbidden_columns=tuple(forbidden),
    )


# --- a self-contained oracle, written from the criterion rather than from the module ------
#
# The functions below import nothing from ``double_ml_uplift.causal.graph``. They re-derive the
# back-door criterion from ``spec.edges`` with their own adjacency, trail enumeration and
# collider handling, so the completeness claim below is checked against an implementation that
# shares no code with the one under test. The repo-primitive oracle that follows is kept for the
# soundness and minimality checks -- those reuse ``path_is_blocked``, which row 1 of CLAIMS.md
# pins separately against brute-force path enumeration -- and the two are asserted to agree.


def _children(spec: CausalSpec) -> dict[str, set[str]]:
    table: dict[str, set[str]] = {name: set() for name in spec.nodes}
    for parent, child in spec.edges:
        table[parent].add(child)
    return table


def _parents(spec: CausalSpec) -> dict[str, set[str]]:
    table: dict[str, set[str]] = {name: set() for name in spec.nodes}
    for parent, child in spec.edges:
        table[child].add(parent)
    return table


def _reachable(start: str, step: dict[str, set[str]]) -> set[str]:
    """Proper descendants (or ancestors) of ``start`` under the given adjacency."""
    seen: set[str] = set()
    stack = list(step[start])
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(step[node])
    return seen


def _trails(spec: CausalSpec, start: str, end: str) -> list[list[str]]:
    """Every simple undirected path from ``start`` to ``end``, edge directions ignored."""
    neighbours: dict[str, set[str]] = {name: set() for name in spec.nodes}
    for parent, child in spec.edges:
        neighbours[parent].add(child)
        neighbours[child].add(parent)
    found: list[list[str]] = []

    def walk(path: list[str]) -> None:
        node = path[-1]
        if node == end:
            found.append(list(path))
            return
        for nxt in sorted(neighbours[node]):
            if nxt not in path:
                path.append(nxt)
                walk(path)
                path.pop()

    walk([start])
    return found


def _trail_is_blocked(spec: CausalSpec, trail: list[str], conditioned: set[str]) -> bool:
    """Pearl's d-separation rule, applied node by node along one trail.

    A trail is blocked when some intermediate node blocks it: a non-collider that is conditioned
    on, or a collider with neither itself nor any of its descendants conditioned on.
    """
    children = _children(spec)
    for position in range(1, len(trail) - 1):
        before, node, after = trail[position - 1], trail[position], trail[position + 1]
        collider = node in children[before] and node in children[after]
        if collider:
            opened = {node} | _reachable(node, children)
            if not (opened & conditioned):
                return True
        elif node in conditioned:
            return True
    return False


def _backdoor_trails(spec: CausalSpec) -> list[list[str]]:
    """Trails leaving the treatment through an arrowhead into it."""
    parents = _parents(spec)
    return [
        trail
        for trail in _trails(spec, spec.treatment, spec.outcome)
        if len(trail) > 1 and trail[1] in parents[spec.treatment]
    ]


def independent_pool(spec: CausalSpec) -> list[str]:
    """Variables the criterion allows: observed, not on or below the treatment or the outcome."""
    children = _children(spec)
    banned = {spec.treatment, spec.outcome}
    banned |= _reachable(spec.treatment, children) | _reachable(spec.outcome, children)
    banned |= set(spec.forbidden_columns)
    banned |= {name for name, node in spec.nodes.items() if not node.observed}
    return sorted(set(spec.nodes) - banned)


def independent_blocks(spec: CausalSpec, adjustment: Iterable[str]) -> bool:
    conditioned = set(adjustment)
    return all(_trail_is_blocked(spec, trail, conditioned) for trail in _backdoor_trails(spec))


def independent_has_any_valid_set(spec: CausalSpec) -> bool:
    """Brute force over every subset of the admissible pool, using only the code above."""
    pool = independent_pool(spec)
    return any(
        independent_blocks(spec, subset)
        for size in range(len(pool) + 1)
        for subset in combinations(pool, size)
    )


# --- an oracle built on the path primitives that CLAIMS.md row 1 pins independently -------


def oracle_pool(spec: CausalSpec) -> list[str]:
    """Variables this repository is willing to adjust for, stated independently of the module.

    Observed, not the treatment or the outcome, and not downstream of either (descendants of the
    outcome are excluded because conditioning on them opens collider paths), and not forbidden.
    """
    graph = spec.graph()
    banned = graph.descendants([spec.treatment]) | graph.descendants([spec.outcome])
    banned |= set(spec.forbidden_columns) | set(spec.unobserved_nodes())
    return sorted(set(spec.nodes) - banned)


def oracle_blocks(spec: CausalSpec, adjustment: Iterable[str]) -> bool:
    """Every backdoor trail from treatment to outcome is blocked by ``adjustment``."""
    graph = spec.graph()
    conditioned = set(adjustment)
    return all(
        graph.path_is_blocked(path, conditioned)
        for path in graph.backdoor_paths(spec.treatment, spec.outcome)
    )


def oracle_has_any_valid_set(spec: CausalSpec) -> bool:
    """Brute force: does *any* subset of the admissible pool block every backdoor trail?"""
    pool = oracle_pool(spec)
    return any(
        oracle_blocks(spec, subset)
        for size in range(len(pool) + 1)
        for subset in combinations(pool, size)
    )


# --- the bundled specifications ----------------------------------------------------------


def test_retention_offer_drops_the_design_column_from_the_minimal_set() -> None:
    identification = identify_backdoor(bundled_spec("retention_offer"))
    assert identification.identified
    assert identification.adjustment_set == (
        "discount_count",
        "engagement_score",
        "monthly_spend",
        "protected_group",
        "support_tickets",
        "tenure_months",
    )
    # `pilot` only points into the treatment, so it blocks nothing and is not needed.
    assert "pilot" in identification.considered
    assert "pilot" not in identification.adjustment_set
    assert identification.backdoor_paths > 0


def test_an_unmeasured_common_cause_refuses_and_names_the_open_path() -> None:
    identification = identify_backdoor(bundled_spec("unobserved"))
    assert not identification.identified
    assert identification.adjustment_set == ()
    assert any("churn_risk" in reason for reason in identification.reasons)
    with pytest.raises(NotIdentifiedError, match="churn_risk"):
        identification.require()


def test_require_returns_the_set_when_the_effect_is_identified() -> None:
    identification = identify_backdoor(bundled_spec("retention_offer"))
    assert identification.require() == identification.adjustment_set


# --- the leakage guard -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "fragment"),
    [
        ("post_offer_logins", "forbidden_columns"),
        ("offer_redeemed", "forbidden_columns"),
        ("csat_after_offer", "forbidden_columns"),
        ("offer_sent", "is the treatment"),
        ("retained_revenue", "is the outcome"),
        ("not_a_column", "not declared"),
    ],
)
def test_leakage_guard_names_the_offender_and_the_reason(name: str, fragment: str) -> None:
    spec = bundled_spec("retention_offer")
    with pytest.raises(LeakageError) as info:
        leakage_guard(spec, ["tenure_months", name])
    offenders = dict(info.value.offenders)
    assert name in offenders
    assert fragment in offenders[name]


def test_a_mediator_not_listed_as_forbidden_is_still_refused_by_its_role() -> None:
    spec = make_spec(["t", "y", "m"], [("t", "m"), ("m", "y"), ("t", "y")])
    assert spec.nodes["m"].role == "post_treatment"
    with pytest.raises(LeakageError, match="declared post-treatment"):
        leakage_guard(spec, ["m"])


def test_leakage_guard_refuses_unobserved_and_forbidden_variables() -> None:
    spec = bundled_spec("unobserved")
    with pytest.raises(LeakageError, match="declared unobserved"):
        leakage_guard(spec, ["churn_risk"])

    forbidden = make_spec(["t", "y", "z"], [("t", "y"), ("z", "t"), ("z", "y")], forbidden=["z"])
    with pytest.raises(LeakageError, match="forbidden_columns"):
        leakage_guard(forbidden, ["z"])


def test_the_guard_still_refuses_when_the_declared_roles_are_wrong() -> None:
    """Defence in depth: a valid spec cannot mislabel a downstream node, but the guard does not
    rely on that. ``model_construct`` skips validation, so these are the branches that would
    catch a specification written by an older, laxer version of the validator."""
    mediator = CausalSpec.model_construct(
        version=1,
        name="mislabelled",
        unit_id="row_id",
        treatment="t",
        outcome="y",
        pilot_column=None,
        protected_attribute=None,
        nodes={
            "t": NodeSpec(kind="binary", role="treatment"),
            "y": NodeSpec(kind="continuous", role="outcome"),
            "m": NodeSpec(kind="continuous", role="confounder"),
        },
        edges=(("t", "m"), ("m", "y"), ("t", "y")),
        forbidden_columns=(),
    )
    with pytest.raises(LeakageError, match="descendant of the treatment"):
        leakage_guard(mediator, ["m"])

    downstream = CausalSpec.model_construct(
        version=1,
        name="mislabelled-outcome-child",
        unit_id="row_id",
        treatment="t",
        outcome="y",
        pilot_column=None,
        protected_attribute=None,
        nodes={
            "t": NodeSpec(kind="binary", role="treatment"),
            "y": NodeSpec(kind="continuous", role="outcome"),
            "d": NodeSpec(kind="continuous", role="confounder"),
        },
        edges=(("t", "y"), ("y", "d")),
        forbidden_columns=(),
    )
    with pytest.raises(LeakageError, match="descendant of the outcome"):
        leakage_guard(downstream, ["d"])


def test_a_pre_treatment_variable_with_a_post_treatment_name_is_refused() -> None:
    spec = make_spec(["t", "y", "logins_after_offer"], [("t", "y"), ("logins_after_offer", "t")])
    with pytest.raises(LeakageError, match="name looks post-treatment"):
        leakage_guard(spec, ["logins_after_offer"])
    # ...and identification never proposes it either.
    assert "logins_after_offer" not in identify_backdoor(spec).considered


def test_leakage_guard_returns_sorted_names_for_an_admissible_request() -> None:
    spec = bundled_spec("retention_offer")
    assert leakage_guard(spec, ["monthly_spend", "tenure_months", "monthly_spend"]) == (
        "monthly_spend",
        "tenure_months",
    )


# --- caller-supplied adjustment sets ------------------------------------------------------


def test_verify_adjustment_set_accepts_a_blocking_set_and_refuses_an_incomplete_one() -> None:
    spec = make_spec(
        ["t", "y", "u", "w"], [("t", "y"), ("u", "t"), ("u", "y"), ("w", "t"), ("w", "y")]
    )
    assert verify_adjustment_set(spec, ["u", "w"]).identified
    refusal = verify_adjustment_set(spec, ["u"])
    assert not refusal.identified
    assert refusal.adjustment_set == ()
    assert any("open backdoor path" in reason for reason in refusal.reasons)


def test_format_path_renders_arrowheads_in_both_directions() -> None:
    graph = CausalGraph.from_edges(["t", "u", "y"], [("u", "t"), ("u", "y")])
    assert format_path(graph, ["t", "u", "y"]) == "t <- u -> y"


# --- 30 fixture DAGs with known answers ---------------------------------------------------

# (name, nodes, edges, unobserved, identified, expected minimal adjustment set)
FIXTURES: list[tuple[str, list[str], list[Edge], list[str], bool, tuple[str, ...]]] = [
    ("no_confounding", ["t", "y"], [("t", "y")], [], True, ()),
    ("single_confounder", ["t", "y", "u"], [("t", "y"), ("u", "t"), ("u", "y")], [], True, ("u",)),
    (
        "two_confounders",
        ["t", "y", "u", "w"],
        [("t", "y"), ("u", "t"), ("u", "y"), ("w", "t"), ("w", "y")],
        [],
        True,
        ("u", "w"),
    ),
    (
        "latent_confounder",
        ["t", "y", "u"],
        [("t", "y"), ("u", "t"), ("u", "y")],
        ["u"],
        False,
        (),
    ),
    ("mediator_only", ["t", "y", "m"], [("t", "m"), ("m", "y"), ("t", "y")], [], True, ()),
    (
        "mediator_plus_confounder",
        ["t", "y", "m", "u"],
        [("t", "m"), ("m", "y"), ("t", "y"), ("u", "t"), ("u", "y")],
        [],
        True,
        ("u",),
    ),
    ("instrument_only", ["t", "y", "z"], [("z", "t"), ("t", "y")], [], True, ()),
    (
        "instrument_and_confounder",
        ["t", "y", "z", "u"],
        [("z", "t"), ("t", "y"), ("u", "t"), ("u", "y")],
        [],
        True,
        ("u",),
    ),
    (
        "instrument_with_latent_confounder",
        ["t", "y", "z", "u"],
        [("z", "t"), ("t", "y"), ("u", "t"), ("u", "y")],
        ["u"],
        False,
        (),
    ),
    ("collider_on_t_and_y", ["t", "y", "c"], [("t", "y"), ("t", "c"), ("y", "c")], [], True, ()),
    (
        "m_bias",
        ["t", "y", "a", "b", "m"],
        [("t", "y"), ("a", "t"), ("a", "m"), ("b", "m"), ("b", "y")],
        [],
        True,
        (),
    ),
    (
        "m_bias_with_latent_arms",
        ["t", "y", "a", "b", "m"],
        [("t", "y"), ("a", "t"), ("a", "m"), ("b", "m"), ("b", "y")],
        ["a", "b"],
        True,
        (),
    ),
    (
        "confounder_behind_a_chain",
        ["t", "y", "u", "v"],
        [("t", "y"), ("u", "v"), ("v", "t"), ("u", "y")],
        [],
        True,
        ("v",),
    ),
    (
        "chain_confounder_latent_head",
        ["t", "y", "u", "v"],
        [("t", "y"), ("u", "v"), ("v", "t"), ("u", "y")],
        ["u"],
        True,
        ("v",),
    ),
    (
        "chain_confounder_latent_middle",
        ["t", "y", "u", "v"],
        [("t", "y"), ("u", "v"), ("v", "t"), ("u", "y")],
        ["v"],
        True,
        ("u",),
    ),
    (
        "proxy_cannot_replace_the_confounder",
        ["t", "y", "u", "p"],
        [("t", "y"), ("u", "t"), ("u", "y"), ("u", "p")],
        ["u"],
        False,
        (),
    ),
    (
        "outcome_predictor_is_not_needed",
        ["t", "y", "r"],
        [("t", "y"), ("r", "y")],
        [],
        True,
        (),
    ),
    (
        "treatment_predictor_is_not_needed",
        ["t", "y", "s"],
        [("t", "y"), ("s", "t")],
        [],
        True,
        (),
    ),
    (
        "two_paths_one_shared_blocker",
        ["t", "y", "u", "a", "b"],
        [("t", "y"), ("u", "a"), ("u", "b"), ("a", "t"), ("b", "y")],
        [],
        True,
        ("u",),
    ),
    (
        "descendant_of_outcome_is_never_used",
        ["t", "y", "u", "d"],
        [("t", "y"), ("u", "t"), ("u", "y"), ("y", "d")],
        [],
        True,
        ("u",),
    ),
    (
        "post_treatment_child_is_never_used",
        ["t", "y", "u", "k"],
        [("t", "y"), ("u", "t"), ("u", "y"), ("t", "k")],
        [],
        True,
        ("u",),
    ),
    (
        "confounder_also_causes_a_mediator",
        ["t", "y", "u", "m"],
        [("t", "m"), ("m", "y"), ("t", "y"), ("u", "t"), ("u", "m")],
        [],
        True,
        ("u",),
    ),
    (
        "forbidden_blocker_makes_it_unidentified",
        ["t", "y", "u"],
        [("t", "y"), ("u", "t"), ("u", "y")],
        [],
        False,
        (),
    ),  # forbidden=["u"] applied below
    (
        "three_independent_confounders",
        ["t", "y", "u", "v", "w"],
        [
            ("t", "y"),
            ("u", "t"),
            ("u", "y"),
            ("v", "t"),
            ("v", "y"),
            ("w", "t"),
            ("w", "y"),
        ],
        [],
        True,
        ("u", "v", "w"),
    ),
    (
        "one_of_three_is_latent",
        ["t", "y", "u", "v", "w"],
        [
            ("t", "y"),
            ("u", "t"),
            ("u", "y"),
            ("v", "t"),
            ("v", "y"),
            ("w", "t"),
            ("w", "y"),
        ],
        ["w"],
        False,
        (),
    ),
    (
        "confounder_of_treatment_and_mediator_only",
        ["t", "y", "m", "u"],
        [("t", "m"), ("m", "y"), ("t", "y"), ("u", "t"), ("u", "m")],
        ["u"],
        False,
        (),
    ),
    (
        "collider_between_two_confounders",
        ["t", "y", "u", "v", "c"],
        [("t", "y"), ("u", "t"), ("v", "y"), ("u", "c"), ("v", "c")],
        [],
        True,
        (),
    ),
    (
        "long_backdoor_chain",
        ["t", "y", "a", "b", "c"],
        [("t", "y"), ("c", "b"), ("b", "a"), ("a", "t"), ("c", "y")],
        [],
        True,
        ("c",),
    ),
    (
        "long_backdoor_chain_latent_tail",
        ["t", "y", "a", "b", "c"],
        [("t", "y"), ("c", "b"), ("b", "a"), ("a", "t"), ("c", "y")],
        ["a", "c"],
        True,
        ("b",),
    ),
    (
        "two_disjoint_backdoor_routes",
        ["t", "y", "u", "v"],
        [("t", "y"), ("u", "t"), ("u", "y"), ("v", "t"), ("v", "y")],
        ["u"],
        False,
        (),
    ),
]


@pytest.mark.parametrize(
    ("name", "nodes", "edges", "unobserved", "identified", "expected"),
    FIXTURES,
    ids=[item[0] for item in FIXTURES],
)
def test_fixture_dags_get_the_adjustment_set_theory_predicts(
    name: str,
    nodes: list[str],
    edges: list[Edge],
    unobserved: list[str],
    identified: bool,
    expected: tuple[str, ...],
) -> None:
    forbidden = ["u"] if name == "forbidden_blocker_makes_it_unidentified" else []
    spec = make_spec(nodes, edges, unobserved=unobserved, forbidden=forbidden, name=name)
    result = identify_backdoor(spec)
    assert result.identified is identified, result.reasons
    assert result.adjustment_set == expected
    if identified:
        assert oracle_blocks(spec, result.adjustment_set)


def test_the_fixture_suite_covers_at_least_thirty_dags() -> None:
    assert len(FIXTURES) >= 30
    assert len({item[0] for item in FIXTURES}) == len(FIXTURES)


# --- the same claims, on generated DAGs ---------------------------------------------------


@st.composite
def random_specs(draw: st.DrawFn) -> CausalSpec:
    """A DAG over ``t``, ``y`` and up to four other nodes, with a guaranteed ``t -> y`` edge."""
    extra = draw(st.integers(min_value=0, max_value=4))
    names = ["t", *[f"v{index}" for index in range(extra)], "y"]
    order = {name: position for position, name in enumerate(names)}
    candidates = [
        (left, right)
        for left in names
        for right in names
        if order[left] < order[right] and (left, right) != ("t", "y")
    ]
    edge_strategy = (
        st.lists(st.sampled_from(candidates), max_size=len(candidates), unique=True)
        if candidates
        else st.just([])
    )
    edges: list[Edge] = draw(edge_strategy)
    hidden_strategy = st.lists(st.sampled_from(names[1:-1]), unique=True) if extra else st.just([])
    hidden: list[str] = draw(hidden_strategy)
    return make_spec(names, [("t", "y"), *edges], unobserved=hidden)


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(spec=random_specs())
def test_an_identified_set_really_blocks_every_backdoor_path(spec: CausalSpec) -> None:
    result = identify_backdoor(spec)
    if not result.identified:
        return
    graph = spec.graph()
    assert oracle_blocks(spec, result.adjustment_set)
    for name in result.adjustment_set:
        assert name not in graph.descendants([spec.treatment])
        assert name not in graph.descendants([spec.outcome])
        assert spec.nodes[name].observed


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(spec=random_specs())
def test_a_refusal_means_no_admissible_subset_would_have_worked(spec: CausalSpec) -> None:
    """Completeness, against an oracle that shares no code with the module under test.

    ``independent_has_any_valid_set`` re-derives adjacency, trail enumeration and the collider
    rule from ``spec.edges`` alone -- see the note above it -- so a shared bug in the repository's
    path primitives cannot make both sides agree.
    """
    result = identify_backdoor(spec)
    assert result.identified == independent_has_any_valid_set(spec)


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(spec=random_specs())
def test_the_two_oracles_agree_with_each_other(spec: CausalSpec) -> None:
    """The self-contained oracle and the one built on the repository's path primitives agree.

    If they ever diverge, one of the two is wrong and the completeness result above is not
    trustworthy either, so this failure is worth surfacing on its own.
    """
    assert sorted(independent_pool(spec)) == sorted(oracle_pool(spec))
    assert independent_has_any_valid_set(spec) == oracle_has_any_valid_set(spec)
    identified = identify_backdoor(spec)
    if identified.identified:
        assert independent_blocks(spec, identified.adjustment_set)
        assert oracle_blocks(spec, identified.adjustment_set)


@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
@given(spec=random_specs())
def test_the_returned_adjustment_set_is_locally_minimal(spec: CausalSpec) -> None:
    result = identify_backdoor(spec)
    if not result.identified:
        return
    for name in result.adjustment_set:
        trimmed = [item for item in result.adjustment_set if item != name]
        assert not oracle_blocks(spec, trimmed), f"{name} was not needed"


@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
@given(spec=random_specs())
def test_identification_is_deterministic_and_survives_the_leakage_guard(
    spec: CausalSpec,
) -> None:
    first, second = identify_backdoor(spec), identify_backdoor(spec)
    assert first == second
    if first.identified:
        assert leakage_guard(spec, first.adjustment_set) == first.adjustment_set
        assert verify_adjustment_set(spec, first.adjustment_set).identified
