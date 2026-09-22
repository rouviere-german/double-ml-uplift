"""Budgeted targeting, the value of the resulting policy, and the allocation fairness check."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from double_ml_uplift.causal.targeting import (
    FairnessCheck,
    allocation_disparity,
    build_plan,
    fairness_check,
    greedy_knapsack,
    pilot_policy_value,
    rank_by_value_per_cost,
    true_policy_value,
)

pytestmark = pytest.mark.unit

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


# --- ranking and selection -------------------------------------------------------------------


def test_ranking_is_by_uplift_per_unit_of_cost() -> None:
    uplift = np.array([10.0, 10.0, 1.0])
    cost = np.array([5.0, 1.0, 1.0])
    assert rank_by_value_per_cost(uplift, cost).tolist() == [1, 0, 2]


def test_ties_break_by_row_order_so_the_plan_is_deterministic() -> None:
    uplift = np.array([3.0, 3.0, 3.0])
    cost = np.ones(3)
    assert rank_by_value_per_cost(uplift, cost).tolist() == [0, 1, 2]


def test_ranking_refuses_mismatched_or_non_positive_costs() -> None:
    with pytest.raises(ValueError, match="same length"):
        rank_by_value_per_cost(np.ones(3), np.ones(2))
    with pytest.raises(ValueError, match="every cost must be positive"):
        rank_by_value_per_cost(np.ones(3), np.array([1.0, 0.0, 1.0]))


def test_a_customer_with_non_positive_predicted_uplift_is_never_targeted() -> None:
    uplift = np.array([5.0, 0.0, -3.0, 2.0])
    selected = greedy_knapsack(uplift, np.ones(4), budget=4.0)
    assert selected.tolist() == [True, False, False, True]


def test_the_budget_is_respected_and_spent_on_the_best_rows_first() -> None:
    uplift = np.array([1.0, 9.0, 5.0, 7.0])
    cost = np.ones(4)
    assert greedy_knapsack(uplift, cost, budget=2.0).tolist() == [False, True, False, True]
    assert greedy_knapsack(uplift, cost, budget=0.0).tolist() == [False] * 4
    assert greedy_knapsack(uplift, cost, budget=99.0).tolist() == [True] * 4


def test_an_expensive_row_is_skipped_so_a_cheaper_one_can_still_fit() -> None:
    uplift = np.array([100.0, 3.0])
    cost = np.array([10.0, 1.0])
    # Best value per cost is row 0 (10.0/unit) but it does not fit; row 1 does.
    assert greedy_knapsack(uplift, cost, budget=5.0).tolist() == [False, True]


def test_ineligible_rows_are_excluded_entirely() -> None:
    uplift = np.array([9.0, 8.0, 7.0])
    eligible = np.array([False, True, True])
    assert greedy_knapsack(uplift, np.ones(3), 3.0, eligible=eligible).tolist() == [
        False,
        True,
        True,
    ]


def test_selection_guards_are_enforced() -> None:
    with pytest.raises(ValueError, match="budget must not be negative"):
        greedy_knapsack(np.ones(3), np.ones(3), -1.0)
    with pytest.raises(ValueError, match="eligible mask and uplift"):
        greedy_knapsack(np.ones(3), np.ones(3), 1.0, eligible=np.ones(2, dtype=np.bool_))


@settings(max_examples=200)
@given(
    uplift=st.lists(st.floats(-50.0, 50.0, allow_nan=False), min_size=1, max_size=40),
    budget_units=st.integers(min_value=0, max_value=40),
)
def test_the_plan_never_exceeds_its_budget_and_only_takes_positive_uplift(
    uplift: list[float], budget_units: int
) -> None:
    values = np.asarray(uplift, dtype=np.float64)
    cost = np.ones(values.size)
    selected = greedy_knapsack(values, cost, float(budget_units))
    assert int(selected.sum()) <= budget_units
    assert np.all(values[selected] > 0.0)


@settings(max_examples=100)
@given(uplift=st.lists(st.floats(0.1, 50.0, allow_nan=False), min_size=2, max_size=30))
def test_a_larger_budget_never_deselects_a_customer(uplift: list[float]) -> None:
    """With equal costs, greedy selection is monotone in the budget."""
    values = np.asarray(uplift, dtype=np.float64)
    cost = np.ones(values.size)
    smaller = greedy_knapsack(values, cost, 3.0)
    larger = greedy_knapsack(values, cost, 6.0)
    assert np.all(larger[smaller])


# --- valuing the policy -----------------------------------------------------------------------


def pilot_sample(
    n: int = 4_000, *, effect_targeted: float = 20.0, seed: int = 0
) -> tuple[BoolArray, IntArray, IntArray, FloatArray, FloatArray, FloatArray]:
    rng = np.random.default_rng(seed)
    selected = rng.random(n) < 0.3
    pilot = (rng.random(n) < 0.5).astype(np.int64)
    treatment = rng.binomial(1, 0.5, size=n).astype(np.int64)
    effect = np.where(selected, effect_targeted, -5.0)
    y0 = rng.normal(100.0, 10.0, size=n)
    y1 = y0 + effect
    outcome = np.where(treatment == 1, y1, y0)
    return np.asarray(selected), pilot, treatment, outcome, y0, y1


def test_the_pilot_estimate_brackets_the_truth_it_is_estimating() -> None:
    selected, pilot, treatment, outcome, y0, y1 = pilot_sample()
    estimate = pilot_policy_value(selected, pilot, treatment, outcome)
    truth = true_policy_value(selected, y0, y1)
    assert estimate.covers(truth)
    assert estimate.method == "randomized-pilot"
    assert estimate.targeted_share == pytest.approx(float(selected.mean()))
    assert estimate.n_pilot_treated > 100
    assert estimate.n_pilot_control > 100


def test_the_value_of_targeting_nobody_is_exactly_zero() -> None:
    selected, pilot, treatment, outcome, y0, y1 = pilot_sample()
    empty = np.zeros_like(selected)
    estimate = pilot_policy_value(empty, pilot, treatment, outcome)
    assert estimate.value == 0.0
    assert estimate.targeted_share == 0.0
    assert true_policy_value(empty, y0, y1) == 0.0


def test_the_true_value_is_the_average_gain_over_treating_nobody() -> None:
    y0 = np.array([1.0, 2.0, 3.0, 4.0])
    y1 = np.array([5.0, 2.0, 0.0, 9.0])
    selected = np.array([True, False, True, False])
    assert true_policy_value(selected, y0, y1) == pytest.approx((4.0 + -3.0) / 4.0)


def test_a_pilot_too_thin_to_value_the_policy_is_refused_rather_than_guessed() -> None:
    n = 200
    rng = np.random.default_rng(1)
    selected = np.asarray(rng.random(n) < 0.02)
    pilot = np.zeros(n, dtype=np.int64)
    pilot[:4] = 1
    treatment = rng.binomial(1, 0.5, size=n).astype(np.int64)
    outcome = rng.normal(size=n)
    with pytest.raises(ValueError, match="at least 5 of each are required"):
        pilot_policy_value(selected, pilot, treatment, outcome)


def test_policy_value_inputs_must_line_up() -> None:
    with pytest.raises(ValueError, match="same length"):
        pilot_policy_value(
            np.ones(4, dtype=np.bool_),
            np.ones(4, dtype=np.int64),
            np.ones(3, dtype=np.int64),
            np.ones(4),
        )
    with pytest.raises(ValueError, match="same length"):
        true_policy_value(np.ones(3, dtype=np.bool_), np.ones(3), np.ones(2))


def test_a_targeted_policy_beats_treating_everyone_when_effects_differ_in_sign() -> None:
    selected, _pilot, _treatment, _outcome, y0, y1 = pilot_sample(seed=2)
    everyone = np.ones_like(selected)
    assert true_policy_value(selected, y0, y1) > true_policy_value(everyone, y0, y1)


# --- plans ----------------------------------------------------------------------------------


def test_build_plan_prices_a_baseline_ranking_with_the_uplift_model() -> None:
    uplift = np.array([10.0, -2.0, 6.0, 1.0])
    propensity = np.array([0.9, 0.8, 0.2, 0.1])
    cost = np.ones(4)
    plan = build_plan(propensity, cost, 2.0, ranking="propensity", uplift=uplift)
    # Selection follows the propensity, but the value reported is the predicted uplift.
    assert plan.selected.tolist() == [True, True, False, False]
    assert plan.n_selected == 2
    assert plan.total_cost == pytest.approx(2.0)
    assert plan.predicted_gain == pytest.approx((10.0 - 2.0) / 4.0)
    assert plan.to_dict()["ranking"] == "propensity"


def test_build_plan_defaults_to_valuing_its_own_score() -> None:
    uplift = np.array([4.0, 8.0])
    plan = build_plan(uplift, np.ones(2), 1.0, ranking="uplift")
    assert plan.selected.tolist() == [False, True]
    assert plan.predicted_gain == pytest.approx(8.0 / 2.0)


# --- fairness ---------------------------------------------------------------------------------


def test_equal_allocation_rates_mean_no_disparity() -> None:
    selected = np.array([True, False, True, False])
    protected = np.array([1, 1, 0, 0], dtype=np.int64)
    disparity, rate_in, rate_out = allocation_disparity(selected, protected)
    assert disparity == pytest.approx(0.0)
    assert rate_in == pytest.approx(0.5)
    assert rate_out == pytest.approx(0.5)


def test_a_one_sided_allocation_is_blocked() -> None:
    selected = np.array([True, True, False, False])
    protected = np.array([1, 1, 0, 0], dtype=np.int64)
    check = fairness_check(selected, protected, threshold=0.15)
    assert check.disparity == pytest.approx(1.0)
    assert check.blocked
    assert check.to_dict()["blocked"] is True


def test_a_small_disparity_passes_the_release_check() -> None:
    selected = np.array([True, True, True, False, True, True, False, False])
    protected = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int64)
    # 0.75 in the protected group against 0.50 outside it: a disparity of 0.25.
    assert not fairness_check(selected, protected, threshold=0.30).blocked
    assert fairness_check(selected, protected, threshold=0.20).blocked


def test_a_single_group_population_cannot_be_disparate() -> None:
    selected = np.array([True, False, True, True])
    protected = np.ones(4, dtype=np.int64)
    disparity, rate_in, rate_out = allocation_disparity(selected, protected)
    assert disparity == 0.0
    assert rate_in == rate_out == pytest.approx(0.75)


def test_a_measured_check_says_so_and_an_unmeasured_one_reports_no_numbers() -> None:
    selected = np.array([True, False, True, False])
    protected = np.array([1, 1, 0, 0], dtype=np.int64)
    measured = fairness_check(selected, protected, threshold=0.15)
    assert measured.checked is True
    assert measured.to_dict()["checked"] is True

    unmeasured = FairnessCheck.not_checked(0.15)
    assert unmeasured.checked is False
    assert unmeasured.blocked is False
    assert unmeasured.to_dict() == {
        "checked": False,
        "disparity": None,
        "threshold": 0.15,
        "blocked": False,
        "rate_protected": None,
        "rate_other": None,
    }


def test_fairness_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="same length"):
        allocation_disparity(np.ones(3, dtype=np.bool_), np.ones(2, dtype=np.int64))
    with pytest.raises(ValueError, match="must be binary"):
        allocation_disparity(np.ones(3, dtype=np.bool_), np.array([0, 1, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="threshold must lie"):
        fairness_check(np.ones(2, dtype=np.bool_), np.ones(2, dtype=np.int64), threshold=0.0)


@settings(max_examples=150)
@given(
    flags=st.lists(st.booleans(), min_size=2, max_size=60),
    groups=st.lists(st.integers(0, 1), min_size=2, max_size=60),
)
def test_the_disparity_is_a_rate_difference_in_zero_to_one(
    flags: list[bool], groups: list[int]
) -> None:
    n = min(len(flags), len(groups))
    assume(n >= 2)
    selected = np.asarray(flags[:n], dtype=np.bool_)
    protected = np.asarray(groups[:n], dtype=np.int64)
    disparity, rate_in, rate_out = allocation_disparity(selected, protected)
    assert 0.0 <= disparity <= 1.0
    assert 0.0 <= rate_in <= 1.0
    assert 0.0 <= rate_out <= 1.0
    assert disparity == pytest.approx(abs(rate_in - rate_out))
