"""The overlap and balance gate: what it measures, and when it refuses to report a number."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from double_ml_uplift.causal.diagnostics import (
    GatePolicy,
    NotEstimableError,
    abstain_mask,
    apply_gate,
    overlap_report,
    standardized_mean_difference,
    trim_mask,
)

pytestmark = pytest.mark.unit

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def balanced(n: int = 400, *, seed: int = 0) -> tuple[FloatArray, IntArray, FloatArray]:
    """A randomised campaign: nothing for the gate to complain about."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 3))
    treatment = rng.binomial(1, 0.5, size=n).astype(np.int64)
    return features, treatment, np.full(n, 0.5)


# --- the standardised mean difference --------------------------------------------------------


def test_smd_is_zero_when_the_arms_have_the_same_mean() -> None:
    values = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    treatment = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
    assert standardized_mean_difference(values, treatment) == pytest.approx(0.0)


def test_smd_uses_the_unweighted_pooled_denominator_in_both_directions() -> None:
    values = np.array([0.0, 0.0, 2.0, 2.0])
    treatment = np.array([1, 1, 0, 0], dtype=np.int64)
    # Both arms are constant, so the pooled within-arm SD is zero and the SMD is defined as 0.
    assert standardized_mean_difference(values, treatment) == pytest.approx(0.0)

    spread = np.array([0.0, 1.0, 2.0, 3.0])
    smd = standardized_mean_difference(spread, treatment)
    assert smd == pytest.approx(-2.0 / np.sqrt((0.25 + 0.25) / 2.0))


def test_weighting_can_remove_an_imbalance() -> None:
    values = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    treatment = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int64)
    before = standardized_mean_difference(values, treatment)
    # Up-weight the single control customer whose covariate matches the treated arm's mix.
    weights = np.ones(values.size, dtype=np.float64)
    weights[4] = 3.0
    after = standardized_mean_difference(values, treatment, weights)
    assert abs(before) > 0.5
    assert after == pytest.approx(0.0)


def test_smd_refuses_an_empty_arm_or_vanished_weights() -> None:
    values = np.arange(4.0)
    with pytest.raises(ValueError, match="both arms must be non-empty"):
        standardized_mean_difference(values, np.ones(4, dtype=np.int64))
    treatment = np.array([1, 1, 0, 0], dtype=np.int64)
    with pytest.raises(ValueError, match="weights vanished"):
        standardized_mean_difference(values, treatment, np.array([0.0, 0.0, 1.0, 1.0]))


@settings(max_examples=100)
@given(scale=st.floats(min_value=0.1, max_value=100.0), shift=st.floats(-50.0, 50.0))
def test_smd_is_invariant_to_rescaling_the_covariate(scale: float, shift: float) -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(size=60)
    treatment = rng.binomial(1, 0.5, size=60).astype(np.int64)
    base = standardized_mean_difference(values, treatment)
    moved = standardized_mean_difference(values * scale + shift, treatment)
    assert moved == pytest.approx(base, rel=1e-9, abs=1e-9)


# --- the report --------------------------------------------------------------------------------


def test_a_randomised_campaign_passes_every_gate() -> None:
    features, treatment, propensity = balanced()
    policy = GatePolicy()
    report = overlap_report(features, ["a", "b", "c"], treatment, propensity, policy)
    decision = apply_gate(report, policy)
    assert decision.estimable
    assert decision.reasons == ()
    assert report.fraction_outside == 0.0
    assert report.n == 400
    assert report.propensity_min == pytest.approx(0.5)
    assert report.max_abs_smd_after < policy.max_abs_smd
    decision.require()  # does not raise


def test_a_positivity_violation_is_refused_with_a_reason() -> None:
    rng = np.random.default_rng(2)
    n = 400
    features = rng.normal(size=(n, 2))
    treatment = rng.binomial(1, 0.5, size=n).astype(np.int64)
    propensity = np.where(np.arange(n) < n // 2, 0.001, 0.999)
    policy = GatePolicy()
    report = overlap_report(features, ["a", "b"], treatment, propensity, policy)
    decision = apply_gate(report, policy)
    assert not decision.estimable
    assert any(reason.startswith("positivity") for reason in decision.reasons)
    assert report.fraction_outside == pytest.approx(1.0)
    with pytest.raises(NotEstimableError, match="positivity") as info:
        decision.require()
    assert info.value.report is report


def test_residual_imbalance_after_weighting_is_refused() -> None:
    n = 400
    rng = np.random.default_rng(3)
    treatment = rng.binomial(1, 0.5, size=n).astype(np.int64)
    # A covariate that separates the arms perfectly, with a propensity that claims otherwise.
    feature = treatment.astype(np.float64) + rng.normal(0.0, 0.05, size=n)
    features = feature.reshape(-1, 1)
    propensity = np.full(n, 0.5)
    policy = GatePolicy()
    report = overlap_report(features, ["separator"], treatment, propensity, policy)
    decision = apply_gate(report, policy)
    assert not decision.estimable
    assert any(reason.startswith("balance") for reason in decision.reasons)
    assert report.worst_balanced_feature() == "separator"


def test_a_tiny_effective_sample_is_refused() -> None:
    n = 60
    rng = np.random.default_rng(4)
    features = rng.normal(size=(n, 1))
    treatment = rng.binomial(1, 0.5, size=n).astype(np.int64)
    propensity = np.full(n, 0.5)
    policy = GatePolicy(min_effective_sample=1_000.0)
    decision = apply_gate(overlap_report(features, ["a"], treatment, propensity, policy), policy)
    assert not decision.estimable
    assert sum(reason.startswith("effective sample") for reason in decision.reasons) == 2


def test_every_failing_threshold_contributes_its_own_reason() -> None:
    n = 200
    rng = np.random.default_rng(5)
    treatment = rng.binomial(1, 0.5, size=n).astype(np.int64)
    features = (treatment.astype(np.float64) + rng.normal(0.0, 0.05, size=n)).reshape(-1, 1)
    propensity = np.where(treatment == 1, 0.999, 0.001)
    policy = GatePolicy(min_effective_sample=10_000.0)
    decision = apply_gate(
        overlap_report(features, ["separator"], treatment, propensity, policy), policy
    )
    assert len(decision.reasons) >= 3


def test_the_report_is_json_serialisable_and_rounded() -> None:
    features, treatment, propensity = balanced(100, seed=6)
    policy = GatePolicy()
    payload = overlap_report(features, ["a", "b", "c"], treatment, propensity, policy).to_dict()
    assert payload["n"] == 100
    assert payload["bounds"] == [0.05, 0.95]
    assert set(payload) == {
        "n",
        "n_treated",
        "propensity_min",
        "propensity_max",
        "fraction_outside",
        "bounds",
        "effective_sample_treated",
        "effective_sample_control",
        "max_abs_smd_before",
        "max_abs_smd_after",
        "worst_balanced_feature",
    }


def test_an_empty_balance_list_reports_zero_imbalance() -> None:
    n = 40
    rng = np.random.default_rng(7)
    treatment = rng.binomial(1, 0.5, size=n).astype(np.int64)
    report = overlap_report(np.zeros((n, 0)), [], treatment, np.full(n, 0.5), GatePolicy())
    assert report.max_abs_smd_before == 0.0
    assert report.max_abs_smd_after == 0.0
    assert report.worst_balanced_feature() is None


def test_report_inputs_are_validated() -> None:
    features, treatment, propensity = balanced(20, seed=8)
    policy = GatePolicy()
    with pytest.raises(ValueError, match="features and feature_names disagree"):
        overlap_report(features, ["only_one"], treatment, propensity, policy)
    with pytest.raises(ValueError, match="same length"):
        overlap_report(features, ["a", "b", "c"], treatment[:5], propensity, policy)


# --- policy and masks ----------------------------------------------------------------------


def test_the_overlap_region_must_bracket_the_coin_flip() -> None:
    """The field bounds are what guarantee ``propensity_low < 0.5 < propensity_high``, so a
    randomised campaign can never be refused for lack of overlap."""
    for overrides in (
        {"propensity_low": 0.0},
        {"propensity_low": 0.5},
        {"propensity_low": 0.6},
        {"propensity_high": 0.5},
        {"propensity_high": 1.0},
        {"propensity_high": 0.4},
        {"max_abs_smd": 0.0},
        {"max_outside_fraction": 1.5},
        {"min_effective_sample": 0.0},
    ):
        with pytest.raises(ValidationError):
            GatePolicy(**overrides)
    policy = GatePolicy()
    assert policy.propensity_low < 0.5 < policy.propensity_high
    with pytest.raises(ValidationError, match="unexpected"):
        GatePolicy.model_validate({"unexpected": 1})


def test_trim_and_abstain_masks_are_exact_complements() -> None:
    propensity = np.array([0.0, 0.04, 0.05, 0.5, 0.95, 0.96, 1.0])
    policy = GatePolicy()
    keep = trim_mask(propensity, policy)
    drop = abstain_mask(propensity, policy)
    assert keep.tolist() == [False, False, True, True, True, False, False]
    assert np.array_equal(keep, ~drop)


@settings(max_examples=100)
@given(
    values=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=50
    )
)
def test_every_row_is_either_kept_or_abstained_on(values: list[float]) -> None:
    propensity = np.asarray(values, dtype=np.float64)
    policy = GatePolicy()
    assert int(trim_mask(propensity, policy).sum() + abstain_mask(propensity, policy).sum()) == (
        propensity.size
    )
