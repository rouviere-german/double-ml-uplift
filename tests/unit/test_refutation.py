"""Refuters: each one states a consequence the estimate should have, and flags when it does not."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from double_ml_uplift.causal.estimators import AteEstimate, naive_difference
from double_ml_uplift.causal.refutation import (
    EValue,
    RefutationPolicy,
    RefutationReport,
    RefutationResult,
    e_value,
    pilot_consistency,
    placebo_treatment,
    random_common_cause,
    refute_all,
    subset_stability,
)

pytestmark = pytest.mark.unit

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def honest_refit(
    features: FloatArray, treatment: IntArray, outcome: FloatArray, unit_ids: Sequence[str]
) -> AteEstimate:
    """A refitter with no leakage: the difference in means on whatever it is handed."""
    return naive_difference(treatment, outcome)


def leaky_refit(
    features: FloatArray, treatment: IntArray, outcome: FloatArray, unit_ids: Sequence[str]
) -> AteEstimate:
    """A broken pipeline: it reports a large effect no matter how the treatment was assigned."""
    return AteEstimate(
        estimator="leaky",
        ate=50.0,
        se=1.0,
        ci_low=48.0,
        ci_high=52.0,
        ci_method="influence",
        n=int(treatment.size),
        influence=np.ones(treatment.size, dtype=np.float64),
    )


def randomised(
    n: int = 600, *, effect: float = 5.0, seed: int = 0
) -> tuple[FloatArray, IntArray, FloatArray, list[str]]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 3))
    treatment = rng.binomial(1, 0.5, size=n).astype(np.int64)
    outcome = 10.0 + features[:, 0] + effect * treatment + rng.normal(0.0, 4.0, size=n)
    return features, treatment, outcome, [f"u_{index:05d}" for index in range(n)]


# --- placebo -----------------------------------------------------------------------------------


def test_a_placebo_treatment_passes_on_an_honest_pipeline() -> None:
    features, treatment, outcome, unit_ids = randomised()
    result = placebo_treatment(honest_refit, features, treatment, outcome, unit_ids, seed=1)
    assert result.name == "placebo_treatment"
    assert result.passed
    assert result.severity == "blocking"
    assert "placebo estimate" in result.detail


def test_a_placebo_treatment_catches_a_pipeline_that_ignores_the_treatment() -> None:
    features, treatment, outcome, unit_ids = randomised()
    result = placebo_treatment(leaky_refit, features, treatment, outcome, unit_ids, seed=1)
    assert not result.passed
    assert result.statistic == pytest.approx(50.0)


def test_the_placebo_refuter_does_not_look_at_the_real_treatment_effect() -> None:
    """Permuting the treatment destroys the effect, so the estimate should collapse to zero."""
    features, treatment, outcome, unit_ids = randomised(effect=60.0, seed=2)
    real = naive_difference(treatment, outcome)
    result = placebo_treatment(honest_refit, features, treatment, outcome, unit_ids, seed=3)
    assert real.ate > 40.0
    assert result.passed
    assert abs(float(result.detail.split()[2])) < 20.0


# --- random common cause -------------------------------------------------------------------


def test_adding_an_independent_covariate_barely_moves_an_honest_estimate() -> None:
    features, treatment, outcome, unit_ids = randomised()
    baseline = naive_difference(treatment, outcome)
    result = random_common_cause(
        honest_refit, features, treatment, outcome, unit_ids, baseline, RefutationPolicy(), seed=4
    )
    assert result.passed
    assert result.statistic == pytest.approx(0.0, abs=1e-12)


def test_an_unstable_estimator_is_flagged_by_the_random_common_cause() -> None:
    features, treatment, outcome, unit_ids = randomised()
    baseline = naive_difference(treatment, outcome)
    result = random_common_cause(
        leaky_refit, features, treatment, outcome, unit_ids, baseline, RefutationPolicy(), seed=4
    )
    assert not result.passed
    assert result.statistic > RefutationPolicy().random_common_cause_z


# --- subset stability ---------------------------------------------------------------------------


def test_nested_subsets_agree_with_the_full_sample() -> None:
    features, treatment, outcome, unit_ids = randomised(800, seed=5)
    baseline = naive_difference(treatment, outcome)
    result = subset_stability(
        honest_refit, features, treatment, outcome, unit_ids, baseline, RefutationPolicy(), seed=6
    )
    assert result.passed
    assert "subset of" in result.detail or result.detail == "no subset deviated"


def test_a_subset_that_disagrees_wildly_is_flagged() -> None:
    features, treatment, outcome, unit_ids = randomised(400, seed=7)
    baseline = AteEstimate(
        estimator="x", ate=0.0, se=0.5, ci_low=-1.0, ci_high=1.0, ci_method="influence", n=400
    )
    result = subset_stability(
        honest_refit, features, treatment, outcome, unit_ids, baseline, RefutationPolicy(), seed=8
    )
    assert not result.passed


def test_the_subset_fraction_is_validated() -> None:
    features, treatment, outcome, unit_ids = randomised(200, seed=9)
    baseline = naive_difference(treatment, outcome)
    policy = RefutationPolicy.model_construct(
        random_common_cause_z=1.5,
        subset_stability_z=3.0,
        pilot_consistency_z=2.5,
        min_e_value=1.25,
        subset_fraction=0.05,
        n_subsets=1,
        min_pilot_per_arm=25,
    )
    with pytest.raises(ValueError, match="subset_fraction must lie"):
        subset_stability(
            honest_refit, features, treatment, outcome, unit_ids, baseline, policy, seed=1
        )


# --- pilot consistency: the only refuter that sees unobserved confounding -------------------


def test_the_pilot_check_agrees_with_an_unconfounded_estimate() -> None:
    n = 2_000
    rng = np.random.default_rng(10)
    treatment = rng.binomial(1, 0.5, size=n).astype(np.int64)
    outcome = 10.0 + 6.0 * treatment + rng.normal(0.0, 5.0, size=n)
    pilot = (rng.random(n) < 0.4).astype(np.int64)
    baseline = naive_difference(treatment, outcome)
    result = pilot_consistency(baseline, treatment, outcome, pilot, RefutationPolicy())
    assert result.passed
    assert "randomized pilot" in result.detail


def test_the_pilot_check_catches_an_observational_estimate_that_is_far_off() -> None:
    n = 2_000
    rng = np.random.default_rng(11)
    treatment = rng.binomial(1, 0.5, size=n).astype(np.int64)
    outcome = 10.0 + 6.0 * treatment + rng.normal(0.0, 5.0, size=n)
    pilot = (rng.random(n) < 0.4).astype(np.int64)
    inflated = AteEstimate(
        estimator="biased",
        ate=90.0,
        se=1.0,
        ci_low=88.0,
        ci_high=92.0,
        ci_method="influence",
        n=n,
    )
    result = pilot_consistency(inflated, treatment, outcome, pilot, RefutationPolicy())
    assert not result.passed
    assert result.statistic > RefutationPolicy().pilot_consistency_z


def test_a_pilot_slice_that_is_too_small_is_skipped_rather_than_guessed() -> None:
    n = 200
    rng = np.random.default_rng(12)
    treatment = rng.binomial(1, 0.5, size=n).astype(np.int64)
    outcome = rng.normal(size=n)
    pilot = np.zeros(n, dtype=np.int64)
    pilot[:6] = 1
    result = pilot_consistency(
        naive_difference(treatment, outcome), treatment, outcome, pilot, RefutationPolicy()
    )
    assert result.passed
    assert result.detail.startswith("skipped")
    assert result.statistic == 0.0


# --- E-values ------------------------------------------------------------------------------------


def test_the_e_value_of_a_null_effect_is_one() -> None:
    estimate = AteEstimate(
        estimator="x", ate=0.0, se=1.0, ci_low=-2.0, ci_high=2.0, ci_method="influence", n=100
    )
    value = e_value(estimate, outcome_sd=10.0)
    assert value.point == pytest.approx(1.0)
    assert value.ci_limit == pytest.approx(1.0)


def test_a_larger_standardised_effect_needs_a_stronger_confounder() -> None:
    small = AteEstimate(
        estimator="x", ate=1.0, se=0.2, ci_low=0.6, ci_high=1.4, ci_method="influence", n=100
    )
    large = AteEstimate(
        estimator="x", ate=8.0, se=0.2, ci_low=7.6, ci_high=8.4, ci_method="influence", n=100
    )
    assert e_value(large, 10.0).point > e_value(small, 10.0).point
    assert e_value(large, 10.0).ci_limit > e_value(small, 10.0).ci_limit


def test_the_e_value_is_symmetric_in_the_sign_of_the_effect() -> None:
    positive = AteEstimate(
        estimator="x", ate=5.0, se=1.0, ci_low=3.0, ci_high=7.0, ci_method="influence", n=100
    )
    negative = AteEstimate(
        estimator="x", ate=-5.0, se=1.0, ci_low=-7.0, ci_high=-3.0, ci_method="influence", n=100
    )
    assert e_value(positive, 10.0).point == pytest.approx(e_value(negative, 10.0).point)
    assert e_value(positive, 10.0).ci_limit == pytest.approx(e_value(negative, 10.0).ci_limit)


@settings(max_examples=200)
@given(
    ate=st.floats(min_value=-200.0, max_value=200.0),
    sd=st.floats(min_value=0.5, max_value=500.0),
)
def test_the_e_value_is_never_below_one_and_rises_with_the_effect(ate: float, sd: float) -> None:
    estimate = AteEstimate(
        estimator="x",
        ate=ate,
        se=1.0,
        ci_low=ate - 0.5,
        ci_high=ate + 0.5,
        ci_method="influence",
        n=100,
    )
    value = e_value(estimate, sd)
    assert value.point >= 1.0 - 1e-12
    assert value.ci_limit >= 1.0 - 1e-12
    assert math.isfinite(value.point)


def test_an_absurd_standardised_effect_stays_a_finite_json_number() -> None:
    """An E-value squares its risk ratio, so an unbounded exponent overflows to infinity, which
    is not valid JSON and would break every consumer of the refutation report."""
    huge = AteEstimate(
        estimator="x",
        ate=1e6,
        se=1.0,
        ci_low=1e6 - 2.0,
        ci_high=1e6 + 2.0,
        ci_method="influence",
        n=100,
    )
    value = e_value(huge, outcome_sd=0.5)
    assert math.isfinite(value.point)
    assert math.isfinite(value.ci_limit)
    assert math.isfinite(value.risk_ratio)
    assert json.loads(json.dumps(value.to_dict(), allow_nan=False))["point"] > 1e100
    mirrored = e_value(
        AteEstimate(
            estimator="x",
            ate=-1e6,
            se=1.0,
            ci_low=-1e6 - 2.0,
            ci_high=-1e6 + 2.0,
            ci_method="influence",
            n=100,
        ),
        outcome_sd=0.5,
    )
    assert mirrored.point == pytest.approx(value.point)


def test_a_non_positive_outcome_sd_is_refused() -> None:
    estimate = AteEstimate(
        estimator="x", ate=1.0, se=1.0, ci_low=0.0, ci_high=2.0, ci_method="influence", n=10
    )
    with pytest.raises(ValueError, match="outcome_sd must be positive"):
        e_value(estimate, 0.0)


# --- the whole suite --------------------------------------------------------------------------


def test_refute_all_reports_every_refuter_including_the_ones_that_passed() -> None:
    features, treatment, outcome, unit_ids = randomised(800, seed=13)
    pilot = (np.arange(800) % 2).astype(np.int64)
    baseline = naive_difference(treatment, outcome)
    report = refute_all(
        honest_refit,
        features,
        treatment,
        outcome,
        unit_ids,
        baseline,
        RefutationPolicy(),
        pilot=pilot,
        seed=14,
    )
    names = [item.name for item in report.results]
    assert names == [
        "placebo_treatment",
        "random_common_cause",
        "subset_stability",
        "pilot_consistency",
        "e_value",
    ]
    assert report.passed
    assert report.flagged == ()
    payload = report.to_dict()
    assert payload["passed"] is True
    assert len(payload["results"]) == 5  # type: ignore[arg-type]
    e_value_payload = report.e_value.to_dict()
    assert set(e_value_payload) == {"risk_ratio", "point", "ci_limit"}


def test_without_a_pilot_the_suite_runs_the_other_four() -> None:
    features, treatment, outcome, unit_ids = randomised(500, seed=15)
    report = refute_all(
        honest_refit,
        features,
        treatment,
        outcome,
        unit_ids,
        naive_difference(treatment, outcome),
        RefutationPolicy(),
        seed=16,
    )
    assert [item.name for item in report.results] == [
        "placebo_treatment",
        "random_common_cause",
        "subset_stability",
        "e_value",
    ]


def test_an_advisory_failure_does_not_block_but_is_reported() -> None:
    advisory = RefutationResult(
        name="e_value", passed=False, statistic=1.0, threshold=2.0, detail="", severity="advisory"
    )
    blocking = RefutationResult(
        name="placebo_treatment", passed=False, statistic=9.0, threshold=2.0, detail=""
    )
    only_advisory = RefutationReport(
        results=(advisory,), e_value=EValue(risk_ratio=1.0, point=1.0, ci_limit=1.0)
    )
    assert only_advisory.passed
    assert only_advisory.flagged == ()
    assert only_advisory.advisories == ("e_value",)

    with_blocking = RefutationReport(
        results=(advisory, blocking), e_value=EValue(risk_ratio=1.0, point=1.0, ci_limit=1.0)
    )
    assert not with_blocking.passed
    assert with_blocking.flagged == ("placebo_treatment",)


def test_policy_bounds_are_enforced() -> None:
    for overrides in (
        {"random_common_cause_z": 0.0},
        {"subset_stability_z": 20.0},
        {"min_e_value": 0.5},
        {"subset_fraction": 1.0},
        {"n_subsets": 0},
        {"min_pilot_per_arm": 1},
    ):
        with pytest.raises(ValidationError):
            RefutationPolicy(**overrides)
