"""Estimator algebra: the doubly robust identity, the FWL differential, and interval behaviour."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from double_ml_uplift.causal.estimators import (
    ATE_ESTIMATORS,
    AteEstimate,
    aipw,
    aipw_pseudo_outcome,
    bootstrap_interval,
    clipped_ipw,
    dml_partialling_out,
    dr_learner,
    estimate_all,
    naive_difference,
    outcome_regression,
    pehe,
    summarize,
)
from double_ml_uplift.causal.folds import make_fold_plan
from double_ml_uplift.causal.nuisance import (
    CrossFitted,
    NuisanceConfig,
    cross_fit,
    stack_features,
)

pytestmark = pytest.mark.unit

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def unit_ids(n: int) -> list[str]:
    return [f"row_{index:06d}" for index in range(n)]


def fitted_from(
    propensity: FloatArray,
    mu0: FloatArray,
    mu1: FloatArray,
    outcome_hat: FloatArray | None = None,
    *,
    n_folds: int = 2,
) -> CrossFitted:
    """A :class:`CrossFitted` with nuisances we chose, so estimator algebra is testable exactly."""
    n = propensity.size
    plan = make_fold_plan(unit_ids(n), n_folds, salt="test")
    pooled = propensity * mu1 + (1.0 - propensity) * mu0 if outcome_hat is None else outcome_hat
    return CrossFitted(
        propensity=propensity,
        mu0=mu0,
        mu1=mu1,
        outcome_hat=np.asarray(pooled, dtype=np.float64),
        plan=plan,
        config=NuisanceConfig(kind="ols"),
    )


# --- the doubly robust identity, exactly -------------------------------------------------

_reals = st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False)
_propensities = st.floats(min_value=0.05, max_value=0.95, allow_nan=False)


@settings(max_examples=200)
@given(
    y0=st.lists(_reals, min_size=6, max_size=40),
    gap=st.lists(_reals, min_size=6, max_size=40),
    propensity=st.lists(_propensities, min_size=6, max_size=40),
    mu0=st.lists(_reals, min_size=6, max_size=40),
    mu1=st.lists(_reals, min_size=6, max_size=40),
)
def test_aipw_pseudo_outcome_averages_to_the_true_effect_for_any_outcome_model(
    y0: list[float],
    gap: list[float],
    propensity: list[float],
    mu0: list[float],
    mu1: list[float],
) -> None:
    """With the true propensity, ``E_T[psi | X] = Y1 - Y0`` whatever ``mu`` is. This is the
    algebraic core of double robustness, so it is checked without any sampling noise: average
    the pseudo-outcome over both arms, weighted by the propensity that generated them."""
    n = min(len(y0), len(gap), len(propensity), len(mu0), len(mu1))
    assume(n >= 6)
    y0a = np.asarray(y0[:n], dtype=np.float64)
    y1a = y0a + np.asarray(gap[:n], dtype=np.float64)
    e = np.asarray(propensity[:n], dtype=np.float64)
    fitted = fitted_from(e, np.asarray(mu0[:n]), np.asarray(mu1[:n]))

    treated_all = np.ones(n, dtype=np.int64)
    control_all = np.zeros(n, dtype=np.int64)
    treated_all[0], control_all[0] = 0, 1  # both arms present, as the estimator requires
    psi_treated = aipw_pseudo_outcome(treated_all, y1a, fitted, clip=0.001)
    psi_control = aipw_pseudo_outcome(control_all, y0a, fitted, clip=0.001)
    # Row 0 carries the flipped arm in each vector, so compare the rows that are genuinely
    # treated in the first call and untreated in the second.
    expected = y1a[1:] - y0a[1:]
    mixed = e[1:] * psi_treated[1:] + (1.0 - e[1:]) * psi_control[1:]
    assert np.allclose(mixed, expected, atol=1e-8, rtol=1e-9)


@settings(max_examples=100)
@given(
    mu0=st.lists(_reals, min_size=8, max_size=30),
    gap=st.lists(_reals, min_size=8, max_size=30),
)
def test_outcome_regression_reports_the_mean_of_its_own_arm_difference(
    mu0: list[float], gap: list[float]
) -> None:
    n = min(len(mu0), len(gap))
    assume(n >= 8)
    mu0a = np.asarray(mu0[:n], dtype=np.float64)
    mu1a = mu0a + np.asarray(gap[:n], dtype=np.float64)
    fitted = fitted_from(np.full(n, 0.5), mu0a, mu1a)
    estimate = outcome_regression(fitted)
    assert estimate.ate == pytest.approx(float((mu1a - mu0a).mean()), abs=1e-9)
    assert estimate.n == n


@settings(max_examples=100)
@given(shift=_reals, scale=st.floats(min_value=0.1, max_value=50.0))
def test_self_normalised_ipw_is_shift_invariant_and_scale_equivariant(
    shift: float, scale: float
) -> None:
    rng = np.random.default_rng(3)
    n = 60
    propensity = rng.uniform(0.2, 0.8, size=n)
    treatment = (rng.random(n) < propensity).astype(np.int64)
    assume(0 < treatment.sum() < n)
    outcome = rng.normal(10.0, 3.0, size=n)
    fitted = fitted_from(propensity, np.zeros(n), np.zeros(n))

    base = clipped_ipw(treatment, outcome, fitted).ate
    shifted = clipped_ipw(treatment, outcome + shift, fitted).ate
    scaled = clipped_ipw(treatment, outcome * scale, fitted).ate
    assert shifted == pytest.approx(base, abs=1e-8)
    assert scaled == pytest.approx(base * scale, rel=1e-9, abs=1e-8)


@settings(max_examples=100)
@given(
    offset=st.lists(_reals, min_size=10, max_size=30),
    scale=st.floats(min_value=0.1, max_value=20.0),
)
def test_dml_is_invariant_to_anything_already_removed_from_both_residuals(
    offset: list[float], scale: float
) -> None:
    """Adding ``g(X)`` to ``Y`` and to ``E[Y|X]`` cannot move the partialling-out estimate."""
    rng = np.random.default_rng(11)
    n = len(offset)
    propensity = rng.uniform(0.25, 0.75, size=n)
    treatment = (rng.random(n) < propensity).astype(np.int64)
    assume(0 < treatment.sum() < n)
    outcome = rng.normal(0.0, 5.0, size=n)
    pooled = rng.normal(0.0, 1.0, size=n)
    fitted = fitted_from(propensity, np.zeros(n), np.zeros(n), outcome_hat=pooled)
    extra = np.asarray(offset, dtype=np.float64)

    base = dml_partialling_out(treatment, outcome, fitted).ate
    moved = dml_partialling_out(
        treatment,
        outcome + extra,
        fitted_from(propensity, np.zeros(n), np.zeros(n), outcome_hat=pooled + extra),
    ).ate
    scaled = dml_partialling_out(
        treatment,
        outcome * scale,
        fitted_from(propensity, np.zeros(n), np.zeros(n), outcome_hat=pooled * scale),
    ).ate
    assert moved == pytest.approx(base, abs=1e-8)
    assert scaled == pytest.approx(base * scale, rel=1e-9, abs=1e-8)


# --- the exact differential against statsmodels ------------------------------------------


def test_dml_without_crossfitting_equals_the_ols_coefficient() -> None:
    """Frisch-Waugh-Lovell: partialling out with OLS nuisances and one fold is the OLS
    coefficient on the treatment in the long regression, to machine precision."""
    import statsmodels.api as sm

    rng = np.random.default_rng(20260921)
    n = 900
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.binomial(1, 0.4, size=n).astype(np.float64)
    features = stack_features([x1, x2, x3])
    propensity = 1.0 / (1.0 + np.exp(-(0.8 * x1 - 0.5 * x2 + 0.3 * x3)))
    treatment = (rng.random(n) < propensity).astype(np.int64)
    outcome = 3.0 + 2.5 * x1 - 1.5 * x2 + 0.7 * x3 + 4.0 * treatment + rng.normal(0.0, 2.0, n)

    plan = make_fold_plan(unit_ids(n), 1, salt="fwl")
    fitted = cross_fit(features, treatment, outcome, plan, NuisanceConfig(kind="ols"))
    estimate = dml_partialling_out(treatment, outcome, fitted)

    design = sm.add_constant(
        np.column_stack([treatment.astype(np.float64), features]), has_constant="add"
    )
    ols = sm.OLS(outcome, design).fit()
    assert estimate.ate == pytest.approx(float(ols.params[1]), abs=1e-8, rel=0.0)


def test_one_fold_plans_fit_and_predict_on_the_same_rows() -> None:
    plan = make_fold_plan(unit_ids(12), 1, salt="s")
    train, test = plan.indices(0)
    assert np.array_equal(train, test)
    assert plan.sizes() == (12,)


# --- intervals, bootstrap and guards ------------------------------------------------------


def test_naive_difference_is_the_difference_in_observed_means() -> None:
    treatment = np.array([0, 0, 1, 1, 1, 0], dtype=np.int64)
    outcome = np.array([1.0, 3.0, 10.0, 12.0, 14.0, 5.0])
    estimate = naive_difference(treatment, outcome)
    assert estimate.ate == pytest.approx(12.0 - 3.0)
    assert estimate.ci_method == "analytic"
    assert estimate.influence is None
    assert estimate.covers(12.0 - 3.0)


def test_estimate_inputs_are_validated() -> None:
    fitted = fitted_from(np.full(8, 0.5), np.zeros(8), np.zeros(8))
    with pytest.raises(ValueError, match="same length"):
        naive_difference(np.zeros(3, dtype=np.int64), np.zeros(4))
    with pytest.raises(ValueError, match="at least four rows"):
        naive_difference(np.array([0, 1], dtype=np.int64), np.zeros(2))
    with pytest.raises(ValueError, match="both a treated and an untreated"):
        naive_difference(np.ones(8, dtype=np.int64), np.zeros(8))
    with pytest.raises(ValueError, match="each arm needs at least two rows"):
        naive_difference(np.array([1, 0, 0, 0], dtype=np.int64), np.zeros(4))
    with pytest.raises(ValueError, match="clip must lie"):
        fitted.clipped_propensity(0.9)
    with pytest.raises(ValueError, match="level must lie"):
        naive_difference(np.array([0, 0, 1, 1], dtype=np.int64), np.arange(4.0), level=0.4)


def test_ipw_refuses_when_an_arm_has_no_weight() -> None:
    n = 8
    treatment = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    outcome = np.arange(n, dtype=np.float64)
    fitted = fitted_from(np.full(n, 0.5), np.zeros(n), np.zeros(n))
    # A propensity vector of exactly zero in the treated arm cannot happen after clipping,
    # so the guard is exercised through an all-one-arm weight vector instead.
    with pytest.raises(ValueError, match="both a treated and an untreated"):
        clipped_ipw(np.ones(n, dtype=np.int64), outcome, fitted)
    assert clipped_ipw(treatment, outcome, fitted).n == n


def test_dml_refuses_when_the_treatment_residual_is_degenerate() -> None:
    n = 10
    treatment = np.array([0, 1] * (n // 2), dtype=np.int64)
    outcome = np.arange(n, dtype=np.float64)
    fitted = fitted_from(treatment.astype(np.float64), np.zeros(n), np.zeros(n))
    with pytest.raises(ValueError, match="no variation"):
        dml_partialling_out(treatment, outcome, fitted)


def test_bootstrap_interval_is_seeded_bounded_and_needs_an_influence_function() -> None:
    n = 200
    rng = np.random.default_rng(5)
    treatment = rng.binomial(1, 0.5, size=n).astype(np.int64)
    outcome = rng.normal(size=n) + 2.0 * treatment
    fitted = fitted_from(np.full(n, 0.5), np.zeros(n), np.zeros(n), outcome_hat=np.zeros(n))
    estimate = dml_partialling_out(treatment, outcome, fitted, n_bootstrap=300, seed=7)

    assert estimate.bootstrap_ci is not None
    low, high = estimate.bootstrap_ci
    assert low < estimate.ate < high
    assert bootstrap_interval(estimate, n_bootstrap=300, seed=7) == (low, high)
    assert bootstrap_interval(estimate, n_bootstrap=300, seed=8) != (low, high)
    with pytest.raises(ValueError, match="n_bootstrap must lie"):
        bootstrap_interval(estimate, n_bootstrap=0)
    with pytest.raises(ValueError, match="exposes no influence function"):
        bootstrap_interval(naive_difference(treatment, outcome))


def test_a_wider_level_never_produces_a_narrower_interval() -> None:
    n = 120
    rng = np.random.default_rng(2)
    treatment = rng.binomial(1, 0.5, size=n).astype(np.int64)
    outcome = rng.normal(size=n)
    fitted = fitted_from(np.full(n, 0.5), np.zeros(n), np.zeros(n))
    narrow = aipw(treatment, outcome, fitted, level=0.90)
    wide = aipw(treatment, outcome, fitted, level=0.99)
    assert wide.ci_low <= narrow.ci_low
    assert wide.ci_high >= narrow.ci_high
    assert narrow.ate == pytest.approx(wide.ate)


def test_significance_and_dict_view() -> None:
    estimate = AteEstimate(
        estimator="dml", ate=2.0, se=0.5, ci_low=1.0, ci_high=3.0, ci_method="influence", n=100
    )
    assert estimate.significant()
    assert not estimate.covers(0.0)
    payload = estimate.to_dict()
    assert payload["estimator"] == "dml"
    assert payload["bootstrap_ci"] is None
    assert set(summarize({"dml": estimate})["dml"]) == {
        "estimator",
        "ate",
        "se",
        "ci_low",
        "ci_high",
        "ci_method",
        "n",
        "level",
    }


def test_estimate_all_returns_every_named_estimator() -> None:
    n = 80
    rng = np.random.default_rng(4)
    propensity = rng.uniform(0.3, 0.7, size=n)
    treatment = (rng.random(n) < propensity).astype(np.int64)
    outcome = rng.normal(size=n) + 3.0 * treatment
    fitted = fitted_from(propensity, np.zeros(n), np.ones(n))
    estimates = estimate_all(treatment, outcome, fitted, n_bootstrap=50, seed=1)
    assert tuple(estimates) == ATE_ESTIMATORS
    assert all(estimate.n == n for estimate in estimates.values())
    assert estimates["dml"].bootstrap_ci is not None


# --- the CATE learner ---------------------------------------------------------------------


def test_dr_learner_predicts_out_of_fold_and_beats_a_constant_on_a_split_population() -> None:
    rng = np.random.default_rng(1234)
    n = 1_200
    group = rng.binomial(1, 0.5, size=n).astype(np.float64)
    noise = rng.normal(size=n)
    features = stack_features([group, noise])
    treatment = rng.binomial(1, 0.5, size=n).astype(np.int64)
    true_effect = np.where(group == 1, 8.0, -4.0)
    outcome = 1.0 + 0.5 * noise + true_effect * treatment + rng.normal(0.0, 1.0, size=n)

    plan = make_fold_plan(unit_ids(n), 4, salt="cate")
    fitted = cross_fit(features, treatment, outcome, plan, NuisanceConfig(kind="ridge"))
    cate = dr_learner(features, treatment, outcome, fitted, NuisanceConfig(kind="ridge"))

    assert cate.out_of_fold.size == n
    assert cate.pseudo_outcome.size == n
    constant = np.full(n, float(np.mean(cate.pseudo_outcome)))
    assert pehe(cate.out_of_fold, true_effect) < pehe(constant, true_effect)
    # The refitted model predicts for new rows with the same feature layout.
    assert cate.predict(features[:10]).shape == (10,)


def test_pehe_is_zero_for_a_perfect_prediction_and_rejects_mismatched_lengths() -> None:
    truth = np.array([1.0, -2.0, 3.0])
    assert pehe(truth, truth) == pytest.approx(0.0)
    assert pehe(truth + 2.0, truth) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="same length"):
        pehe(truth, truth[:2])
