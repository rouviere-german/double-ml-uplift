"""Cross-fitting must never let a row's own fold inform its prediction."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError

from double_ml_uplift.causal.folds import make_fold_plan
from double_ml_uplift.causal.nuisance import (
    CrossFitted,
    LinearProbabilityModel,
    NuisanceConfig,
    cross_fit,
    fit_arm_models,
    make_classifier,
    make_regressor,
    stack_features,
)

pytestmark = pytest.mark.unit

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def sample(n: int = 400, *, seed: int = 0) -> tuple[FloatArray, IntArray, FloatArray, list[str]]:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    features = stack_features([x1, x2])
    propensity = 1.0 / (1.0 + np.exp(-(0.9 * x1 - 0.4 * x2)))
    treatment = (rng.random(n) < propensity).astype(np.int64)
    outcome = 2.0 + 1.5 * x1 - 0.8 * x2 + 3.0 * treatment + rng.normal(0.0, 1.0, size=n)
    return features, treatment, outcome, [f"u_{index:05d}" for index in range(n)]


# --- learner families ----------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["ols", "ridge", "gbm"])
def test_every_learner_family_fits_and_predicts_the_right_shapes(kind: str) -> None:
    features, treatment, outcome, _ = sample(200, seed=1)
    config = NuisanceConfig(kind=kind, n_estimators=20)
    regressor = make_regressor(config)
    regressor.fit(features, outcome)
    assert np.asarray(regressor.predict(features)).shape == (200,)

    classifier = make_classifier(config)
    classifier.fit(features, treatment)
    proba = np.asarray(classifier.predict_proba(features))
    assert proba.shape == (200, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("kind", ["ols", "ridge", "gbm"])
def test_learners_are_deterministic(kind: str) -> None:
    features, _, outcome, _ = sample(150, seed=2)
    config = NuisanceConfig(kind=kind, n_estimators=20)
    first, second = make_regressor(config), make_regressor(config)
    first.fit(features, outcome)
    second.fit(features, outcome)
    assert np.allclose(np.asarray(first.predict(features)), np.asarray(second.predict(features)))


def test_the_linear_probability_model_is_an_unclipped_projection() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 1)) * 4.0
    treatment = (x[:, 0] > 0).astype(np.int64)
    model = LinearProbabilityModel().fit(x, treatment)
    proba = model.predict_proba(x)
    assert proba.shape == (300, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    # It is a least-squares projection, so some fitted values fall outside [0, 1] on purpose.
    assert proba[:, 1].min() < 0.0 or proba[:, 1].max() > 1.0
    assert model.inner.coef_.shape == (1,)


def test_a_classifier_that_does_not_return_two_columns_is_refused() -> None:
    class Broken:
        def fit(self, x: FloatArray, y: IntArray) -> Broken:
            return self

        def predict_proba(self, x: FloatArray) -> FloatArray:
            return np.zeros((x.shape[0], 3))

    from double_ml_uplift.causal.nuisance import _positive_probability

    with pytest.raises(ValueError, match="two-class probabilities"):
        _positive_probability(Broken(), np.zeros((4, 2)))


# --- cross-fitting -------------------------------------------------------------------------


def test_every_prediction_comes_from_a_model_that_never_saw_its_own_row() -> None:
    """The definitive cross-fitting check: refit each fold by hand and compare."""
    features, treatment, outcome, unit_ids = sample(500, seed=4)
    plan = make_fold_plan(unit_ids, 5, salt="xf")
    config = NuisanceConfig(kind="ridge")
    fitted = cross_fit(features, treatment, outcome, plan, config)

    for fold in range(plan.n_folds):
        train, test = plan.indices(fold)
        propensity_model, mu0_model, mu1_model, pooled = fit_arm_models(
            features[train], treatment[train], outcome[train], config
        )
        expected_e = np.asarray(propensity_model.predict_proba(features[test]))[:, 1]
        assert np.allclose(fitted.propensity[test], expected_e)
        assert np.allclose(fitted.mu0[test], np.asarray(mu0_model.predict(features[test])))
        assert np.allclose(fitted.mu1[test], np.asarray(mu1_model.predict(features[test])))
        assert np.allclose(fitted.outcome_hat[test], np.asarray(pooled.predict(features[test])))


def test_cross_fitting_is_reproducible_and_covers_every_row() -> None:
    features, treatment, outcome, unit_ids = sample(300, seed=5)
    plan = make_fold_plan(unit_ids, 4, salt="xf")
    config = NuisanceConfig(kind="ridge")
    first = cross_fit(features, treatment, outcome, plan, config)
    second = cross_fit(features, treatment, outcome, plan, config)
    assert np.allclose(first.propensity, second.propensity)
    assert first.n == 300
    assert np.all(np.isfinite(first.propensity))
    assert np.all(np.isfinite(first.outcome_hat))


def test_out_of_fold_predictions_differ_from_in_sample_ones() -> None:
    features, treatment, outcome, unit_ids = sample(400, seed=6)
    config = NuisanceConfig(kind="gbm", n_estimators=40)
    out_of_fold = cross_fit(
        features, treatment, outcome, make_fold_plan(unit_ids, 5, salt="a"), config
    )
    in_sample = cross_fit(
        features, treatment, outcome, make_fold_plan(unit_ids, 1, salt="a"), config
    )
    residual_out = float(np.mean((outcome - out_of_fold.outcome_hat) ** 2))
    residual_in = float(np.mean((outcome - in_sample.outcome_hat) ** 2))
    # Fitting and predicting on the same rows always looks better. That gap is the bias
    # cross-fitting exists to remove.
    assert residual_in < residual_out


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda f, t, y: (f[:, 0], t, y), "two-dimensional"),
        (lambda f, t, y: (f[:10], t, y), "same number of rows"),
        (lambda f, t, y: (f[:3], t[:3], y[:3]), "at least four rows"),
        (lambda f, t, y: (f, np.full(t.size, 2, dtype=np.int64), y), "coded 0/1"),
        (lambda f, t, y: (f, np.ones(t.size, dtype=np.int64), y), "both a treated"),
    ],
)
def test_cross_fitting_inputs_are_validated(mutate: object, message: str) -> None:
    features, treatment, outcome, unit_ids = sample(40, seed=7)
    bad_features, bad_treatment, bad_outcome = mutate(features, treatment, outcome)  # type: ignore[operator]
    plan = make_fold_plan(unit_ids, 2, salt="v")
    with pytest.raises(ValueError, match=message):
        cross_fit(bad_features, bad_treatment, bad_outcome, plan, NuisanceConfig(kind="ols"))


def test_non_finite_inputs_are_refused() -> None:
    features, treatment, outcome, unit_ids = sample(40, seed=8)
    plan = make_fold_plan(unit_ids, 2, salt="v")
    broken = outcome.copy()
    broken[0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        cross_fit(features, treatment, broken, plan, NuisanceConfig(kind="ols"))


def test_a_fold_plan_for_different_rows_is_refused() -> None:
    features, treatment, outcome, unit_ids = sample(40, seed=9)
    plan = make_fold_plan(unit_ids[:20], 2, salt="v")
    with pytest.raises(ValueError, match="does not cover the same rows"):
        cross_fit(features, treatment, outcome, plan, NuisanceConfig(kind="ols"))


def test_a_training_fold_with_a_single_arm_is_refused() -> None:
    n = 40
    features = np.zeros((n, 2), dtype=np.float64)
    outcome = np.arange(n, dtype=np.float64)
    unit_ids = [f"u_{index:05d}" for index in range(n)]
    plan = make_fold_plan(unit_ids, 2, salt="arm")
    treatment = np.zeros(n, dtype=np.int64)
    # Every treated row lands in fold 0, so fold 0's training set is all-control.
    treatment[plan.folds == 0] = 1
    with pytest.raises(ValueError, match="single treatment arm"):
        cross_fit(features, treatment, outcome, plan, NuisanceConfig(kind="ols"))


# --- the container -------------------------------------------------------------------------


def test_cross_fitted_rejects_inconsistent_lengths_and_clips_on_request() -> None:
    plan = make_fold_plan([f"u_{index}" for index in range(6)], 2, salt="c")
    config = NuisanceConfig(kind="ols")
    with pytest.raises(ValueError, match="inconsistent lengths"):
        CrossFitted(
            propensity=np.zeros(6),
            mu0=np.zeros(5),
            mu1=np.zeros(6),
            outcome_hat=np.zeros(6),
            plan=plan,
            config=config,
        )
    fitted = CrossFitted(
        propensity=np.array([0.001, 0.5, 0.999, 0.2, 0.8, 0.4]),
        mu0=np.zeros(6),
        mu1=np.zeros(6),
        outcome_hat=np.zeros(6),
        plan=plan,
        config=config,
    )
    clipped = fitted.clipped_propensity(0.01)
    assert clipped.min() == pytest.approx(0.01)
    assert clipped.max() == pytest.approx(0.99)


def test_stack_features_needs_at_least_one_column() -> None:
    with pytest.raises(ValueError, match="at least one feature column"):
        stack_features([])
    assert stack_features([np.zeros(3), np.ones(3)]).shape == (3, 2)


def test_nuisance_configuration_bounds_are_enforced() -> None:
    with pytest.raises(ValidationError, match="kind"):
        NuisanceConfig(kind="nope")
    with pytest.raises(ValidationError, match="num_leaves"):
        NuisanceConfig(num_leaves=1)
    with pytest.raises(ValidationError, match="learning_rate"):
        NuisanceConfig(learning_rate=0.0)
    assert NuisanceConfig().kind == "gbm"
