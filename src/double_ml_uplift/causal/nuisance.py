"""Nuisance learners and K-fold cross-fitting.

Double machine learning needs two nuisance functions: the propensity ``e(x) = P(T = 1 | X = x)``
and the outcome regressions ``mu_t(x) = E[Y | T = t, X = x]`` (plus the pooled ``m(x) = E[Y | X]``
used by the partialling-out form). Fitting them on the same rows that the effect is estimated on
leaves a first-order bias term, so every prediction here is out-of-fold: for each fold ``k`` the
learners see only the other folds (Chernozhukov et al., "Double/debiased machine learning for
treatment and structural parameters", *Econometrics Journal* 21(1), 2018).

Three learner families are available, all CPU-only and all deterministic:

- ``ols``: ordinary least squares for the outcome and a *linear probability model* for the
  treatment. This is the configuration under which partialling-out DML without cross-fitting
  coincides exactly with the OLS treatment coefficient (Frisch-Waugh-Lovell), which is what
  ``tests/unit/test_estimators.py::test_dml_without_crossfitting_equals_the_ols_coefficient``
  checks to 1e-8.
- ``ridge``: standardised ridge regression and L2 logistic regression.
- ``gbm``: LightGBM, for the scenarios where the nuisances are genuinely nonlinear.

``n_folds = 1`` deliberately disables cross-fitting (fit and predict on the same rows). It exists
for the FWL differential test and for teaching material; the engine refuses it for real work.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from double_ml_uplift.causal.folds import FoldPlan

__all__ = [
    "Classifier",
    "CrossFitted",
    "LinearProbabilityModel",
    "NuisanceConfig",
    "NuisanceKind",
    "Regressor",
    "cross_fit",
    "fit_arm_models",
    "make_classifier",
    "make_regressor",
    "stack_features",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
NuisanceKind = Literal["ols", "ridge", "gbm"]


class Regressor(Protocol):
    """The slice of the scikit-learn regressor API this package relies on."""

    def fit(self, x: FloatArray, y: FloatArray) -> Any: ...

    def predict(self, x: FloatArray) -> FloatArray: ...


class Classifier(Protocol):
    """The slice of the scikit-learn classifier API this package relies on."""

    def fit(self, x: FloatArray, y: IntArray) -> Any: ...

    def predict_proba(self, x: FloatArray) -> FloatArray: ...


class NuisanceConfig(BaseModel):
    """Hyper-parameters of the nuisance learners. Hashed into the run manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: NuisanceKind = "gbm"
    n_estimators: int = Field(default=80, ge=10, le=2_000)
    num_leaves: int = Field(default=8, ge=2, le=255)
    learning_rate: float = Field(default=0.12, gt=0.0, le=1.0)
    min_child_samples: int = Field(default=40, ge=1, le=10_000)
    ridge_alpha: float = Field(default=1.0, gt=0.0, le=1e6)
    random_state: int = Field(default=0, ge=0, le=2**31 - 1)


class LinearProbabilityModel:
    """Least-squares regression of a 0/1 treatment, exposing the classifier interface.

    Fitted values are the linear projection of ``T`` on ``[1, X]`` and may fall outside ``[0, 1]``.
    That is intentional: the partialling-out estimator only uses the residual ``T - e(X)``, and
    keeping the projection unclipped is what makes the Frisch-Waugh-Lovell identity exact. The
    overlap gate reports the out-of-range mass, so nothing silently depends on it.
    """

    def __init__(self) -> None:
        from sklearn.linear_model import LinearRegression  # noqa: PLC0415 - heavy optional import

        self._model = LinearRegression()

    @property
    def inner(self) -> Any:
        """The underlying least-squares model, exposed so the bundle can serialize it as JSON."""
        return self._model

    def fit(self, x: FloatArray, y: IntArray) -> LinearProbabilityModel:
        self._model.fit(x, np.asarray(y, dtype=np.float64))
        return self

    def predict_proba(self, x: FloatArray) -> FloatArray:
        positive = np.asarray(self._model.predict(x), dtype=np.float64)
        return np.column_stack([1.0 - positive, positive])


def make_regressor(config: NuisanceConfig) -> Regressor:
    """Build an unfitted outcome learner."""
    if config.kind == "ols":
        from sklearn.linear_model import LinearRegression  # noqa: PLC0415

        return cast("Regressor", LinearRegression())
    if config.kind == "ridge":
        from sklearn.linear_model import Ridge  # noqa: PLC0415
        from sklearn.pipeline import make_pipeline  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        return cast("Regressor", make_pipeline(StandardScaler(), Ridge(alpha=config.ridge_alpha)))
    from lightgbm import LGBMRegressor  # noqa: PLC0415

    return cast(
        "Regressor",
        LGBMRegressor(
            n_estimators=config.n_estimators,
            num_leaves=config.num_leaves,
            learning_rate=config.learning_rate,
            min_child_samples=config.min_child_samples,
            random_state=config.random_state,
            n_jobs=1,
            verbose=-1,
            deterministic=True,
            force_row_wise=True,
        ),
    )


def make_classifier(config: NuisanceConfig) -> Classifier:
    """Build an unfitted treatment (propensity) learner."""
    if config.kind == "ols":
        return LinearProbabilityModel()
    if config.kind == "ridge":
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
        from sklearn.pipeline import make_pipeline  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        return cast(
            "Classifier",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0 / config.ridge_alpha, max_iter=2_000),
            ),
        )
    from lightgbm import LGBMClassifier  # noqa: PLC0415

    return cast(
        "Classifier",
        LGBMClassifier(
            n_estimators=config.n_estimators,
            num_leaves=config.num_leaves,
            learning_rate=config.learning_rate,
            min_child_samples=config.min_child_samples,
            random_state=config.random_state,
            n_jobs=1,
            verbose=-1,
            deterministic=True,
            force_row_wise=True,
        ),
    )


def _positive_probability(model: Classifier, x: FloatArray) -> FloatArray:
    proba = np.asarray(model.predict_proba(x), dtype=np.float64)
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ValueError("the treatment model must expose two-class probabilities")
    return np.asarray(proba[:, 1], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class CrossFitted:
    """Out-of-fold nuisance predictions for every row."""

    propensity: FloatArray
    """Raw out-of-fold P(T = 1 | X). Not clipped; the overlap gate inspects it."""

    mu0: FloatArray
    mu1: FloatArray
    outcome_hat: FloatArray
    """Pooled out-of-fold E[Y | X], used by the partialling-out estimator."""

    plan: FoldPlan
    config: NuisanceConfig

    def __post_init__(self) -> None:
        sizes = {self.propensity.size, self.mu0.size, self.mu1.size, self.outcome_hat.size}
        if len(sizes) != 1:
            raise ValueError("cross-fitted predictions have inconsistent lengths")

    @property
    def n(self) -> int:
        return int(self.propensity.size)

    def clipped_propensity(self, clip: float) -> FloatArray:
        if not 0.0 < clip < 0.5:
            raise ValueError("clip must lie in (0, 0.5)")
        return np.clip(self.propensity, clip, 1.0 - clip)


def _check_inputs(features: FloatArray, treatment: IntArray, outcome: FloatArray) -> None:
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if not (features.shape[0] == treatment.size == outcome.size):
        raise ValueError("features, treatment and outcome must have the same number of rows")
    if features.shape[0] < 4:
        raise ValueError("at least four rows are required")
    unique = set(np.unique(treatment).tolist())
    if not unique <= {0, 1}:
        raise ValueError("treatment must be coded 0/1")
    if unique != {0, 1}:
        raise ValueError("treatment must contain both a treated and an untreated row")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(outcome)):
        raise ValueError("features and outcome must be finite")


def fit_arm_models(
    features: FloatArray,
    treatment: IntArray,
    outcome: FloatArray,
    config: NuisanceConfig,
) -> tuple[Classifier, Regressor, Regressor, Regressor]:
    """Fit the propensity, both arm regressions and the pooled outcome regression on all rows."""
    _check_inputs(features, treatment, outcome)
    propensity_model = make_classifier(config)
    propensity_model.fit(features, treatment)
    mu0_model = make_regressor(config)
    mu0_model.fit(features[treatment == 0], outcome[treatment == 0])
    mu1_model = make_regressor(config)
    mu1_model.fit(features[treatment == 1], outcome[treatment == 1])
    pooled = make_regressor(config)
    pooled.fit(features, outcome)
    return propensity_model, mu0_model, mu1_model, pooled


def cross_fit(
    features: FloatArray,
    treatment: IntArray,
    outcome: FloatArray,
    plan: FoldPlan,
    config: NuisanceConfig,
) -> CrossFitted:
    """Fit the nuisances fold by fold and return out-of-fold predictions for every row."""
    _check_inputs(features, treatment, outcome)
    if plan.folds.size != treatment.size:
        raise ValueError("the fold plan does not cover the same rows as the data")

    n = treatment.size
    propensity = np.empty(n, dtype=np.float64)
    mu0 = np.empty(n, dtype=np.float64)
    mu1 = np.empty(n, dtype=np.float64)
    outcome_hat = np.empty(n, dtype=np.float64)

    for fold in range(plan.n_folds):
        train, test = plan.indices(fold)
        arms = treatment[train]
        if arms.min() == arms.max():
            raise ValueError(f"fold {fold} has a training set with a single treatment arm")
        models = fit_arm_models(features[train], arms, outcome[train], config)
        propensity_model, mu0_model, mu1_model, pooled = models
        test_features = features[test]
        propensity[test] = _positive_probability(propensity_model, test_features)
        mu0[test] = np.asarray(mu0_model.predict(test_features), dtype=np.float64)
        mu1[test] = np.asarray(mu1_model.predict(test_features), dtype=np.float64)
        outcome_hat[test] = np.asarray(pooled.predict(test_features), dtype=np.float64)

    return CrossFitted(
        propensity=propensity,
        mu0=mu0,
        mu1=mu1,
        outcome_hat=outcome_hat,
        plan=plan,
        config=config,
    )


def stack_features(columns: Sequence[FloatArray]) -> FloatArray:
    """Column-stack 1-D feature arrays into the design matrix used by the learners."""
    if not columns:
        raise ValueError("at least one feature column is required")
    return np.column_stack(columns).astype(np.float64, copy=False)
