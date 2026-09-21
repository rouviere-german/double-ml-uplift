"""Average and conditional treatment effect estimators.

Five ATE estimators are implemented so that the failures of the weak ones are visible next to
the successes of the doubly robust ones, and one CATE learner:

``naive``
    Difference in observed means. Unbiased only under randomisation; it is here to show the
    confounding bias, not to be used.
``outcome_regression``
    ``mean(mu1(X) - mu0(X))`` from cross-fitted arm regressions (a "T-learner" plug-in). Its
    reported standard error deliberately ignores nuisance estimation error, which is why its
    intervals under-cover when the outcome model is misspecified.
``ipw``
    Self-normalised (Hajek) inverse-probability weighting with the propensity clipped away from
    0 and 1. Consistent when the propensity is correct, high variance under weak overlap.
``aipw``
    The augmented IPW / doubly robust estimating equation
    ``psi = mu1 - mu0 + T (Y - mu1) / e - (1 - T)(Y - mu0) / (1 - e)`` (Robins, Rotnitzky and
    Zhao, JASA 89, 1994). Consistent if *either* nuisance is right, and its influence function is
    exactly ``psi - theta``, so the standard error needs no bootstrap.
``dml``
    Partialling-out (partially linear) double machine learning: regress the residual
    ``Y - E[Y|X]`` on the residual ``T - E[T|X]`` (Chernozhukov et al., 2018; Robinson,
    *Econometrica* 56, 1988). Reported with an influence-function standard error and, optionally,
    a multiplier bootstrap interval.
``dr_learner``
    Kennedy's doubly robust learner (Kennedy, "Towards optimal doubly robust estimation of
    heterogeneous causal effects", 2020/2023): regress the AIPW pseudo-outcome on the covariates,
    cross-fitting both stages, to obtain ``tau(x)``.

All of them take the *out-of-fold* nuisance predictions produced by
:mod:`double_ml_uplift.causal.nuisance`; none of them refits anything on the rows it scores.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from double_ml_uplift.causal.folds import FoldPlan
from double_ml_uplift.causal.nuisance import (
    CrossFitted,
    NuisanceConfig,
    Regressor,
    make_regressor,
)

__all__ = [
    "ATE_ESTIMATORS",
    "AteEstimate",
    "CateFit",
    "aipw",
    "aipw_pseudo_outcome",
    "bootstrap_interval",
    "clipped_ipw",
    "dml_partialling_out",
    "dr_learner",
    "estimate_all",
    "naive_difference",
    "outcome_regression",
    "pehe",
    "summarize",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
CiMethod = Literal["analytic", "influence", "bootstrap"]

ATE_ESTIMATORS: Final = ("naive", "outcome_regression", "ipw", "aipw", "dml")
DEFAULT_LEVEL: Final = 0.95
MAX_BOOTSTRAP: Final = 5_000


@dataclass(frozen=True, slots=True)
class AteEstimate:
    """A point estimate with an interval and the machinery that produced it."""

    estimator: str
    ate: float
    se: float
    ci_low: float
    ci_high: float
    ci_method: CiMethod
    n: int
    level: float = DEFAULT_LEVEL
    bootstrap_ci: tuple[float, float] | None = None
    influence: FloatArray | None = field(default=None, repr=False, compare=False)

    def covers(self, value: float) -> bool:
        """Whether the reported interval contains ``value`` (used to measure coverage)."""
        return bool(self.ci_low <= value <= self.ci_high)

    def significant(self) -> bool:
        """Whether the interval excludes zero."""
        return not self.covers(0.0)

    def to_dict(self) -> dict[str, float | str | int | list[float] | None]:
        return {
            "estimator": self.estimator,
            "ate": self.ate,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_method": self.ci_method,
            "n": self.n,
            "level": self.level,
            "bootstrap_ci": list(self.bootstrap_ci) if self.bootstrap_ci else None,
        }


def _z(level: float) -> float:
    if not 0.5 < level < 1.0:
        raise ValueError("level must lie in (0.5, 1)")
    return float(norm.ppf(0.5 + level / 2.0))


def _from_influence(
    name: str, point: float, influence: FloatArray, level: float, *, method: CiMethod = "influence"
) -> AteEstimate:
    n = int(influence.size)
    se = float(np.std(influence, ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    half = _z(level) * se
    return AteEstimate(
        estimator=name,
        ate=float(point),
        se=se,
        ci_low=float(point - half),
        ci_high=float(point + half),
        ci_method=method,
        n=n,
        level=level,
        influence=influence,
    )


def _check(treatment: IntArray, outcome: FloatArray) -> None:
    if treatment.size != outcome.size:
        raise ValueError("treatment and outcome must have the same length")
    if treatment.size < 4:
        raise ValueError("at least four rows are required")
    arms = set(np.unique(treatment).tolist())
    if arms != {0, 1}:
        raise ValueError("treatment must contain both a treated and an untreated row")


def naive_difference(
    treatment: IntArray, outcome: FloatArray, *, level: float = DEFAULT_LEVEL
) -> AteEstimate:
    """Difference in observed means, with a Welch standard error. Biased under confounding."""
    _check(treatment, outcome)
    treated, control = outcome[treatment == 1], outcome[treatment == 0]
    if treated.size < 2 or control.size < 2:
        raise ValueError("each arm needs at least two rows")
    point = float(treated.mean() - control.mean())
    se = float(np.sqrt(treated.var(ddof=1) / treated.size + control.var(ddof=1) / control.size))
    half = _z(level) * se
    return AteEstimate(
        estimator="naive",
        ate=point,
        se=se,
        ci_low=point - half,
        ci_high=point + half,
        ci_method="analytic",
        n=int(outcome.size),
        level=level,
    )


def outcome_regression(fitted: CrossFitted, *, level: float = DEFAULT_LEVEL) -> AteEstimate:
    """Plug-in ``mean(mu1 - mu0)``.

    The standard error treats the fitted arm regressions as if they were known functions. That is
    wrong, and knowingly so: it is the interval a naive "predict both arms and subtract" pipeline
    would report, and the Monte Carlo bench measures how badly it under-covers.
    """
    difference = np.asarray(fitted.mu1 - fitted.mu0, dtype=np.float64)
    point = float(difference.mean())
    return _from_influence("outcome_regression", point, difference - point, level)


def clipped_ipw(
    treatment: IntArray,
    outcome: FloatArray,
    fitted: CrossFitted,
    *,
    clip: float = 0.01,
    level: float = DEFAULT_LEVEL,
) -> AteEstimate:
    """Self-normalised inverse-probability weighting with clipped propensities."""
    _check(treatment, outcome)
    propensity = fitted.clipped_propensity(clip)
    treated_weight = np.where(treatment == 1, 1.0 / propensity, 0.0)
    control_weight = np.where(treatment == 0, 1.0 / (1.0 - propensity), 0.0)
    mean_treated_weight = float(treated_weight.mean())
    mean_control_weight = float(control_weight.mean())
    if mean_treated_weight <= 0.0 or mean_control_weight <= 0.0:
        raise ValueError("inverse-probability weights vanished in one arm")
    mu1 = float((treated_weight * outcome).sum() / treated_weight.sum())
    mu0 = float((control_weight * outcome).sum() / control_weight.sum())
    point = mu1 - mu0
    influence = (
        treated_weight * (outcome - mu1) / mean_treated_weight
        - control_weight * (outcome - mu0) / mean_control_weight
    )
    return _from_influence("ipw", point, np.asarray(influence, dtype=np.float64), level)


def aipw_pseudo_outcome(
    treatment: IntArray, outcome: FloatArray, fitted: CrossFitted, *, clip: float = 0.01
) -> FloatArray:
    """The doubly robust pseudo-outcome whose mean is the AIPW estimate of the ATE."""
    _check(treatment, outcome)
    propensity = fitted.clipped_propensity(clip)
    treated = np.asarray(treatment, dtype=np.float64)
    correction = treated * (outcome - fitted.mu1) / propensity
    correction = correction - (1.0 - treated) * (outcome - fitted.mu0) / (1.0 - propensity)
    return np.asarray(fitted.mu1 - fitted.mu0 + correction, dtype=np.float64)


def aipw(
    treatment: IntArray,
    outcome: FloatArray,
    fitted: CrossFitted,
    *,
    clip: float = 0.01,
    level: float = DEFAULT_LEVEL,
) -> AteEstimate:
    """Augmented IPW. Consistent when either the propensity or the outcome model is correct."""
    pseudo = aipw_pseudo_outcome(treatment, outcome, fitted, clip=clip)
    point = float(pseudo.mean())
    return _from_influence("aipw", point, pseudo - point, level)


def dml_partialling_out(
    treatment: IntArray,
    outcome: FloatArray,
    fitted: CrossFitted,
    *,
    level: float = DEFAULT_LEVEL,
    n_bootstrap: int = 0,
    seed: int = 0,
) -> AteEstimate:
    """Partially linear DML: regress the outcome residual on the treatment residual."""
    _check(treatment, outcome)
    treatment_residual = np.asarray(treatment, dtype=np.float64) - fitted.propensity
    outcome_residual = np.asarray(outcome, dtype=np.float64) - fitted.outcome_hat
    denominator = float(np.mean(treatment_residual**2))
    if denominator <= 1e-12:
        raise ValueError("the treatment residual has no variation; theta is not identified")
    point = float(np.mean(treatment_residual * outcome_residual) / denominator)
    influence = treatment_residual * (outcome_residual - point * treatment_residual) / denominator
    estimate = _from_influence("dml", point, np.asarray(influence, dtype=np.float64), level)
    if n_bootstrap > 0:
        low, high = bootstrap_interval(estimate, n_bootstrap=n_bootstrap, seed=seed, level=level)
        estimate = AteEstimate(
            estimator=estimate.estimator,
            ate=estimate.ate,
            se=estimate.se,
            ci_low=estimate.ci_low,
            ci_high=estimate.ci_high,
            ci_method=estimate.ci_method,
            n=estimate.n,
            level=estimate.level,
            bootstrap_ci=(low, high),
            influence=estimate.influence,
        )
    return estimate


def bootstrap_interval(
    estimate: AteEstimate,
    *,
    n_bootstrap: int = 500,
    seed: int = 0,
    level: float = DEFAULT_LEVEL,
) -> tuple[float, float]:
    """Multiplier-bootstrap percentile interval built from the estimate's influence function.

    Draws ``xi ~ Exponential(1)`` weights and forms ``theta_b = theta + mean((xi - 1) * psi)``.
    This is the Bayesian/weighted bootstrap of Chernozhukov et al. (2018, section 3.2): it reuses
    the fitted nuisances, so it costs one pass over the influence values per replication instead
    of refitting the learners.
    """
    if estimate.influence is None:
        raise ValueError(f"the {estimate.estimator} estimator exposes no influence function")
    if not 1 <= n_bootstrap <= MAX_BOOTSTRAP:
        raise ValueError(f"n_bootstrap must lie in [1, {MAX_BOOTSTRAP}]")
    rng = np.random.default_rng(seed)
    influence = estimate.influence
    weights = rng.exponential(1.0, size=(n_bootstrap, influence.size)) - 1.0
    draws = estimate.ate + weights @ influence / influence.size
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return float(low), float(high)


def estimate_all(
    treatment: IntArray,
    outcome: FloatArray,
    fitted: CrossFitted,
    *,
    clip: float = 0.01,
    level: float = DEFAULT_LEVEL,
    n_bootstrap: int = 0,
    seed: int = 0,
) -> dict[str, AteEstimate]:
    """Run every ATE estimator on one cross-fitted sample."""
    return {
        "naive": naive_difference(treatment, outcome, level=level),
        "outcome_regression": outcome_regression(fitted, level=level),
        "ipw": clipped_ipw(treatment, outcome, fitted, clip=clip, level=level),
        "aipw": aipw(treatment, outcome, fitted, clip=clip, level=level),
        "dml": dml_partialling_out(
            treatment, outcome, fitted, level=level, n_bootstrap=n_bootstrap, seed=seed
        ),
    }


@dataclass(frozen=True, slots=True)
class CateFit:
    """A fitted conditional-effect model with honest out-of-fold predictions."""

    out_of_fold: FloatArray
    """Second-stage predictions for the training rows, produced without seeing their own fold."""

    pseudo_outcome: FloatArray
    model: Regressor = field(repr=False, compare=False)
    plan: FoldPlan = field(repr=False, compare=False)

    def predict(self, features: FloatArray) -> FloatArray:
        """Predict ``tau(x)`` for new rows with the model fitted on every pseudo-outcome."""
        return np.asarray(self.model.predict(features), dtype=np.float64)


def dr_learner(
    features: FloatArray,
    treatment: IntArray,
    outcome: FloatArray,
    fitted: CrossFitted,
    config: NuisanceConfig,
    *,
    clip: float = 0.01,
) -> CateFit:
    """Fit the doubly robust learner: cross-fitted AIPW pseudo-outcomes, regressed on ``X``."""
    pseudo = aipw_pseudo_outcome(treatment, outcome, fitted, clip=clip)
    plan = fitted.plan
    out_of_fold = np.empty(pseudo.size, dtype=np.float64)
    for fold in range(plan.n_folds):
        train, test = plan.indices(fold)
        stage_two = make_regressor(config)
        stage_two.fit(features[train], pseudo[train])
        out_of_fold[test] = np.asarray(stage_two.predict(features[test]), dtype=np.float64)
    final = make_regressor(config)
    final.fit(features, pseudo)
    return CateFit(out_of_fold=out_of_fold, pseudo_outcome=pseudo, model=final, plan=plan)


def pehe(predicted: FloatArray, truth: FloatArray) -> float:
    """Root mean squared error of conditional-effect predictions (the usual PEHE)."""
    if predicted.size != truth.size:
        raise ValueError("predicted and true effects must have the same length")
    return float(np.sqrt(np.mean((predicted - truth) ** 2)))


def summarize(estimates: Mapping[str, AteEstimate]) -> dict[str, dict[str, float | str | int]]:
    """JSON-friendly view of a set of estimates (used by the CLI and the service)."""
    return {
        name: {
            key: value
            for key, value in estimate.to_dict().items()
            if value is not None and not isinstance(value, list)
        }
        for name, estimate in estimates.items()
    }
