"""Overlap and balance diagnostics, and the gate that turns them into a refusal.

Backdoor adjustment needs positivity: every customer must have had a non-degenerate chance of
receiving and of not receiving the offer. When the historical targeting rule was close to
deterministic, inverse-probability weights explode, the doubly robust correction term blows up,
and every estimator reports a confident number that is mostly extrapolation.

This module measures that instead of hoping:

* the mass of estimated propensities outside ``[low, high]``;
* the standardised mean difference (SMD) of every covariate before and after weighting, the usual
  balance check for weighting estimators (Austin, "Balance diagnostics for comparing the
  distribution of baseline covariates", *Statistics in Medicine* 28, 2009);
* the Kish effective sample size of each weighted arm.

:func:`apply_gate` turns the report into an "estimable / not estimable" decision with reasons.
The engine treats a refusal as a hard failure: no point estimate is returned at all. At scoring
time, :func:`abstain_mask` marks individual customers whose propensity falls outside the same
bounds, and the service returns an abstention for them rather than a number.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "BalanceStat",
    "GateDecision",
    "GatePolicy",
    "NotEstimableError",
    "OverlapReport",
    "abstain_mask",
    "apply_gate",
    "overlap_report",
    "standardized_mean_difference",
    "trim_mask",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


class GatePolicy(BaseModel):
    """Thresholds at which the diagnostics stop being a warning and become a refusal.

    The overlap region must bracket ``0.5``: a gate that excluded the coin flip would refuse a
    randomised experiment. The field bounds encode exactly that, and they also make
    ``propensity_low < propensity_high`` true by construction, so no cross-field check is needed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    propensity_low: float = Field(default=0.05, gt=0.0, lt=0.5)
    propensity_high: float = Field(default=0.95, gt=0.5, lt=1.0)
    max_outside_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    max_abs_smd: float = Field(default=0.20, gt=0.0, le=2.0)
    min_effective_sample: float = Field(default=30.0, ge=1.0)


@dataclass(frozen=True, slots=True)
class BalanceStat:
    feature: str
    smd_before: float
    smd_after: float


@dataclass(frozen=True, slots=True)
class OverlapReport:
    """What the diagnostics measured. Stored with every estimate, refused or not."""

    n: int
    n_treated: int
    propensity_min: float
    propensity_max: float
    fraction_outside: float
    bounds: tuple[float, float]
    effective_sample_treated: float
    effective_sample_control: float
    balance: tuple[BalanceStat, ...]

    @property
    def max_abs_smd_before(self) -> float:
        return max((abs(item.smd_before) for item in self.balance), default=0.0)

    @property
    def max_abs_smd_after(self) -> float:
        return max((abs(item.smd_after) for item in self.balance), default=0.0)

    def worst_balanced_feature(self) -> str | None:
        if not self.balance:
            return None
        return max(self.balance, key=lambda item: abs(item.smd_after)).feature

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "n_treated": self.n_treated,
            "propensity_min": round(self.propensity_min, 6),
            "propensity_max": round(self.propensity_max, 6),
            "fraction_outside": round(self.fraction_outside, 6),
            "bounds": list(self.bounds),
            "effective_sample_treated": round(self.effective_sample_treated, 3),
            "effective_sample_control": round(self.effective_sample_control, 3),
            "max_abs_smd_before": round(self.max_abs_smd_before, 6),
            "max_abs_smd_after": round(self.max_abs_smd_after, 6),
            "worst_balanced_feature": self.worst_balanced_feature(),
        }


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Whether an estimate may be reported, and if not, exactly why."""

    estimable: bool
    reasons: tuple[str, ...]
    report: OverlapReport

    def require(self) -> None:
        if not self.estimable:
            raise NotEstimableError(self.reasons, self.report)


class NotEstimableError(ValueError):
    """Diagnostics failed, so no effect estimate is reported."""

    def __init__(self, reasons: Sequence[str], report: OverlapReport | None = None) -> None:
        super().__init__("; ".join(reasons) or "the effect is not estimable")
        self.reasons = tuple(reasons)
        self.report = report


def standardized_mean_difference(
    values: FloatArray,
    treatment: IntArray,
    weights: FloatArray | None = None,
) -> float:
    """SMD of one covariate between arms, optionally weighted.

    The denominator always uses the *unweighted* pooled within-arm standard deviation, so the
    before/after numbers are on the same scale and a weighting scheme cannot flatter itself by
    shrinking its own denominator.
    """
    treated = values[treatment == 1]
    control = values[treatment == 0]
    if treated.size == 0 or control.size == 0:
        raise ValueError("both arms must be non-empty")
    pooled = np.sqrt((treated.var(ddof=0) + control.var(ddof=0)) / 2.0)
    if pooled <= 1e-12:
        return 0.0
    if weights is None:
        difference = float(treated.mean() - control.mean())
    else:
        treated_weights = weights[treatment == 1]
        control_weights = weights[treatment == 0]
        if treated_weights.sum() <= 0 or control_weights.sum() <= 0:
            raise ValueError("weights vanished in one arm")
        difference = float(
            np.average(treated, weights=treated_weights)
            - np.average(control, weights=control_weights)
        )
    return float(difference / pooled)


def _effective_sample_size(weights: FloatArray) -> float:
    total = float(weights.sum())
    squared = float((weights**2).sum())
    return 0.0 if squared <= 0.0 else total * total / squared


def overlap_report(
    features: FloatArray,
    feature_names: Sequence[str],
    treatment: IntArray,
    propensity: FloatArray,
    policy: GatePolicy,
) -> OverlapReport:
    """Measure overlap and covariate balance before and after inverse-probability weighting."""
    if features.ndim != 2 or features.shape[1] != len(feature_names):
        raise ValueError("features and feature_names disagree")
    if not (features.shape[0] == treatment.size == propensity.size):
        raise ValueError("features, treatment and propensity must have the same length")
    clipped = np.clip(propensity, policy.propensity_low, policy.propensity_high)
    weights = np.where(treatment == 1, 1.0 / clipped, 1.0 / (1.0 - clipped))
    outside = np.mean((propensity < policy.propensity_low) | (propensity > policy.propensity_high))
    balance = tuple(
        BalanceStat(
            feature=name,
            smd_before=standardized_mean_difference(features[:, index], treatment),
            smd_after=standardized_mean_difference(features[:, index], treatment, weights),
        )
        for index, name in enumerate(feature_names)
    )
    return OverlapReport(
        n=int(treatment.size),
        n_treated=int((treatment == 1).sum()),
        propensity_min=float(propensity.min()),
        propensity_max=float(propensity.max()),
        fraction_outside=float(outside),
        bounds=(policy.propensity_low, policy.propensity_high),
        effective_sample_treated=_effective_sample_size(weights[treatment == 1]),
        effective_sample_control=_effective_sample_size(weights[treatment == 0]),
        balance=balance,
    )


def apply_gate(report: OverlapReport, policy: GatePolicy) -> GateDecision:
    """Turn a report into a decision. Every failing threshold contributes its own reason."""
    reasons: list[str] = []
    if report.fraction_outside > policy.max_outside_fraction:
        reasons.append(
            f"positivity: {report.fraction_outside:.1%} of estimated propensities fall outside "
            f"[{policy.propensity_low:g}, {policy.propensity_high:g}] "
            f"(limit {policy.max_outside_fraction:.1%})"
        )
    if report.max_abs_smd_after > policy.max_abs_smd:
        worst = max(report.balance, key=lambda item: abs(item.smd_after))
        reasons.append(
            f"balance: after weighting, {worst.feature} still has an absolute standardized mean "
            f"difference of {abs(worst.smd_after):.3f} (limit {policy.max_abs_smd:g})"
        )
    for label, value in (
        ("treated", report.effective_sample_treated),
        ("control", report.effective_sample_control),
    ):
        if value < policy.min_effective_sample:
            reasons.append(
                f"effective sample: the weighted {label} arm is worth {value:.1f} rows "
                f"(minimum {policy.min_effective_sample:g})"
            )
    return GateDecision(estimable=not reasons, reasons=tuple(reasons), report=report)


def trim_mask(propensity: FloatArray, policy: GatePolicy) -> BoolArray:
    """Rows kept by trimming: those inside the overlap region."""
    inside = (propensity >= policy.propensity_low) & (propensity <= policy.propensity_high)
    return np.asarray(inside, dtype=np.bool_)


def abstain_mask(propensity: FloatArray, policy: GatePolicy) -> BoolArray:
    """Rows a scoring request must abstain on: those outside the overlap region."""
    return np.asarray(~trim_mask(propensity, policy), dtype=np.bool_)
