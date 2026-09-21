"""Refuters: cheap attempts to make the estimate fail before anyone acts on it.

None of these can prove an estimate right. Each one states a consequence the estimate should
have if the assumptions hold, and reports when that consequence does not hold:

``placebo_treatment``
    Permute the treatment assignment. The true effect of a permuted treatment is zero, so an
    interval that excludes zero indicates a broken pipeline (leakage, a mis-specified residual,
    a fold that saw its own rows). With a correct 95% interval this flags 5% of the time by
    construction, which is what the bench measures.
``random_common_cause``
    Add an independent random covariate to the adjustment set. The estimate should barely move;
    a large shift means the estimator is unstable in the number of covariates.
``subset_stability``
    Re-estimate on random subsets. Subset estimates are nested in the full sample, so the
    difference is compared against ``sqrt(se_subset^2 - se_full^2)``, the standard error of the
    difference under nesting, rather than against the subset standard error alone.
``pilot_consistency``
    Compare the observational estimate against the difference in means on the randomized pilot
    slice. This is the only refuter here that can detect *unobserved* confounding, because the
    pilot estimate does not rely on the unconfoundedness assumption at all. The two estimates
    share rows, so treating them as independent slightly overstates the standard error of their
    difference; the check is therefore conservative.
``e_value``
    Sensitivity analysis rather than a test: the minimum strength of association (on the risk
    ratio scale) that an unmeasured confounder would need with both treatment and outcome to
    explain the estimate away (VanderWeele and Ding, "Sensitivity analysis in observational
    research: introducing the E-value", *Annals of Internal Medicine* 167, 2017). Continuous
    outcomes are converted with the approximation ``RR ~ exp(0.91 * d)`` for a standardised mean
    difference ``d``, as recommended in the same paper.

A refuter has three outcomes, not two. ``passed`` means it ran and found nothing, ``failed``
means it ran and contradicted the estimate, and ``skipped`` means it could not run at all -- the
randomized slice is too thin, for instance. Collapsing ``skipped`` into ``passed`` would be fail
open, which is exactly what ADR 0002 forbids, so ``RefutationReport.passed`` is true only when
every *blocking* refuter actually ran and passed, and the names of the ones that did not run are
reported separately in ``skipped``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from double_ml_uplift.causal.estimators import AteEstimate, naive_difference

__all__ = [
    "EValue",
    "Refitter",
    "RefutationPolicy",
    "RefutationReport",
    "RefutationResult",
    "Severity",
    "e_value",
    "pilot_consistency",
    "placebo_treatment",
    "random_common_cause",
    "refute_all",
    "subset_stability",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class Refitter(Protocol):
    """Re-runs the primary estimator on a (possibly perturbed) sample."""

    def __call__(
        self,
        features: FloatArray,
        treatment: IntArray,
        outcome: FloatArray,
        unit_ids: Sequence[str],
    ) -> AteEstimate: ...


class RefutationPolicy(BaseModel):
    """Thresholds at which each refuter raises a flag."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    random_common_cause_z: float = Field(default=1.5, gt=0.0, le=10.0)
    subset_stability_z: float = Field(default=3.0, gt=0.0, le=10.0)
    pilot_consistency_z: float = Field(default=2.5, gt=0.0, le=10.0)
    min_e_value: float = Field(default=1.25, ge=1.0, le=100.0)
    subset_fraction: float = Field(default=0.7, ge=0.2, lt=1.0)
    n_subsets: int = Field(default=3, ge=1, le=20)
    min_pilot_per_arm: int = Field(default=25, ge=2, le=100_000)


Severity = Literal["blocking", "advisory"]
Status = Literal["passed", "failed", "skipped"]


@dataclass(frozen=True, slots=True)
class RefutationResult:
    """One refuter's verdict.

    ``severity`` separates a broken estimate from a fragile one. A *blocking* refuter that fails
    says the estimate contradicts something it should imply, so the pipeline should not be
    trusted. The E-value is *advisory*: a small E-value does not mean the estimate is wrong, only
    that a modest unmeasured confounder would suffice to explain it away, which is the normal
    situation for a small standardised effect on a continuous outcome.
    """

    name: str
    status: Status
    statistic: float
    threshold: float
    detail: str
    severity: Severity = "blocking"

    @property
    def passed(self) -> bool:
        """True only when the refuter ran and found nothing. A skipped refuter is not a pass."""
        return self.status == "passed"

    @property
    def failed(self) -> bool:
        """True only when the refuter ran and contradicted the estimate."""
        return self.status == "failed"

    @classmethod
    def verdict(
        cls,
        name: str,
        *,
        ok: bool,
        statistic: float,
        threshold: float,
        detail: str,
        severity: Severity = "blocking",
    ) -> RefutationResult:
        """Build the result of a refuter that actually ran."""
        return cls(
            name=name,
            status="passed" if ok else "failed",
            statistic=statistic,
            threshold=threshold,
            detail=detail,
            severity=severity,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "passed": self.passed,
            "severity": self.severity,
            "statistic": round(self.statistic, 6),
            "threshold": self.threshold,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class EValue:
    """Minimum confounder strength needed to explain away the point estimate and the interval."""

    risk_ratio: float
    point: float
    ci_limit: float

    def to_dict(self) -> dict[str, float]:
        return {
            "risk_ratio": round(self.risk_ratio, 6),
            "point": round(self.point, 4),
            "ci_limit": round(self.ci_limit, 4),
        }


@dataclass(frozen=True, slots=True)
class RefutationReport:
    results: tuple[RefutationResult, ...]
    e_value: EValue

    @property
    def flagged(self) -> tuple[str, ...]:
        """Blocking refuters that ran and failed. Empty means nothing contradicted the estimate."""
        return tuple(
            item.name for item in self.results if item.failed and item.severity == "blocking"
        )

    @property
    def advisories(self) -> tuple[str, ...]:
        return tuple(
            item.name for item in self.results if item.failed and item.severity == "advisory"
        )

    @property
    def skipped(self) -> tuple[str, ...]:
        """Refuters that could not run at all, blocking or advisory."""
        return tuple(item.name for item in self.results if item.status == "skipped")

    @property
    def passed(self) -> bool:
        """Every blocking refuter ran and passed.

        A blocking refuter that could not run leaves the estimate unchecked in the one way that
        refuter existed to check, so it withholds the pass rather than granting it. Callers that
        want to distinguish "contradicted" from "unchecked" read ``flagged`` and ``skipped``.
        """
        blocking_skipped = any(
            item.status == "skipped" and item.severity == "blocking" for item in self.results
        )
        return not self.flagged and not blocking_skipped

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "flagged": list(self.flagged),
            "skipped": list(self.skipped),
            "advisories": list(self.advisories),
            "results": [item.to_dict() for item in self.results],
            "e_value": self.e_value.to_dict(),
        }


# The E-value squares its risk ratio, so the log ratio is capped where the square still fits in
# a float. exp(300) is about 2e130 and its E-value about 4e130, comfortably finite. A capped value
# means "a standardised effect of more than ~330 outcome standard deviations": not a real effect,
# but a number that still serialises as JSON instead of an infinity that does not.
_MAX_LOG_RISK_RATIO: Final = 300.0


def _risk_ratio(difference: float, outcome_sd: float) -> float:
    """``exp(0.91 * d)`` for a standardised mean difference ``d``, capped to stay finite."""
    exponent = 0.91 * difference / outcome_sd
    return math.exp(max(-_MAX_LOG_RISK_RATIO, min(_MAX_LOG_RISK_RATIO, exponent)))


def _e_value_from_risk_ratio(risk_ratio: float) -> float:
    ratio = risk_ratio if risk_ratio >= 1.0 else 1.0 / risk_ratio
    return float(ratio + math.sqrt(ratio * (ratio - 1.0)))


def e_value(estimate: AteEstimate, outcome_sd: float) -> EValue:
    """E-value for a continuous outcome, via the standardised-mean-difference approximation."""
    if outcome_sd <= 0.0:
        raise ValueError("outcome_sd must be positive")
    risk_ratio = _risk_ratio(estimate.ate, outcome_sd)
    point = _e_value_from_risk_ratio(risk_ratio)
    if estimate.covers(0.0):
        limit = 1.0
    else:
        bound = estimate.ci_low if estimate.ate > 0 else estimate.ci_high
        limit = _e_value_from_risk_ratio(_risk_ratio(bound, outcome_sd))
    return EValue(risk_ratio=risk_ratio, point=point, ci_limit=limit)


def placebo_treatment(
    refit: Refitter,
    features: FloatArray,
    treatment: IntArray,
    outcome: FloatArray,
    unit_ids: Sequence[str],
    *,
    seed: int = 0,
) -> RefutationResult:
    """Permute the treatment; a placebo effect should be indistinguishable from zero."""
    rng = np.random.default_rng(seed)
    permuted = np.asarray(rng.permutation(treatment), dtype=np.int64)
    estimate = refit(features, permuted, outcome, unit_ids)
    z = abs(estimate.ate) / estimate.se if estimate.se > 0 else math.inf
    passed = estimate.covers(0.0)
    return RefutationResult.verdict(
        "placebo_treatment",
        ok=passed,
        statistic=float(z),
        threshold=float(abs(estimate.ate - estimate.ci_low) / estimate.se if estimate.se else 0.0),
        detail=(
            f"placebo estimate {estimate.ate:.3f} [{estimate.ci_low:.3f}, {estimate.ci_high:.3f}]"
        ),
    )


def random_common_cause(
    refit: Refitter,
    features: FloatArray,
    treatment: IntArray,
    outcome: FloatArray,
    unit_ids: Sequence[str],
    baseline: AteEstimate,
    policy: RefutationPolicy,
    *,
    seed: int = 0,
) -> RefutationResult:
    """Add an independent covariate; the estimate should move by much less than its own noise."""
    rng = np.random.default_rng(seed)
    extra = rng.normal(0.0, 1.0, size=(features.shape[0], 1))
    augmented = np.hstack([features, extra]).astype(np.float64, copy=False)
    estimate = refit(augmented, treatment, outcome, unit_ids)
    scale = baseline.se if baseline.se > 0 else math.inf
    z = abs(estimate.ate - baseline.ate) / scale
    return RefutationResult.verdict(
        "random_common_cause",
        ok=bool(z <= policy.random_common_cause_z),
        statistic=float(z),
        threshold=policy.random_common_cause_z,
        detail=f"estimate moved from {baseline.ate:.3f} to {estimate.ate:.3f}",
    )


def subset_stability(
    refit: Refitter,
    features: FloatArray,
    treatment: IntArray,
    outcome: FloatArray,
    unit_ids: Sequence[str],
    baseline: AteEstimate,
    policy: RefutationPolicy,
    *,
    seed: int = 0,
) -> RefutationResult:
    """Re-estimate on random subsets; nested estimates should agree within sampling error."""
    if not 0.2 <= policy.subset_fraction < 1.0:
        raise ValueError("subset_fraction must lie in [0.2, 1)")
    rng = np.random.default_rng(seed)
    n = treatment.size
    size = max(round(policy.subset_fraction * n), 20)
    worst = 0.0
    worst_detail = "no subset deviated"
    for _ in range(max(policy.n_subsets, 1)):
        rows = rng.choice(n, size=size, replace=False)
        rows.sort()
        subset_ids = [unit_ids[int(index)] for index in rows]
        estimate = refit(features[rows], treatment[rows], outcome[rows], subset_ids)
        variance = estimate.se**2 - baseline.se**2
        scale = math.sqrt(variance) if variance > 1e-12 else max(estimate.se, 1e-12)
        z = abs(estimate.ate - baseline.ate) / scale
        if z > worst:
            worst = z
            worst_detail = f"subset of {size} rows estimated {estimate.ate:.3f} (z = {z:.2f})"
    return RefutationResult.verdict(
        "subset_stability",
        ok=bool(worst <= policy.subset_stability_z),
        statistic=float(worst),
        threshold=policy.subset_stability_z,
        detail=worst_detail,
    )


def pilot_consistency(
    baseline: AteEstimate,
    treatment: IntArray,
    outcome: FloatArray,
    pilot: IntArray,
    policy: RefutationPolicy,
) -> RefutationResult:
    """Compare the observational estimate with the randomized pilot's difference in means."""
    rows = np.flatnonzero(pilot == 1)
    pilot_treatment = treatment[rows]
    per_arm = min(int((pilot_treatment == 1).sum()), int((pilot_treatment == 0).sum()))
    if per_arm < policy.min_pilot_per_arm:
        return RefutationResult(
            name="pilot_consistency",
            status="skipped",
            statistic=0.0,
            threshold=policy.pilot_consistency_z,
            detail=(
                f"skipped: the randomized slice has only {per_arm} rows in its smaller arm "
                f"(minimum {policy.min_pilot_per_arm})"
            ),
        )
    pilot_estimate = naive_difference(pilot_treatment, outcome[rows], level=baseline.level)
    scale = math.sqrt(baseline.se**2 + pilot_estimate.se**2)
    z = abs(baseline.ate - pilot_estimate.ate) / scale if scale > 0 else math.inf
    return RefutationResult.verdict(
        "pilot_consistency",
        ok=bool(z <= policy.pilot_consistency_z),
        statistic=float(z),
        threshold=policy.pilot_consistency_z,
        detail=(
            f"observational {baseline.ate:.3f} vs randomized pilot {pilot_estimate.ate:.3f} "
            f"+/- {1.96 * pilot_estimate.se:.3f} on {rows.size} rows"
        ),
    )


def refute_all(
    refit: Refitter,
    features: FloatArray,
    treatment: IntArray,
    outcome: FloatArray,
    unit_ids: Sequence[str],
    baseline: AteEstimate,
    policy: RefutationPolicy,
    *,
    pilot: IntArray | None = None,
    seed: int = 0,
) -> RefutationReport:
    """Run the whole suite. Every refuter is reported, including the ones that passed."""
    results = [
        placebo_treatment(refit, features, treatment, outcome, unit_ids, seed=seed + 1),
        random_common_cause(
            refit, features, treatment, outcome, unit_ids, baseline, policy, seed=seed + 2
        ),
        subset_stability(
            refit, features, treatment, outcome, unit_ids, baseline, policy, seed=seed + 3
        ),
    ]
    if pilot is not None:
        results.append(pilot_consistency(baseline, treatment, outcome, pilot, policy))
    value = e_value(baseline, float(np.std(outcome, ddof=1)))
    results.append(
        RefutationResult.verdict(
            "e_value",
            ok=bool(value.ci_limit >= policy.min_e_value),
            statistic=value.ci_limit,
            threshold=policy.min_e_value,
            severity="advisory",
            detail=(
                f"an unmeasured confounder would need risk ratios of {value.point:.2f} with both "
                f"treatment and outcome to explain away the point estimate "
                f"({value.ci_limit:.2f} for the interval limit)"
            ),
        )
    )
    return RefutationReport(results=tuple(results), e_value=value)
