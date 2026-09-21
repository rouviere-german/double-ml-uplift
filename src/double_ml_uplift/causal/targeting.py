"""Budget-constrained targeting, and checking the resulting policy before it is used.

Ranking customers by predicted uplift is the easy half. The half that decides whether a campaign
is worth running is the value of the resulting policy, which is a causal quantity in its own
right and is estimated here on the randomized pilot slice rather than on the same observational
data that trained the CATE model.

For a policy ``pi`` that targets a subset of customers, the gain per customer over treating
nobody is

    V(pi) = P(pi = 1) * E[tau | pi = 1]

The second factor is estimated from the pilot as a difference in means between the pilot's
treated and untreated customers *among those the policy would target*, which needs no
unconfoundedness assumption. The first factor is known exactly, because ``pi`` is computed for
every customer.

Selection itself is the classic greedy heuristic for the fractional knapsack: sort by predicted
uplift per unit of cost, take while the budget lasts, and never target a customer whose predicted
uplift is not positive. For equal costs this is exactly "top-k by uplift".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

__all__ = [
    "FairnessCheck",
    "PolicyValue",
    "TargetingPlan",
    "allocation_disparity",
    "build_plan",
    "fairness_check",
    "greedy_knapsack",
    "pilot_policy_value",
    "rank_by_value_per_cost",
    "true_policy_value",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


def rank_by_value_per_cost(uplift: FloatArray, cost: FloatArray) -> IntArray:
    """Row order by predicted uplift per unit of cost, best first. Ties break by row order."""
    if uplift.size != cost.size:
        raise ValueError("uplift and cost must have the same length")
    if np.any(cost <= 0.0):
        raise ValueError("every cost must be positive")
    ratio = uplift / cost
    return np.asarray(np.lexsort((np.arange(ratio.size), -ratio)), dtype=np.int64)


def greedy_knapsack(
    uplift: FloatArray,
    cost: FloatArray,
    budget: float,
    *,
    eligible: BoolArray | None = None,
) -> BoolArray:
    """Select customers by uplift per cost while the budget lasts.

    Customers with non-positive predicted uplift are never selected, and ``eligible`` (used for
    abstentions outside the overlap region) removes rows from consideration entirely.
    """
    if budget < 0.0:
        raise ValueError("budget must not be negative")
    order = rank_by_value_per_cost(uplift, cost)
    allowed = np.ones(uplift.size, dtype=np.bool_) if eligible is None else np.asarray(eligible)
    if allowed.size != uplift.size:
        raise ValueError("eligible mask and uplift must have the same length")
    selected = np.zeros(uplift.size, dtype=np.bool_)
    spent = 0.0
    for index in order.tolist():
        if not allowed[index] or uplift[index] <= 0.0:
            continue
        price = float(cost[index])
        if spent + price > budget:
            continue
        selected[index] = True
        spent += price
    return selected


@dataclass(frozen=True, slots=True)
class PolicyValue:
    """Estimated gain per customer of a policy over treating nobody."""

    value: float
    se: float
    ci_low: float
    ci_high: float
    targeted_share: float
    n_pilot_treated: int
    n_pilot_control: int
    method: str

    def covers(self, truth: float) -> bool:
        return bool(self.ci_low <= truth <= self.ci_high)

    def to_dict(self) -> dict[str, object]:
        return {
            "value": round(self.value, 4),
            "se": round(self.se, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "targeted_share": round(self.targeted_share, 6),
            "n_pilot_treated": self.n_pilot_treated,
            "n_pilot_control": self.n_pilot_control,
            "method": self.method,
        }


def pilot_policy_value(
    selected: BoolArray,
    pilot: IntArray,
    treatment: IntArray,
    outcome: FloatArray,
    *,
    level: float = 0.95,
    min_per_arm: int = 5,
) -> PolicyValue:
    """Estimate ``V(pi)`` from the randomized slice alone."""
    sizes = {selected.size, pilot.size, treatment.size, outcome.size}
    if len(sizes) != 1:
        raise ValueError("selected, pilot, treatment and outcome must have the same length")
    share = float(np.mean(selected))
    rows = np.flatnonzero((pilot == 1) & selected)
    treated = outcome[rows][treatment[rows] == 1]
    control = outcome[rows][treatment[rows] == 0]
    if share == 0.0:
        return PolicyValue(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, "randomized-pilot")
    if treated.size < min_per_arm or control.size < min_per_arm:
        raise ValueError(
            f"the randomized slice has {treated.size} treated and {control.size} untreated rows "
            f"among the targeted customers; at least {min_per_arm} of each are required"
        )
    within = float(treated.mean() - control.mean())
    within_se = float(
        np.sqrt(treated.var(ddof=1) / treated.size + control.var(ddof=1) / control.size)
    )
    value = share * within
    se = share * within_se
    half = float(norm.ppf(0.5 + level / 2.0)) * se
    return PolicyValue(
        value=value,
        se=se,
        ci_low=value - half,
        ci_high=value + half,
        targeted_share=share,
        n_pilot_treated=int(treated.size),
        n_pilot_control=int(control.size),
        method="randomized-pilot",
    )


def true_policy_value(selected: BoolArray, y0: FloatArray, y1: FloatArray) -> float:
    """Ground-truth ``V(pi)`` from the potential outcomes. Evaluation only, never in serving."""
    if not (selected.size == y0.size == y1.size):
        raise ValueError("selected, y0 and y1 must have the same length")
    return float(np.mean(np.where(selected, y1 - y0, 0.0)))


@dataclass(frozen=True, slots=True)
class TargetingPlan:
    """A concrete selection of customers, with what it costs and what it is expected to earn."""

    selected: BoolArray
    ranking: str
    budget: float
    total_cost: float
    n_selected: int
    predicted_gain: float
    """Sum of predicted uplift over the selected customers, divided by the population size."""

    def to_dict(self) -> dict[str, object]:
        return {
            "ranking": self.ranking,
            "budget": round(self.budget, 4),
            "total_cost": round(self.total_cost, 4),
            "n_selected": self.n_selected,
            "predicted_gain": round(self.predicted_gain, 4),
        }


def build_plan(
    score: FloatArray,
    cost: FloatArray,
    budget: float,
    *,
    ranking: str,
    uplift: FloatArray | None = None,
    eligible: BoolArray | None = None,
) -> TargetingPlan:
    """Select under the budget by ``score`` and price the result with ``uplift``.

    ``score`` and ``uplift`` differ for the baseline policies: propensity ranking selects by the
    historical propensity but is still valued with the predicted uplift.
    """
    selected = greedy_knapsack(score, cost, budget, eligible=eligible)
    values = score if uplift is None else uplift
    return TargetingPlan(
        selected=selected,
        ranking=ranking,
        budget=budget,
        total_cost=float(cost[selected].sum()),
        n_selected=int(selected.sum()),
        predicted_gain=float(np.sum(np.where(selected, values, 0.0)) / max(score.size, 1)),
    )


@dataclass(frozen=True, slots=True)
class FairnessCheck:
    """Demographic-parity check on who receives the offer."""

    disparity: float
    threshold: float
    blocked: bool
    rate_protected: float
    rate_other: float

    def to_dict(self) -> dict[str, object]:
        return {
            "disparity": round(self.disparity, 6),
            "threshold": self.threshold,
            "blocked": self.blocked,
            "rate_protected": round(self.rate_protected, 6),
            "rate_other": round(self.rate_other, 6),
        }


def allocation_disparity(selected: BoolArray, protected: IntArray) -> tuple[float, float, float]:
    """Return ``(disparity, rate in group 1, rate in group 0)`` for a binary attribute."""
    if selected.size != protected.size:
        raise ValueError("selected and protected must have the same length")
    groups = set(np.unique(protected).tolist())
    if not groups <= {0, 1}:
        raise ValueError("the protected attribute must be binary")
    in_group = protected == 1
    out_group = protected == 0
    if not in_group.any() or not out_group.any():
        return 0.0, float(selected.mean()), float(selected.mean())
    rate_protected = float(selected[in_group].mean())
    rate_other = float(selected[out_group].mean())
    return abs(rate_protected - rate_other), rate_protected, rate_other


def fairness_check(
    selected: BoolArray, protected: IntArray, *, threshold: float = 0.15
) -> FairnessCheck:
    """Block a targeting list whose allocation rates differ too much between groups."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must lie in (0, 1]")
    disparity, rate_protected, rate_other = allocation_disparity(selected, protected)
    return FairnessCheck(
        disparity=disparity,
        threshold=threshold,
        blocked=bool(disparity > threshold),
        rate_protected=rate_protected,
        rate_other=rate_other,
    )
