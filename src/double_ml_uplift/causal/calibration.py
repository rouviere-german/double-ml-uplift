"""Uplift calibration: what a predicted effect is actually worth, and how sure we are.

A CATE model returns a number per customer, but nothing in the data ever reveals that customer's
individual effect, so a per-customer standard error from the second-stage model would be honest
only about the model's own wobble. What *can* be estimated is the average effect in a group of
customers with similar predictions, because the doubly robust pseudo-outcome is an unbiased
estimate of the effect at the unit level (conditional on the nuisances being right).

So the model is calibrated in deciles of its own out-of-fold prediction: for each bin, the mean
and standard error of the pseudo-outcomes in it. That gives

* an uncertainty band to report with every score -- the uncertainty of the *group* effect, which
  is what a targeting decision actually needs;
* a calibration curve and slope, the standard diagnostic for an uplift model: a well-calibrated
  model has bin means that rise with bin predictions on a slope near one.

This is the "uplift curve / Qini-style" binning used throughout the uplift-modelling literature
(Radcliffe, "Using control groups to target on predicted lift", 2007), with AIPW pseudo-outcomes
in place of the raw treated-minus-control difference.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

__all__ = ["CateCalibration", "calibrate"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class CateCalibration:
    """Binned relationship between predicted uplift and the doubly robust pseudo-outcome."""

    edges: FloatArray
    """Interior bin edges, ``n_bins - 1`` of them, in increasing order."""

    bin_predicted: FloatArray
    bin_actual: FloatArray
    bin_se: FloatArray
    bin_count: IntArray

    @property
    def n_bins(self) -> int:
        return int(self.bin_actual.size)

    def assign(self, predicted: FloatArray) -> IntArray:
        """Bin index for each predicted uplift."""
        return np.asarray(
            np.searchsorted(self.edges, np.asarray(predicted, dtype=np.float64), side="right"),
            dtype=np.int64,
        )

    def interval(
        self, predicted: FloatArray, *, level: float = 0.95
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Return ``(group_effect, se, ci_low, ci_high)`` for each predicted uplift."""
        if not 0.5 < level < 1.0:
            raise ValueError("level must lie in (0.5, 1)")
        index = self.assign(predicted)
        effect = self.bin_actual[index]
        se = self.bin_se[index]
        half = float(norm.ppf(0.5 + level / 2.0)) * se
        return effect, se, effect - half, effect + half

    def slope(self) -> float:
        """Least-squares slope of bin means on bin predictions. One means well calibrated."""
        weights = self.bin_count.astype(np.float64)
        if weights.sum() <= 0.0:
            return 0.0
        mean_x = float(np.average(self.bin_predicted, weights=weights))
        mean_y = float(np.average(self.bin_actual, weights=weights))
        centred = self.bin_predicted - mean_x
        variance = float(np.sum(weights * centred**2))
        if variance <= 1e-12:
            return 0.0
        return float(np.sum(weights * centred * (self.bin_actual - mean_y)) / variance)

    def to_dict(self) -> dict[str, list[float] | list[int] | float]:
        return {
            "edges": [round(value, 6) for value in self.edges.tolist()],
            "bin_predicted": [round(value, 4) for value in self.bin_predicted.tolist()],
            "bin_actual": [round(value, 4) for value in self.bin_actual.tolist()],
            "bin_se": [round(value, 4) for value in self.bin_se.tolist()],
            "bin_count": [int(value) for value in self.bin_count.tolist()],
            "slope": round(self.slope(), 4),
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, object]) -> CateCalibration:
        def _floats(key: str) -> FloatArray:
            return np.asarray(document[key], dtype=np.float64)

        return cls(
            edges=_floats("edges"),
            bin_predicted=_floats("bin_predicted"),
            bin_actual=_floats("bin_actual"),
            bin_se=_floats("bin_se"),
            bin_count=np.asarray(document["bin_count"], dtype=np.int64),
        )


def _drop_empty_bins(predicted: FloatArray, edges: FloatArray) -> FloatArray:
    """Merge away any bin no training row fell into.

    Quantile edges can coincide with a value that actually occurs -- in the limit, a model that
    predicts one number for everybody produces an edge equal to that number and leaves the bin
    below it empty. An empty bin has no mean and no standard error, so :meth:`interval` would
    report an effect of exactly zero with a zero-width band for any customer who landed there:
    the most confident possible answer, from no data at all. Dropping the edge merges the empty
    bin into its neighbour instead, so every reported interval rests on at least one observation.
    """
    while edges.size:
        counts = np.bincount(
            np.searchsorted(edges, predicted, side="right"), minlength=edges.size + 1
        )
        empty = np.flatnonzero(counts == 0)
        if empty.size == 0:
            break
        first = int(empty[0])
        edges = np.delete(edges, max(first - 1, 0))
    return edges


def calibrate(
    predicted: FloatArray, pseudo_outcome: FloatArray, *, n_bins: int = 10
) -> CateCalibration:
    """Bin ``predicted`` into quantile groups and summarise the pseudo-outcome in each."""
    if predicted.size != pseudo_outcome.size:
        raise ValueError("predicted and pseudo_outcome must have the same length")
    if not 2 <= n_bins <= 50:
        raise ValueError("n_bins must lie in [2, 50]")
    if predicted.size < 2 * n_bins:
        raise ValueError(f"at least {2 * n_bins} rows are needed for {n_bins} bins")
    quantiles = np.quantile(predicted, np.linspace(0.0, 1.0, n_bins + 1))
    edges = _drop_empty_bins(predicted, np.unique(quantiles[1:-1]))
    index = np.searchsorted(edges, predicted, side="right")
    count = edges.size + 1

    bin_predicted = np.zeros(count, dtype=np.float64)
    bin_actual = np.zeros(count, dtype=np.float64)
    bin_se = np.zeros(count, dtype=np.float64)
    bin_count = np.zeros(count, dtype=np.int64)
    for position in range(count):
        rows = index == position
        size = int(rows.sum())
        bin_count[position] = size
        if size == 0:
            continue
        values = pseudo_outcome[rows]
        bin_predicted[position] = float(predicted[rows].mean())
        bin_actual[position] = float(values.mean())
        bin_se[position] = float(values.std(ddof=1) / np.sqrt(size)) if size > 1 else float("inf")
    return CateCalibration(
        edges=edges,
        bin_predicted=bin_predicted,
        bin_actual=bin_actual,
        bin_se=bin_se,
        bin_count=bin_count,
    )
