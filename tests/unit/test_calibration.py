"""Uplift calibration: what a predicted effect is worth in the group it lands in."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from double_ml_uplift.causal.calibration import CateCalibration, calibrate

pytestmark = pytest.mark.unit


def linear_sample(
    n: int = 2_000, *, slope: float = 1.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    predicted = rng.normal(0.0, 5.0, size=n)
    pseudo = slope * predicted + rng.normal(0.0, 1.0, size=n)
    return predicted, pseudo


def test_a_well_calibrated_model_has_a_slope_near_one() -> None:
    predicted, pseudo = linear_sample()
    calibration = calibrate(predicted, pseudo, n_bins=10)
    assert calibration.n_bins == 10
    assert calibration.slope() == pytest.approx(1.0, abs=0.1)
    assert int(calibration.bin_count.sum()) == 2_000


def test_a_model_that_exaggerates_has_a_slope_below_one() -> None:
    predicted, pseudo = linear_sample(slope=0.4, seed=1)
    assert calibrate(predicted, pseudo).slope() == pytest.approx(0.4, abs=0.1)


def test_a_model_with_no_signal_has_a_slope_near_zero() -> None:
    rng = np.random.default_rng(2)
    predicted = rng.normal(size=3_000)
    pseudo = rng.normal(size=3_000)
    assert abs(calibrate(predicted, pseudo).slope()) < 0.2


def test_bin_means_rise_with_bin_predictions_for_a_good_model() -> None:
    predicted, pseudo = linear_sample(seed=3)
    calibration = calibrate(predicted, pseudo, n_bins=5)
    assert np.all(np.diff(calibration.bin_predicted) > 0)
    assert np.all(np.diff(calibration.bin_actual) > 0)


def test_assign_places_a_prediction_in_the_bin_its_value_implies() -> None:
    predicted = np.arange(100.0)
    pseudo = predicted.copy()
    calibration = calibrate(predicted, pseudo, n_bins=4)
    assert calibration.edges.size == 3
    assigned = calibration.assign(np.array([-1.0, 50.0, 1_000.0]))
    assert assigned[0] == 0
    assert assigned[-1] == calibration.n_bins - 1
    assert 0 <= assigned[1] < calibration.n_bins


def test_the_interval_reports_the_group_effect_not_a_per_customer_guess() -> None:
    predicted, pseudo = linear_sample(seed=4)
    calibration = calibrate(predicted, pseudo, n_bins=10)
    effect, se, low, high = calibration.interval(predicted[:50], level=0.95)
    assert effect.shape == se.shape == low.shape == high.shape == (50,)
    assert np.all(low <= effect)
    assert np.all(effect <= high)
    # Two customers in the same bin get the same interval: that is the point.
    index = calibration.assign(predicted[:50])
    for bin_id in np.unique(index):
        rows = index == bin_id
        assert np.allclose(effect[rows], effect[rows][0])


def test_a_wider_level_widens_the_band() -> None:
    predicted, pseudo = linear_sample(seed=5)
    calibration = calibrate(predicted, pseudo)
    _, _, low90, high90 = calibration.interval(predicted[:20], level=0.90)
    _, _, low99, high99 = calibration.interval(predicted[:20], level=0.99)
    assert np.all(low99 <= low90)
    assert np.all(high99 >= high90)
    with pytest.raises(ValueError, match="level must lie"):
        calibration.interval(predicted[:5], level=1.0)


def test_calibration_inputs_are_validated() -> None:
    predicted, pseudo = linear_sample(100, seed=6)
    with pytest.raises(ValueError, match="same length"):
        calibrate(predicted, pseudo[:50])
    with pytest.raises(ValueError, match="n_bins must lie"):
        calibrate(predicted, pseudo, n_bins=1)
    with pytest.raises(ValueError, match="n_bins must lie"):
        calibrate(predicted, pseudo, n_bins=100)
    with pytest.raises(ValueError, match="at least 20 rows"):
        calibrate(predicted[:15], pseudo[:15], n_bins=10)


def test_constant_predictions_collapse_to_a_single_bin_with_slope_zero() -> None:
    predicted = np.zeros(100)
    pseudo = np.arange(100.0)
    calibration = calibrate(predicted, pseudo, n_bins=5)
    assert calibration.n_bins == 1
    assert calibration.slope() == 0.0
    assert calibration.bin_actual[0] == pytest.approx(float(pseudo.mean()))
    # Even a prediction below everything seen in training gets that bin's honest interval.
    effect, se, low, high = calibration.interval(np.array([-99.0]))
    assert effect[0] == pytest.approx(float(pseudo.mean()))
    assert se[0] > 0.0
    assert low[0] < high[0]


def test_no_bin_is_ever_empty_so_no_interval_is_built_from_nothing() -> None:
    """A heavily regularised second stage predicts few distinct values, which is exactly when
    quantile edges land on real data and leave a bin with no rows in it."""
    rng = np.random.default_rng(9)
    predicted = np.repeat(np.array([-2.0, -2.0, 0.0, 5.0]), 60)
    pseudo = predicted * 0.8 + rng.normal(size=predicted.size)
    calibration = calibrate(predicted, pseudo, n_bins=10)
    assert calibration.bin_count.min() >= 1
    assert np.all(calibration.bin_se > 0.0)
    assert int(calibration.bin_count.sum()) == predicted.size


def test_a_calibration_round_trips_through_its_dict_form() -> None:
    predicted, pseudo = linear_sample(seed=7)
    calibration = calibrate(predicted, pseudo, n_bins=8)
    payload = calibration.to_dict()
    assert set(payload) == {
        "edges",
        "bin_predicted",
        "bin_actual",
        "bin_se",
        "bin_count",
        "slope",
    }
    restored = CateCalibration.from_dict(payload)
    assert np.allclose(restored.edges, calibration.edges)
    assert np.allclose(restored.bin_actual, calibration.bin_actual, atol=1e-4)
    assert restored.n_bins == calibration.n_bins
    assert np.array_equal(restored.assign(predicted[:100]), calibration.assign(predicted[:100]))


@settings(max_examples=60)
@given(
    n_bins=st.integers(min_value=2, max_value=12),
    seed=st.integers(min_value=0, max_value=1_000),
)
def test_every_row_lands_in_exactly_one_bin(n_bins: int, seed: int) -> None:
    predicted, pseudo = linear_sample(400, seed=seed)
    calibration = calibrate(predicted, pseudo, n_bins=n_bins)
    index = calibration.assign(predicted)
    assert index.min() >= 0
    assert index.max() < calibration.n_bins
    assert int(calibration.bin_count.sum()) == predicted.size
    assert np.array_equal(np.bincount(index, minlength=calibration.n_bins), calibration.bin_count)


@settings(max_examples=60)
@given(shift=st.floats(min_value=-100.0, max_value=100.0))
def test_shifting_every_pseudo_outcome_shifts_every_bin_mean(shift: float) -> None:
    predicted, pseudo = linear_sample(400, seed=8)
    base = calibrate(predicted, pseudo, n_bins=5)
    moved = calibrate(predicted, pseudo + shift, n_bins=5)
    assert np.allclose(moved.bin_actual, base.bin_actual + shift)
    assert moved.slope() == pytest.approx(base.slope(), abs=1e-8)
