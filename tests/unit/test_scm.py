"""The simulator is the ground truth, so its own invariants are tested before anything uses it."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from double_ml_uplift.causal.scenarios import (
    MAX_SCENARIO_BYTES,
    load_scenario,
    scenario_names,
)
from double_ml_uplift.causal.scm import (
    CONFOUNDER_COLUMNS,
    ORACLE_COLUMNS,
    OUTCOME_COLUMN,
    PILOT_COLUMN,
    POST_TREATMENT_COLUMNS,
    TREATMENT_COLUMN,
    UNIT_COLUMN,
    ScmConfig,
    SimulatedCampaign,
    simulate_campaign,
    standardize,
)
from double_ml_uplift.causal.spec import bundled_spec
from double_ml_uplift.ml.contracts import FORBIDDEN_COLUMNS, validate_campaign_frame

pytestmark = pytest.mark.unit

SMALL = 600


def config(**overrides: object) -> ScmConfig:
    base: dict[str, object] = {"name": "t", "n_customers": SMALL, "seed": 7}
    base.update(overrides)
    return ScmConfig.model_validate(base)


# --- determinism and hashing ---------------------------------------------------------------


def test_the_same_configuration_produces_byte_identical_data() -> None:
    first = simulate_campaign(config())
    second = simulate_campaign(config())
    assert first.frame.equals(second.frame)
    assert first.oracle.equals(second.oracle)
    assert first.config_sha256 == second.config_sha256


def test_a_different_seed_produces_different_data_but_the_same_schema() -> None:
    first = simulate_campaign(config(seed=1))
    second = simulate_campaign(config(seed=2))
    assert list(first.frame.columns) == list(second.frame.columns)
    assert not first.frame.equals(second.frame)
    assert first.config_sha256 != second.config_sha256


def test_the_parameter_hash_moves_with_every_parameter() -> None:
    base = config().sha256()
    assert config(confounding_strength=0.5).sha256() != base
    assert config(nonlinear=True).sha256() != base
    assert config(direct_effect=(1.0, 2.0, 3.0, 4.0)).sha256() != base


# --- the oracle never reaches the estimator -------------------------------------------------


def test_the_observed_frame_carries_no_oracle_column() -> None:
    campaign = simulate_campaign(config())
    assert set(campaign.frame.columns).isdisjoint(FORBIDDEN_COLUMNS)
    assert set(campaign.frame.columns).isdisjoint(set(ORACLE_COLUMNS) - {UNIT_COLUMN})
    assert set(campaign.oracle.columns) == set(ORACLE_COLUMNS)


def test_the_observed_frame_is_exactly_what_the_specification_declares() -> None:
    campaign = simulate_campaign(config())
    spec = bundled_spec("retention_offer")
    assert set(campaign.frame.columns) == set(spec.frame_columns())
    assert len(validate_campaign_frame(campaign.frame, spec)) == SMALL


def test_the_frame_and_the_oracle_line_up_row_by_row() -> None:
    campaign = simulate_campaign(config())
    assert campaign.frame[UNIT_COLUMN].tolist() == campaign.oracle[UNIT_COLUMN].tolist()
    assert campaign.frame[UNIT_COLUMN].is_unique


# --- the structural equations hold ----------------------------------------------------------


def test_the_observed_outcome_is_the_potential_outcome_of_the_arm_actually_assigned() -> None:
    campaign = simulate_campaign(config())
    y0, y1 = campaign.potential_outcomes
    treatment = campaign.frame[TREATMENT_COLUMN].to_numpy()
    expected = np.where(treatment == 1, y1, y0)
    assert np.allclose(campaign.frame[OUTCOME_COLUMN].to_numpy(), np.round(expected, 4))


def test_the_true_effect_is_the_difference_of_the_potential_outcomes() -> None:
    campaign = simulate_campaign(config())
    y0, y1 = campaign.potential_outcomes
    assert np.allclose(campaign.true_cate, y1 - y0)
    assert campaign.true_ate == pytest.approx(float(np.mean(y1 - y0)))


def test_the_effect_is_heterogeneous_and_negative_for_a_sizeable_minority() -> None:
    campaign = simulate_campaign(load_scenario("heterogeneous", n_customers=4_000, seed=3))
    effects = campaign.true_cate
    assert effects.std() > 1.0
    harmed = float(np.mean(effects < 0.0))
    assert 0.1 < harmed < 0.6, harmed


def test_the_pilot_slice_is_a_genuine_coin_flip() -> None:
    campaign = simulate_campaign(config(pilot_fraction=0.3, confounding_strength=2.0))
    pilot = campaign.frame[PILOT_COLUMN].to_numpy()
    truth = campaign.oracle["true_propensity"].to_numpy()
    assert np.allclose(truth[pilot == 1], 0.5)
    assert not np.allclose(truth[pilot == 0], 0.5)
    assert 0.2 < float(pilot.mean()) < 0.4


def test_no_confounding_means_a_constant_propensity() -> None:
    campaign = simulate_campaign(config(confounding_strength=0.0, pilot_fraction=0.0))
    truth = campaign.oracle["true_propensity"].to_numpy()
    assert np.allclose(truth, 0.5)


def fraction_outside_overlap(campaign: SimulatedCampaign) -> float:
    truth = campaign.oracle["true_propensity"].to_numpy()
    return float(np.mean((truth < 0.05) | (truth > 0.95)))


def test_tilting_the_treatment_index_pushes_propensities_to_the_boundaries() -> None:
    gentle = simulate_campaign(config(overlap_tilt=1.0, pilot_fraction=0.0))
    harsh = simulate_campaign(config(overlap_tilt=6.0, pilot_fraction=0.0))
    assert fraction_outside_overlap(gentle) < 0.10
    assert fraction_outside_overlap(harsh) > 0.30
    assert fraction_outside_overlap(harsh) > 3 * fraction_outside_overlap(gentle)


def test_the_hidden_confounder_moves_both_the_treatment_and_the_outcome() -> None:
    campaign = simulate_campaign(
        config(hidden_confounder_strength=2.0, pilot_fraction=0.0, outcome_noise_sd=1.0)
    )
    hidden = campaign.oracle["hidden_confounder"].to_numpy()
    propensity = campaign.oracle["true_propensity"].to_numpy()
    y0, _ = campaign.potential_outcomes
    assert np.corrcoef(hidden, propensity)[0, 1] > 0.5
    assert np.corrcoef(hidden, y0)[0, 1] > 0.3


def test_without_a_hidden_confounder_it_influences_nothing() -> None:
    campaign = simulate_campaign(config(hidden_confounder_strength=0.0, pilot_fraction=0.0))
    hidden = campaign.oracle["hidden_confounder"].to_numpy()
    propensity = campaign.oracle["true_propensity"].to_numpy()
    assert abs(float(np.corrcoef(hidden, propensity)[0, 1])) < 0.15


def test_post_treatment_columns_react_to_the_treatment() -> None:
    campaign = simulate_campaign(config(n_customers=3_000))
    frame = campaign.frame
    treated = frame[frame[TREATMENT_COLUMN] == 1]
    control = frame[frame[TREATMENT_COLUMN] == 0]
    for column in POST_TREATMENT_COLUMNS:
        assert treated[column].mean() != pytest.approx(control[column].mean(), abs=1e-6)


def test_standardize_centres_with_fixed_constants_not_sample_statistics() -> None:
    campaign = simulate_campaign(config())
    z = standardize(campaign.frame)
    assert z.shape == (SMALL, len(CONFOUNDER_COLUMNS))
    half = standardize(campaign.frame.iloc[: SMALL // 2])
    assert np.allclose(z[: SMALL // 2], half)


# --- configuration bounds ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_customers": 10},
        {"seed": -1},
        {"confounding_strength": -0.5},
        {"overlap_tilt": 0.1},
        {"pilot_fraction": 0.9},
        {"outcome_noise_sd": 0.0},
        {"name": "Not A Name"},
        {"offer_cost": 0.0},
    ],
)
def test_out_of_range_parameters_are_refused(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        config(**overrides)


def test_the_configuration_is_frozen_and_closed() -> None:
    parameters = config()
    with pytest.raises(ValidationError, match="frozen"):
        parameters.seed = 3  # type: ignore[misc]
    with pytest.raises(ValidationError, match="unexpected"):
        ScmConfig.model_validate({**parameters.model_dump(), "unexpected": 1})


# --- bundled scenarios ----------------------------------------------------------------------


def test_every_bundled_scenario_loads_and_simulates() -> None:
    assert scenario_names() == (
        "heterogeneous",
        "hidden_confounder",
        "linear",
        "nonlinear",
        "randomized",
        "weak_overlap",
    )
    for name in scenario_names():
        campaign = simulate_campaign(load_scenario(name, n_customers=200, seed=1))
        assert len(campaign.frame) == 200
        assert campaign.config.name == name


def test_scenario_overrides_apply_and_unknown_names_are_refused() -> None:
    scenario = load_scenario("linear", n_customers=321, seed=99)
    assert (scenario.n_customers, scenario.seed) == (321, 99)
    with pytest.raises(ValueError, match="unknown scenario"):
        load_scenario("nope")


def test_the_scenario_files_stay_within_their_size_budget() -> None:
    from importlib import resources

    package = resources.files("double_ml_uplift.configs.scenarios")
    for name in scenario_names():
        text = package.joinpath(f"{name}.yaml").read_text(encoding="utf-8")
        assert len(text.encode("utf-8")) <= MAX_SCENARIO_BYTES


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    nonlinear=st.booleans(),
    strength=st.floats(min_value=0.0, max_value=2.0),
)
def test_every_draw_satisfies_the_data_contract(
    seed: int, nonlinear: bool, strength: float
) -> None:
    campaign = simulate_campaign(
        config(n_customers=120, seed=seed, nonlinear=nonlinear, confounding_strength=strength)
    )
    validate_campaign_frame(campaign.frame, bundled_spec("retention_offer"))
    assert np.all(np.isfinite(campaign.true_cate))
