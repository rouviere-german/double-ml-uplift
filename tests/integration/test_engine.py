"""The whole pipeline, against ground truth the estimators never see."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from double_ml_uplift.causal.bundle import ScoringBundle
from double_ml_uplift.causal.diagnostics import GatePolicy, NotEstimableError
from double_ml_uplift.causal.engine import (
    EstimationConfig,
    EstimationResult,
    UpliftEngine,
    dataset_digest,
)
from double_ml_uplift.causal.estimators import pehe
from double_ml_uplift.causal.identification import LeakageError, NotIdentifiedError
from double_ml_uplift.causal.nuisance import NuisanceConfig
from double_ml_uplift.causal.scenarios import load_scenario
from double_ml_uplift.causal.scm import SimulatedCampaign, simulate_campaign
from double_ml_uplift.causal.spec import CausalSpec, NodeSpec, bundled_spec
from double_ml_uplift.causal.targeting import true_policy_value
from double_ml_uplift.ml.contracts import (
    ContractLimits,
    DataContractError,
    PotentialOutcomeLeakError,
)

pytestmark = pytest.mark.integration

ROWS = 2_500
# The configuration the repository ships, with the refuters switched off so the tests that do
# not examine them stay quick. Tests that do examine them turn them back on.
FAST = EstimationConfig(run_refuters=False)


@pytest.fixture(scope="module")
def spec() -> CausalSpec:
    return bundled_spec("retention_offer")


def draw(scenario: str, *, rows: int = ROWS, seed: int = 4242) -> SimulatedCampaign:
    return simulate_campaign(load_scenario(scenario, n_customers=rows, seed=seed))


@pytest.fixture(scope="module")
def confounded() -> SimulatedCampaign:
    return draw("heterogeneous")


@pytest.fixture(scope="module")
def fitted(spec: CausalSpec, confounded: SimulatedCampaign) -> EstimationResult:
    return UpliftEngine(spec, FAST).fit(confounded.frame)


# --- the headline claim -----------------------------------------------------------------------


def test_the_naive_estimate_is_wrong_and_the_doubly_robust_ones_are_not(
    fitted: EstimationResult, confounded: SimulatedCampaign
) -> None:
    """One draw cannot demonstrate interval coverage -- a correct 95% interval misses one draw
    in twenty by construction, and ``bench/bench_montecarlo.py`` is what measures coverage over
    50 of them. What one draw *can* show is the bias: the naive difference in means is far from
    the truth and excludes it, and adjusting removes most of that error."""
    truth = confounded.true_ate
    naive = fitted.estimates["naive"]
    naive_error = abs(naive.ate - truth)
    assert not naive.covers(truth)
    assert naive_error > 5.0
    for name in ("aipw", "dml"):
        estimate = fitted.estimates[name]
        assert abs(estimate.ate - truth) < naive_error / 2.0, f"{name} = {estimate.ate}"
    assert fitted.estimates["dml"].covers(truth)


def test_the_dr_learner_beats_predicting_one_number_for_everybody(
    fitted: EstimationResult, confounded: SimulatedCampaign
) -> None:
    truth = confounded.true_cate
    constant = np.full(truth.size, fitted.primary.ate)
    assert pehe(fitted.cate.out_of_fold, truth) < pehe(constant, truth)
    assert fitted.calibration.slope() == pytest.approx(1.0, abs=0.45)


def test_the_result_records_every_hash_needed_to_reproduce_it(
    fitted: EstimationResult,
) -> None:
    payload = fitted.to_dict()
    for key in ("spec_sha256", "config_sha256", "dataset_sha256", "fold_digest"):
        assert len(payload[key]) == 64
    assert payload["estimable"] is True
    assert payload["refusal_reasons"] == []
    assert payload["primary"] == "aipw"
    assert set(payload["estimates"]) == {"naive", "outcome_regression", "ipw", "aipw", "dml"}
    assert payload["adjustment_set"] == list(fitted.adjustment_set)
    assert payload["backdoor_paths"] > 0


# --- reproducibility ---------------------------------------------------------------------------


def test_the_same_inputs_produce_the_same_number_and_the_same_hashes(
    spec: CausalSpec, confounded: SimulatedCampaign
) -> None:
    first = UpliftEngine(spec, FAST).fit(confounded.frame)
    second = UpliftEngine(spec, FAST).fit(confounded.frame)
    assert first.dataset_sha256 == second.dataset_sha256
    assert first.fold_digest == second.fold_digest
    assert first.config_sha256 == second.config_sha256
    assert first.primary.ate == pytest.approx(second.primary.ate, abs=1e-12)


def test_shuffling_the_rows_does_not_change_the_fold_a_customer_lands_in(
    spec: CausalSpec, confounded: SimulatedCampaign
) -> None:
    engine = UpliftEngine(spec, FAST)
    straight = engine.prepare(confounded.frame)
    shuffled_frame = confounded.frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    shuffled = engine.prepare(shuffled_frame)

    order = {unit: index for index, unit in enumerate(straight.unit_ids)}
    plan_a = UpliftEngine(spec, FAST)._cross_fit(
        straight.features, straight.treatment, straight.outcome, straight.unit_ids
    )[1]
    plan_b = UpliftEngine(spec, FAST)._cross_fit(
        shuffled.features, shuffled.treatment, shuffled.outcome, shuffled.unit_ids
    )[1]
    for position, unit in enumerate(shuffled.unit_ids):
        assert plan_b.folds[position] == plan_a.folds[order[unit]]


def test_changing_the_fold_salt_changes_the_digest_and_the_configuration_hash(
    spec: CausalSpec, confounded: SimulatedCampaign
) -> None:
    other = FAST.model_copy(update={"fold_salt": "a-different-salt"})
    first = UpliftEngine(spec, FAST).fit(confounded.frame)
    second = UpliftEngine(spec, other).fit(confounded.frame)
    assert first.fold_digest != second.fold_digest
    assert first.config_sha256 != second.config_sha256
    assert first.dataset_sha256 == second.dataset_sha256


def test_the_dataset_digest_moves_with_every_input() -> None:
    unit_ids = ["a", "b", "c", "d"]
    features = np.arange(8.0).reshape(4, 2)
    treatment = np.array([0, 1, 0, 1], dtype=np.int64)
    outcome = np.arange(4.0)
    base = dataset_digest(unit_ids, features, treatment, outcome)
    assert base == dataset_digest(unit_ids, features, treatment, outcome)
    assert base != dataset_digest(["a", "b", "c", "e"], features, treatment, outcome)
    assert base != dataset_digest(unit_ids, features + 1.0, treatment, outcome)
    assert base != dataset_digest(unit_ids, features, 1 - treatment, outcome)
    assert base != dataset_digest(unit_ids, features, treatment, outcome + 1.0)


# --- every gate fails closed ---------------------------------------------------------------------


def test_a_positivity_violation_refuses_before_a_number_exists(spec: CausalSpec) -> None:
    campaign = draw("weak_overlap")
    with pytest.raises(NotEstimableError) as info:
        UpliftEngine(spec, FAST).fit(campaign.frame)
    assert any(reason.startswith("positivity") for reason in info.value.reasons)
    assert info.value.report is not None
    assert info.value.report.fraction_outside > 0.05


def test_the_gate_can_be_measured_without_being_enforced(spec: CausalSpec) -> None:
    campaign = draw("weak_overlap")
    config = FAST.model_copy(update={"enforce_gate": False})
    result = UpliftEngine(spec, config).fit(campaign.frame)
    assert not result.diagnostics.estimable
    assert result.to_dict()["estimable"] is False
    assert result.to_dict()["refusal_reasons"]


def test_an_unidentified_specification_refuses_before_touching_the_data() -> None:
    engine = UpliftEngine(bundled_spec("unobserved"), FAST)
    with pytest.raises(NotIdentifiedError, match="churn_risk"):
        engine.adjustment_set()
    with pytest.raises(NotIdentifiedError):
        engine.fit(draw("linear").frame)


def test_a_frame_carrying_the_potential_outcomes_is_refused(spec: CausalSpec) -> None:
    campaign = draw("linear", rows=300)
    leaked = campaign.frame.copy()
    leaked["y1"] = campaign.oracle["y1"].to_numpy()
    with pytest.raises(PotentialOutcomeLeakError) as info:
        UpliftEngine(spec, FAST).fit(leaked)
    assert info.value.columns == ["y1"]


def test_an_upload_larger_than_the_configured_limit_is_refused(spec: CausalSpec) -> None:
    campaign = draw("linear", rows=300)
    engine = UpliftEngine(spec, FAST, limits=ContractLimits(max_rows=100))
    with pytest.raises(DataContractError) as info:
        engine.fit(campaign.frame)
    assert info.value.failures[0]["check"] == "max_rows(100)"


def test_a_specification_whose_adjustment_set_would_be_empty_is_refused() -> None:
    """The engine has nothing to adjust for, so it says so instead of running an unadjusted
    regression and calling the answer causal."""
    spec = CausalSpec(
        version=1,
        name="no-covariates",
        unit_id="row_id",
        treatment="t",
        outcome="y",
        nodes={
            "t": NodeSpec(kind="binary", role="treatment"),
            "y": NodeSpec(kind="continuous", role="outcome"),
        },
        edges=(("t", "y"),),
    )
    frame = pd.DataFrame(
        {
            "row_id": [f"r{index:04d}" for index in range(100)],
            "t": np.tile([0, 1], 50).astype(np.int64),
            "y": np.arange(100.0),
        }
    )
    with pytest.raises(ValueError, match="adjustment set is empty"):
        UpliftEngine(spec, FAST).fit(frame)


def test_an_unmeasured_confounder_is_caught_by_the_pilot(spec: CausalSpec) -> None:
    campaign = draw("hidden_confounder", rows=3_000)
    config = FAST.model_copy(update={"run_refuters": True})
    result = UpliftEngine(spec, config).fit(campaign.frame)
    assert result.diagnostics.estimable
    assert result.refutation is not None
    pilot = next(item for item in result.refutation.results if item.name == "pilot_consistency")
    assert not pilot.passed
    assert "pilot_consistency" in result.refutation.flagged
    assert abs(result.primary.ate - campaign.true_ate) > 20.0


def test_the_refuters_pass_on_a_well_specified_campaign(spec: CausalSpec) -> None:
    campaign = draw("linear", rows=3_000, seed=909)
    config = FAST.model_copy(update={"run_refuters": True})
    result = UpliftEngine(spec, config).fit(campaign.frame)
    assert result.refutation is not None
    assert result.refutation.flagged == ()


# --- targeting ------------------------------------------------------------------------------------


def test_uplift_targeting_beats_propensity_targeting_in_true_value(
    spec: CausalSpec, fitted: EstimationResult, confounded: SimulatedCampaign
) -> None:
    outcome = UpliftEngine(spec, FAST).target(fitted, budget_fraction=0.2, cost_per_offer=1.0)
    y0, y1 = confounded.potential_outcomes
    truths = {
        name: true_policy_value(plan.selected, y0, y1) for name, plan in outcome.plans.items()
    }
    assert truths["uplift"] > truths["propensity"]
    assert truths["uplift"] > truths["treat_all"]
    assert outcome.pilot_values["uplift"].covers(truths["uplift"])
    assert outcome.plans["uplift"].n_selected <= int(0.2 * confounded.frame.shape[0]) + 1
    assert not outcome.fairness.blocked
    assert outcome.to_dict()["budget"] == pytest.approx(0.2 * ROWS)


def test_customers_outside_the_overlap_region_are_never_targeted(
    spec: CausalSpec, fitted: EstimationResult
) -> None:
    tight = FAST.model_copy(update={"gate": GatePolicy(propensity_low=0.3, propensity_high=0.7)})
    engine = UpliftEngine(spec, tight)
    outcome = engine.target(fitted, budget_fraction=1.0)
    from double_ml_uplift.causal.diagnostics import abstain_mask

    abstained = abstain_mask(fitted.fitted.propensity, tight.gate)
    assert outcome.abstained == int(abstained.sum())
    assert outcome.abstained > 0
    assert not outcome.plans["uplift"].selected[abstained].any()


def test_targeting_arguments_are_validated(spec, fitted: EstimationResult) -> None:  # type: ignore[no-untyped-def]
    engine = UpliftEngine(spec, FAST)
    with pytest.raises(ValueError, match="budget_fraction must lie"):
        engine.target(fitted, budget_fraction=0.0)
    with pytest.raises(ValueError, match="cost_per_offer must be positive"):
        engine.target(fitted, cost_per_offer=0.0)


def test_a_disparate_allocation_is_blocked_at_the_release_threshold(
    spec: CausalSpec,
    fitted: EstimationResult,
) -> None:
    engine = UpliftEngine(spec, FAST)
    outcome = engine.target(fitted, budget_fraction=0.2, fairness_threshold=1e-6)
    assert outcome.fairness.blocked
    assert outcome.fairness.disparity > 0.0


# --- serving --------------------------------------------------------------------------------------


def test_the_bundle_scores_exactly_like_the_fitted_engine(spec, fitted: EstimationResult) -> None:  # type: ignore[no-untyped-def]
    engine = UpliftEngine(spec, FAST)
    manifest, artifacts = engine.bundle(fitted, metrics={"ate": fitted.primary.ate})
    bundle = ScoringBundle.load(manifest, artifacts)
    rows = fitted.prepared.features[:200]

    direct_uplift, direct_propensity, direct_abstain = engine.score_features(fitted, rows)
    bundle_uplift, bundle_propensity, bundle_abstain = bundle.predict(rows)
    assert np.allclose(direct_uplift, bundle_uplift)
    assert np.allclose(direct_propensity, bundle_propensity)
    assert np.array_equal(direct_abstain, bundle_abstain)
    assert manifest.adjustment_set == fitted.adjustment_set
    assert manifest.spec_sha256 == fitted.spec_sha256
    assert manifest.metrics["ate"] == pytest.approx(fitted.primary.ate)


def test_scoring_features_that_do_not_match_the_adjustment_set_is_refused(
    spec: CausalSpec, fitted: EstimationResult
) -> None:
    engine = UpliftEngine(spec, FAST)
    with pytest.raises(ValueError, match="do not match the adjustment set"):
        engine.score_features(fitted, fitted.prepared.features[:, :2])


def test_a_customer_far_outside_the_training_population_gets_an_abstention(
    spec: CausalSpec, fitted: EstimationResult
) -> None:
    engine = UpliftEngine(spec, FAST)
    extreme = fitted.prepared.features[:1].copy()
    extreme[:, :] = fitted.prepared.features.max(axis=0) * 50.0
    _, propensity, abstain = engine.score_features(fitted, extreme)
    assert abstain.shape == (1,)
    assert bool(abstain[0]) == bool(
        propensity[0] < FAST.gate.propensity_low or propensity[0] > FAST.gate.propensity_high
    )


# --- configuration -----------------------------------------------------------------------------


def test_the_configuration_hash_covers_everything_that_changes_a_number() -> None:
    base = EstimationConfig()
    assert base.sha256() == EstimationConfig().sha256()
    for update in (
        {"n_folds": 7},
        {"propensity_clip": 0.02},
        {"level": 0.9},
        {"primary": "dml"},
        {"nuisance": NuisanceConfig(kind="ridge")},
        {"gate": GatePolicy(max_abs_smd=0.3)},
    ):
        assert base.model_copy(update=update).sha256() != base.sha256()


def test_the_configuration_refuses_settings_that_would_disable_cross_fitting() -> None:
    with pytest.raises(ValidationError, match="n_folds"):
        EstimationConfig(n_folds=1)
    with pytest.raises(ValidationError, match="primary"):
        EstimationConfig(primary="ipw")
    with pytest.raises(ValidationError, match="level"):
        EstimationConfig(level=1.0)


def test_switching_the_primary_estimator_switches_what_the_refuters_examine(
    spec: CausalSpec, confounded: SimulatedCampaign
) -> None:
    config = FAST.model_copy(update={"primary": "dml"})
    result = UpliftEngine(spec, config).fit(confounded.frame)
    assert result.primary_name == "dml"
    assert result.primary is result.estimates["dml"]


def test_a_bootstrap_interval_is_attached_when_asked_for(
    spec: CausalSpec, confounded: SimulatedCampaign
) -> None:
    config = FAST.model_copy(update={"n_bootstrap": 200})
    result = UpliftEngine(spec, config).fit(confounded.frame)
    low, high = result.estimates["dml"].bootstrap_ci or (0.0, 0.0)
    assert low < result.estimates["dml"].ate < high


def test_requesting_a_post_treatment_variable_is_refused_by_the_guard(spec: CausalSpec) -> None:
    from double_ml_uplift.causal.identification import verify_adjustment_set

    with pytest.raises(LeakageError) as info:
        verify_adjustment_set(spec, ["tenure_months", "post_offer_logins"])
    assert [name for name, _ in info.value.offenders] == ["post_offer_logins"]
