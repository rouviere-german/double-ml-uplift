"""Offline end-to-end demo. No network, no API key, a few seconds on one CPU.

It walks the whole story the repository is about:

1. a specification that mislabels a post-treatment column is refused at load;
2. a frame that still carries the potential outcomes is refused by the data contract;
3. asking to adjust for a mediator is refused by the leakage guard;
4. a specification with an unmeasured common cause is refused by identification;
5. on a well-specified confounded campaign, the naive difference in means is wrong while AIPW
   and DML recover the true average effect, and the refuters pass;
6. a campaign whose historical targeting was near-deterministic is refused by the overlap gate,
   with reasons, instead of returning a number;
7. a campaign with an unmeasured confounder passes every gate but is caught by the randomized
   pilot consistency check;
8. a budget-constrained targeting list is built, valued on the randomized pilot, checked for
   allocation fairness, and then held for approval: the requesting operator cannot approve it,
   a second operator can, and the whole sequence is in a verifiable hash chain;
9. the scoring bundle round-trips without pickle, and a tampered artifact is refused.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from double_ml_uplift.causal.bundle import ArtifactError, ScoringBundle
from double_ml_uplift.causal.diagnostics import NotEstimableError
from double_ml_uplift.causal.engine import EstimationConfig, UpliftEngine
from double_ml_uplift.causal.estimators import pehe
from double_ml_uplift.causal.identification import (
    LeakageError,
    NotIdentifiedError,
    identify_backdoor,
    verify_adjustment_set,
)
from double_ml_uplift.causal.scenarios import load_scenario
from double_ml_uplift.causal.scm import simulate_campaign
from double_ml_uplift.causal.spec import CausalSpec, SpecError, bundled_spec
from double_ml_uplift.causal.targeting import true_policy_value
from double_ml_uplift.ml.contracts import PotentialOutcomeLeakError, validate_campaign_frame
from double_ml_uplift.ml.drift import DriftMonitor
from double_ml_uplift.service.approvals import ApprovalError, ApprovalStore, ExportPayload
from double_ml_uplift.service.audit import AuditLog, verify_file

__all__ = ["run_uplift_demo"]

_DEMO_ROWS = 4_000


def _refused_specification() -> dict[str, Any]:
    try:
        bundled_spec("leaky")
    except SpecError as exc:
        return {"refused": True, "reason": str(exc).splitlines()[0][:160]}
    return {"refused": False, "reason": ""}  # pragma: no cover - the bundled spec is invalid


def _refused_contract(spec: CausalSpec, frame: Any, oracle: Any) -> dict[str, Any]:
    leaked = frame.copy()
    leaked["y0"] = oracle["y0"].to_numpy()
    leaked["true_cate"] = oracle["true_cate"].to_numpy()
    try:
        validate_campaign_frame(leaked, spec)
    except PotentialOutcomeLeakError as exc:
        return {"refused": True, "columns": exc.columns}
    return {"refused": False, "columns": []}  # pragma: no cover - the columns are always refused


def _refused_leakage(spec: CausalSpec) -> dict[str, Any]:
    try:
        verify_adjustment_set(spec, ["tenure_months", "post_offer_logins"])
    except LeakageError as exc:
        return {"refused": True, "offenders": [name for name, _ in exc.offenders]}
    return {"refused": False, "offenders": []}  # pragma: no cover - the mediator is always refused


def _refused_identification() -> dict[str, Any]:
    identification = identify_backdoor(bundled_spec("unobserved"))
    try:
        identification.require()
    except NotIdentifiedError as exc:
        return {"refused": True, "reasons": list(exc.reasons)}
    return {"refused": False, "reasons": []}  # pragma: no cover - the latent cause always refuses


def _refused_overlap(spec: CausalSpec) -> dict[str, Any]:
    campaign = simulate_campaign(load_scenario("weak_overlap", n_customers=_DEMO_ROWS))
    engine = UpliftEngine(spec, EstimationConfig(run_refuters=False))
    try:
        engine.fit(campaign.frame)
    except NotEstimableError as exc:
        report = exc.report.to_dict() if exc.report else {}
        return {
            "refused": True,
            "reasons": list(exc.reasons),
            "fraction_outside_overlap": report.get("fraction_outside"),
        }
    return {"refused": False, "reasons": []}  # pragma: no cover - the gate always refuses here


def _hidden_confounder(spec: CausalSpec) -> dict[str, Any]:
    campaign = simulate_campaign(load_scenario("hidden_confounder", n_customers=_DEMO_ROWS))
    engine = UpliftEngine(spec, EstimationConfig())
    result = engine.fit(campaign.frame)
    pilot = next(
        item
        for item in (result.refutation.results if result.refutation else ())
        if item.name == "pilot_consistency"
    )
    return {
        "true_ate": round(campaign.true_ate, 3),
        "reported_ate": round(result.primary.ate, 3),
        "gates_passed": result.diagnostics.estimable,
        "pilot_consistency_flagged": not pilot.passed,
        "pilot_detail": pilot.detail,
        "flagged": list(result.refutation.flagged) if result.refutation else [],
    }


def _approval_flow(workdir: Path, payload: ExportPayload) -> dict[str, Any]:
    audit = AuditLog(workdir / "audit.jsonl")
    store = ApprovalStore(workdir / "approvals.json", ttl_seconds=600.0)
    record = store.request("operator-a", payload)
    audit.append("operator-a", "export_requested", {"request_id": record.request_id})

    self_approval_refused = False
    try:
        store.decide(record.request_id, "operator-a", approve=True)
    except ApprovalError:
        self_approval_refused = True
        audit.append("operator-a", "self_approval_refused", {"request_id": record.request_id})

    store.decide(record.request_id, "operator-b", approve=True)
    audit.append("operator-b", "export_decision", {"request_id": record.request_id})
    store.consume(record.request_id, "operator-a", payload)
    audit.append("operator-a", "export_released", {"request_id": record.request_id})

    replay_refused = False
    try:
        store.consume(record.request_id, "operator-a", payload)
    except ApprovalError:
        replay_refused = True

    tampered = replace(payload, pseudonyms=payload.pseudonyms[:-1])
    wrong_payload_refused = False
    try:
        store.consume(record.request_id, "operator-a", tampered)
    except ApprovalError:
        wrong_payload_refused = True

    return {
        "request_id": record.request_id,
        "self_approval_refused": self_approval_refused,
        "replay_refused": replay_refused,
        "wrong_payload_refused": wrong_payload_refused,
        "audit_entries": verify_file(audit.path, expect_head=audit.head),
        "audit_head": audit.head,
    }


def _bundle_round_trip(engine: UpliftEngine, result: Any, workdir: Path) -> dict[str, Any]:
    manifest, artifacts = engine.bundle(result)
    ScoringBundle.write(workdir / "bundle", manifest, artifacts)
    loaded = ScoringBundle.read(workdir / "bundle")
    uplift, _, _ = loaded.predict(result.prepared.features[:64])
    direct, _, _ = engine.score_features(result, result.prepared.features[:64])
    tampered = dict(artifacts)
    tampered["cate.bin"] = artifacts["cate.bin"] + b" "
    tamper_refused = False
    try:
        ScoringBundle.load(manifest, tampered)
    except ArtifactError:
        tamper_refused = True
    return {
        "formats": [manifest.propensity.format, manifest.cate.format],
        "predictions_match": bool(np.allclose(uplift, direct)),
        "tampered_artifact_refused": tamper_refused,
        "calibration_slope": round(result.calibration.slope(), 3),
    }


def run_uplift_demo(workdir: Path) -> dict[str, Any]:
    """Run the whole story and return a JSON-serializable summary."""
    workdir.mkdir(parents=True, exist_ok=True)
    spec = bundled_spec("retention_offer")
    campaign = simulate_campaign(load_scenario("heterogeneous", n_customers=_DEMO_ROWS))
    truth = campaign.true_ate

    engine = UpliftEngine(spec, EstimationConfig(n_bootstrap=400))
    result = engine.fit(campaign.frame)
    estimates = {
        name: {
            "ate": round(estimate.ate, 3),
            "ci": [round(estimate.ci_low, 3), round(estimate.ci_high, 3)],
            "covers_truth": estimate.covers(truth),
        }
        for name, estimate in result.estimates.items()
    }

    outcome = engine.target(result, budget_fraction=0.2, cost_per_offer=1.0)
    y0, y1 = campaign.potential_outcomes
    policies = {
        name: {
            "n_selected": plan.n_selected,
            "fairness": outcome.fairness[name].to_dict(),
            "pilot_value": outcome.pilot_values[name].to_dict(),
            "true_value": round(true_policy_value(plan.selected, y0, y1), 3),
            "pilot_ci_covers_true_value": outcome.pilot_values[name].covers(
                true_policy_value(plan.selected, y0, y1)
            ),
        }
        for name, plan in outcome.plans.items()
    }
    selected = outcome.plans["uplift"].selected
    payload = ExportPayload(
        spec_sha256=result.spec_sha256,
        config_sha256=result.config_sha256,
        dataset_sha256=result.dataset_sha256,
        ranking="uplift",
        budget_fraction=0.2,
        cost_per_offer=1.0,
        pseudonyms=tuple(
            f"cid_{index:040x}" for index, chosen in enumerate(selected.tolist()) if chosen
        ),
    )

    monitor = DriftMonitor({"propensity": result.fitted.propensity})
    shifted = np.clip(result.fitted.propensity * 1.6, 0.001, 0.999)
    drift = monitor.check({"propensity": shifted})

    return {
        "scenario": campaign.config.name,
        "rows": result.prepared.n,
        "spec_sha256": result.spec_sha256[:16],
        "scm_config_sha256": campaign.config_sha256[:16],
        "fold_digest": result.fold_digest[:16],
        "adjustment_set": list(result.adjustment_set),
        "true_ate": round(truth, 3),
        "estimates": estimates,
        "dml_bootstrap_ci": [
            round(value, 3) for value in (result.estimates["dml"].bootstrap_ci or ())
        ],
        "refutation": result.refutation.to_dict() if result.refutation else None,
        "cate_pehe": {
            "dr_learner": round(pehe(result.cate.out_of_fold, campaign.true_cate), 3),
            "outcome_regression": round(
                pehe(result.fitted.mu1 - result.fitted.mu0, campaign.true_cate), 3
            ),
            "constant_ate": round(
                pehe(np.full(result.prepared.n, result.primary.ate), campaign.true_cate), 3
            ),
        },
        "targeting": {
            "budget_fraction": 0.2,
            "abstained": outcome.abstained,
            "fairness": {name: check.to_dict() for name, check in outcome.fairness.items()},
            "policies": policies,
        },
        "approval": _approval_flow(workdir, payload),
        "bundle": _bundle_round_trip(engine, result, workdir),
        "drift": {
            "propensity_psi_shifted": round(drift.features[0].psi, 4),
            "drifted": drift.drifted,
        },
        "refusals": {
            "specification": _refused_specification(),
            "data_contract": _refused_contract(spec, campaign.frame, campaign.oracle),
            "leakage_guard": _refused_leakage(spec),
            "identification": _refused_identification(),
            "overlap_gate": _refused_overlap(spec),
        },
        "unmeasured_confounding": _hidden_confounder(spec),
    }
