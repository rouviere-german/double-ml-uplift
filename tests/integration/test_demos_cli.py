"""The offline demo and the command line entry points."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace as otel_trace

from double_ml_uplift import cli
from double_ml_uplift.causal.demo import run_uplift_demo
from double_ml_uplift.core import tracing
from double_ml_uplift.core.security import hash_api_key
from double_ml_uplift.service.audit import AuditLog

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return run_uplift_demo(tmp_path_factory.mktemp("demo"))


# --- the story the demo tells -------------------------------------------------------------------


def test_the_demo_shows_the_naive_estimate_missing_the_truth(demo: dict[str, Any]) -> None:
    truth = demo["true_ate"]
    assert demo["estimates"]["naive"]["covers_truth"] is False
    assert abs(demo["estimates"]["naive"]["ate"] - truth) > 5.0
    for name in ("aipw", "dml"):
        assert demo["estimates"][name]["covers_truth"] is True
    low, high = demo["dml_bootstrap_ci"]
    assert low < demo["estimates"]["dml"]["ate"] < high


def test_the_demo_records_the_hashes_that_make_the_run_reproducible(
    demo: dict[str, Any],
) -> None:
    for key in ("spec_sha256", "scm_config_sha256", "fold_digest"):
        assert len(demo[key]) == 16
    assert demo["rows"] == 4_000
    assert demo["scenario"] == "heterogeneous"
    assert "post_offer_logins" not in demo["adjustment_set"]


def test_the_demo_shows_every_refusal(demo: dict[str, Any]) -> None:
    refusals = demo["refusals"]
    assert refusals["specification"]["refused"] is True
    assert refusals["data_contract"]["refused"] is True
    assert set(refusals["data_contract"]["columns"]) == {"y0", "true_cate"}
    assert refusals["leakage_guard"]["offenders"] == ["post_offer_logins"]
    assert refusals["identification"]["refused"] is True
    reasons = refusals["identification"]["reasons"]
    assert any("churn_risk" in reason for reason in reasons)
    assert refusals["overlap_gate"]["refused"] is True
    assert refusals["overlap_gate"]["fraction_outside_overlap"] > 0.05


def test_the_demo_shows_the_pilot_catching_unmeasured_confounding(
    demo: dict[str, Any],
) -> None:
    hidden = demo["unmeasured_confounding"]
    assert hidden["gates_passed"] is True
    assert hidden["pilot_consistency_flagged"] is True
    assert hidden["flagged"] == ["pilot_consistency"]
    assert abs(hidden["reported_ate"] - hidden["true_ate"]) > 20.0


def test_the_demo_refuters_pass_on_the_well_specified_campaign(demo: dict[str, Any]) -> None:
    assert demo["refutation"]["passed"] is True
    assert demo["refutation"]["flagged"] == []
    names = {item["name"] for item in demo["refutation"]["results"]}
    assert names == {
        "placebo_treatment",
        "random_common_cause",
        "subset_stability",
        "pilot_consistency",
        "e_value",
    }


def test_the_demo_shows_uplift_targeting_beating_the_alternatives(
    demo: dict[str, Any],
) -> None:
    policies = demo["targeting"]["policies"]
    assert policies["uplift"]["true_value"] > policies["propensity"]["true_value"]
    assert policies["uplift"]["true_value"] > policies["treat_all"]["true_value"]
    for policy in policies.values():
        assert policy["pilot_ci_covers_true_value"] is True
    assert set(demo["targeting"]["fairness"]) == set(demo["targeting"]["policies"])
    assert all(check["blocked"] is False for check in demo["targeting"]["fairness"].values())


def test_the_demo_holds_the_targeting_list_for_a_second_operator(
    demo: dict[str, Any],
) -> None:
    approval = demo["approval"]
    assert approval["self_approval_refused"] is True
    assert approval["replay_refused"] is True
    assert approval["wrong_payload_refused"] is True
    assert approval["audit_entries"] == 4
    assert len(approval["audit_head"]) == 64


def test_the_demo_round_trips_the_bundle_without_pickle(demo: dict[str, Any]) -> None:
    bundle = demo["bundle"]
    assert bundle["formats"] == ["lightgbm-text/v1", "lightgbm-text/v1"]
    assert bundle["predictions_match"] is True
    assert bundle["tampered_artifact_refused"] is True


def test_the_demo_publishes_the_run_and_reads_it_back_through_its_content_address(
    demo: dict[str, Any],
) -> None:
    registry = demo["bundle"]["registry"]
    assert registry["predictions_match"] is True
    assert registry["corrupted_blob_refused"] is True
    assert registry["alias"] == "latest"
    recorded = registry["recorded"]
    assert recorded["python"].startswith("3.")
    assert set(recorded["packages"]) >= {"numpy", "pandas", "scikit-learn"}
    assert len(recorded["fold_digest"]) == 16
    # The demo runs from the checkout, so the lockfile is visible and its digest is recorded.
    assert recorded["lockfile_sha256"] is not None


def test_the_demo_shows_the_dr_learner_beating_both_baselines(demo: dict[str, Any]) -> None:
    pehe = demo["cate_pehe"]
    assert pehe["dr_learner"] < pehe["outcome_regression"]
    assert pehe["dr_learner"] < pehe["constant_ate"]


def test_the_demo_detects_a_shifted_propensity_population(demo: dict[str, Any]) -> None:
    assert demo["drift"]["drifted"] is True
    assert demo["drift"]["propensity_psi_shifted"] > 0.2


def test_the_demo_is_json_serialisable(demo: dict[str, Any]) -> None:
    assert json.loads(json.dumps(demo, sort_keys=True))["rows"] == 4_000


# --- the command line ------------------------------------------------------------------------


@pytest.mark.usefixtures("restore_logging")
def test_cli_demo_prints_one_json_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["demo", "--workdir", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["demo"] == "uplift"
    assert payload["rows"] == 4_000


def test_cli_spec_reports_the_adjustment_set(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["spec", "retention_offer"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["identified"] is True
    assert payload["treatment"] == "offer_sent"
    assert len(payload["sha256"]) == 64
    assert "pilot" not in payload["adjustment_set"]


def test_cli_spec_exits_non_zero_when_the_effect_is_not_identified(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["spec", "unobserved"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["identified"] is False
    assert any("churn_risk" in reason for reason in payload["reasons"])


def test_cli_spec_exits_non_zero_on_an_invalid_specification(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["spec", "leaky"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is False
    assert "leak" in payload["error"]


def test_cli_generate_api_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["generate-api-key"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["api_key_sha256"] == hash_api_key(payload["api_key"])


def test_cli_issue_token_needs_a_configured_secret(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from double_ml_uplift.core import config

    config.get_settings.cache_clear()
    assert cli.main(["issue-token", "analyst-1"]) == 1
    assert "APP_JWT_SECRET" in capsys.readouterr().err

    monkeypatch.setenv("APP_JWT_SECRET", "cli-test-signing-key-" + "v" * 24)
    config.get_settings.cache_clear()
    try:
        assert cli.main(["issue-token", "operator-a", "--role", "operator"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["role"] == "operator"
        assert payload["token"].count(".") == 2
    finally:
        config.get_settings.cache_clear()


def test_cli_audit_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("operator-a", "export_requested", {})
    assert cli.main(["audit", "verify", str(path), "--expect-head", log.head]) == 0
    assert json.loads(capsys.readouterr().out)["entries"] == 1
    assert cli.main(["audit", "verify", str(path), "--expect-head", "0" * 64]) == 1
    assert json.loads(capsys.readouterr().out)["valid"] is False


def test_cli_audit_verify_reports_a_missing_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["audit", "verify", str(tmp_path / "nope.jsonl")]) == 1
    assert "does not exist" in json.loads(capsys.readouterr().out)["error"]


@pytest.mark.usefixtures("restore_logging")
def test_cli_serve_configures_uvicorn_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    from double_ml_uplift.core import config

    monkeypatch.setenv("APP_CUSTOMER_ID_PEPPER", "cli-test-pepper-" + "u" * 24)
    config.get_settings.cache_clear()
    captured: dict[str, Any] = {}

    def fake_run(app: object, **kwargs: Any) -> None:
        captured.update(kwargs, app=app)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(tracing, "_installed", [])
    monkeypatch.setattr(otel_trace, "set_tracer_provider", lambda provider: None)
    try:
        assert cli.main(["serve", "--port", "8123"]) == 0
    finally:
        config.get_settings.cache_clear()
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
    assert captured["server_header"] is False
    assert captured["access_log"] is False


def test_module_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["double_ml_uplift", "generate-api-key"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("double_ml_uplift", run_name="__main__")
    assert exit_info.value.code == 0
    assert "api_key_sha256" in capsys.readouterr().out


def test_an_unknown_command_is_refused() -> None:
    with pytest.raises(SystemExit):
        cli.main(["not-a-command"])
