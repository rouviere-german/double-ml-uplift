"""The HTTP service: roles, refusals, pseudonymous scoring and the approved export."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from double_ml_uplift.causal.engine import EstimationConfig
from double_ml_uplift.causal.nuisance import NuisanceConfig
from double_ml_uplift.causal.publish import MODEL_NAME, load_published_bundle
from double_ml_uplift.causal.scenarios import load_scenario
from double_ml_uplift.causal.scm import SimulatedCampaign, simulate_campaign
from double_ml_uplift.causal.spec import bundled_spec
from double_ml_uplift.core.config import Settings
from double_ml_uplift.core.security import hash_api_key
from double_ml_uplift.ml.registry import ModelRegistry
from double_ml_uplift.service.app import create_service_app
from double_ml_uplift.service.approvals import ApprovalStore
from double_ml_uplift.service.auth import issue_token
from double_ml_uplift.service.identifiers import (
    PSEUDONYM_PATTERN,
    CustomerIdHasher,
    MissingPepperError,
)

pytestmark = pytest.mark.integration

SECRET = "service-test-signing-key-" + "z" * 24
PEPPER = "service-test-pepper-" + "q" * 24
API_KEY = "service-test-api-key-" + "k" * 24
ROWS = 1_200

CONFIG = EstimationConfig(
    n_folds=3,
    nuisance=NuisanceConfig(kind="ridge"),
    cate_learner=NuisanceConfig(kind="ridge"),
    run_refuters=False,
)


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "test",
        "jwt_secret": SECRET,
        "customer_id_pepper": PEPPER,
        "data_dir": tmp_path,
        "min_cohort_size": 50,
        "rate_limit_per_second": 1_000.0,
        "rate_limit_burst": 1_000,
        "max_body_bytes": 32 * 1024 * 1024,
        "api_key_sha256": [hash_api_key(API_KEY)],
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def campaign(scenario: str = "heterogeneous", *, rows: int = ROWS) -> SimulatedCampaign:
    return simulate_campaign(load_scenario(scenario, n_customers=rows, seed=31337))


def rows_payload(simulated: SimulatedCampaign) -> list[dict[str, Any]]:
    return [
        {str(key): value for key, value in record.items()}
        for record in simulated.frame.to_dict(orient="records")
    ]


def token(settings: Settings, subject: str, role: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_token(settings, subject, role)}"}  # type: ignore[arg-type]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_service_app(
        bundled_spec("retention_offer"),
        settings=settings,
        config=CONFIG,
        registry=CollectorRegistry(),
    )
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def analyst(settings: Settings) -> dict[str, str]:
    return token(settings, "analyst-1", "analyst")


@pytest.fixture
def operator(settings: Settings) -> dict[str, str]:
    return token(settings, "operator-a", "operator")


def estimate(client: TestClient, headers: dict[str, str], **body: Any) -> Any:
    payload = {"rows": rows_payload(campaign()), **body}
    return client.post("/v1/estimate", json=payload, headers=headers)


# --- probes and roles -------------------------------------------------------------------------


def test_the_probes_are_public_and_readiness_reports_the_jwt_check(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["checks"] == {"jwt": "ok"}


def test_without_a_signing_key_every_role_route_fails_closed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, jwt_secret=None)
    app = create_service_app(
        bundled_spec("retention_offer"), settings=settings, registry=CollectorRegistry()
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/readyz").status_code == 503
        assert client.post("/v1/estimate", json={"rows": [{"a": 1}, {"a": 2}]}).status_code == 401


@pytest.mark.parametrize("path", ["/v1/estimate", "/v1/score", "/v1/targeting/requests"])
def test_every_route_requires_a_bearer_token(client: TestClient, path: str) -> None:
    response = client.post(path, json={})
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert "error" in response.json()


def test_an_analyst_may_not_request_an_export(client: TestClient, analyst: dict[str, str]) -> None:
    response = client.post("/v1/targeting/requests", json={}, headers=analyst)
    assert response.status_code == 403
    assert "operator role" in response.json()["error"]["message"]


@pytest.mark.parametrize(
    "bad",
    ["", "not-a-token", "Bearer", "x" * 5_000],
)
def test_malformed_tokens_are_refused(client: TestClient, bad: str) -> None:
    response = client.post(
        "/v1/estimate", json={"rows": [{"a": 1}, {"a": 2}]}, headers={"Authorization": bad}
    )
    assert response.status_code == 401


def test_a_token_signed_with_another_key_is_refused(client: TestClient, tmp_path: Path) -> None:
    foreign = make_settings(tmp_path, jwt_secret="a-completely-different-key-" + "y" * 24)
    response = client.post(
        "/v1/estimate",
        json={"rows": [{"a": 1}, {"a": 2}]},
        headers=token(foreign, "attacker", "operator"),
    )
    assert response.status_code == 401


def test_a_token_that_outlives_the_configured_maximum_is_refused(
    client: TestClient, settings: Settings
) -> None:
    long_lived = issue_token(
        settings, "analyst-1", "analyst", lifetime_seconds=settings.jwt_max_lifetime_seconds + 60
    )
    response = client.post(
        "/v1/estimate",
        json={"rows": [{"a": 1}, {"a": 2}]},
        headers={"Authorization": f"Bearer {long_lived}"},
    )
    assert response.status_code == 401


# --- estimation -------------------------------------------------------------------------------


def test_an_estimate_returns_every_estimator_with_its_diagnostics(
    client: TestClient, analyst: dict[str, str]
) -> None:
    response = estimate(client, analyst)
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body["estimates"]) == {"naive", "outcome_regression", "ipw", "aipw", "dml"}
    assert body["primary"] == "aipw"
    assert body["estimand"] == "backdoor-adjustment"
    assert body["n_rows"] == ROWS
    assert len(body["dataset_sha256"]) == 64
    assert body["diagnostics"]["n"] == ROWS
    assert "post_offer_logins" not in body["adjustment_set"]


def test_an_estimate_is_recorded_in_the_registry_it_reports(
    client: TestClient, settings: Settings, analyst: dict[str, str]
) -> None:
    """The run version in the response is a real registry entry that scores like the service.

    The reproducibility record the specification asks for -- specification, configuration,
    dataset and fold digests plus the environment and the lockfile -- lives in that entry, so an
    answer can be traced back to the exact thing that produced it.
    """
    body = estimate(client, analyst).json()
    assert len(body["run_version"]) == 64

    registry = ModelRegistry(Path(settings.data_dir) / "registry")
    version = registry.get(MODEL_NAME, body["run_version"])
    assert registry.get(MODEL_NAME, "latest").digest == body["run_version"]

    run = version.manifest
    assert run.params["spec_sha256"] == body["spec_sha256"]
    assert run.params["config_sha256"] == body["config_sha256"]
    assert run.params["dataset_sha256"] == body["dataset_sha256"]
    assert run.params["fold_digest"] == body["fold_digest"]
    assert run.params["n_rows"] == body["n_rows"]
    assert run.python.startswith("3.")
    assert set(run.packages) >= {"numpy", "pandas", "scikit-learn"}

    # The stored bundle is the one the service is scoring with, not a copy that drifted.
    published = load_published_bundle(registry, body["run_version"])
    scored = client.post(
        "/v1/score", json={"customers": rows_payload(campaign())[:32]}, headers=analyst
    ).json()
    features = np.column_stack(
        [campaign().frame[name].to_numpy(dtype=float)[:32] for name in published.adjustment_set]
    )
    uplift, _, _ = published.predict(features)
    assert [round(float(value), 4) for value in uplift] == [
        row["uplift"] for row in scored["scored"]
    ]


def test_a_deployment_salt_changes_the_folds_and_only_its_hash_leaves_the_process(
    tmp_path: Path, analyst: dict[str, str]
) -> None:
    """The bundled fold salt is a public literal; a deployment can key it per environment.

    Two services differing only in `APP_FOLD_SALT` must assign the same customers to different
    folds, and the salt itself must never appear in a response.
    """
    digests = []
    for salt in ("first-deployment-salt", "second-deployment-salt"):
        settings = make_settings(tmp_path / salt, fold_salt=salt)
        app = create_service_app(
            bundled_spec("retention_offer"),
            settings=settings,
            config=CONFIG,
            registry=CollectorRegistry(),
        )
        with TestClient(app, raise_server_exceptions=False) as scoped:
            body = estimate(scoped, analyst).json()
        digests.append((body["fold_digest"], body["config_sha256"], json.dumps(body)))

    assert digests[0][0] != digests[1][0], "a different salt must move customers between folds"
    assert digests[0][1] != digests[1][1], "the salt is part of what defines the configuration"
    for _, _, document in digests:
        assert "deployment-salt" not in document


def test_without_a_deployment_salt_the_bundled_default_is_used(
    client: TestClient, settings: Settings, analyst: dict[str, str]
) -> None:
    assert settings.fold_salt is None
    body = estimate(client, analyst).json()
    assert body["config_sha256"] == CONFIG.sha256()


def test_a_cohort_below_the_minimum_is_refused(client: TestClient, analyst: dict[str, str]) -> None:
    response = client.post(
        "/v1/estimate", json={"rows": rows_payload(campaign())[:10]}, headers=analyst
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "cohort_too_small"


def test_a_positivity_violation_returns_reasons_and_no_estimate(
    client: TestClient, analyst: dict[str, str]
) -> None:
    payload = {"rows": rows_payload(campaign("weak_overlap"))}
    response = client.post("/v1/estimate", json=payload, headers=analyst)
    assert response.status_code == 422
    body = response.json()["error"]
    assert body["code"] == "not_estimable"
    assert "positivity" in body["message"]
    # No point estimate anywhere in the refusal: the gate runs before one exists.
    assert set(response.json()) == {"error"}
    for field in ("ci_low", "ci_high", "estimates", "aipw", "adjustment_set"):
        assert field not in response.text


def test_a_frame_carrying_the_oracle_is_refused_by_the_contract(
    client: TestClient, analyst: dict[str, str]
) -> None:
    simulated = campaign()
    rows = rows_payload(simulated)
    truth = simulated.oracle["true_cate"].tolist()
    for row, value in zip(rows, truth, strict=True):
        row["true_cate"] = value
    response = client.post("/v1/estimate", json={"rows": rows}, headers=analyst)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "data_contract"
    assert "true_cate" in response.json()["error"]["message"]


def test_asking_to_adjust_for_a_mediator_is_refused(
    client: TestClient, analyst: dict[str, str]
) -> None:
    response = estimate(client, analyst, adjustment_set=["tenure_months", "post_offer_logins"])
    assert response.status_code == 422
    body = response.json()["error"]
    assert body["code"] == "leakage"
    assert "post_offer_logins" in body["message"]


def test_an_adjustment_set_that_leaves_a_backdoor_open_is_refused(
    client: TestClient, analyst: dict[str, str]
) -> None:
    response = estimate(client, analyst, adjustment_set=["tenure_months"])
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "not_identified"


@pytest.mark.parametrize(("field", "value"), [("n_folds", 11), ("n_bootstrap", 4_000)])
def test_per_request_caps_are_enforced(
    client: TestClient, analyst: dict[str, str], field: str, value: int
) -> None:
    response = estimate(client, analyst, **{field: value})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "cap_exceeded"


def test_a_row_that_is_not_a_flat_record_is_refused_before_pandas(
    client: TestClient, analyst: dict[str, str]
) -> None:
    for bad in ({"a": None}, {"a": [1, 2]}, {"Bad-Name": 1}, {"a": "x" * 500}):
        response = client.post("/v1/estimate", json={"rows": [bad, {"a": 1}]}, headers=analyst)
        assert response.status_code == 422, bad


# --- scoring ----------------------------------------------------------------------------------


def test_scoring_before_an_estimate_is_a_conflict(
    client: TestClient, analyst: dict[str, str]
) -> None:
    response = client.post(
        "/v1/score", json={"customers": [{"customer_id": "cust_0000001"}]}, headers=analyst
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "no_estimate"


def test_scored_customers_come_back_pseudonymised_with_group_intervals(
    client: TestClient, analyst: dict[str, str]
) -> None:
    assert estimate(client, analyst).status_code == 200
    customers = rows_payload(campaign())[:40]
    response = client.post(
        "/v1/score", json={"customers": customers, "cost_per_offer": 1.0}, headers=analyst
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["scored"]) == 40
    raw_ids = {row["customer_id"] for row in customers}
    for scored in body["scored"]:
        assert PSEUDONYM_PATTERN.fullmatch(scored["customer_ref"])
        assert scored["customer_ref"] not in raw_ids
        if not scored["abstain"]:
            assert scored["ci_low"] <= scored["group_effect"] <= scored["ci_high"]
            assert scored["recommended_action"] in {"send_offer", "hold", "uncertain"}
    assert not any(raw in response.text for raw in raw_ids)
    assert 0.0 <= body["recommended_treatment_rate"] <= 1.0
    assert body["propensity_psi"] >= 0.0


def test_the_pseudonym_is_stable_and_keyed(settings: Settings, tmp_path: Path) -> None:
    hasher = CustomerIdHasher.from_settings(settings)
    first = hasher.hash("cust_0000001")
    assert first == hasher.hash("cust_0000001")
    assert first != hasher.hash("cust_0000002")

    other = CustomerIdHasher.from_settings(
        make_settings(tmp_path, customer_id_pepper="another-pepper-" + "w" * 24)
    )
    assert other.hash("cust_0000001") != first
    assert hasher.hash_all(["a", "b"]) == [hasher.hash("a"), hasher.hash("b")]


def test_without_a_pepper_the_service_refuses_to_start(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, customer_id_pepper=None)
    with pytest.raises(MissingPepperError, match="raw customer ids"):
        create_service_app(
            bundled_spec("retention_offer"), settings=settings, registry=CollectorRegistry()
        )


def test_an_over_long_customer_id_is_refused_by_the_hasher(settings: Settings) -> None:
    hasher = CustomerIdHasher.from_settings(settings)
    with pytest.raises(ValueError, match="between 1 and 256"):
        hasher.hash("")
    with pytest.raises(ValueError, match="between 1 and 256"):
        hasher.hash("x" * 300)


def test_a_scoring_request_missing_adjustment_columns_is_refused(
    client: TestClient, analyst: dict[str, str]
) -> None:
    assert estimate(client, analyst).status_code == 200
    response = client.post(
        "/v1/score", json={"customers": [{"customer_id": "cust_0000001"}]}, headers=analyst
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "missing_columns"


def test_a_scoring_request_without_a_customer_id_is_refused(
    client: TestClient, analyst: dict[str, str]
) -> None:
    assert estimate(client, analyst).status_code == 200
    customers = rows_payload(campaign())[:5]
    for row in customers:
        row.pop("customer_id")
    response = client.post("/v1/score", json={"customers": customers}, headers=analyst)
    assert response.status_code == 422


def test_too_many_customers_in_one_batch_is_refused(client: TestClient, tmp_path: Path) -> None:
    settings = make_settings(tmp_path, max_score_rows=5)
    app = create_service_app(
        bundled_spec("retention_offer"),
        settings=settings,
        config=CONFIG,
        registry=CollectorRegistry(),
    )
    with TestClient(app, raise_server_exceptions=False) as scoped:
        headers = token(settings, "analyst-1", "analyst")
        assert estimate(scoped, headers).status_code == 200
        response = scoped.post(
            "/v1/score", json={"customers": rows_payload(campaign())[:6]}, headers=headers
        )
        assert response.status_code == 413


# --- the approved export -----------------------------------------------------------------------


def request_export(client: TestClient, operator: dict[str, str], **body: Any) -> Any:
    payload = {"budget_fraction": 0.2, "cost_per_offer": 1.0, "ranking": "uplift", **body}
    return client.post("/v1/targeting/requests", json=payload, headers=operator)


def test_an_export_needs_an_estimate_first(client: TestClient, operator: dict[str, str]) -> None:
    assert request_export(client, operator).status_code == 409


def test_metrics_require_an_api_key_not_a_role_token(
    client: TestClient, analyst: dict[str, str]
) -> None:
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers=analyst).status_code == 401
    assert client.get("/metrics", headers={"X-API-Key": API_KEY}).status_code == 200


def test_the_full_approval_flow_releases_only_pseudonyms(
    client: TestClient, settings: Settings, analyst: dict[str, str], operator: dict[str, str]
) -> None:
    assert estimate(client, analyst).status_code == 200
    requested = request_export(client, operator)
    assert requested.status_code == 200, requested.text
    body = requested.json()
    request_id = body["request_id"]
    assert body["n_selected"] > 0
    assert body["fairness"]["blocked"] is False
    assert body["pilot_value"]["method"] == "randomized-pilot"
    first_head = body["audit_head"]

    # The requester cannot approve its own export.
    refused = client.post(
        f"/v1/targeting/requests/{request_id}/decision", json={"approve": True}, headers=operator
    )
    assert refused.status_code == 409
    message = refused.json()["error"]["message"]
    assert "cannot be approved by the operator that asked for it" in message

    second = token(settings, "operator-b", "operator")
    approved = client.post(
        f"/v1/targeting/requests/{request_id}/decision", json={"approve": True}, headers=second
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"

    # Only the requesting operator may spend the approval.
    stolen = client.post("/v1/targeting/export", json={"request_id": request_id}, headers=second)
    assert stolen.status_code == 409

    released = client.post(
        "/v1/targeting/export", json={"request_id": request_id}, headers=operator
    )
    assert released.status_code == 200, released.text
    payload = released.json()
    assert payload["ranking"] == "uplift"
    assert len(payload["customer_refs"]) == body["n_selected"]
    assert all(PSEUDONYM_PATTERN.fullmatch(ref) for ref in payload["customer_refs"])
    assert payload["audit_head"] != first_head

    # An approval is spent once.
    replay = client.post("/v1/targeting/export", json={"request_id": request_id}, headers=operator)
    assert replay.status_code == 409


def test_a_denied_export_cannot_be_released(
    client: TestClient, settings: Settings, analyst: dict[str, str], operator: dict[str, str]
) -> None:
    assert estimate(client, analyst).status_code == 200
    request_id = request_export(client, operator).json()["request_id"]
    second = token(settings, "operator-b", "operator")
    denied = client.post(
        f"/v1/targeting/requests/{request_id}/decision", json={"approve": False}, headers=second
    )
    assert denied.json()["status"] == "denied"
    response = client.post(
        "/v1/targeting/export", json={"request_id": request_id}, headers=operator
    )
    assert response.status_code == 409
    assert "not approved" in response.json()["error"]["message"]


def test_an_unapproved_export_cannot_be_released(
    client: TestClient, analyst: dict[str, str], operator: dict[str, str]
) -> None:
    assert estimate(client, analyst).status_code == 200
    request_id = request_export(client, operator).json()["request_id"]
    response = client.post(
        "/v1/targeting/export", json={"request_id": request_id}, headers=operator
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "approval_refused"


def test_an_approval_does_not_carry_over_to_a_different_plan(
    client: TestClient, settings: Settings, analyst: dict[str, str], operator: dict[str, str]
) -> None:
    """The approval commits to the exact list of customers, so a new budget invalidates it."""
    assert estimate(client, analyst).status_code == 200
    request_id = request_export(client, operator, budget_fraction=0.2).json()["request_id"]
    second = token(settings, "operator-b", "operator")
    client.post(
        f"/v1/targeting/requests/{request_id}/decision", json={"approve": True}, headers=second
    )
    # A second request replaces the stored plan with a different one.
    assert request_export(client, operator, budget_fraction=0.5).status_code == 200
    response = client.post(
        "/v1/targeting/export", json={"request_id": request_id}, headers=operator
    )
    assert response.status_code == 409
    assert "does not match the approval" in response.json()["error"]["message"]


def test_a_disparate_allocation_is_blocked_and_audited(client: TestClient, tmp_path: Path) -> None:
    settings = make_settings(tmp_path, fairness_max_disparity=1e-6)
    app = create_service_app(
        bundled_spec("retention_offer"),
        settings=settings,
        config=CONFIG,
        registry=CollectorRegistry(),
    )
    with TestClient(app, raise_server_exceptions=False) as scoped:
        analyst_headers = token(settings, "analyst-1", "analyst")
        operator_headers = token(settings, "operator-a", "operator")
        assert estimate(scoped, analyst_headers).status_code == 200
        response = request_export(scoped, operator_headers)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "fairness_disparity"

    from double_ml_uplift.service.audit import verify_file

    log = tmp_path / "audit" / "exports.jsonl"
    assert verify_file(log) >= 1
    assert "export_blocked_by_fairness" in log.read_text(encoding="utf-8")


def test_a_full_approval_store_refuses_rather_than_evicting_a_live_request(
    tmp_path: Path, analyst: dict[str, str], operator: dict[str, str], settings: Settings
) -> None:
    """Flooding the store must not silently drop somebody else's pending export request."""
    app = create_service_app(
        bundled_spec("retention_offer"),
        settings=settings,
        config=CONFIG,
        registry=CollectorRegistry(),
    )
    app.state.service.approvals = ApprovalStore(
        Path(settings.data_dir) / "approvals.json",
        ttl_seconds=settings.approval_ttl_seconds,
        max_records=2,
    )
    with TestClient(app, raise_server_exceptions=False) as scoped:
        assert estimate(scoped, analyst).status_code == 200
        first = request_export(scoped, operator, budget_fraction=0.2).json()["request_id"]
        assert request_export(scoped, operator, budget_fraction=0.3).status_code == 200

        full = request_export(scoped, operator, budget_fraction=0.4)
        assert full.status_code == 503, full.text
        assert full.json()["error"]["code"] == "approval_store_full"

        # The first request is still there to be decided on.
        second = token(settings, "operator-b", "operator")
        decided = scoped.post(
            f"/v1/targeting/requests/{first}/decision", json={"approve": True}, headers=second
        )
        assert decided.status_code == 200
        assert decided.json()["status"] == "approved"


def test_a_targeting_list_below_the_minimum_cohort_is_refused(
    client: TestClient, analyst: dict[str, str], operator: dict[str, str]
) -> None:
    assert estimate(client, analyst).status_code == 200
    response = request_export(client, operator, budget_fraction=0.01)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "cohort_too_small"


def test_a_policy_the_pilot_cannot_value_reports_no_value_rather_than_a_guess(
    client: TestClient, analyst: dict[str, str], operator: dict[str, str], tmp_path: Path
) -> None:
    """A tight budget leaves too few randomized rows among the targeted customers to estimate
    the policy's value. The plan is still built; the value is simply absent."""
    from double_ml_uplift.causal.engine import UpliftEngine

    assert estimate(client, analyst).status_code == 200
    engine = UpliftEngine(bundled_spec("retention_offer"), CONFIG)
    result = engine.fit(campaign().frame)
    outcome = engine.target(result, budget_fraction=0.005, cost_per_offer=1.0)
    assert outcome.plans["uplift"].n_selected > 0
    assert "uplift" not in outcome.pilot_values
    assert "treat_all" in outcome.pilot_values


def test_an_unknown_request_id_is_refused(
    client: TestClient, analyst: dict[str, str], operator: dict[str, str]
) -> None:
    assert estimate(client, analyst).status_code == 200
    assert request_export(client, operator).status_code == 200
    unknown = "exp_" + "0" * 32
    response = client.post(
        f"/v1/targeting/requests/{unknown}/decision", json={"approve": True}, headers=operator
    )
    assert response.status_code == 409
    assert client.post(
        "/v1/targeting/export", json={"request_id": unknown}, headers=operator
    ).status_code in {404, 409, 500}


# --- audit and metrics ----------------------------------------------------------------------------


def test_the_audit_chain_covers_the_whole_export_and_verifies(
    client: TestClient, settings: Settings, analyst: dict[str, str], operator: dict[str, str]
) -> None:
    from double_ml_uplift.service.audit import AuditLog, verify_file

    assert estimate(client, analyst).status_code == 200
    request_id = request_export(client, operator).json()["request_id"]
    second = token(settings, "operator-b", "operator")
    client.post(
        f"/v1/targeting/requests/{request_id}/decision", json={"approve": True}, headers=second
    )
    released = client.post(
        "/v1/targeting/export", json={"request_id": request_id}, headers=operator
    )
    assert released.status_code == 200

    path = Path(settings.data_dir) / "audit" / "exports.jsonl"
    assert verify_file(path, expect_head=released.json()["audit_head"]) == 3
    actions = [entry.action for entry in AuditLog(path).entries()]
    assert actions == ["export_requested", "export_decision", "export_released"]

    # Tampering with any entry breaks the chain.
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("export_requested", "export_ignored")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="does not match its digest"):
        verify_file(path)


def test_the_audit_log_never_records_a_raw_customer_id(
    client: TestClient, settings: Settings, analyst: dict[str, str], operator: dict[str, str]
) -> None:
    assert estimate(client, analyst).status_code == 200
    assert request_export(client, operator).status_code == 200
    text = (Path(settings.data_dir) / "audit" / "exports.jsonl").read_text(encoding="utf-8")
    assert "cust_000" not in text


def test_refusals_and_estimates_are_counted_by_reason(
    client: TestClient, analyst: dict[str, str]
) -> None:
    assert estimate(client, analyst).status_code == 200
    assert (
        client.post(
            "/v1/estimate", json={"rows": rows_payload(campaign("weak_overlap"))}, headers=analyst
        ).status_code
        == 422
    )
    body = client.get("/metrics", headers={"X-API-Key": API_KEY}).text
    assert 'uplift_estimates_total{estimator="aipw"} 1.0' in body
    assert 'uplift_refusals_total{reason="positivity"} 1.0' in body


def test_an_unknown_refusal_reason_is_folded_into_a_bounded_label() -> None:
    from double_ml_uplift.service.metrics import REFUSAL_REASONS, create_causal_metrics

    registry = CollectorRegistry()
    metrics = create_causal_metrics(registry)
    metrics.refuse("something_new")
    metrics.refuse("positivity")
    assert registry.get_sample_value("uplift_refusals_total", {"reason": "other"}) == 1.0
    assert registry.get_sample_value("uplift_refusals_total", {"reason": "positivity"}) == 1.0
    assert "other" not in REFUSAL_REASONS


def test_the_fairness_check_is_computed_on_the_ranking_being_released(
    tmp_path: Path, analyst: dict[str, str], operator: dict[str, str]
) -> None:
    """The gate must judge the list the operator asked for, not always the uplift list.

    On this fixture the uplift list has a disparity of about 0.0146 and the propensity list about
    0.0729. A release limit strictly between them must accept the first and refuse the second; a
    gate that always looks at the uplift list would wave the propensity list through and, worse,
    show the operator a diagnostic describing a different list.
    """
    settings = make_settings(tmp_path, fairness_max_disparity=0.04)
    app = create_service_app(
        bundled_spec("retention_offer"),
        settings=settings,
        config=CONFIG,
        registry=CollectorRegistry(),
    )
    with TestClient(app, raise_server_exceptions=False) as scoped:
        assert estimate(scoped, analyst).status_code == 200

        allowed = request_export(scoped, operator, ranking="uplift")
        assert allowed.status_code == 200, allowed.text
        assert allowed.json()["fairness"]["blocked"] is False
        assert allowed.json()["fairness"]["disparity"] < 0.04

        refused = request_export(scoped, operator, ranking="propensity")
        assert refused.status_code == 422, refused.text
        assert refused.json()["error"]["code"] == "fairness_disparity"

    from double_ml_uplift.service.audit import verify_file

    log = tmp_path / "audit" / "exports.jsonl"
    assert verify_file(log) >= 1
    blocked = [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["action"] == "export_blocked_by_fairness"
    ]
    assert len(blocked) == 1
    assert blocked[0]["data"]["ranking"] == "propensity"
    assert blocked[0]["data"]["disparity"] > 0.04


def test_releasing_an_unknown_request_id_is_refused_not_a_server_error(
    client: TestClient, analyst: dict[str, str], operator: dict[str, str]
) -> None:
    """A well-formed id that the store never issued is an approval refusal, like every sibling."""
    assert estimate(client, analyst).status_code == 200
    assert request_export(client, operator).status_code == 200
    response = client.post(
        "/v1/targeting/export", json={"request_id": "exp_" + "0" * 32}, headers=operator
    )
    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "approval_refused"
    assert "unknown export request" in response.json()["error"]["message"]
