"""The HTTP service: estimate, score, and release a targeting list under approval.

State is deliberately small and explicit. The service holds one causal specification and one
estimation configuration, and it remembers the *last successful estimation* so that scoring and
targeting refer to something that has actually passed every gate. There is no model store and no
multi-tenancy; both are listed as non-goals in the README.

Every route fails closed:

* identification, contract, overlap and balance failures return ``422`` with the reasons and
  never a point estimate;
* aggregates below the configured minimum cohort size are refused;
* scoring abstains per customer outside the overlap region;
* releasing a targeting list needs an approval from a second operator, bound to the exact payload
  and spent once, and every step is written to the hash-chained audit log.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from fastapi import APIRouter, FastAPI, Request
from prometheus_client import CollectorRegistry

from double_ml_uplift.causal.bundle import ScoringBundle
from double_ml_uplift.causal.diagnostics import NotEstimableError
from double_ml_uplift.causal.engine import (
    EstimationConfig,
    EstimationResult,
    TargetingOutcome,
    UpliftEngine,
)
from double_ml_uplift.causal.identification import (
    LeakageError,
    NotIdentifiedError,
    verify_adjustment_set,
)
from double_ml_uplift.causal.spec import CausalSpec
from double_ml_uplift.core.app import create_app
from double_ml_uplift.core.config import Settings, get_settings
from double_ml_uplift.core.errors import AppError
from double_ml_uplift.core.logging import get_logger
from double_ml_uplift.ml.contracts import ContractLimits, DataContractError
from double_ml_uplift.ml.drift import psi
from double_ml_uplift.service.approvals import (
    ApprovalError,
    ApprovalStore,
    ExportPayload,
    payload_digest,
)
from double_ml_uplift.service.audit import AuditLog
from double_ml_uplift.service.auth import AnalystPrincipal, JwtVerifier, OperatorPrincipal
from double_ml_uplift.service.identifiers import CustomerIdHasher
from double_ml_uplift.service.metrics import CausalMetrics, create_causal_metrics
from double_ml_uplift.service.models import (
    ApprovalDecision,
    EstimateRequest,
    EstimateResponse,
    ExportRelease,
    ExportRequestBody,
    ExportRequestResponse,
    ExportResponse,
    ScoredCustomer,
    ScoreRequest,
    ScoreResponse,
)

__all__ = ["ServiceState", "build_router", "create_service_app"]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

_log = get_logger("double_ml_uplift.service")
_GATE_REASONS = {
    "positivity": "positivity",
    "balance": "balance",
    "effective sample": "effective_sample",
}


@dataclass
class ServiceState:
    """Everything the routes share. One specification, one configuration, one fitted result."""

    spec: CausalSpec
    config: EstimationConfig
    settings: Settings
    hasher: CustomerIdHasher
    audit: AuditLog
    approvals: ApprovalStore
    metrics: CausalMetrics
    result: EstimationResult | None = None
    bundle: ScoringBundle | None = None
    targeting: TargetingOutcome | None = None
    targeting_request: ExportRequestBody | None = None

    def engine(self, *, n_folds: int | None = None, n_bootstrap: int | None = None) -> UpliftEngine:
        updates: dict[str, Any] = {}
        if n_folds is not None:
            updates["n_folds"] = n_folds
        if n_bootstrap is not None:
            updates["n_bootstrap"] = n_bootstrap
        config = self.config.model_copy(update=updates) if updates else self.config
        limits = ContractLimits(max_rows=self.settings.max_upload_rows)
        return UpliftEngine(self.spec, config, limits=limits)

    def require_result(self) -> EstimationResult:
        if self.result is None:
            raise AppError(409, "no_estimate", "Run an estimation before scoring or targeting.")
        return self.result

    def require_bundle(self) -> ScoringBundle:
        if self.bundle is None:
            raise AppError(409, "no_estimate", "Run an estimation before scoring or targeting.")
        return self.bundle


def _state(request: Request) -> ServiceState:
    state: ServiceState = request.app.state.service
    return state


def _gate_reason(reasons: tuple[str, ...]) -> str:
    for text in reasons:
        for prefix, label in _GATE_REASONS.items():
            if text.startswith(prefix):
                return label
    return "positivity"


def _frame(rows: list[dict[str, Any]], state: ServiceState) -> pd.DataFrame:
    if len(rows) > state.settings.max_upload_rows:
        state.metrics.refuse("contract")
        raise AppError(413, "payload_too_large", "The upload exceeds the configured row limit.")
    frame = pd.DataFrame.from_records(rows)
    for name, node in state.spec.nodes.items():
        if name in frame.columns and node.kind in {"binary", "count"}:
            frame[name] = frame[name].astype("int64", errors="ignore")
    return frame


def _features(frame: pd.DataFrame, names: tuple[str, ...]) -> FloatArray:
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise AppError(
            422, "missing_columns", f"The request is missing {len(missing)} adjustment column(s)."
        )
    return np.column_stack([frame[name].to_numpy(dtype=np.float64) for name in names]).astype(
        np.float64, copy=False
    )


def _recommend(ci_low: float, ci_high: float, cost: float) -> str:
    if ci_low > cost:
        return "send_offer"
    if ci_high <= cost:
        return "hold"
    return "uncertain"


def build_router() -> APIRouter:  # noqa: PLR0915 - one router, one route per endpoint
    """Build the service routes. They are mounted inside the baseline middleware stack."""
    router = APIRouter(prefix="/v1")

    @router.post("/estimate", response_model=EstimateResponse)
    async def estimate(
        body: EstimateRequest, principal: AnalystPrincipal, request: Request
    ) -> EstimateResponse:
        state = _state(request)
        settings = state.settings
        if len(body.rows) < settings.min_cohort_size:
            state.metrics.refuse("cohort_too_small")
            raise AppError(
                422,
                "cohort_too_small",
                f"Aggregate results need at least {settings.min_cohort_size} customers.",
            )
        if body.n_folds is not None and body.n_folds > settings.max_request_folds:
            raise AppError(422, "cap_exceeded", "n_folds exceeds the configured maximum.")
        if body.n_bootstrap is not None and body.n_bootstrap > settings.max_request_bootstrap:
            raise AppError(422, "cap_exceeded", "n_bootstrap exceeds the configured maximum.")

        engine = state.engine(n_folds=body.n_folds, n_bootstrap=body.n_bootstrap)
        if body.adjustment_set is not None:
            _verify_requested_set(engine, body.adjustment_set, state)
        try:
            result = engine.fit(_frame(body.rows, state))
        except DataContractError as exc:
            state.metrics.refuse("contract")
            raise AppError(422, "data_contract", _contract_message(exc)) from None
        except LeakageError:
            state.metrics.refuse("leakage")
            raise AppError(422, "leakage", "A requested variable is post-treatment.") from None
        except NotIdentifiedError as exc:
            state.metrics.refuse("not_identified")
            raise AppError(422, "not_identified", " | ".join(exc.reasons)) from None
        except NotEstimableError as exc:
            state.metrics.refuse(_gate_reason(exc.reasons))
            raise AppError(422, "not_estimable", " | ".join(exc.reasons)) from None

        manifest, artifacts = engine.bundle(result)
        state.result = result
        state.bundle = ScoringBundle.load(manifest, artifacts)
        state.targeting = None
        state.metrics.estimates_total.labels(estimator=result.primary_name).inc()
        _log.info(
            "estimate_completed",
            actor=principal.subject,
            rows=result.prepared.n,
            dataset_sha256=result.dataset_sha256,
            estimable=True,
        )
        return EstimateResponse(
            spec_sha256=result.spec_sha256,
            config_sha256=result.config_sha256,
            dataset_sha256=result.dataset_sha256,
            fold_digest=result.fold_digest,
            adjustment_set=list(result.adjustment_set),
            estimand=result.identification.estimand,
            primary=result.primary_name,
            estimates={name: est.to_dict() for name, est in sorted(result.estimates.items())},
            diagnostics=result.diagnostics.report.to_dict(),
            refutation=result.refutation.to_dict() if result.refutation else None,
            calibration=result.calibration.to_dict(),
            n_rows=result.prepared.n,
        )

    @router.post("/score", response_model=ScoreResponse)
    async def score(
        body: ScoreRequest, principal: AnalystPrincipal, request: Request
    ) -> ScoreResponse:
        state = _state(request)
        if len(body.customers) > state.settings.max_score_rows:
            raise AppError(413, "payload_too_large", "Too many customers in one scoring request.")
        result = state.require_result()
        bundle = state.require_bundle()
        frame = pd.DataFrame.from_records(body.customers)
        unit = state.spec.unit_id
        if unit not in frame.columns:
            raise AppError(422, "missing_columns", f"Every customer needs a {unit}.")
        features = _features(frame, bundle.adjustment_set)
        uplift, propensity, abstain = bundle.predict(features)
        effect, _, low, high = bundle.calibration.interval(uplift, level=state.config.level)
        refs = state.hasher.hash_all([str(value) for value in frame[unit].tolist()])

        scored: list[ScoredCustomer] = []
        recommended = 0
        for index, ref in enumerate(refs):
            if bool(abstain[index]):
                scored.append(
                    ScoredCustomer(
                        customer_ref=ref,
                        abstain=True,
                        reason="outside the overlap region this model was fitted on",
                    )
                )
                continue
            action = _recommend(float(low[index]), float(high[index]), body.cost_per_offer)
            recommended += int(action == "send_offer")
            scored.append(
                ScoredCustomer(
                    customer_ref=ref,
                    abstain=False,
                    uplift=round(float(uplift[index]), 4),
                    group_effect=round(float(effect[index]), 4),
                    ci_low=round(float(low[index]), 4),
                    ci_high=round(float(high[index]), 4),
                    propensity=round(float(propensity[index]), 6),
                    recommended_action=action,
                )
            )
        n_abstained = int(abstain.sum())
        rate = recommended / len(scored) if scored else 0.0
        drift = psi(np.asarray(bundle.manifest.reference_propensity), propensity)
        state.metrics.scored_total.inc(len(scored))
        state.metrics.abstentions_total.inc(n_abstained)
        state.metrics.recommended_treatment_rate.set(rate)
        state.metrics.propensity_psi.set(drift)
        _log.info(
            "scored_batch",
            actor=principal.subject,
            customers=len(scored),
            abstained=n_abstained,
            propensity_psi=round(drift, 4),
        )
        return ScoreResponse(
            spec_sha256=result.spec_sha256,
            dataset_sha256=result.dataset_sha256,
            scored=scored,
            n_abstained=n_abstained,
            recommended_treatment_rate=round(rate, 6),
            propensity_psi=round(drift, 6),
        )

    @router.post("/targeting/requests", response_model=ExportRequestResponse)
    async def request_export(
        body: ExportRequestBody, principal: OperatorPrincipal, request: Request
    ) -> ExportRequestResponse:
        state = _state(request)
        result = state.require_result()
        engine = state.engine()
        outcome = engine.target(
            result,
            budget_fraction=body.budget_fraction,
            cost_per_offer=body.cost_per_offer,
            fairness_threshold=state.settings.fairness_max_disparity,
        )
        state.targeting = outcome
        state.targeting_request = body
        plan = outcome.plans[body.ranking]
        # The list being released is the list being judged: read this ranking's own check.
        fairness = outcome.fairness[body.ranking]
        if plan.n_selected < state.settings.min_cohort_size:
            state.metrics.refuse("cohort_too_small")
            raise AppError(
                422,
                "cohort_too_small",
                f"A targeting list needs at least {state.settings.min_cohort_size} customers.",
            )
        if fairness.blocked:
            state.metrics.refuse("fairness")
            state.audit.append(
                principal.subject,
                "export_blocked_by_fairness",
                {"disparity": round(fairness.disparity, 6), "ranking": body.ranking},
            )
            raise AppError(
                422,
                "fairness_disparity",
                "The allocation disparity between protected groups exceeds the release limit.",
            )
        payload = _payload(state, result, plan.selected, body)
        record = state.approvals.request(principal.subject, payload)
        state.audit.append(
            principal.subject,
            "export_requested",
            {
                "request_id": record.request_id,
                "payload_sha256": record.payload_sha256,
                "ranking": body.ranking,
                "n_selected": plan.n_selected,
                "budget_fraction": body.budget_fraction,
            },
        )
        pilot = outcome.pilot_values.get(body.ranking)
        return ExportRequestResponse(
            request_id=record.request_id,
            payload_sha256=record.payload_sha256,
            n_selected=plan.n_selected,
            expires_at=record.expires_at,
            fairness=fairness.to_dict(),
            pilot_value=pilot.to_dict() if pilot else None,
            audit_head=state.audit.head,
        )

    @router.post("/targeting/requests/{request_id}/decision")
    async def decide_export(
        request_id: str, body: ApprovalDecision, principal: OperatorPrincipal, request: Request
    ) -> dict[str, str]:
        state = _state(request)
        try:
            record = state.approvals.decide(request_id, principal.subject, approve=body.approve)
        except ApprovalError as exc:
            state.metrics.refuse("approval")
            raise AppError(409, "approval_refused", str(exc)) from None
        state.audit.append(
            principal.subject,
            "export_decision",
            {"request_id": record.request_id, "status": record.status},
        )
        return {"request_id": record.request_id, "status": record.status}

    @router.post("/targeting/export", response_model=ExportResponse)
    async def release_export(
        body: ExportRelease, principal: OperatorPrincipal, request: Request
    ) -> ExportResponse:
        state = _state(request)
        result = state.require_result()
        outcome, requested = state.targeting, state.targeting_request
        if outcome is None or requested is None:
            raise AppError(409, "no_targeting_plan", "Request an export before releasing one.")
        record = state.approvals.get(body.request_id)
        payload = _payload(state, result, outcome.plans[requested.ranking].selected, requested)
        if payload_digest(payload) != record.payload_sha256:
            state.metrics.refuse("approval")
            raise AppError(
                409, "approval_refused", "The current targeting plan does not match the approval."
            )
        try:
            state.approvals.consume(body.request_id, principal.subject, payload)
        except ApprovalError as exc:
            state.metrics.refuse("approval")
            raise AppError(409, "approval_refused", str(exc)) from None
        state.metrics.exports_total.inc()
        state.audit.append(
            principal.subject,
            "export_released",
            {
                "request_id": body.request_id,
                "payload_sha256": record.payload_sha256,
                "n_customers": len(payload.pseudonyms),
            },
        )
        return ExportResponse(
            request_id=body.request_id,
            payload_sha256=record.payload_sha256,
            ranking=payload.ranking,
            customer_refs=list(payload.pseudonyms),
            audit_head=state.audit.head,
        )

    return router


def _contract_message(exc: DataContractError) -> str:
    parts = [f"{item['column']}:{item['check']} x{item['count']}" for item in exc.failures[:5]]
    return "The upload violates the data contract (" + "; ".join(parts) + ")."


def _verify_requested_set(engine: UpliftEngine, requested: list[str], state: ServiceState) -> None:
    try:
        identification = verify_adjustment_set(engine.spec, requested)
    except LeakageError as exc:
        state.metrics.refuse("leakage")
        raise AppError(
            422,
            "leakage",
            "Rejected adjustment variable(s): "
            + "; ".join(f"{name} ({reason})" for name, reason in exc.offenders),
        ) from None
    if not identification.identified:
        state.metrics.refuse("not_identified")
        raise AppError(422, "not_identified", " | ".join(identification.reasons))


def _payload(
    state: ServiceState,
    result: EstimationResult,
    selected: BoolArray,
    body: ExportRequestBody,
) -> ExportPayload:
    ids = [
        state.hasher.hash(unit)
        for unit, chosen in zip(result.prepared.unit_ids, selected.tolist(), strict=True)
        if chosen
    ]
    return ExportPayload(
        spec_sha256=result.spec_sha256,
        config_sha256=result.config_sha256,
        dataset_sha256=result.dataset_sha256,
        ranking=body.ranking,
        budget_fraction=body.budget_fraction,
        cost_per_offer=body.cost_per_offer,
        pseudonyms=tuple(ids),
    )


def create_service_app(
    spec: CausalSpec,
    *,
    settings: Settings | None = None,
    config: EstimationConfig | None = None,
    registry: CollectorRegistry | None = None,
) -> FastAPI:
    """Build the FastAPI app: baseline hardening plus this service's routes and state."""
    cfg = settings if settings is not None else get_settings()
    metrics_registry = registry if registry is not None else CollectorRegistry()
    data_dir = Path(cfg.data_dir)
    state = ServiceState(
        spec=spec,
        config=config or EstimationConfig(),
        settings=cfg,
        hasher=CustomerIdHasher.from_settings(cfg),
        audit=AuditLog(data_dir / "audit" / "exports.jsonl"),
        approvals=ApprovalStore(data_dir / "approvals.json", ttl_seconds=cfg.approval_ttl_seconds),
        metrics=create_causal_metrics(metrics_registry),
    )
    app = create_app(
        cfg,
        registry=metrics_registry,
        routers=[build_router()],
        readiness_checks={"jwt": lambda: JwtVerifier(cfg).configured},
    )
    app.state.jwt_verifier = JwtVerifier(cfg)
    app.state.service = state
    return app
