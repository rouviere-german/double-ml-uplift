"""Prometheus families specific to this service, on the same per-app registry as the HTTP ones.

The four things worth alerting on here are not latency and error rate:

``uplift_estimates_total``
    Successful estimations, labelled by the primary estimator.
``uplift_refusals_total{reason}``
    Refusals by cause -- ``not_identified``, ``positivity``, ``balance``, ``effective_sample``,
    ``contract``, ``leakage``, ``cohort_too_small``, ``fairness``. A rising ``positivity`` rate
    means the targeting policy has become deterministic and the historical data can no longer
    answer the question.
``uplift_abstentions_total``
    Scored customers the service refused to score, because their propensity fell outside the
    overlap region the model was fitted on.
``uplift_propensity_psi``
    Population Stability Index of the scoring population's propensity distribution against the
    training population's. The single most useful drift signal for this system: it moves as soon
    as the mix of customers being scored stops looking like the mix that was estimated on.
``uplift_recommended_treatment_rate``
    Share of scored customers the model would target. A silent jump here is how a broken CATE
    model shows up before anyone reads a dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge

__all__ = ["REFUSAL_REASONS", "CausalMetrics", "create_causal_metrics"]

REFUSAL_REASONS = (
    "not_identified",
    "positivity",
    "balance",
    "effective_sample",
    "contract",
    "leakage",
    "cohort_too_small",
    "fairness",
    "approval",
)


@dataclass(frozen=True, slots=True)
class CausalMetrics:
    registry: CollectorRegistry
    estimates_total: Counter
    refusals_total: Counter
    abstentions_total: Counter
    scored_total: Counter
    exports_total: Counter
    propensity_psi: Gauge
    recommended_treatment_rate: Gauge

    def refuse(self, reason: str) -> None:
        """Count a refusal. Unknown reasons are folded into ``other`` so labels stay bounded."""
        self.refusals_total.labels(reason=reason if reason in REFUSAL_REASONS else "other").inc()


def create_causal_metrics(registry: CollectorRegistry, namespace: str = "uplift") -> CausalMetrics:
    return CausalMetrics(
        registry=registry,
        estimates_total=Counter(
            "estimates_total",
            "Estimation runs that produced a reported effect.",
            ["estimator"],
            namespace=namespace,
            registry=registry,
        ),
        refusals_total=Counter(
            "refusals_total",
            "Requests refused, by reason.",
            ["reason"],
            namespace=namespace,
            registry=registry,
        ),
        abstentions_total=Counter(
            "abstentions_total",
            "Customers the scorer refused to score because they fall outside the overlap region.",
            namespace=namespace,
            registry=registry,
        ),
        scored_total=Counter(
            "scored_total",
            "Customers scored, including abstentions.",
            namespace=namespace,
            registry=registry,
        ),
        exports_total=Counter(
            "exports_total",
            "Targeting lists released after approval.",
            namespace=namespace,
            registry=registry,
        ),
        propensity_psi=Gauge(
            "propensity_psi",
            "PSI of the scoring population's propensity against the training population's.",
            namespace=namespace,
            registry=registry,
        ),
        recommended_treatment_rate=Gauge(
            "recommended_treatment_rate",
            "Share of the last scoring batch the model recommends treating.",
            namespace=namespace,
            registry=registry,
        ),
    )
