"""Seeded structural causal model for a retention-offer campaign.

The generator produces a customer table whose *observed* columns are exactly what an analyst
would have, plus an ``oracle`` table that is never handed to an estimator. Structural equations,
in order:

1. Pre-treatment covariates ``X`` and a synthetic protected attribute ``A`` are drawn
   independently. A hidden confounder ``U`` is drawn and used only when
   ``hidden_confounder_strength`` is non-zero.
2. A pilot slice is drawn: ``pilot = 1`` for a random fraction of customers.
3. Treatment ``T`` follows the historical targeting policy
   ``P(T = 1 | X, U) = sigmoid(tilt * (beta . z(X) + gamma * U))`` outside the pilot, and
   ``0.5`` inside it, so the pilot is a genuine randomized slice.
4. A post-treatment mediator ``M`` (logins after the offer) follows ``M = g(z) + delta * T + e_m``.
5. The outcome follows ``Y = f(z) + gamma_y * U + b * M + direct(z) * T + e_y``. ``M`` shares its
   noise across arms, so the true individual effect is ``tau = b * (M1 - M0) + direct(z)``, which
   equals ``b * delta + direct(z)`` except for the few customers whose untreated mediator is
   clipped at zero. It is heterogeneous and negative for a sizeable minority of customers.
6. A post-treatment collider ``C`` (satisfaction survey) follows ``C = c1 * T + c2 * Y + e_c``.

``f`` and ``g`` are linear or nonlinear depending on ``nonlinear``; ``tilt`` widens the treatment
index to create positivity violations. Nothing here is fitted, so every estimator can be scored
against ``tau`` and against the true policy value.

Reference for the potential-outcome notation: Rubin, "Estimating causal effects of treatments in
randomized and nonrandomized studies" (1974); Imbens and Rubin (2015), chapter 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from double_ml_uplift.core.canonical import canonical_json, sha256_hex

__all__ = [
    "CONFOUNDER_COLUMNS",
    "ORACLE_COLUMNS",
    "OUTCOME_COLUMN",
    "PILOT_COLUMN",
    "POST_TREATMENT_COLUMNS",
    "PROTECTED_COLUMN",
    "SCM_FEATURE_SCALES",
    "TREATMENT_COLUMN",
    "UNIT_COLUMN",
    "ScmConfig",
    "SimulatedCampaign",
    "simulate_campaign",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

UNIT_COLUMN: Final = "customer_id"
TREATMENT_COLUMN: Final = "offer_sent"
OUTCOME_COLUMN: Final = "retained_revenue"
PILOT_COLUMN: Final = "pilot"
PROTECTED_COLUMN: Final = "protected_group"
CONFOUNDER_COLUMNS: Final = (
    "tenure_months",
    "monthly_spend",
    "support_tickets",
    "discount_count",
    "engagement_score",
    PROTECTED_COLUMN,
)
POST_TREATMENT_COLUMNS: Final = ("post_offer_logins", "offer_redeemed", "csat_after_offer")
ORACLE_COLUMNS: Final = (
    UNIT_COLUMN,
    "y0",
    "y1",
    "true_cate",
    "true_propensity",
    "hidden_confounder",
)

# Fixed centering and scaling constants, so the structural equations mean the same thing for
# every seed and sample size. They approximate the marginal mean and standard deviation of each
# covariate; nothing depends on them being exact.
SCM_FEATURE_SCALES: Final[dict[str, tuple[float, float]]] = {
    "tenure_months": (28.0, 14.0),
    "monthly_spend": (110.0, 52.0),
    "support_tickets": (1.8, 1.34),
    "discount_count": (1.1, 1.05),
    "engagement_score": (0.0, 1.0),
    PROTECTED_COLUMN: (0.35, 0.48),
}


class ScmConfig(BaseModel):
    """Parameters of the generator. Hashed into the reproducibility manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    n_customers: int = Field(ge=50, le=5_000_000)
    seed: int = Field(ge=0, le=2**31 - 1)
    # Strength of the historical targeting rule: 0 means the campaign was randomized.
    confounding_strength: float = Field(default=1.0, ge=0.0, le=5.0)
    # Multiplies the treatment index; > 1 pushes propensities towards 0 and 1.
    overlap_tilt: float = Field(default=1.0, ge=0.5, le=12.0)
    # Coefficient of the hidden confounder in both the treatment and the outcome equation.
    hidden_confounder_strength: float = Field(default=0.0, ge=0.0, le=5.0)
    nonlinear: bool = False
    pilot_fraction: float = Field(default=0.05, ge=0.0, le=0.5)
    outcome_noise_sd: float = Field(default=45.0, gt=0.0, le=1000.0)
    # Direct (non-mediated) effect: intercept plus heterogeneity in tenure, tickets, engagement.
    direct_effect: tuple[float, float, float, float] = (14.0, 18.0, -26.0, 9.0)
    mediator_effect: float = Field(default=4.0, ge=0.0, le=100.0)
    mediator_coefficient: float = Field(default=5.0, ge=0.0, le=100.0)
    offer_cost: float = Field(default=12.0, gt=0.0, le=10_000.0)

    def sha256(self) -> str:
        """Content hash of the parameters, recorded with every run."""
        return sha256_hex(canonical_json(self.model_dump(mode="json")))


@dataclass(frozen=True, slots=True)
class SimulatedCampaign:
    """An observed campaign table plus the oracle that only the evaluation code may read."""

    config: ScmConfig
    frame: pd.DataFrame
    oracle: pd.DataFrame

    @property
    def true_ate(self) -> float:
        return float(self.oracle["true_cate"].to_numpy(dtype=np.float64).mean())

    @property
    def true_cate(self) -> FloatArray:
        return self.oracle["true_cate"].to_numpy(dtype=np.float64)

    @property
    def potential_outcomes(self) -> tuple[FloatArray, FloatArray]:
        y0 = self.oracle["y0"].to_numpy(dtype=np.float64)
        y1 = self.oracle["y1"].to_numpy(dtype=np.float64)
        return y0, y1

    @property
    def config_sha256(self) -> str:
        return self.config.sha256()


def _sigmoid(values: FloatArray) -> FloatArray:
    return np.asarray(0.5 * (1.0 + np.tanh(0.5 * values)), dtype=np.float64)


def standardize(frame: pd.DataFrame, columns: tuple[str, ...] = CONFOUNDER_COLUMNS) -> FloatArray:
    """Centre and scale covariates with the fixed structural constants (not sample statistics)."""
    columns_out = []
    for name in columns:
        centre, scale = SCM_FEATURE_SCALES[name]
        columns_out.append((frame[name].to_numpy(dtype=np.float64) - centre) / scale)
    return np.column_stack(columns_out).astype(np.float64, copy=False)


def _draw_covariates(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenure_months": np.clip(np.round(rng.gamma(4.0, 7.0, size=n), 0), 0.0, 240.0),
            "monthly_spend": np.clip(np.round(rng.lognormal(4.6, 0.45, size=n), 2), 0.0, 5_000.0),
            "support_tickets": np.minimum(rng.poisson(1.8, size=n), 200).astype(np.int64),
            "discount_count": np.minimum(rng.poisson(1.1, size=n), 50).astype(np.int64),
            "engagement_score": np.clip(np.round(rng.normal(0.0, 1.0, size=n), 4), -10.0, 10.0),
            PROTECTED_COLUMN: (rng.random(n) < 0.35).astype(np.int64),
        }
    )


def _baseline_index(z: FloatArray, *, nonlinear: bool) -> FloatArray:
    """f(z): systematic part of the untreated outcome, on the revenue scale."""
    linear = 68.0 * z[:, 0] + 95.0 * z[:, 1] - 38.0 * z[:, 2] - 21.0 * z[:, 3] + 44.0 * z[:, 4]
    linear = linear - 16.0 * z[:, 5]
    if not nonlinear:
        return np.asarray(390.0 + linear, dtype=np.float64)
    curved = 52.0 * np.sin(1.4 * z[:, 1]) + 30.0 * z[:, 0] * z[:, 4] - 24.0 * np.square(z[:, 2])
    return np.asarray(390.0 + linear + curved, dtype=np.float64)


def _treatment_index(z: FloatArray, *, nonlinear: bool) -> FloatArray:
    """beta . z: the historical targeting rule's score, before tilt and hidden confounding."""
    linear = -0.55 * z[:, 0] + 0.85 * z[:, 1] + 0.62 * z[:, 2] + 0.40 * z[:, 3] - 0.35 * z[:, 4]
    linear = linear + 0.25 * z[:, 5]
    if not nonlinear:
        return np.asarray(linear, dtype=np.float64)
    curved = 0.45 * np.tanh(2.0 * z[:, 1] * z[:, 2]) - 0.30 * np.square(z[:, 0])
    return np.asarray(linear + curved, dtype=np.float64)


def _mediator_index(z: FloatArray, *, nonlinear: bool) -> FloatArray:
    """g(z): logins in the period after the offer, before the offer's own contribution."""
    linear = 3.1 + 1.25 * z[:, 4] + 0.85 * z[:, 1] - 0.42 * z[:, 2]
    if not nonlinear:
        return np.asarray(linear, dtype=np.float64)
    return np.asarray(linear + 0.7 * np.sin(1.1 * z[:, 0]), dtype=np.float64)


def _direct_effect(z: FloatArray, coefficients: tuple[float, float, float, float]) -> FloatArray:
    """direct(z): the part of the treatment effect that does not run through the mediator."""
    intercept, tenure, tickets, engagement = coefficients
    return np.asarray(
        intercept
        + tenure * np.tanh(z[:, 0])
        + tickets * np.maximum(z[:, 2], 0.0)
        + engagement * z[:, 4],
        dtype=np.float64,
    )


def simulate_campaign(config: ScmConfig) -> SimulatedCampaign:
    """Draw one campaign. Deterministic in ``config`` (including its seed)."""
    rng = np.random.default_rng(config.seed)
    n = config.n_customers

    covariates = _draw_covariates(rng, n)
    z = standardize(covariates)
    hidden = rng.normal(0.0, 1.0, size=n)
    pilot = (rng.random(n) < config.pilot_fraction).astype(np.int64)

    index = config.confounding_strength * _treatment_index(z, nonlinear=config.nonlinear)
    index = config.overlap_tilt * (index + config.hidden_confounder_strength * hidden)
    propensity = np.where(pilot == 1, 0.5, _sigmoid(index))
    treatment = (rng.random(n) < propensity).astype(np.int64)

    mediator_noise = rng.normal(0.0, 1.1, size=n)
    mediator_base = _mediator_index(z, nonlinear=config.nonlinear) + mediator_noise
    mediator0 = np.maximum(mediator_base, 0.0)
    mediator1 = np.maximum(mediator_base + config.mediator_effect, 0.0)
    mediator = np.where(treatment == 1, mediator1, mediator0)

    outcome_noise = rng.normal(0.0, config.outcome_noise_sd, size=n)
    baseline = _baseline_index(z, nonlinear=config.nonlinear)
    baseline = baseline + 34.0 * config.hidden_confounder_strength * hidden + outcome_noise
    direct = _direct_effect(z, config.direct_effect)
    y0 = baseline + config.mediator_coefficient * mediator0
    y1 = baseline + config.mediator_coefficient * mediator1 + direct
    outcome = np.where(treatment == 1, y1, y0)

    csat = np.clip(
        4.6 + 1.1 * treatment + 0.004 * (outcome - 400.0) + rng.normal(0.0, 0.8, size=n), 0.0, 10.0
    )
    redeemed = ((treatment == 1) & (rng.random(n) < 0.55)).astype(np.int64)

    unit_ids = [f"cust_{index:07d}" for index in range(n)]
    frame = pd.DataFrame({UNIT_COLUMN: unit_ids})
    frame = pd.concat([frame, covariates], axis=1)
    frame[PILOT_COLUMN] = pilot
    frame[TREATMENT_COLUMN] = treatment
    frame["post_offer_logins"] = np.round(mediator, 4)
    frame["offer_redeemed"] = redeemed
    frame["csat_after_offer"] = np.round(csat, 4)
    frame[OUTCOME_COLUMN] = np.round(outcome, 4)

    oracle = pd.DataFrame(
        {
            UNIT_COLUMN: unit_ids,
            "y0": y0,
            "y1": y1,
            "true_cate": y1 - y0,
            "true_propensity": propensity,
            "hidden_confounder": hidden,
        }
    )
    return SimulatedCampaign(config=config, frame=frame, oracle=oracle)
