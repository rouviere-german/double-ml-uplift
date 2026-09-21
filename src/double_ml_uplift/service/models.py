"""Request and response schemas.

Every model is closed (``extra="forbid"``) and every collection is bounded, so a request is
rejected on size and shape before a single row reaches pandas. Uploaded rows are typed loosely on
purpose -- the real column-by-column contract is the pandera schema derived from the causal
specification -- but their keys, value types and string lengths are checked here so that a
malformed upload cannot turn into a large, slow or surprising frame.
"""

from __future__ import annotations

import math
import re
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "MAX_CELL_LENGTH",
    "MAX_ROW_KEYS",
    "ApprovalDecision",
    "EstimateRequest",
    "EstimateResponse",
    "ExportRelease",
    "ExportRequestBody",
    "ExportRequestResponse",
    "ExportResponse",
    "ScoreRequest",
    "ScoreResponse",
    "ScoredCustomer",
]

MAX_ROW_KEYS: Final = 64
MAX_CELL_LENGTH: Final = 128
_COLUMN = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
_Row = dict[str, Any]


def _check_rows(rows: list[_Row]) -> list[_Row]:
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each row must be an object")
        if len(row) > MAX_ROW_KEYS:
            raise ValueError(f"a row may have at most {MAX_ROW_KEYS} columns")
        for key, value in row.items():
            if _COLUMN.fullmatch(key) is None:
                raise ValueError("column names must be lowercase snake_case identifiers")
            if isinstance(value, bool) or value is None:
                raise ValueError(f"column {key} must be a number or a string")
            if isinstance(value, int | float):
                if not math.isfinite(float(value)):
                    raise ValueError(f"column {key} must be finite")
            elif isinstance(value, str):
                if len(value) > MAX_CELL_LENGTH:
                    raise ValueError(f"column {key} exceeds {MAX_CELL_LENGTH} characters")
            else:
                raise ValueError(f"column {key} must be a number or a string")
    return rows


class EstimateRequest(BaseModel):
    """Run the pipeline on an uploaded campaign table."""

    model_config = ConfigDict(extra="forbid")

    rows: list[_Row] = Field(min_length=2, max_length=5_000_000)
    n_folds: int | None = Field(default=None, ge=2, le=20)
    n_bootstrap: int | None = Field(default=None, ge=0, le=5_000)
    adjustment_set: list[Annotated[str, Field(max_length=64)]] | None = Field(
        default=None, max_length=MAX_ROW_KEYS
    )

    @field_validator("rows")
    @classmethod
    def _validate_rows(cls, value: list[_Row]) -> list[_Row]:
        return _check_rows(value)


class EstimateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_sha256: str
    config_sha256: str
    dataset_sha256: str
    fold_digest: str
    adjustment_set: list[str]
    estimand: str
    primary: str
    estimates: dict[str, dict[str, Any]]
    diagnostics: dict[str, Any]
    refutation: dict[str, Any] | None
    calibration: dict[str, Any]
    n_rows: int
    run_version: str = Field(pattern=r"^[0-9a-f]{64}$")
    """Content address of this run in the local model registry: SHA-256 of its version record."""


class ScoreRequest(BaseModel):
    """Score customers with the bundle produced by the last successful estimation."""

    model_config = ConfigDict(extra="forbid")

    customers: list[_Row] = Field(min_length=1, max_length=100_000)
    cost_per_offer: float = Field(default=0.0, ge=0.0, le=1e6)
    """Recommend an offer only when the lower confidence limit clears this cost."""

    @field_validator("customers")
    @classmethod
    def _validate_customers(cls, value: list[_Row]) -> list[_Row]:
        return _check_rows(value)


class ScoredCustomer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_ref: str
    """HMAC pseudonym. The raw customer id never appears in a response or a log."""

    abstain: bool
    reason: str | None = None
    uplift: float | None = None
    group_effect: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    propensity: float | None = None
    recommended_action: Literal["send_offer", "hold", "uncertain", "abstain"] = "abstain"


class ScoreResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec_sha256: str
    dataset_sha256: str
    scored: list[ScoredCustomer]
    n_abstained: int
    recommended_treatment_rate: float
    propensity_psi: float


class ExportRequestBody(BaseModel):
    """Ask for permission to release a targeting list."""

    model_config = ConfigDict(extra="forbid")

    budget_fraction: float = Field(default=0.2, gt=0.0, le=1.0)
    cost_per_offer: float = Field(default=1.0, gt=0.0, le=1e6)
    ranking: Literal["uplift", "propensity"] = "uplift"


class ExportRequestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    payload_sha256: str
    n_selected: int
    expires_at: float
    fairness: dict[str, Any]
    pilot_value: dict[str, Any] | None
    audit_head: str


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approve: bool


class ExportRelease(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(pattern=r"^exp_[0-9a-f]{32}$")


class ExportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    payload_sha256: str
    ranking: str
    customer_refs: list[str]
    audit_head: str
