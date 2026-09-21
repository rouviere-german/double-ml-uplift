"""Typed configuration loaded from environment variables (prefix ``APP_``).

Only ``.env.example`` is committed. Secrets are ``SecretStr`` so they never appear in ``repr``
or logs. API keys are configured as SHA-256 digests, so the process never holds raw keys.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

__all__ = ["Settings", "get_settings"]

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_ORIGIN = re.compile(r"^https?://[A-Za-z0-9.-]+(:\d{1,5})?$")


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    """Service settings. Every field can be set with ``APP_<FIELD_NAME>``."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- service ---
    service_name: str = Field(default="double-ml-uplift", pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    enable_docs: bool = False

    # --- HTTP security ---
    api_key_sha256: Annotated[list[str], NoDecode] = Field(default_factory=list)
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    rate_limit_per_second: float = Field(default=5.0, gt=0)
    rate_limit_burst: int = Field(default=20, ge=1)
    rate_limit_max_clients: int = Field(default=10_000, ge=1)
    max_body_bytes: int = Field(default=1_048_576, ge=1)
    metrics_require_auth: bool = True

    # --- observability ---
    tracing_exporter: Literal["none", "console"] = "none"

    # --- role-based access (JWT) ---
    # HS256 signing key shared with the identity provider that mints analyst/operator tokens.
    # Unset means no token can be verified, so every role-protected route fails closed.
    jwt_secret: SecretStr | None = None
    jwt_issuer: str = Field(default="double-ml-uplift", min_length=1, max_length=128)
    jwt_audience: str = Field(default="double-ml-uplift-api", min_length=1, max_length=128)
    jwt_max_lifetime_seconds: float = Field(default=3_600.0, gt=0, le=86_400.0)

    # --- privacy ---
    # HMAC key for pseudonymising customer ids in responses, logs and exports. Fail closed:
    # without it the service refuses to start, because raw ids would otherwise leave the process.
    customer_id_pepper: SecretStr | None = None
    min_cohort_size: int = Field(default=50, ge=2, le=1_000_000)

    # --- request caps ---
    max_upload_rows: int = Field(default=200_000, ge=100, le=5_000_000)
    max_request_folds: int = Field(default=10, ge=2, le=20)
    max_request_bootstrap: int = Field(default=2_000, ge=0, le=5_000)
    max_score_rows: int = Field(default=5_000, ge=1, le=100_000)

    # --- targeting release ---
    fairness_max_disparity: float = Field(default=0.15, gt=0.0, le=1.0)
    approval_ttl_seconds: float = Field(default=900.0, gt=0, le=86_400.0)

    # --- estimation ---
    # Keys the cross-fitting fold hash. The bundled default is a public literal, which is fine
    # for reproducibility and useless against an attacker who can choose customer ids and read
    # this repository. Set it per environment when fold assignment is adversarially relevant.
    # Only its digest leaves the process, inside config_sha256.
    fold_salt: SecretStr | None = None

    # --- storage ---
    data_dir: Path = Path("var")

    @field_validator("api_key_sha256", mode="before")
    @classmethod
    def _parse_key_hashes(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("api_key_sha256")
    @classmethod
    def _check_key_hashes(cls, value: list[str]) -> list[str]:
        for digest in value:
            if not _SHA256_HEX.fullmatch(digest):
                raise ValueError("api_key_sha256 entries must be lowercase hex SHA-256 digests")
        return value

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("cors_allow_origins")
    @classmethod
    def _check_origins(cls, value: list[str]) -> list[str]:
        for origin in value:
            if not _ORIGIN.fullmatch(origin):
                raise ValueError(
                    "cors_allow_origins must be explicit scheme://host[:port] origins "
                    "('*' is refused)"
                )
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings (cached). Tests construct ``Settings(...)`` directly instead."""
    return Settings()
