"""JWT bearer authentication with two roles.

``analyst``
    May run estimations and read aggregate results and scores. Aggregates are refused below the
    configured minimum cohort size, and customer ids come back pseudonymised.
``operator``
    Everything an analyst may do, plus requesting, approving and releasing a targeting export.

Verification is deliberately narrow:

* only ``HS256`` is accepted, so a token whose header claims ``none`` or an RSA algorithm cannot
  trick the verifier into skipping or re-interpreting the signature;
* issuer, audience, expiry and issued-at are all required, and a token whose lifetime exceeds
  ``jwt_max_lifetime_seconds`` is rejected even if it is otherwise valid, so a stolen long-lived
  token is not usable forever;
* with no signing key configured, every role-protected route fails closed.

Tokens are verified, never minted, in production paths. :func:`issue_token` exists for the tests,
the demo and local development and is not reachable over HTTP.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Final, Literal

import jwt
from fastapi import Depends, Request

from double_ml_uplift.core.config import Settings
from double_ml_uplift.core.errors import AppError

__all__ = [
    "ROLES",
    "AnalystPrincipal",
    "JwtVerifier",
    "OperatorPrincipal",
    "Role",
    "RolePrincipal",
    "issue_token",
    "require_role",
]

Role = Literal["analyst", "operator"]
ROLES: Final[dict[str, int]] = {"analyst": 1, "operator": 2}
_ALGORITHM: Final = "HS256"
_SUBJECT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,63}$")
_MAX_TOKEN_BYTES: Final = 4_096


@dataclass(frozen=True, slots=True)
class RolePrincipal:
    """An authenticated caller: who they are and what they are allowed to do."""

    subject: str
    role: Role

    def may(self, required: Role) -> bool:
        return ROLES[self.role] >= ROLES[required]


class JwtVerifier:
    """Verifies bearer tokens against one configured HS256 key."""

    def __init__(self, settings: Settings) -> None:
        secret = settings.jwt_secret
        self._secret = secret.get_secret_value() if secret is not None else None
        self._issuer = settings.jwt_issuer
        self._audience = settings.jwt_audience
        self._max_lifetime = settings.jwt_max_lifetime_seconds

    @property
    def configured(self) -> bool:
        return bool(self._secret)

    def verify(self, token: str | None) -> RolePrincipal | None:
        """Return the principal for a valid token, else ``None``. No key means no principal."""
        if not self._secret or not token or len(token) > _MAX_TOKEN_BYTES:
            return None
        try:
            claims: dict[str, Any] = jwt.decode(
                token,
                self._secret,
                algorithms=[_ALGORITHM],
                issuer=self._issuer,
                audience=self._audience,
                options={
                    "require": ["exp", "iat", "iss", "aud", "sub", "role"],
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_iss": True,
                    "verify_aud": True,
                },
                leeway=0,
            )
        except jwt.PyJWTError:
            return None
        return self._principal(claims)

    def _principal(self, claims: Mapping[str, Any]) -> RolePrincipal | None:
        subject = claims.get("sub")
        role = claims.get("role")
        issued_at, expires_at = claims.get("iat"), claims.get("exp")
        if not isinstance(subject, str) or _SUBJECT.fullmatch(subject) is None:
            return None
        if role not in ROLES:
            return None
        if not isinstance(issued_at, int | float) or not isinstance(expires_at, int | float):
            return None
        if float(expires_at) - float(issued_at) > self._max_lifetime:
            return None
        return RolePrincipal(subject=subject, role=role)


def issue_token(
    settings: Settings,
    subject: str,
    role: Role,
    *,
    lifetime_seconds: float = 600.0,
    now: float | None = None,
) -> str:
    """Mint a token. For tests, the demo and local development only."""
    secret = settings.jwt_secret
    if secret is None:
        raise ValueError("APP_JWT_SECRET must be set before a token can be issued")
    if _SUBJECT.fullmatch(subject) is None:
        raise ValueError("subject must be a short identifier")
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    issued = time.time() if now is None else now
    return jwt.encode(
        {
            "sub": subject,
            "role": role,
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
            "iat": int(issued),
            "exp": int(issued + lifetime_seconds),
        },
        secret.get_secret_value(),
        algorithm=_ALGORITHM,
    )


def _bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization")
    if header and header[:7].lower() == "bearer ":
        return header[7:].strip() or None
    return None


def require_role(required: Role) -> Any:
    """FastAPI dependency factory: authenticate the caller and check its role."""

    def dependency(request: Request) -> RolePrincipal:
        verifier: JwtVerifier = request.app.state.jwt_verifier
        principal = verifier.verify(_bearer_token(request))
        if principal is None:
            request.app.state.metrics.auth_failures_total.inc()
            raise AppError(
                401,
                "unauthorized",
                "A valid bearer token is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not principal.may(required):
            raise AppError(403, "forbidden", f"This action requires the {required} role.")
        return principal

    return Depends(dependency)


AnalystPrincipal = Annotated[RolePrincipal, require_role("analyst")]
OperatorPrincipal = Annotated[RolePrincipal, require_role("operator")]
