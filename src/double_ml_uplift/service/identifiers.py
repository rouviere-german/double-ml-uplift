"""Pseudonymous customer identifiers.

Every response, log line, audit record and export refers to customers by
``cid_<40 hex characters>``, the first 20 bytes of ``HMAC-SHA256(pepper, customer_id)``. HMAC
rather than a bare hash, because a plain SHA-256 of a customer id is trivially reversible by
enumerating the id space; the pepper is a secret that never leaves the process.

The pepper is required. Without it the service refuses to start rather than silently emitting raw
ids or deriving a per-process key that would make yesterday's export unmatchable to today's.
"""

from __future__ import annotations

import hmac
import re
from dataclasses import dataclass
from typing import Final

from double_ml_uplift.core.config import Settings

__all__ = ["PSEUDONYM_PATTERN", "CustomerIdHasher", "MissingPepperError"]

PSEUDONYM_PATTERN: Final = re.compile(r"^cid_[0-9a-f]{40}$")
_DIGEST_BYTES: Final = 20
_MAX_ID_LENGTH: Final = 256


class MissingPepperError(RuntimeError):
    """No HMAC pepper is configured, so customer ids cannot be pseudonymised."""


@dataclass(frozen=True, slots=True)
class CustomerIdHasher:
    """Keyed, deterministic pseudonymisation of customer identifiers."""

    pepper: bytes

    @classmethod
    def from_settings(cls, settings: Settings) -> CustomerIdHasher:
        pepper = settings.customer_id_pepper
        if pepper is None or not pepper.get_secret_value():
            raise MissingPepperError(
                "APP_CUSTOMER_ID_PEPPER must be set; the service will not emit raw customer ids"
            )
        return cls(pepper=pepper.get_secret_value().encode("utf-8"))

    def hash(self, customer_id: str) -> str:
        if not customer_id or len(customer_id) > _MAX_ID_LENGTH:
            raise ValueError("customer id must be between 1 and 256 characters")
        digest = hmac.new(self.pepper, customer_id.encode("utf-8"), "sha256").digest()
        return f"cid_{digest[:_DIGEST_BYTES].hex()}"

    def hash_all(self, customer_ids: list[str]) -> list[str]:
        return [self.hash(value) for value in customer_ids]
