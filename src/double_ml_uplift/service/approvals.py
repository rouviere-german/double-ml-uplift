"""Approvals for releasing a targeting list.

Exporting a list of customers to contact is the one action in this service with a consequence
outside it, so it is the one action that needs a second person. The rules, all of them tested:

* an approval is bound to the SHA-256 of the exact export payload (specification, configuration
  and dataset hashes, budget, ranking, cost and the ordered pseudonymous ids), so approving one
  list never releases another;
* the requester cannot approve its own request;
* only the requesting principal may consume the approval, so a second operator cannot pick up
  someone else's approved export;
* an approval is consumed once. Two processes sharing the store take a file lock and reload
  before deciding, so a concurrent second release is refused;
* approvals expire;
* the store is capped at ``max_records`` (1,000 by default) so the file cannot grow without
  bound. The cap is applied to expired and already-decided records first. Applying it by age
  alone would let anyone holding an operator token evict another operator's pending or approved
  record simply by making a thousand fresh requests, and the release would then fail with
  "unknown export request" -- no escalation, but a quiet availability failure in a control-plane
  store. A request that would evict a *live* record is refused instead.

The store is a JSON file. It is not itself tamper-evident: whoever can write it can forge an
approved record. It is protected the way the audit log is -- by file permissions -- and every
decision is also written to the hash-chained audit log, which is where a forged approval would
show up as a release with no matching decision.
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from double_ml_uplift.core.canonical import canonical_json, sha256_hex
from double_ml_uplift.core.fileio import atomic_write_text, file_lock

__all__ = [
    "MAX_RECORDS",
    "ApprovalError",
    "ApprovalRecord",
    "ApprovalStore",
    "ApprovalStoreFullError",
    "ExportPayload",
    "payload_digest",
]

Status = Literal["pending", "approved", "denied", "consumed"]

MAX_RECORDS: Final = 1_000
"""Default cap on stored records. See the module docstring for why it is not applied by age."""

# A record still capable of being acted on: nothing else is worth protecting from eviction.
_LIVE_STATUSES: Final = frozenset({"pending", "approved"})


class ApprovalError(ValueError):
    """An approval is missing, expired, already used, or does not match the request."""


class ApprovalStoreFullError(ApprovalError):
    """Every slot holds a live request, so a new one would evict somebody else's."""


class ApprovalRecord(BaseModel):
    """One export request and its decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(pattern=r"^exp_[0-9a-f]{32}$")
    requester: str
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: float
    expires_at: float
    status: Status = "pending"
    decided_by: str | None = None
    decided_at: float | None = None


@dataclass(frozen=True, slots=True)
class ExportPayload:
    """The exact thing being approved. Its digest is what an approval commits to."""

    spec_sha256: str
    config_sha256: str
    dataset_sha256: str
    ranking: str
    budget_fraction: float
    cost_per_offer: float
    pseudonyms: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_sha256": self.spec_sha256,
            "config_sha256": self.config_sha256,
            "dataset_sha256": self.dataset_sha256,
            "ranking": self.ranking,
            "budget_fraction": self.budget_fraction,
            "cost_per_offer": self.cost_per_offer,
            "pseudonyms": list(self.pseudonyms),
        }


def payload_digest(payload: ExportPayload) -> str:
    """Canonical SHA-256 of an export payload."""
    return sha256_hex(canonical_json(payload.to_dict()))


class ApprovalStore:
    """A small persisted approval store with cross-process locking."""

    def __init__(
        self,
        path: Path,
        *,
        ttl_seconds: float = 900.0,
        clock: Any = None,
        max_records: int = MAX_RECORDS,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_records < 1:
            raise ValueError("max_records must be positive")
        self.path = path
        self.ttl_seconds = ttl_seconds
        self.max_records = max_records
        self._clock = clock or time.time
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _is_live(self, record: ApprovalRecord, now: float) -> bool:
        return record.status in _LIVE_STATUSES and now <= record.expires_at

    # --- persistence ---------------------------------------------------------------------

    def _load(self) -> dict[str, ApprovalRecord]:
        if not self.path.is_file():
            return {}
        document = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ApprovalError("approval store is not a JSON object")
        return {str(key): ApprovalRecord.model_validate(value) for key, value in document.items()}

    def _prune(self, records: dict[str, ApprovalRecord]) -> list[ApprovalRecord]:
        """Apply the cap to spent records first, and only then to live ones.

        ``request`` refuses before the live set can exceed the cap, so the final fallback here
        is unreachable in the normal flow; it exists so that a store restored from an oversized
        file still converges instead of growing forever.
        """
        if len(records) <= self.max_records:
            return sorted(records.values(), key=lambda item: item.created_at)
        now = float(self._clock())
        live = [item for item in records.values() if self._is_live(item, now)]
        spent = [item for item in records.values() if not self._is_live(item, now)]
        room = self.max_records - len(live)
        if room <= 0:
            return sorted(live, key=lambda item: item.created_at)[-self.max_records :]
        keep = live + sorted(spent, key=lambda item: item.created_at)[-room:]
        return sorted(keep, key=lambda item: item.created_at)

    def _save(self, records: dict[str, ApprovalRecord]) -> None:
        pruned = self._prune(records)
        atomic_write_text(
            self.path,
            json.dumps(
                {item.request_id: item.model_dump(mode="json") for item in pruned},
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    # --- operations ----------------------------------------------------------------------

    def request(self, requester: str, payload: ExportPayload) -> ApprovalRecord:
        """Record a pending export request for one exact payload."""
        now = float(self._clock())
        record = ApprovalRecord(
            request_id=f"exp_{secrets.token_hex(16)}",
            requester=requester,
            payload_sha256=payload_digest(payload),
            created_at=now,
            expires_at=now + self.ttl_seconds,
        )
        with file_lock(self.path):
            records = self._load()
            if sum(self._is_live(item, now) for item in records.values()) >= self.max_records:
                raise ApprovalStoreFullError(
                    f"the approval store already holds {self.max_records} live export requests; "
                    "decide or let some expire before asking for another"
                )
            records[record.request_id] = record
            self._save(records)
        return record

    def decide(self, request_id: str, approver: str, *, approve: bool) -> ApprovalRecord:
        """Approve or deny a pending request. The requester may not decide its own."""
        with file_lock(self.path):
            records = self._load()
            record = records.get(request_id)
            if record is None:
                raise ApprovalError("unknown export request")
            if record.status != "pending":
                raise ApprovalError(f"request is already {record.status}")
            if float(self._clock()) > record.expires_at:
                raise ApprovalError("request has expired")
            if approver == record.requester:
                raise ApprovalError(
                    "an export cannot be approved by the operator that asked for it"
                )
            decided = record.model_copy(
                update={
                    "status": "approved" if approve else "denied",
                    "decided_by": approver,
                    "decided_at": float(self._clock()),
                }
            )
            records[request_id] = decided
            self._save(records)
        return decided

    def consume(self, request_id: str, principal: str, payload: ExportPayload) -> ApprovalRecord:
        """Spend an approval on exactly the payload it was granted for."""
        digest = payload_digest(payload)
        with file_lock(self.path):
            records = self._load()
            record = records.get(request_id)
            if record is None:
                raise ApprovalError("unknown export request")
            if record.status != "approved":
                raise ApprovalError(f"request is {record.status}, not approved")
            if record.requester != principal:
                raise ApprovalError(
                    "an approval can only be used by the operator that requested it"
                )
            if float(self._clock()) > record.expires_at:
                raise ApprovalError("approval has expired")
            if record.payload_sha256 != digest:
                raise ApprovalError("the export payload does not match the approved one")
            consumed = record.model_copy(update={"status": "consumed"})
            records[request_id] = consumed
            self._save(records)
        return consumed

    def get(self, request_id: str) -> ApprovalRecord:
        record = self._load().get(request_id)
        if record is None:
            raise ApprovalError("unknown export request")
        return record
