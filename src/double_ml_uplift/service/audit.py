"""A hash-chained, append-only audit log for targeting exports.

Every export request, approval decision, refusal and release is appended as one JSON line whose
digest covers the previous digest:

    digest_i = SHA-256(canonical_json({index, timestamp, actor, action, data, previous}))

so changing, deleting, reordering or re-encoding any entry breaks every digest after it. The
first entry chains from 64 zeros. Verification recomputes the whole chain; supplying the head
digest that was recorded elsewhere also detects truncation of the tail, which is otherwise a
perfectly consistent shorter chain.

Records pass through the repository's redaction before they are hashed, so a field that
accidentally carries a credential is redacted *and* the redacted form is what the chain commits
to. Customer identifiers are pseudonymised by the caller before they ever reach this module.

Writers take a cross-process file lock and re-read the tail before appending, so two processes
sharing one file extend a single chain instead of forking it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from double_ml_uplift.core.canonical import canonical_json, sha256_hex
from double_ml_uplift.core.fileio import file_lock
from double_ml_uplift.core.redaction import redact_value

__all__ = ["GENESIS", "AuditEntry", "AuditError", "AuditLog", "verify_file"]

GENESIS: Final = "0" * 64
_MAX_LINE_BYTES: Final = 64 * 1024


class AuditError(ValueError):
    """The audit log is malformed or its hash chain is broken."""


@dataclass(frozen=True, slots=True)
class AuditEntry:
    index: int
    timestamp: str
    actor: str
    action: str
    data: dict[str, Any]
    previous: str
    digest: str

    def payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "data": self.data,
            "previous": self.previous,
        }

    def recompute(self) -> str:
        return sha256_hex(canonical_json(self.payload()))

    def to_json(self) -> str:
        return json.dumps(
            {**self.payload(), "digest": self.digest}, sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_json(cls, line: str) -> AuditEntry:
        try:
            document = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditError(f"audit line is not valid JSON: {exc.msg}") from None
        if not isinstance(document, dict):
            raise AuditError("audit line is not a JSON object")
        try:
            entry = cls(
                index=_exact_int(document["index"]),
                timestamp=_exact_str(document["timestamp"]),
                actor=_exact_str(document["actor"]),
                action=_exact_str(document["action"]),
                data=_exact_mapping(document["data"]),
                previous=_exact_str(document["previous"]),
                digest=_exact_str(document["digest"]),
            )
        except KeyError as exc:
            raise AuditError(f"audit line is missing the field {exc.args[0]!r}") from None
        return entry


def _exact_int(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise AuditError("expected an integer field")
    return value


def _exact_str(value: object) -> str:
    if not isinstance(value, str):
        raise AuditError("expected a string field")
    return value


def _exact_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AuditError("expected an object field")
    return value


def _parse(text: str) -> list[AuditEntry]:
    entries = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if len(line.encode("utf-8")) > _MAX_LINE_BYTES:
            raise AuditError("audit line exceeds the maximum length")
        entries.append(AuditEntry.from_json(line))
    return entries


def verify_entries(entries: list[AuditEntry], *, expect_head: str | None = None) -> None:
    """Raise :class:`AuditError` if the chain is broken, out of order or truncated."""
    previous = GENESIS
    for position, entry in enumerate(entries):
        if entry.index != position:
            raise AuditError(f"entry {position} claims index {entry.index}")
        if entry.previous != previous:
            raise AuditError(f"entry {position} does not chain from its predecessor")
        if entry.recompute() != entry.digest:
            raise AuditError(f"entry {position} does not match its digest")
        previous = entry.digest
    if expect_head is not None and previous != expect_head:
        raise AuditError("the log does not end at the expected head digest")


def verify_file(path: Path, *, expect_head: str | None = None) -> int:
    """Verify a log file and return the number of entries."""
    if not path.is_file():
        raise AuditError(f"audit log {path.name} does not exist")
    entries = _parse(path.read_text(encoding="utf-8"))
    verify_entries(entries, expect_head=expect_head)
    return len(entries)


class AuditLog:
    """An append-only log. Opening one verifies the existing chain before writing to it."""

    def __init__(self, path: Path, *, clock: Any = None) -> None:
        self.path = path
        self._clock = clock or (lambda: datetime.now(UTC))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entries = self._read()
        self._head = entries[-1].digest if entries else GENESIS
        self._count = len(entries)

    def _read(self) -> list[AuditEntry]:
        if not self.path.is_file():
            return []
        entries = _parse(self.path.read_text(encoding="utf-8"))
        verify_entries(entries)
        return entries

    @property
    def head(self) -> str:
        return self._head

    @property
    def count(self) -> int:
        return self._count

    def entries(self) -> list[AuditEntry]:
        return self._read()

    def append(self, actor: str, action: str, data: Mapping[str, Any]) -> AuditEntry:
        """Append one redacted record, extending whatever chain is on disk right now."""
        redacted = redact_value(dict(data))
        if not isinstance(redacted, dict):  # pragma: no cover - redact_value preserves mappings
            raise AuditError("redaction did not preserve the record shape")
        with file_lock(self.path):
            existing = self._read()
            previous = existing[-1].digest if existing else GENESIS
            entry = AuditEntry(
                index=len(existing),
                timestamp=self._clock().isoformat(),
                actor=actor,
                action=action,
                data=redacted,
                previous=previous,
                digest="",
            )
            entry = AuditEntry(**{**entry.payload(), "digest": entry.recompute()})
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(entry.to_json() + "\n")
                handle.flush()
            self._head = entry.digest
            self._count = entry.index + 1
        return entry
