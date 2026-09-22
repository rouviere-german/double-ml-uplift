"""Content-addressed model registry on the local filesystem.

Layout under ``root``::

    blobs/sha256/<2 hex>/<digest>          artifact bytes, named by their SHA-256
    models/<name>/versions/<digest>.json   version record (file -> blob digest, manifest, metrics)
    models/<name>/aliases.json             alias -> version digest (for example "production")

A version digest is the SHA-256 of the canonical record, so identical content always yields the
same version and any edit to a record or blob is detected on load.

Retention: :meth:`ModelRegistry.prune` keeps the newest ``keep`` versions of a model plus every
version an alias points at, then deletes the blobs that no remaining version of *any* model
references. A caller that registers on every request (the service publishes every successful
estimate) must prune, or the directory grows by one bundle per request. Writers and the pruner
share one advisory lock, so a blob that a concurrent ``register`` has just re-used is never
collected underneath it. A record that fails verification is never deleted, and while one exists
blob collection is skipped entirely: its references are unknown, so no blob can be proven orphaned.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from double_ml_uplift.core.canonical import canonical_json, sha256_hex
from double_ml_uplift.core.fileio import atomic_write_bytes, atomic_write_text, file_lock
from double_ml_uplift.ml.manifest import RunManifest

__all__ = ["ModelRegistry", "ModelVersion", "RegistryError", "RegistryIntegrityError"]

_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class RegistryError(ValueError):
    """Invalid registry request (bad name, unknown version or alias)."""


class RegistryIntegrityError(RegistryError):
    """Stored bytes or records do not match their content address."""


class ModelVersion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: dict[str, str]
    manifest: RunManifest
    metrics: dict[str, float]


def _check_name(kind: str, value: str, pattern: re.Pattern[str]) -> str:
    if not pattern.fullmatch(value) or ".." in value:
        raise RegistryError(f"invalid {kind}: {value!r}")
    return value


def _created_at(version: ModelVersion) -> datetime:
    """Order versions by when their run was recorded; an unparseable stamp sorts oldest."""
    try:
        stamp = datetime.fromisoformat(version.manifest.created_at)
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)
    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=UTC)


class ModelRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _lock(self) -> Path:
        # ``file_lock`` takes its lock on the sidecar ``registry.lock`` next to this path.
        return self.root / "registry"

    def _blob_path(self, digest: str) -> Path:
        _check_name("digest", digest, _DIGEST)
        return self.root / "blobs" / "sha256" / digest[:2] / digest

    def _model_dir(self, name: str) -> Path:
        return self.root / "models" / _check_name("model name", name, _NAME)

    def put_blob(self, data: bytes) -> str:
        """Store ``data`` under its SHA-256 and return the digest.

        An existing file at that address is re-used only while it still holds exactly these
        bytes. A damaged one is rewritten: the address alone says what it must contain, and
        re-using it would hand the damage on to every later version with the same content.
        """
        digest = sha256_hex(data)
        path = self._blob_path(digest)
        if not path.exists() or path.read_bytes() != data:
            atomic_write_bytes(path, data)
        return digest

    def read_blob(self, digest: str) -> bytes:
        path = self._blob_path(digest)
        if not path.exists():
            raise RegistryIntegrityError(f"missing blob {digest}")
        data = path.read_bytes()
        if sha256_hex(data) != digest:
            raise RegistryIntegrityError(f"blob {digest} does not match its content hash")
        return data

    def register(
        self,
        name: str,
        files: Mapping[str, bytes],
        manifest: RunManifest,
        metrics: Mapping[str, float] | None = None,
    ) -> ModelVersion:
        model_dir = self._model_dir(name)
        if not files:
            raise RegistryError("a model version needs at least one file")
        # Every name is checked before any byte is written, so a bad name leaves no orphan blob.
        checked = {
            _check_name("file name", file_name, _FILE): data for file_name, data in files.items()
        }
        # Held across the blob writes and the record write: ``put_blob`` skips a blob that already
        # exists, so without the lock ``prune`` could collect it between the skip and the record.
        with file_lock(self._lock()):
            blob_refs = {
                file_name: self.put_blob(data) for file_name, data in sorted(checked.items())
            }
            record: dict[str, JsonValue] = {
                "name": name,
                "files": dict(blob_refs),
                "manifest": manifest.model_dump(mode="json"),
                "metrics": dict(metrics or {}),
            }
            digest = sha256_hex(canonical_json(record))
            version = ModelVersion.model_validate({**record, "digest": digest})
            path = model_dir / "versions" / f"{digest}.json"
            if not path.exists():
                atomic_write_text(path, version.model_dump_json(indent=2) + "\n")
        return version

    def versions(self, name: str) -> list[str]:
        directory = self._model_dir(name) / "versions"
        return sorted(p.stem for p in directory.glob("*.json")) if directory.exists() else []

    def _aliases_path(self, name: str) -> Path:
        return self._model_dir(name) / "aliases.json"

    def _aliases(self, name: str) -> dict[str, str]:
        path = self._aliases_path(name)
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items()}

    def set_alias(self, name: str, alias: str, digest: str) -> None:
        _check_name("alias", alias, _NAME)
        self.get(name, digest)  # must exist and verify
        aliases = self._aliases(name)
        aliases[alias] = digest
        atomic_write_text(
            self._aliases_path(name), json.dumps(aliases, indent=2, sort_keys=True) + "\n"
        )

    def get(self, name: str, ref: str) -> ModelVersion:
        """Load a version by digest or alias and verify its record hash."""
        digest = ref if _DIGEST.fullmatch(ref) else self._aliases(name).get(ref)
        if digest is None:
            raise RegistryError(f"unknown alias {ref!r} for model {name}")
        path = self._model_dir(name) / "versions" / f"{digest}.json"
        if not path.exists():
            raise RegistryError(f"unknown version {digest} for model {name}")
        version = ModelVersion.model_validate_json(path.read_text(encoding="utf-8"))
        record = version.model_dump(mode="json")
        record.pop("digest")
        if version.digest != digest or sha256_hex(canonical_json(record)) != digest:
            raise RegistryIntegrityError(f"version record {digest} was modified")
        return version

    def load_files(self, version: ModelVersion) -> dict[str, bytes]:
        return {file_name: self.read_blob(blob) for file_name, blob in version.files.items()}

    # --- retention -----------------------------------------------------------------------

    def _readable_versions(self) -> tuple[dict[str, list[ModelVersion]], bool]:
        """Every verifiable version of every model, and whether any record failed to verify."""
        found: dict[str, list[ModelVersion]] = {}
        unreadable = False
        models = self.root / "models"
        for model_dir in sorted(models.iterdir()) if models.is_dir() else []:
            try:
                digests = self.versions(model_dir.name)
            except RegistryError:
                # Not a name ``register`` could have written, so whatever it holds is unknown.
                unreadable = True
                continue
            for digest in digests:
                try:
                    found.setdefault(model_dir.name, []).append(self.get(model_dir.name, digest))
                except (RegistryError, ValueError):
                    unreadable = True
        return found, unreadable

    def prune(self, name: str, *, keep: int) -> list[str]:
        """Delete all but the newest ``keep`` versions of ``name`` and the blobs left orphaned.

        A version an alias points at always survives as well, even when it is older than the
        newest ``keep``. Returns the digests of the version records removed, oldest first.
        """
        if keep < 1:
            raise RegistryError("keep must be at least 1")
        versions_dir = self._model_dir(name) / "versions"
        with file_lock(self._lock()):
            found, unreadable = self._readable_versions()
            pinned = set(self._aliases(name).values())
            ordered = sorted(found.get(name, []), key=lambda item: (_created_at(item), item.digest))
            doomed = [item for item in ordered[:-keep] if item.digest not in pinned]
            for item in doomed:
                (versions_dir / f"{item.digest}.json").unlink(missing_ok=True)
            if not unreadable and doomed:
                gone = {item.digest for item in doomed}
                live = {
                    blob
                    for items in found.values()
                    for item in items
                    if item.digest not in gone
                    for blob in item.files.values()
                }
                for blob in {blob for item in doomed for blob in item.files.values()} - live:
                    self._blob_path(blob).unlink(missing_ok=True)
        return [item.digest for item in doomed]
