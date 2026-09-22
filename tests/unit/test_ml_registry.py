from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from double_ml_uplift.ml.manifest import RunManifest, create_manifest
from double_ml_uplift.ml.registry import ModelRegistry, RegistryError, RegistryIntegrityError

pytestmark = pytest.mark.unit


@pytest.fixture
def manifest() -> RunManifest:
    return create_manifest(seed=1, datasets=[], run_id="run", packages=[])


def test_register_resolve_and_load(tmp_path: Path, manifest: RunManifest) -> None:
    registry = ModelRegistry(tmp_path)
    files = {"weights.npz": b"\x00\x01", "config.json": b"{}"}
    version = registry.register("demo-model", files, manifest, metrics={"auc": 0.8})
    assert registry.register("demo-model", files, manifest, metrics={"auc": 0.8}) == version
    assert registry.versions("demo-model") == [version.digest]
    registry.set_alias("demo-model", "production", version.digest)
    assert registry.get("demo-model", "production") == version
    assert registry.load_files(registry.get("demo-model", version.digest)) == files

    other = registry.register("demo-model", files, manifest, metrics={"auc": 0.81})
    assert other.digest != version.digest
    assert registry.versions("missing-model") == []


def test_tampered_blob_is_detected(tmp_path: Path, manifest: RunManifest) -> None:
    registry = ModelRegistry(tmp_path)
    version = registry.register("m", {"w.bin": b"original"}, manifest)
    blob = version.files["w.bin"]
    (tmp_path / "blobs" / "sha256" / blob[:2] / blob).write_bytes(b"tampered")
    with pytest.raises(RegistryIntegrityError, match="content hash"):
        registry.load_files(version)
    (tmp_path / "blobs" / "sha256" / blob[:2] / blob).unlink()
    with pytest.raises(RegistryIntegrityError, match="missing blob"):
        registry.load_files(version)


def test_registering_the_same_content_again_repairs_a_corrupted_blob(
    tmp_path: Path, manifest: RunManifest
) -> None:
    """The address says what the bytes must be, so a damaged blob is rewritten, not re-used.

    Skipping the write because *a* file exists at the address would hand the damage on to every
    later version with the same content, and none of them could ever be loaded.
    """
    registry = ModelRegistry(tmp_path)
    first = registry.register("m", {"w.bin": b"original"}, manifest)
    blob = first.files["w.bin"]
    (tmp_path / "blobs" / "sha256" / blob[:2] / blob).write_bytes(b"tampered")
    with pytest.raises(RegistryIntegrityError):
        registry.load_files(first)

    second = registry.register("m", {"w.bin": b"original"}, manifest, metrics={"auc": 0.9})

    assert second.files["w.bin"] == blob
    assert registry.load_files(second) == {"w.bin": b"original"}
    assert registry.load_files(first) == {"w.bin": b"original"}


def test_tampered_version_record_is_detected(tmp_path: Path, manifest: RunManifest) -> None:
    registry = ModelRegistry(tmp_path)
    version = registry.register("m", {"w.bin": b"data"}, manifest, metrics={"auc": 0.5})
    record_path = tmp_path / "models" / "m" / "versions" / f"{version.digest}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["metrics"]["auc"] = 0.99
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RegistryIntegrityError, match="modified"):
        registry.get("m", version.digest)


@pytest.mark.parametrize(
    ("name", "file_name"),
    [("../evil", "w.bin"), ("Upper", "w.bin"), ("m", "../w.bin"), ("m", "a..b"), ("m", "")],
)
def test_path_traversal_and_invalid_names_are_rejected(
    tmp_path: Path, manifest: RunManifest, name: str, file_name: str
) -> None:
    with pytest.raises(RegistryError):
        ModelRegistry(tmp_path).register(name, {file_name: b"x"}, manifest)


def test_unknown_references_and_empty_versions(tmp_path: Path, manifest: RunManifest) -> None:
    registry = ModelRegistry(tmp_path)
    with pytest.raises(RegistryError, match="at least one file"):
        registry.register("m", {}, manifest)
    with pytest.raises(RegistryError, match="unknown alias"):
        registry.get("m", "production")
    with pytest.raises(RegistryError, match="unknown version"):
        registry.get("m", "0" * 64)
    with pytest.raises(RegistryError, match="unknown version"):
        registry.set_alias("m", "production", "0" * 64)


# --- retention ----------------------------------------------------------------------------------


def _manifest_at(run_id: str, minute: int) -> RunManifest:
    stamp = datetime(2026, 1, 1, 0, minute, tzinfo=UTC)
    return create_manifest(seed=1, datasets=[], run_id=run_id, packages=[], clock=lambda: stamp)


def _blob_path(root: Path, digest: str) -> Path:
    return root / "blobs" / "sha256" / digest[:2] / digest


def test_prune_keeps_the_newest_versions_and_every_alias_target(tmp_path: Path) -> None:
    """Retention bounds the directory without ever deleting what an alias points at."""
    registry = ModelRegistry(tmp_path)
    versions = [
        registry.register("m", {"w.bin": f"weights-{i}".encode()}, _manifest_at(f"run-{i}", i))
        for i in range(5)
    ]
    registry.set_alias("m", "production", versions[0].digest)  # the oldest, but pinned

    removed = registry.prune("m", keep=2)

    assert sorted(removed) == sorted(item.digest for item in versions[1:3])
    survivors = {versions[0].digest, versions[3].digest, versions[4].digest}
    assert set(registry.versions("m")) == survivors
    for gone in versions[1:3]:
        assert not _blob_path(tmp_path, gone.files["w.bin"]).exists()
    for index in (0, 3, 4):
        kept = registry.get("m", versions[index].digest)
        assert registry.load_files(kept) == {"w.bin": f"weights-{index}".encode()}
    assert registry.get("m", "production").digest == versions[0].digest


def test_prune_keeps_a_blob_that_any_surviving_version_still_references(tmp_path: Path) -> None:
    """Blobs are shared by content, across versions and across models; only orphans go."""
    registry = ModelRegistry(tmp_path)
    old = registry.register(
        "m", {"w.bin": b"shared-weights", "x.bin": b"old-only"}, _manifest_at("a", 0)
    )
    new = registry.register("m", {"w.bin": b"shared-weights"}, _manifest_at("b", 1))
    other = registry.register("other", {"w.bin": b"old-only"}, _manifest_at("c", 0))
    lonely = registry.register("m", {"w.bin": b"nobody-else"}, _manifest_at("d", 0))

    registry.prune("m", keep=1)

    assert registry.versions("m") == [new.digest]
    assert registry.load_files(new) == {"w.bin": b"shared-weights"}
    # ``x.bin`` of the pruned version is still what model "other" is made of.
    assert registry.load_files(registry.get("other", other.digest)) == {"w.bin": b"old-only"}
    assert _blob_path(tmp_path, old.files["x.bin"]).exists()
    assert not _blob_path(tmp_path, lonely.files["w.bin"]).exists()


def test_prune_never_deletes_what_it_cannot_read(tmp_path: Path) -> None:
    """A record that fails verification is evidence, and its references are unknown.

    It is neither counted nor deleted, and blob collection is skipped for the whole pass, because
    a blob that looks orphaned may be exactly what the unreadable record points at.
    """
    registry = ModelRegistry(tmp_path)
    broken = registry.register("m", {"w.bin": b"broken-weights"}, _manifest_at("a", 0))
    stale = registry.register("m", {"w.bin": b"stale-weights"}, _manifest_at("b", 1))
    newest = registry.register("m", {"w.bin": b"newest-weights"}, _manifest_at("c", 2))
    record_path = tmp_path / "models" / "m" / "versions" / f"{broken.digest}.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["metrics"]["forged"] = 1.0
    record_path.write_text(json.dumps(record), encoding="utf-8")

    removed = registry.prune("m", keep=1)

    assert removed == [stale.digest]
    assert set(registry.versions("m")) == {broken.digest, newest.digest}
    assert _blob_path(tmp_path, broken.files["w.bin"]).exists()
    assert _blob_path(tmp_path, stale.files["w.bin"]).exists()


def test_prune_validates_its_arguments(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    with pytest.raises(RegistryError, match="keep"):
        registry.prune("m", keep=0)
    assert registry.prune("never-registered", keep=1) == []


def test_prune_orders_by_recorded_time_even_when_a_stamp_is_naive_or_garbled(
    tmp_path: Path,
) -> None:
    """A naive stamp is read as UTC and a garbled one sorts oldest, so neither crashes a prune."""
    registry = ModelRegistry(tmp_path)
    base = create_manifest(seed=1, datasets=[], run_id="base", packages=[])

    def at(stamp: str) -> RunManifest:
        return base.model_copy(update={"created_at": stamp, "run_id": stamp})

    garbled = registry.register("m", {"w.bin": b"g"}, at("yesterday"))
    naive = registry.register("m", {"w.bin": b"n"}, at("2026-01-01T00:05:00"))
    registry.register("m", {"w.bin": b"a"}, at("2026-01-01T00:10:00+00:00"))

    assert registry.prune("m", keep=1) == [garbled.digest, naive.digest]


def test_a_stray_directory_under_models_suspends_blob_collection(tmp_path: Path) -> None:
    """Something the registry did not write may reference any blob, so none is proven orphaned."""
    registry = ModelRegistry(tmp_path)
    old = registry.register("m", {"w.bin": b"old"}, _manifest_at("a", 0))
    registry.register("m", {"w.bin": b"new"}, _manifest_at("b", 1))
    (tmp_path / "models" / "Not A Model").mkdir()

    assert registry.prune("m", keep=1) == [old.digest]
    assert _blob_path(tmp_path, old.files["w.bin"]).exists()
