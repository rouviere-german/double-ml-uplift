"""Publishing a fitted run: the reproducibility record and the content-addressed registry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from double_ml_uplift.causal.bundle import BundleManifest
from double_ml_uplift.causal.engine import EstimationConfig, EstimationResult, UpliftEngine
from double_ml_uplift.causal.nuisance import NuisanceConfig
from double_ml_uplift.causal.publish import (
    BUNDLE_MANIFEST_FILE,
    MODEL_NAME,
    find_lockfile,
    finite_metrics,
    load_published_bundle,
    publish_run,
    run_manifest,
)
from double_ml_uplift.causal.scenarios import load_scenario
from double_ml_uplift.causal.scm import simulate_campaign
from double_ml_uplift.causal.spec import bundled_spec
from double_ml_uplift.ml.registry import ModelRegistry, RegistryError, RegistryIntegrityError

pytestmark = pytest.mark.unit

FAST = EstimationConfig(
    n_folds=3,
    nuisance=NuisanceConfig(kind="ridge"),
    cate_learner=NuisanceConfig(kind="ridge"),
    run_refuters=False,
    n_bootstrap=0,
)


Published = tuple[UpliftEngine, EstimationResult, BundleManifest, dict[str, bytes]]


@pytest.fixture(scope="module")
def published() -> Published:
    campaign = simulate_campaign(load_scenario("linear", n_customers=800, seed=11))
    engine = UpliftEngine(bundled_spec("retention_offer"), FAST)
    result = engine.fit(campaign.frame)
    manifest, artifacts = engine.bundle(
        result, metrics={"ate": result.primary.ate, "se": result.primary.se}
    )
    return engine, result, manifest, artifacts


def test_the_run_manifest_records_the_environment_alongside_the_content_digests(
    published: Published, tmp_path: Path
) -> None:
    """Four digests say what was fitted; the environment says where. Both are needed to repeat."""
    _, result, manifest, _ = published
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("version = 1\n", encoding="utf-8")

    run = run_manifest(result, seed=7, metrics=manifest.metrics, lockfile=lockfile)

    assert run.params["spec_sha256"] == result.spec_sha256
    assert run.params["config_sha256"] == result.config_sha256
    assert run.params["dataset_sha256"] == result.dataset_sha256
    assert run.params["fold_digest"] == result.fold_digest
    assert run.params["n_rows"] == result.prepared.n
    assert isinstance(run.params["lockfile_sha256"], str)
    assert len(str(run.params["lockfile_sha256"])) == 64
    assert run.python.startswith("3.")
    assert set(run.packages) >= {"numpy", "pandas", "scikit-learn"}
    assert "not-installed" not in set(run.packages.values())
    # The whole record has to survive a JSON round trip, because that is how it is stored.
    json.dumps(run.model_dump(mode="json"), allow_nan=False)


def test_a_missing_lockfile_is_recorded_as_absent_rather_than_raising(
    published: Published, tmp_path: Path
) -> None:
    """A deployed image installs from the lockfile and discards it; that is not an error."""
    _, result, _, _ = published
    assert find_lockfile(tmp_path) is None
    run = run_manifest(result, seed=7, lockfile=None)
    assert run.params["lockfile_sha256"] is None


def test_the_lockfile_of_this_repository_is_found_from_inside_it() -> None:
    lockfile = find_lockfile(Path(__file__).parent)
    assert lockfile is not None
    assert lockfile.name == "uv.lock"


def test_a_non_finite_metric_is_dropped_instead_of_losing_the_whole_manifest() -> None:
    kept = finite_metrics({"ate": 1.5, "se": float("inf"), "pehe": float("nan")})
    assert kept == {"ate": 1.5}


def test_a_published_run_scores_exactly_like_the_engine_that_produced_it(
    published: Published, tmp_path: Path
) -> None:
    engine, result, manifest, artifacts = published
    registry = ModelRegistry(tmp_path / "registry")
    run = run_manifest(result, seed=engine.config.seed, metrics=manifest.metrics)

    version = publish_run(registry, manifest, artifacts, run)

    assert version.name == MODEL_NAME
    assert BUNDLE_MANIFEST_FILE in version.files
    assert set(version.files) >= {"propensity.bin", "cate.bin"}
    assert version.manifest.params["fold_digest"] == result.fold_digest

    restored = load_published_bundle(registry, "latest")
    rows = result.prepared.features[:128]
    from_registry, _, _ = restored.predict(rows)
    direct, _, _ = engine.score_features(result, rows)
    assert np.allclose(from_registry, direct)
    assert restored.manifest.sha256() == manifest.sha256()


def test_publishing_the_same_run_twice_yields_the_same_version(
    published: Published, tmp_path: Path
) -> None:
    """Content addressing, not an autoincrementing number: identical content is one version."""
    _, result, manifest, artifacts = published
    registry = ModelRegistry(tmp_path / "registry")
    run = run_manifest(result, seed=3, run_id="fixed", now=None, metrics=manifest.metrics)

    first = publish_run(registry, manifest, artifacts, run)
    second = publish_run(registry, manifest, artifacts, run)

    assert first.digest == second.digest
    assert registry.versions(MODEL_NAME) == [first.digest]


def test_a_rewritten_blob_is_caught_when_the_run_is_read_back(
    published: Published, tmp_path: Path
) -> None:
    """The point of the content address: an artifact cannot be edited under an unchanged record."""
    _, result, manifest, artifacts = published
    registry = ModelRegistry(tmp_path / "registry")
    version = publish_run(
        registry, manifest, artifacts, run_manifest(result, seed=3, metrics=manifest.metrics)
    )

    blob = version.files["cate.bin"]
    path = tmp_path / "registry" / "blobs" / "sha256" / blob[:2] / blob
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(RegistryIntegrityError):
        load_published_bundle(registry, version.digest)


def test_publishing_without_an_alias_leaves_the_alias_alone(
    published: Published, tmp_path: Path
) -> None:
    _, result, manifest, artifacts = published
    registry = ModelRegistry(tmp_path / "registry")
    run = run_manifest(result, seed=3, metrics=manifest.metrics)

    version = publish_run(registry, manifest, artifacts, run, alias=None)

    assert registry.versions(MODEL_NAME) == [version.digest]
    with pytest.raises(RegistryError, match="unknown alias"):
        load_published_bundle(registry, "latest")


def test_a_retention_cap_bounds_the_registry_and_keeps_the_latest_run_loadable(
    published: Published, tmp_path: Path
) -> None:
    """Every estimate publishes, so without a cap the registry grows by one bundle per request."""
    engine, result, manifest, artifacts = published
    registry = ModelRegistry(tmp_path / "registry")
    versions = [
        publish_run(
            registry,
            manifest,
            artifacts,
            run_manifest(result, seed=engine.config.seed, run_id=f"run-{index}"),
            keep=2,
        )
        for index in range(4)
    ]

    assert len(registry.versions(MODEL_NAME)) == 2
    assert registry.get(MODEL_NAME, "latest").digest == versions[-1].digest
    restored = load_published_bundle(registry, "latest")
    assert restored.manifest.sha256() == manifest.sha256()
