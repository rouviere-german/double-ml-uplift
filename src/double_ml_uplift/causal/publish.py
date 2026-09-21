"""Recording a fitted run: the environment it ran in, and a content-addressed copy of it.

:class:`~double_ml_uplift.causal.bundle.BundleManifest` pins *what* was fitted -- the
specification, configuration, dataset and fold-assignment digests -- but nothing about *where*.
Reproducing a number needs both, so this module joins the two halves the repository already has:

* :class:`~double_ml_uplift.ml.manifest.RunManifest` records the interpreter, the platform, the
  installed package versions, the git commit and the SHA-256 of the lockfile, alongside the four
  content digests as parameters;
* :class:`~double_ml_uplift.ml.registry.ModelRegistry` stores the bundle's artifacts under their
  own SHA-256 and the version record under the hash of its canonical form, so neither an artifact
  nor its recorded hash can be edited without the version digest changing.

This is what stands in for the MLflow tracking the specification asks for (ADR 0004): a
directory, no server, no pickle, and a version digest that is a function of the content rather
than an autoincrementing number.

The registry is *not* signed. An attacker who can write the directory can write a consistent
forgery; the threat model says so (T2, T3). What the content addressing buys is detection of a
partial edit -- an artifact swapped under an unchanged record, or a record edited under an
unchanged digest -- which is the accident that actually happens.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import uuid
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import JsonValue

from double_ml_uplift.causal.bundle import BundleManifest, ScoringBundle
from double_ml_uplift.ml.manifest import RunManifest, package_versions, sha256_file
from double_ml_uplift.ml.registry import ModelRegistry, ModelVersion

if TYPE_CHECKING:
    from double_ml_uplift.causal.engine import EstimationResult

__all__ = [
    "BUNDLE_MANIFEST_FILE",
    "MODEL_NAME",
    "RUN_PACKAGES",
    "find_lockfile",
    "finite_metrics",
    "load_published_bundle",
    "publish_run",
    "run_manifest",
]

MODEL_NAME: Final = "uplift-scoring-bundle"
BUNDLE_MANIFEST_FILE: Final = "bundle.json"

# The packages whose versions actually move a number: the nuisance learners, the numerics they
# sit on, and the contract that guards the input. Recording every installed distribution would
# make two identical runs look different because a linter was upgraded.
RUN_PACKAGES: Final = (
    "lightgbm",
    "numpy",
    "pandas",
    "pandera",
    "pydantic",
    "scikit-learn",
    "scipy",
)

# Walking further than this from the start directory would leave the checkout entirely.
_MAX_LOCKFILE_DEPTH: Final = 6


def find_lockfile(start: Path | None = None) -> Path | None:
    """Return the nearest ``uv.lock`` at or above ``start``, or ``None`` if there is none.

    A deployed image installs from the lockfile and then discards it, so its absence is normal
    and is recorded as ``null`` rather than raising: a manifest that says "no lockfile was
    visible" is more useful than no manifest at all.
    """
    directory = (start or Path.cwd()).resolve()
    for candidate in (directory, *directory.parents)[:_MAX_LOCKFILE_DEPTH]:
        lockfile = candidate / "uv.lock"
        if lockfile.is_file():
            return lockfile
    return None


def finite_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Keep only the metrics that can be recorded and read back.

    :class:`RunManifest` refuses a non-finite metric, because ``inf`` and ``nan`` do not survive
    JSON. Dropping the offending entry keeps the reproducibility record -- which is about the
    environment, not about the number -- rather than losing the whole manifest to one degenerate
    standard error.
    """
    return {name: float(value) for name, value in metrics.items() if math.isfinite(float(value))}


def run_manifest(
    result: EstimationResult,
    *,
    seed: int,
    metrics: Mapping[str, float] | None = None,
    lockfile: Path | None = None,
    packages: Iterable[str] = RUN_PACKAGES,
    run_id: str | None = None,
    now: datetime | None = None,
) -> RunManifest:
    """Describe one fitted run: its four content digests plus the environment that produced it.

    ``datasets`` stays empty because the campaign table arrives in memory over HTTP and is never
    written to disk; its SHA-256 is in ``params`` instead, computed by the data contract from the
    canonical form of the frame.
    """
    params: dict[str, JsonValue] = {
        "spec_sha256": result.spec_sha256,
        "config_sha256": result.config_sha256,
        "dataset_sha256": result.dataset_sha256,
        "fold_digest": result.fold_digest,
        "adjustment_set": list(result.adjustment_set),
        "primary_estimator": result.primary_name,
        "n_rows": result.prepared.n,
        "lockfile_sha256": sha256_file(lockfile) if lockfile is not None else None,
    }
    return RunManifest(
        run_id=run_id or uuid.uuid4().hex,
        created_at=(now or datetime.now(UTC)).isoformat(),
        seed=seed,
        python=sys.version.split()[0],
        platform=platform.platform(),
        pythonhashseed=os.environ.get("PYTHONHASHSEED"),
        git_commit=os.environ.get("GIT_COMMIT") or os.environ.get("GITHUB_SHA"),
        packages=package_versions(packages),
        datasets=(),
        params=params,
        metrics=finite_metrics(metrics or {}),
    )


def publish_run(
    registry: ModelRegistry,
    manifest: BundleManifest,
    artifacts: Mapping[str, bytes],
    run: RunManifest,
    *,
    name: str = MODEL_NAME,
    alias: str | None = "latest",
) -> ModelVersion:
    """Store the bundle and its run manifest, and point ``alias`` at the resulting version.

    The bundle manifest travels as one more artifact, so ``load_published_bundle`` can rebuild a
    scoring bundle from the registry alone -- the version record carries no model structure of
    its own, only digests.
    """
    files = dict(artifacts)
    files[BUNDLE_MANIFEST_FILE] = manifest.model_dump_json(indent=2).encode("utf-8")
    version = registry.register(name, files, run, metrics=finite_metrics(manifest.metrics))
    if alias is not None:
        registry.set_alias(name, alias, version.digest)
    return version


def load_published_bundle(
    registry: ModelRegistry, ref: str, *, name: str = MODEL_NAME
) -> ScoringBundle:
    """Rebuild a scoring bundle from a registry version digest or alias.

    Every blob is verified against its content address on the way out, so a tampered artifact is
    a :class:`~double_ml_uplift.ml.registry.RegistryIntegrityError` rather than a silent change
    in what the model predicts.
    """
    version = registry.get(name, ref)
    files = registry.load_files(version)
    manifest = BundleManifest.model_validate_json(files[BUNDLE_MANIFEST_FILE].decode("utf-8"))
    artifacts = {key: value for key, value in files.items() if key != BUNDLE_MANIFEST_FILE}
    return ScoringBundle.load(manifest, artifacts)
