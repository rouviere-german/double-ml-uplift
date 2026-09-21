"""Pickle-free serialization of a fitted scoring bundle.

A scoring bundle is what the service loads: the propensity model, the conditional-effect model,
the adjustment set they expect, the overlap bounds they abstain outside, and the hashes of the
specification, configuration and data they came from.

Every model is written in a format that cannot execute code on load:

* LightGBM boosters as their own text dump, reloaded through ``lightgbm.Booster(model_str=...)``;
* linear models (with or without a leading standardiser) as JSON coefficients, reloaded as an
  explicit ``(x - mean) / scale @ coef + intercept`` evaluation in this module.

There is no ``pickle``, ``joblib`` or ``cloudpickle`` anywhere in the path, which is enforced by
``tests/unit/test_no_pickle.py``, a semgrep rule and ``scripts/check_baseline.py``. Each artifact
carries its SHA-256 in the manifest and is verified on load, so a tampered blob is refused rather
than silently scored with.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from double_ml_uplift.causal.calibration import CateCalibration
from double_ml_uplift.causal.diagnostics import GatePolicy, abstain_mask
from double_ml_uplift.causal.nuisance import Classifier, LinearProbabilityModel, Regressor
from double_ml_uplift.core.canonical import sha256_hex
from double_ml_uplift.core.fileio import atomic_write_bytes, atomic_write_text

__all__ = [
    "ArtifactError",
    "BundleManifest",
    "ScoringBundle",
    "SerializedModel",
    "UnsupportedModelError",
    "deserialize_model",
    "serialize_model",
]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

ModelFormat = Literal["lightgbm-text/v1", "linear-json/v1", "scaled-linear-json/v1"]
ModelTask = Literal["regression", "probability"]
MAX_REFERENCE_SAMPLE = 5_000


class ArtifactError(ValueError):
    """A stored artifact is missing, malformed or does not match its recorded hash."""


class UnsupportedModelError(ArtifactError):
    """The fitted model has no pickle-free representation in this repository."""


class SerializedModel(BaseModel):
    """One model, as bytes plus the metadata needed to evaluate it again."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format: ModelFormat
    task: ModelTask
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    n_features: int = Field(ge=1, le=10_000)
    # The sigmoid link is applied on load for logistic models; linear probability models and
    # LightGBM binary boosters already return a probability.
    logistic: bool = False


def _linear_payload(coef: Any, intercept: Any, mean: Any = None, scale: Any = None) -> bytes:
    document: dict[str, Any] = {
        "coef": [float(value) for value in np.asarray(coef, dtype=np.float64).ravel()],
        "intercept": float(np.asarray(intercept, dtype=np.float64).ravel()[0]),
    }
    if mean is not None and scale is not None:
        document["mean"] = [float(value) for value in np.asarray(mean, dtype=np.float64).ravel()]
        document["scale"] = [float(value) for value in np.asarray(scale, dtype=np.float64).ravel()]
    return json.dumps(document, sort_keys=True, allow_nan=False).encode("utf-8")


def _pipeline_steps(model: object) -> list[Any] | None:
    steps = getattr(model, "steps", None)
    if steps is None:
        return None
    return [step for _, step in steps]


def serialize_model(model: object, task: ModelTask) -> tuple[SerializedModel, bytes]:
    """Convert a fitted learner into ``(metadata, payload)``. Raises on unknown model types."""
    if isinstance(model, LinearProbabilityModel):
        inner = model.inner
        payload = _linear_payload(inner.coef_, inner.intercept_)
        return _describe("linear-json/v1", task, payload, len(inner.coef_), logistic=False), payload

    steps = _pipeline_steps(model)
    if steps is not None:
        if len(steps) != 2:
            raise UnsupportedModelError(
                "only a two-step (scaler, linear model) pipeline is supported"
            )
        scaler, final = steps
        logistic = type(final).__name__ == "LogisticRegression"
        payload = _linear_payload(final.coef_, final.intercept_, scaler.mean_, scaler.scale_)
        n_features = len(np.asarray(scaler.mean_).ravel())
        return (
            _describe("scaled-linear-json/v1", task, payload, n_features, logistic=logistic),
            payload,
        )

    name = type(model).__name__
    if name in {"LGBMRegressor", "LGBMClassifier"}:
        booster = model.booster_  # type: ignore[attr-defined]
        payload = str(booster.model_to_string()).encode("utf-8")
        return (
            _describe(
                "lightgbm-text/v1", task, payload, int(booster.num_feature()), logistic=False
            ),
            payload,
        )
    if name == "LinearRegression":
        payload = _linear_payload(model.coef_, model.intercept_)  # type: ignore[attr-defined]
        n_features = len(np.asarray(model.coef_).ravel())  # type: ignore[attr-defined]
        return _describe("linear-json/v1", task, payload, n_features, logistic=False), payload
    raise UnsupportedModelError(f"no pickle-free representation for {name}")


def _describe(
    fmt: ModelFormat, task: ModelTask, payload: bytes, n_features: int, *, logistic: bool
) -> SerializedModel:
    return SerializedModel(
        format=fmt,
        task=task,
        sha256=sha256_hex(payload),
        n_features=n_features,
        logistic=logistic,
    )


class _LinearEvaluator:
    def __init__(self, document: dict[str, Any], *, logistic: bool) -> None:
        self._coef = np.asarray(document["coef"], dtype=np.float64)
        self._intercept = float(document["intercept"])
        mean = document.get("mean")
        scale = document.get("scale")
        self._mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self._scale = None if scale is None else np.asarray(scale, dtype=np.float64)
        self._logistic = logistic

    def predict(self, features: FloatArray) -> FloatArray:
        values = np.asarray(features, dtype=np.float64)
        if self._mean is not None and self._scale is not None:
            values = (values - self._mean) / self._scale
        linear = values @ self._coef + self._intercept
        if self._logistic:
            return np.asarray(0.5 * (1.0 + np.tanh(0.5 * linear)), dtype=np.float64)
        return np.asarray(linear, dtype=np.float64)


class _BoosterEvaluator:
    def __init__(self, text: str) -> None:
        import lightgbm  # noqa: PLC0415 - only needed when a booster is loaded

        self._booster = lightgbm.Booster(model_str=text)

    def predict(self, features: FloatArray) -> FloatArray:
        return np.asarray(self._booster.predict(features), dtype=np.float64)


def deserialize_model(meta: SerializedModel, payload: bytes) -> Regressor:
    """Rebuild an evaluator from its bytes, verifying the recorded hash first."""
    if sha256_hex(payload) != meta.sha256:
        raise ArtifactError(f"artifact does not match its recorded hash ({meta.format})")
    if meta.format == "lightgbm-text/v1":
        return cast("Regressor", _BoosterEvaluator(payload.decode("utf-8")))
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"artifact is not valid JSON ({meta.format})") from exc
    if not isinstance(document, dict) or "coef" not in document:
        raise ArtifactError(f"artifact is not a linear model document ({meta.format})")
    return cast("Regressor", _LinearEvaluator(document, logistic=meta.logistic))


def _reference_sample(values: npt.ArrayLike) -> tuple[float, ...]:
    """Down-sample a propensity vector to at most MAX_REFERENCE_SAMPLE values, deterministically."""
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0:
        raise ValueError("a reference propensity sample is required")
    if array.size <= MAX_REFERENCE_SAMPLE:
        return tuple(float(value) for value in array)
    step = array.size / MAX_REFERENCE_SAMPLE
    index = (np.arange(MAX_REFERENCE_SAMPLE) * step).astype(np.int64)
    return tuple(float(value) for value in array[index])


class BundleManifest(BaseModel):
    """Everything needed to reproduce, audit and verify a scoring bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fold_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    adjustment_set: tuple[str, ...]
    gate: GatePolicy
    propensity: SerializedModel
    cate: SerializedModel
    # Binned uplift calibration, used to attach a group-level interval to every score.
    calibration: dict[str, Any]
    # A capped sample of training propensities, so drift can be measured at scoring time
    # without keeping the training data.
    reference_propensity: tuple[float, ...] = Field(max_length=MAX_REFERENCE_SAMPLE)
    metrics: dict[str, float] = Field(default_factory=dict)

    def sha256(self) -> str:
        return sha256_hex(self.model_dump_json())


@dataclass(frozen=True, slots=True)
class ScoringBundle:
    """A loaded bundle: predict uplift and propensity, and abstain outside the overlap region."""

    manifest: BundleManifest
    propensity: Regressor
    cate: Regressor

    @property
    def adjustment_set(self) -> tuple[str, ...]:
        return self.manifest.adjustment_set

    @property
    def calibration(self) -> CateCalibration:
        return CateCalibration.from_dict(self.manifest.calibration)

    def predict(self, features: FloatArray) -> tuple[FloatArray, FloatArray, BoolArray]:
        """Return ``(uplift, propensity, abstain)`` for a matrix of adjustment-set columns."""
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.adjustment_set):
            raise ValueError("features do not match the bundle's adjustment set")
        propensity = np.asarray(self.propensity.predict(values), dtype=np.float64)
        uplift = np.asarray(self.cate.predict(values), dtype=np.float64)
        return uplift, propensity, abstain_mask(propensity, self.manifest.gate)

    # --- persistence ---------------------------------------------------------------------

    @staticmethod
    def build(
        *,
        spec_sha256: str,
        config_sha256: str,
        dataset_sha256: str,
        fold_digest: str,
        adjustment_set: tuple[str, ...],
        gate: GatePolicy,
        propensity_model: Classifier,
        cate_model: Regressor,
        calibration: CateCalibration,
        reference_propensity: npt.ArrayLike,
        metrics: dict[str, float] | None = None,
    ) -> tuple[BundleManifest, dict[str, bytes]]:
        """Serialize a fitted pair of models into a manifest and its artifact blobs."""
        propensity_meta, propensity_bytes = serialize_model(propensity_model, "probability")
        cate_meta, cate_bytes = serialize_model(cate_model, "regression")
        manifest = BundleManifest(
            spec_sha256=spec_sha256,
            config_sha256=config_sha256,
            dataset_sha256=dataset_sha256,
            fold_digest=fold_digest,
            adjustment_set=adjustment_set,
            gate=gate,
            propensity=propensity_meta,
            cate=cate_meta,
            calibration=calibration.to_dict(),
            reference_propensity=_reference_sample(reference_propensity),
            metrics=dict(metrics or {}),
        )
        return manifest, {"propensity.bin": propensity_bytes, "cate.bin": cate_bytes}

    @classmethod
    def load(cls, manifest: BundleManifest, artifacts: dict[str, bytes]) -> ScoringBundle:
        missing = sorted({"propensity.bin", "cate.bin"} - set(artifacts))
        if missing:
            raise ArtifactError(f"bundle is missing artifact(s): {', '.join(missing)}")
        return cls(
            manifest=manifest,
            propensity=deserialize_model(manifest.propensity, artifacts["propensity.bin"]),
            cate=deserialize_model(manifest.cate, artifacts["cate.bin"]),
        )

    @staticmethod
    def write(directory: Path, manifest: BundleManifest, artifacts: dict[str, bytes]) -> None:
        atomic_write_text(directory / "manifest.json", manifest.model_dump_json(indent=2) + "\n")
        for name, payload in artifacts.items():
            atomic_write_bytes(directory / name, payload)

    @classmethod
    def read(cls, directory: Path) -> ScoringBundle:
        manifest = BundleManifest.model_validate_json(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        artifacts = {
            name: (directory / name).read_bytes()
            for name in ("propensity.bin", "cate.bin")
            if (directory / name).is_file()
        }
        return cls.load(manifest, artifacts)
