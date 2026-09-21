"""Scoring bundles are serialized without pickle and verified against their recorded hashes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from double_ml_uplift.causal.bundle import (
    MAX_REFERENCE_SAMPLE,
    ArtifactError,
    BundleManifest,
    ScoringBundle,
    UnsupportedModelError,
    _reference_sample,
    deserialize_model,
    serialize_model,
)
from double_ml_uplift.causal.calibration import calibrate
from double_ml_uplift.causal.diagnostics import GatePolicy
from double_ml_uplift.causal.nuisance import NuisanceConfig, make_classifier, make_regressor

pytestmark = pytest.mark.unit

DIGEST = "0" * 64


def training_data(n: int = 400, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 3))
    treatment = rng.binomial(1, 0.5, size=n).astype(np.int64)
    outcome = features[:, 0] * 2.0 + rng.normal(size=n)
    return features, treatment, outcome


# --- serialization, one format per learner family --------------------------------------------


@pytest.mark.parametrize(
    ("kind", "expected_format"),
    [
        ("ols", "linear-json/v1"),
        ("ridge", "scaled-linear-json/v1"),
        ("gbm", "lightgbm-text/v1"),
    ],
)
def test_a_fitted_regressor_round_trips_to_the_same_predictions(
    kind: str, expected_format: str
) -> None:
    features, _, outcome = training_data()
    config = NuisanceConfig(kind=kind, n_estimators=30)
    model = make_regressor(config)
    model.fit(features, outcome)
    meta, payload = serialize_model(model, "regression")
    assert meta.format == expected_format
    assert meta.task == "regression"
    assert not meta.logistic
    restored = deserialize_model(meta, payload)
    assert np.allclose(
        np.asarray(restored.predict(features)), np.asarray(model.predict(features)), atol=1e-8
    )


@pytest.mark.parametrize(
    ("kind", "expected_format", "logistic"),
    [
        ("ols", "linear-json/v1", False),
        ("ridge", "scaled-linear-json/v1", True),
        ("gbm", "lightgbm-text/v1", False),
    ],
)
def test_a_fitted_classifier_round_trips_to_the_same_probabilities(
    kind: str, expected_format: str, logistic: bool
) -> None:
    features, treatment, _ = training_data()
    config = NuisanceConfig(kind=kind, n_estimators=30)
    model = make_classifier(config)
    model.fit(features, treatment)
    meta, payload = serialize_model(model, "probability")
    assert meta.format == expected_format
    assert meta.logistic is logistic
    restored = deserialize_model(meta, payload)
    expected = np.asarray(model.predict_proba(features))[:, 1]
    assert np.allclose(np.asarray(restored.predict(features)), expected, atol=1e-8)


def test_the_payload_is_json_or_text_and_never_a_pickle() -> None:
    features, _, outcome = training_data()
    model = make_regressor(NuisanceConfig(kind="ols"))
    model.fit(features, outcome)
    _, payload = serialize_model(model, "regression")
    document = json.loads(payload.decode("utf-8"))
    assert set(document) == {"coef", "intercept"}
    assert not payload.startswith(b"\x80")  # the pickle protocol marker


def test_an_unknown_model_type_has_no_pickle_free_representation() -> None:
    class Exotic:
        pass

    with pytest.raises(UnsupportedModelError, match="no pickle-free representation"):
        serialize_model(Exotic(), "regression")


def test_a_pipeline_with_more_than_two_steps_is_refused() -> None:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features, _, outcome = training_data(100)
    pipeline = make_pipeline(StandardScaler(), StandardScaler(), Ridge())
    pipeline.fit(features, outcome)
    with pytest.raises(UnsupportedModelError, match="two-step"):
        serialize_model(pipeline, "regression")


# --- hash verification ---------------------------------------------------------------------


def test_a_tampered_payload_is_refused_rather_than_scored_with() -> None:
    features, _, outcome = training_data()
    model = make_regressor(NuisanceConfig(kind="ols"))
    model.fit(features, outcome)
    meta, payload = serialize_model(model, "regression")
    with pytest.raises(ArtifactError, match="does not match its recorded hash"):
        deserialize_model(meta, payload + b" ")


def test_a_payload_that_is_not_the_promised_format_is_refused() -> None:
    from double_ml_uplift.causal.bundle import SerializedModel
    from double_ml_uplift.core.canonical import sha256_hex

    broken = b"{not json"
    meta = SerializedModel(
        format="linear-json/v1", task="regression", sha256=sha256_hex(broken), n_features=2
    )
    with pytest.raises(ArtifactError, match="not valid JSON"):
        deserialize_model(meta, broken)

    wrong_shape = b'{"intercept": 1.0}'
    meta = SerializedModel(
        format="linear-json/v1", task="regression", sha256=sha256_hex(wrong_shape), n_features=2
    )
    with pytest.raises(ArtifactError, match="not a linear model document"):
        deserialize_model(meta, wrong_shape)


# --- building and loading a bundle -------------------------------------------------------------


def build_bundle(kind: str = "gbm") -> tuple[BundleManifest, dict[str, bytes], np.ndarray]:
    features, treatment, outcome = training_data(seed=1)
    config = NuisanceConfig(kind=kind, n_estimators=30)
    propensity_model = make_classifier(config)
    propensity_model.fit(features, treatment)
    cate_model = make_regressor(config)
    cate_model.fit(features, outcome)
    rng = np.random.default_rng(2)
    pseudo = rng.normal(size=400)
    calibration = calibrate(np.asarray(cate_model.predict(features)), pseudo, n_bins=5)
    manifest, artifacts = ScoringBundle.build(
        spec_sha256=DIGEST,
        config_sha256=DIGEST,
        dataset_sha256=DIGEST,
        fold_digest=DIGEST,
        adjustment_set=("a", "b", "c"),
        gate=GatePolicy(),
        propensity_model=propensity_model,
        cate_model=cate_model,
        calibration=calibration,
        reference_propensity=np.asarray(propensity_model.predict_proba(features))[:, 1],
        metrics={"ate": 1.5},
    )
    return manifest, artifacts, features


def test_a_bundle_predicts_uplift_propensity_and_abstentions() -> None:
    manifest, artifacts, features = build_bundle()
    bundle = ScoringBundle.load(manifest, artifacts)
    uplift, propensity, abstain = bundle.predict(features)
    assert uplift.shape == propensity.shape == abstain.shape == (400,)
    assert bundle.adjustment_set == ("a", "b", "c")
    assert bundle.calibration.n_bins == 5
    assert manifest.metrics == {"ate": 1.5}
    assert len(manifest.sha256()) == 64


def test_features_that_do_not_match_the_adjustment_set_are_refused() -> None:
    manifest, artifacts, features = build_bundle()
    bundle = ScoringBundle.load(manifest, artifacts)
    with pytest.raises(ValueError, match="do not match the bundle's adjustment set"):
        bundle.predict(features[:, :2])
    with pytest.raises(ValueError, match="do not match the bundle's adjustment set"):
        bundle.predict(features[:, 0])


def test_a_missing_artifact_is_named() -> None:
    manifest, artifacts, _ = build_bundle()
    with pytest.raises(ArtifactError, match=r"missing artifact\(s\): cate\.bin"):
        ScoringBundle.load(manifest, {"propensity.bin": artifacts["propensity.bin"]})


def test_a_bundle_round_trips_through_the_filesystem(tmp_path: Path) -> None:
    manifest, artifacts, features = build_bundle()
    ScoringBundle.write(tmp_path, manifest, artifacts)
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "propensity.bin").is_file()

    loaded = ScoringBundle.read(tmp_path)
    original = ScoringBundle.load(manifest, artifacts)
    assert loaded.manifest.sha256() == original.manifest.sha256()
    assert np.allclose(loaded.predict(features)[0], original.predict(features)[0])


def test_a_bundle_on_disk_with_a_tampered_artifact_is_refused(tmp_path: Path) -> None:
    manifest, artifacts, _ = build_bundle()
    ScoringBundle.write(tmp_path, manifest, artifacts)
    path = tmp_path / "cate.bin"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ArtifactError, match="does not match its recorded hash"):
        ScoringBundle.read(tmp_path)


def test_a_manifest_with_a_malformed_hash_is_refused() -> None:
    manifest, _, _ = build_bundle()
    document = json.loads(manifest.model_dump_json())
    document["spec_sha256"] = "not-a-digest"
    with pytest.raises(ValidationError, match="spec_sha256"):
        BundleManifest.model_validate(document)


# --- the reference propensity sample -------------------------------------------------------


def test_the_reference_sample_is_capped_and_deterministic() -> None:
    values = np.linspace(0.0, 1.0, MAX_REFERENCE_SAMPLE * 3)
    sample = _reference_sample(values)
    assert len(sample) == MAX_REFERENCE_SAMPLE
    assert sample == _reference_sample(values)
    assert min(sample) >= 0.0
    assert max(sample) <= 1.0


def test_a_short_reference_sample_is_kept_whole() -> None:
    values = np.array([0.1, 0.2, 0.3])
    assert _reference_sample(values) == (0.1, 0.2, 0.3)


def test_an_empty_reference_sample_is_refused() -> None:
    with pytest.raises(ValueError, match="reference propensity sample is required"):
        _reference_sample(np.array([]))


def test_the_bundle_keeps_a_reference_sample_for_drift_but_not_the_training_data() -> None:
    manifest, _, _ = build_bundle()
    assert 0 < len(manifest.reference_propensity) <= MAX_REFERENCE_SAMPLE
    serialized = manifest.model_dump_json()
    assert "customer" not in serialized
    assert "retained_revenue" not in serialized
