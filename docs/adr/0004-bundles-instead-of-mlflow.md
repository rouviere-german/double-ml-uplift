# 4. Hash-verified bundles instead of MLflow

- Status: accepted
- Date: 2026-09-21

## Context

The project scope called for MLflow with a file backend, storing estimator bundles with the
specification hash, nuisance metrics and refutations.

What the service actually needs from a model store is narrow:

- serialise a fitted propensity model and a fitted CATE model;
- record the specification, configuration, dataset and fold hashes they came from, plus the
  adjustment set and the overlap bounds they must abstain outside;
- load them back and be certain they are the bytes that were written;
- never execute code on load.

What the service does *not* need: experiment tracking, a run UI, parameter search, a tracking
server, artifact stores, model staging transitions, or a registry of many models. The service
holds one specification, one configuration and the last successful estimation — that is stated as
a non-goal in the README.

MLflow brings those capabilities and their weight. It also brings a default serialisation path
built on pickle/cloudpickle, which collides directly with the repository-wide rule that no
pickle-family serialisation ships — a rule enforced by a unit test, a semgrep rule and
`scripts/check_baseline.py`. Using MLflow while banning pickle means using MLflow for almost
nothing except a directory layout and a JSON file.

## Decision

Ship `ScoringBundle` instead: a `BundleManifest` (pydantic, frozen) plus one opaque blob per
model, each verified against its SHA-256 on load.

- LightGBM models serialise as their own text dump and reload through
  `lightgbm.Booster(model_str=...)`.
- Linear models, with or without a leading standardiser, serialise as JSON coefficients and are
  evaluated in this repository by an explicit `(x - mean) / scale @ coef + intercept`, so loading
  one runs no third-party deserialiser at all.
- Anything else raises `UnsupportedModelError` rather than falling back to a generic pickler.
- The manifest carries the four hashes, the adjustment set, the gate policy, the calibration bins
  and a capped, deterministically down-sampled vector of training propensities for drift
  comparison — deliberately not the training data.
- `ScoringBundle.write`/`read` put this on disk; the retained `ml/registry.py` provides
  content-addressed storage with aliases for callers who want versioning.

The README records this as a deviation from the original scope rather than quietly dropping it.

## Consequences

**Good.** Loading a model cannot execute code. The threat "model artifact swapped or replaced with
an executable payload" (T2 in the threat model) reduces to "can the attacker also rewrite the
recorded hash", rather than "can the attacker write a file the loader will unpickle".

**Good.** The dependency footprint stays small enough that `uv sync` on a CPU-only laptop is quick
and `pip-audit` has a short surface. MLflow would have pulled in a web framework, a database
layer and a large transitive tree for features this service does not call.

**Good.** The manifest is a pydantic model, so the hashes are validated as hex digests, the
reference sample is length-capped, and a malformed manifest is refused at parse time.

**Bad.** No experiment UI. Comparing twenty estimation runs means reading twenty manifests or
writing the comparison yourself. For a service that keeps one result this is fine; for a research
workflow it would not be.

**Bad.** Only three model families can be serialised. Adding a fourth — say a random forest —
means writing its pickle-free representation by hand, or not using it. That is a real constraint
and it is the price of the guarantee above. `UnsupportedModelError` makes it a loud failure at
bundle time rather than a surprise at load time.

**Neutral.** If experiment tracking is wanted later, MLflow can log the manifest and the metrics
as parameters and artifacts without becoming the serialisation path. The two decisions are
separable, which is why this ADR is about serialisation rather than about tracking.
