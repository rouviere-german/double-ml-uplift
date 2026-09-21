"""The estimation pipeline: contract, identification, cross-fitting, gate, estimate, refute.

The order of the steps is the point of the module. Every gate runs *before* a number exists, so
there is never a point estimate lying around that someone could quote after the refusal:

1. the data contract rejects oracle columns, oversized frames and out-of-range values;
2. identification derives a minimal backdoor adjustment set from the declared DAG, or refuses;
3. the leakage guard re-checks that set (belt and braces: identification never proposes a
   post-treatment variable, and the guard would reject it if it did);
4. folds are assigned by a keyed hash of the customer id, and the nuisances are cross-fitted;
5. overlap and balance diagnostics run, and a failure raises instead of returning an estimate;
6. only then are the five ATE estimators and the DR-learner evaluated;
7. the refuters run against the primary estimate and are stored with it.

``EstimationConfig.enforce_gate`` can disable step 5's refusal. It exists so the Monte Carlo
bench can measure what the gate is protecting against (bias and coverage under a positivity
violation); production code leaves it on, and the service never exposes it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from double_ml_uplift.causal.bundle import BundleManifest, ScoringBundle
from double_ml_uplift.causal.calibration import CateCalibration, calibrate
from double_ml_uplift.causal.diagnostics import (
    GateDecision,
    GatePolicy,
    abstain_mask,
    apply_gate,
    overlap_report,
)
from double_ml_uplift.causal.estimators import (
    AteEstimate,
    CateFit,
    aipw,
    dml_partialling_out,
    dr_learner,
    estimate_all,
)
from double_ml_uplift.causal.folds import MAX_FOLDS, FoldPlan, make_fold_plan
from double_ml_uplift.causal.identification import (
    Identification,
    identify_backdoor,
    leakage_guard,
)
from double_ml_uplift.causal.nuisance import (
    Classifier,
    CrossFitted,
    NuisanceConfig,
    cross_fit,
    make_classifier,
)
from double_ml_uplift.causal.refutation import (
    RefutationPolicy,
    RefutationReport,
    refute_all,
)
from double_ml_uplift.causal.spec import CausalSpec
from double_ml_uplift.causal.targeting import (
    FairnessCheck,
    PolicyValue,
    TargetingPlan,
    build_plan,
    fairness_check,
    pilot_policy_value,
)
from double_ml_uplift.core.canonical import canonical_json, sha256_hex
from double_ml_uplift.ml.contracts import ContractLimits, validate_campaign_frame

__all__ = [
    "EstimationConfig",
    "EstimationResult",
    "PreparedData",
    "TargetingOutcome",
    "UpliftEngine",
    "dataset_digest",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


class EstimationConfig(BaseModel):
    """Everything that changes a number, in one hashable object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_folds: int = Field(default=5, ge=2, le=MAX_FOLDS)
    fold_salt: str = Field(default="retention-offer", max_length=64)
    nuisance: NuisanceConfig = NuisanceConfig()
    # The second stage of the DR-learner regresses noisy pseudo-outcomes, so it is deliberately
    # more regularised than the nuisance learners: shallower trees and a large minimum leaf size.
    # bench/bench_montecarlo.py measures the PEHE this costs and buys.
    cate_learner: NuisanceConfig = NuisanceConfig(
        kind="gbm", num_leaves=4, n_estimators=80, learning_rate=0.05, min_child_samples=150
    )
    gate: GatePolicy = GatePolicy()
    refutation: RefutationPolicy = RefutationPolicy()
    propensity_clip: float = Field(default=0.01, gt=0.0, lt=0.5)
    level: float = Field(default=0.95, gt=0.5, lt=1.0)
    n_bootstrap: int = Field(default=0, ge=0, le=5_000)
    primary: Literal["aipw", "dml"] = "aipw"
    run_refuters: bool = True
    enforce_gate: bool = True
    seed: int = Field(default=0, ge=0, le=2**31 - 1)

    def sha256(self) -> str:
        return sha256_hex(canonical_json(self.model_dump(mode="json")))


def dataset_digest(
    unit_ids: Sequence[str], features: FloatArray, treatment: IntArray, outcome: FloatArray
) -> str:
    """Content hash of the estimation inputs, recorded in the run manifest."""
    hasher = hashlib.sha256()
    hasher.update(b"double-ml-uplift/dataset/v1\n")
    hasher.update("\x00".join(unit_ids).encode("utf-8"))
    for array in (features, treatment.astype(np.int64), outcome):
        hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedData:
    """A contract-checked frame reduced to the arrays the estimators need."""

    unit_ids: tuple[str, ...]
    features: FloatArray
    feature_names: tuple[str, ...]
    treatment: IntArray
    outcome: FloatArray
    pilot: IntArray | None
    protected: IntArray | None
    digest: str

    @property
    def n(self) -> int:
        return int(self.treatment.size)


@dataclass(frozen=True, slots=True)
class TargetingOutcome:
    """Competing targeting policies and what the randomized pilot says each one is worth."""

    budget: float
    cost_per_offer: float
    plans: dict[str, TargetingPlan]
    pilot_values: dict[str, PolicyValue]
    fairness: FairnessCheck
    abstained: int

    def to_dict(self) -> dict[str, object]:
        return {
            "budget": round(self.budget, 4),
            "cost_per_offer": self.cost_per_offer,
            "abstained": self.abstained,
            "plans": {name: plan.to_dict() for name, plan in self.plans.items()},
            "pilot_values": {name: value.to_dict() for name, value in self.pilot_values.items()},
            "fairness": self.fairness.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class EstimationResult:
    """Everything one estimation run produced, including the evidence for its own validity."""

    spec_sha256: str
    config_sha256: str
    dataset_sha256: str
    fold_digest: str
    identification: Identification
    adjustment_set: tuple[str, ...]
    diagnostics: GateDecision
    estimates: dict[str, AteEstimate]
    cate: CateFit = field(repr=False)
    calibration: CateCalibration = field(repr=False)
    refutation: RefutationReport | None
    prepared: PreparedData = field(repr=False)
    fitted: CrossFitted = field(repr=False)
    propensity_model: Classifier = field(repr=False, compare=False)
    primary_name: str

    @property
    def primary(self) -> AteEstimate:
        return self.estimates[self.primary_name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_sha256": self.spec_sha256,
            "config_sha256": self.config_sha256,
            "dataset_sha256": self.dataset_sha256,
            "fold_digest": self.fold_digest,
            "adjustment_set": list(self.adjustment_set),
            "estimand": self.identification.estimand,
            "backdoor_paths": self.identification.backdoor_paths,
            "diagnostics": self.diagnostics.report.to_dict(),
            "estimable": self.diagnostics.estimable,
            "refusal_reasons": list(self.diagnostics.reasons),
            "primary": self.primary_name,
            "estimates": {
                name: estimate.to_dict() for name, estimate in sorted(self.estimates.items())
            },
            "refutation": self.refutation.to_dict() if self.refutation else None,
            "calibration": self.calibration.to_dict(),
        }


class UpliftEngine:
    """Runs the pipeline for one causal specification and one configuration."""

    def __init__(
        self,
        spec: CausalSpec,
        config: EstimationConfig | None = None,
        *,
        limits: ContractLimits | None = None,
    ) -> None:
        self.spec = spec
        self.config = config or EstimationConfig()
        self.limits = limits or ContractLimits()
        self.identification = identify_backdoor(spec)

    # --- preparation ---------------------------------------------------------------------

    def adjustment_set(self) -> tuple[str, ...]:
        """The minimal backdoor adjustment set, or fail closed with the refusal reasons."""
        names = self.identification.require()
        return leakage_guard(self.spec, names)

    def prepare(self, frame: pd.DataFrame) -> PreparedData:
        """Validate the frame and pull out the arrays the estimators need."""
        validated = validate_campaign_frame(frame, self.spec, self.limits)
        names = self.adjustment_set()
        if not names:
            raise ValueError("the backdoor adjustment set is empty; nothing to adjust for")
        features = np.column_stack(
            [validated[name].to_numpy(dtype=np.float64) for name in names]
        ).astype(np.float64, copy=False)
        unit_ids = tuple(str(value) for value in validated[self.spec.unit_id].to_numpy())
        treatment = validated[self.spec.treatment].to_numpy(dtype=np.int64)
        outcome = validated[self.spec.outcome].to_numpy(dtype=np.float64)
        pilot = (
            validated[self.spec.pilot_column].to_numpy(dtype=np.int64)
            if self.spec.pilot_column
            else None
        )
        protected = (
            validated[self.spec.protected_attribute].to_numpy(dtype=np.int64)
            if self.spec.protected_attribute
            else None
        )
        return PreparedData(
            unit_ids=unit_ids,
            features=features,
            feature_names=names,
            treatment=treatment,
            outcome=outcome,
            pilot=pilot,
            protected=protected,
            digest=dataset_digest(unit_ids, features, treatment, outcome),
        )

    # --- estimation ----------------------------------------------------------------------

    def _cross_fit(
        self,
        features: FloatArray,
        treatment: IntArray,
        outcome: FloatArray,
        unit_ids: Sequence[str],
    ) -> tuple[CrossFitted, FoldPlan]:
        plan = make_fold_plan(unit_ids, self.config.n_folds, salt=self.config.fold_salt)
        return cross_fit(features, treatment, outcome, plan, self.config.nuisance), plan

    def _primary_estimate(
        self,
        features: FloatArray,
        treatment: IntArray,
        outcome: FloatArray,
        unit_ids: Sequence[str],
    ) -> AteEstimate:
        """Refit the primary estimator on a perturbed sample. Used by the refuters."""
        fitted, _ = self._cross_fit(features, treatment, outcome, unit_ids)
        if self.config.primary == "dml":
            return dml_partialling_out(treatment, outcome, fitted, level=self.config.level)
        return aipw(
            treatment, outcome, fitted, clip=self.config.propensity_clip, level=self.config.level
        )

    def fit(self, frame: pd.DataFrame) -> EstimationResult:
        """Run the whole pipeline. Raises before producing a number when a gate fails."""
        prepared = self.prepare(frame)
        return self.fit_prepared(prepared)

    def fit_prepared(self, prepared: PreparedData) -> EstimationResult:
        """Same as :meth:`fit` for data that has already passed the contract."""
        fitted, plan = self._cross_fit(
            prepared.features, prepared.treatment, prepared.outcome, prepared.unit_ids
        )
        report = overlap_report(
            prepared.features,
            prepared.feature_names,
            prepared.treatment,
            fitted.propensity,
            self.config.gate,
        )
        decision = apply_gate(report, self.config.gate)
        if self.config.enforce_gate:
            decision.require()

        estimates = estimate_all(
            prepared.treatment,
            prepared.outcome,
            fitted,
            clip=self.config.propensity_clip,
            level=self.config.level,
            n_bootstrap=self.config.n_bootstrap,
            seed=self.config.seed,
        )
        cate = dr_learner(
            prepared.features,
            prepared.treatment,
            prepared.outcome,
            fitted,
            self.config.cate_learner,
            clip=self.config.propensity_clip,
        )
        calibration = calibrate(
            cate.out_of_fold, cate.pseudo_outcome, n_bins=min(10, max(2, prepared.n // 40))
        )
        refutation = None
        if self.config.run_refuters:
            refutation = refute_all(
                self._primary_estimate,
                prepared.features,
                prepared.treatment,
                prepared.outcome,
                prepared.unit_ids,
                estimates[self.config.primary],
                self.config.refutation,
                pilot=prepared.pilot,
                seed=self.config.seed,
            )
        propensity_model = make_classifier(self.config.nuisance)
        propensity_model.fit(prepared.features, prepared.treatment)
        return EstimationResult(
            spec_sha256=self.spec.sha256(),
            config_sha256=self.config.sha256(),
            dataset_sha256=prepared.digest,
            fold_digest=plan.digest,
            identification=self.identification,
            adjustment_set=prepared.feature_names,
            diagnostics=decision,
            estimates=estimates,
            cate=cate,
            calibration=calibration,
            refutation=refutation,
            prepared=prepared,
            fitted=fitted,
            propensity_model=propensity_model,
            primary_name=self.config.primary,
        )

    # --- targeting -----------------------------------------------------------------------

    def target(
        self,
        result: EstimationResult,
        *,
        budget_fraction: float = 0.2,
        cost_per_offer: float = 1.0,
        fairness_threshold: float = 0.15,
    ) -> TargetingOutcome:
        """Build the uplift, propensity and treat-all policies and value them on the pilot."""
        if not 0.0 < budget_fraction <= 1.0:
            raise ValueError("budget_fraction must lie in (0, 1]")
        if cost_per_offer <= 0.0:
            raise ValueError("cost_per_offer must be positive")
        prepared = result.prepared
        uplift = result.cate.out_of_fold
        cost = np.full(prepared.n, cost_per_offer, dtype=np.float64)
        eligible = ~abstain_mask(result.fitted.propensity, self.config.gate)
        budget = budget_fraction * prepared.n * cost_per_offer

        plans = {
            "uplift": build_plan(uplift, cost, budget, ranking="uplift", eligible=eligible),
            "propensity": build_plan(
                result.fitted.propensity,
                cost,
                budget,
                ranking="propensity",
                uplift=uplift,
                eligible=eligible,
            ),
            "treat_all": build_plan(
                np.ones(prepared.n, dtype=np.float64),
                cost,
                float(prepared.n) * cost_per_offer,
                ranking="treat_all",
                uplift=uplift,
            ),
        }
        pilot_values: dict[str, PolicyValue] = {}
        if prepared.pilot is not None:
            for name, plan in plans.items():
                try:
                    pilot_values[name] = pilot_policy_value(
                        plan.selected,
                        prepared.pilot,
                        prepared.treatment,
                        prepared.outcome,
                        level=self.config.level,
                    )
                except ValueError:
                    # The randomized slice does not overlap this policy's selection enough to
                    # value it. Leaving the entry out is the honest answer: callers see a missing
                    # value rather than one computed from a handful of rows. A tight budget is
                    # the usual cause, and the caller still gets the plan itself.
                    continue
        protected = prepared.protected
        check = (
            fairness_check(plans["uplift"].selected, protected, threshold=fairness_threshold)
            if protected is not None
            else FairnessCheck(0.0, fairness_threshold, False, 0.0, 0.0)
        )
        return TargetingOutcome(
            budget=budget,
            cost_per_offer=cost_per_offer,
            plans=plans,
            pilot_values=pilot_values,
            fairness=check,
            abstained=int((~eligible).sum()),
        )

    # --- persistence ---------------------------------------------------------------------

    def bundle(
        self, result: EstimationResult, *, metrics: dict[str, float] | None = None
    ) -> tuple[BundleManifest, dict[str, bytes]]:
        """Serialize the fitted scoring models into a pickle-free, hash-verified bundle."""
        return ScoringBundle.build(
            spec_sha256=result.spec_sha256,
            config_sha256=result.config_sha256,
            dataset_sha256=result.dataset_sha256,
            fold_digest=result.fold_digest,
            adjustment_set=result.adjustment_set,
            gate=self.config.gate,
            propensity_model=result.propensity_model,
            cate_model=result.cate.model,
            calibration=result.calibration,
            reference_propensity=result.fitted.propensity,
            metrics=metrics,
        )

    # --- scoring -------------------------------------------------------------------------

    def score_features(
        self, result: EstimationResult, features: FloatArray
    ) -> tuple[FloatArray, FloatArray, BoolArray]:
        """Return ``(uplift, propensity, abstain)`` for new customers.

        Customers whose estimated propensity falls outside the overlap region get an abstention
        flag: the model was never trained on comparable customers, so any number would be
        extrapolation.
        """
        if features.ndim != 2 or features.shape[1] != len(result.adjustment_set):
            raise ValueError("features do not match the adjustment set used for fitting")
        proba = np.asarray(result.propensity_model.predict_proba(features), dtype=np.float64)
        propensity = np.asarray(proba[:, 1], dtype=np.float64)
        uplift = result.cate.predict(features)
        return uplift, propensity, abstain_mask(propensity, self.config.gate)
