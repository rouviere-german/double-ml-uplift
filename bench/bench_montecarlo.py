"""Monte Carlo evaluation against known ground truth, with statistical tolerance bands.

Usage:
  uv run python bench/bench_montecarlo.py                    # 50 replications (the CI size)
  uv run python bench/bench_montecarlo.py --check            # also enforce bench/mc_thresholds.json
  uv run python bench/bench_montecarlo.py --replications 200 # the full report

Four studies, all on data drawn from the bundled structural causal models, so every number is
compared against a truth the generator knows and the estimators never see:

``ate_accuracy``
    Bias, RMSE and 95% interval coverage per estimator per scenario, plus how often the overlap
    and balance gate would have refused. The gate is *disabled* here on purpose: the point is to
    measure what it protects against.
``cate_quality``
    PEHE of the DR-learner against the two baselines a team would otherwise ship (subtracting two
    arm regressions, or predicting the average effect for everyone), and the calibration slope.
``refuters``
    False-flag rate of each refuter on a well-specified scenario, and the detection rate of the
    randomized-pilot consistency check on a scenario with a strong unmeasured confounder.
``targeting``
    At a 20% budget: how often the pilot-estimated policy value covers the true policy value, and
    how often uplift ranking beats propensity ranking in *true* value.

Coverage is itself estimated from finitely many draws, so ``--check`` widens every target band by
a Monte Carlo allowance of ``2.5 * sqrt(0.95 * 0.05 / replications)``. With 50 replications that
is about 7.7 points; the band tightens as the replication count rises.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from double_ml_uplift.causal.engine import EstimationConfig, EstimationResult, UpliftEngine
from double_ml_uplift.causal.estimators import ATE_ESTIMATORS, pehe
from double_ml_uplift.causal.scenarios import load_scenario
from double_ml_uplift.causal.scm import ScmConfig, SimulatedCampaign, simulate_campaign
from double_ml_uplift.causal.spec import bundled_spec
from double_ml_uplift.causal.targeting import true_policy_value

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / "montecarlo.json"
THRESHOLDS = ROOT / "bench" / "mc_thresholds.json"
ACCURACY_SCENARIOS = ("linear", "nonlinear", "heterogeneous", "weak_overlap")
BASE_SEED = 1_000_000


def _draw(scenario: str, replication: int, rows: int) -> SimulatedCampaign:
    config: ScmConfig = load_scenario(
        scenario, n_customers=rows, seed=BASE_SEED + 7_919 * replication
    )
    return simulate_campaign(config)


def _fit(campaign: SimulatedCampaign, config: EstimationConfig) -> EstimationResult:
    engine = UpliftEngine(bundled_spec("retention_offer"), config)
    return engine.fit(campaign.frame)


@dataclass
class Accumulator:
    errors: list[float]
    covered: list[bool]
    standard_errors: list[float]

    def summary(self) -> dict[str, float]:
        errors = np.asarray(self.errors, dtype=np.float64)
        return {
            "bias": round(float(errors.mean()), 4),
            "rmse": round(float(np.sqrt(np.mean(errors**2))), 4),
            "coverage": round(float(np.mean(self.covered)), 4),
            # Reported next to the empirical spread: a mean standard error far above the spread
            # means the interval is conservative, far below means it under-covers.
            "mean_se": round(float(np.mean(self.standard_errors)), 4),
            "empirical_sd": round(float(errors.std(ddof=1)), 4) if len(errors) > 1 else 0.0,
            "draws": len(self.errors),
        }


def study_ate_accuracy(replications: int, rows: int) -> dict[str, Any]:
    """Bias, RMSE and coverage per estimator, with the gate measured but not enforced."""
    config = EstimationConfig(run_refuters=False, enforce_gate=False)
    out: dict[str, Any] = {}
    for scenario in ACCURACY_SCENARIOS:
        accumulators = {name: Accumulator([], [], []) for name in ATE_ESTIMATORS}
        refusals = 0
        for replication in range(replications):
            campaign = _draw(scenario, replication, rows)
            result = _fit(campaign, config)
            truth = campaign.true_ate
            refusals += int(not result.diagnostics.estimable)
            for name, estimate in result.estimates.items():
                accumulators[name].errors.append(estimate.ate - truth)
                accumulators[name].covered.append(estimate.covers(truth))
                accumulators[name].standard_errors.append(estimate.se)
        out[scenario] = {
            "estimators": {name: item.summary() for name, item in accumulators.items()},
            "gate_refusal_rate": round(refusals / replications, 4),
        }
    return out


def study_cate_quality(replications: int, rows: int) -> dict[str, Any]:
    """PEHE of the DR-learner against the plug-in and constant-effect baselines."""
    config = EstimationConfig(run_refuters=False, enforce_gate=False)
    out: dict[str, Any] = {}
    for scenario in ("heterogeneous", "nonlinear"):
        rows_dr: list[float] = []
        rows_plugin: list[float] = []
        rows_constant: list[float] = []
        slopes: list[float] = []
        for replication in range(replications):
            campaign = _draw(scenario, replication, rows)
            result = _fit(campaign, config)
            truth = campaign.true_cate
            rows_dr.append(pehe(result.cate.out_of_fold, truth))
            rows_plugin.append(pehe(result.fitted.mu1 - result.fitted.mu0, truth))
            rows_constant.append(pehe(np.full(truth.size, result.primary.ate), truth))
            slopes.append(result.calibration.slope())
        out[scenario] = {
            "pehe_dr_learner": round(float(np.mean(rows_dr)), 4),
            "pehe_outcome_regression": round(float(np.mean(rows_plugin)), 4),
            "pehe_constant_ate": round(float(np.mean(rows_constant)), 4),
            "calibration_slope": round(float(np.mean(slopes)), 4),
            "draws": replications,
        }
    return out


def study_refuters(replications: int, detection_replications: int, rows: int) -> dict[str, Any]:
    """False-flag rates on a well-specified scenario; detection under unmeasured confounding."""
    config = EstimationConfig(run_refuters=True, enforce_gate=False)
    flags: dict[str, int] = {}
    for replication in range(replications):
        campaign = _draw("linear", replication, rows)
        result = _fit(campaign, config)
        assert result.refutation is not None
        for item in result.refutation.results:
            flags[item.name] = flags.get(item.name, 0) + int(item.failed)

    detected = 0
    biases: list[float] = []
    for replication in range(detection_replications):
        campaign = _draw("hidden_confounder", replication, rows)
        result = _fit(campaign, config)
        assert result.refutation is not None
        pilot = next(item for item in result.refutation.results if item.name == "pilot_consistency")
        detected += int(pilot.failed)
        biases.append(result.primary.ate - campaign.true_ate)
    return {
        "false_flag_rate": {
            name: round(count / replications, 4) for name, count in sorted(flags.items())
        },
        "false_flag_draws": replications,
        "hidden_confounder_detection_rate": round(detected / detection_replications, 4),
        "hidden_confounder_mean_bias": round(float(np.mean(biases)), 4),
        "detection_draws": detection_replications,
    }


def study_targeting(replications: int, rows: int, budget_fraction: float) -> dict[str, Any]:
    """Does the pilot value the policy honestly, and does uplift ranking beat propensity?"""
    config = EstimationConfig(run_refuters=False, enforce_gate=False)
    engine = UpliftEngine(bundled_spec("retention_offer"), config)
    covered = 0
    beats = 0
    uplift_values: list[float] = []
    propensity_values: list[float] = []
    treat_all_values: list[float] = []
    for replication in range(replications):
        campaign = _draw("heterogeneous", replication, rows)
        result = engine.fit(campaign.frame)
        outcome = engine.target(result, budget_fraction=budget_fraction, cost_per_offer=1.0)
        y0, y1 = campaign.potential_outcomes
        truths = {
            name: true_policy_value(plan.selected, y0, y1) for name, plan in outcome.plans.items()
        }
        covered += int(outcome.pilot_values["uplift"].covers(truths["uplift"]))
        beats += int(truths["uplift"] > truths["propensity"])
        uplift_values.append(truths["uplift"])
        propensity_values.append(truths["propensity"])
        treat_all_values.append(truths["treat_all"])
    return {
        "budget_fraction": budget_fraction,
        "draws": replications,
        "pilot_ci_coverage": round(covered / replications, 4),
        "uplift_beats_propensity_rate": round(beats / replications, 4),
        "mean_true_value_uplift": round(float(np.mean(uplift_values)), 4),
        "mean_true_value_propensity": round(float(np.mean(propensity_values)), 4),
        "mean_true_value_treat_all": round(float(np.mean(treat_all_values)), 4),
    }


def allowance(replications: int) -> float:
    """Monte Carlo slack added to every target band, from the binomial standard error."""
    return 2.5 * math.sqrt(0.95 * 0.05 / max(replications, 1))


def check_thresholds(report: dict[str, Any], replications: int) -> list[str]:  # noqa: PLR0912
    thresholds = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    slack = allowance(replications)
    problems: list[str] = []

    for scenario, bands in thresholds["coverage"].items():
        entry = report["ate_accuracy"][scenario]["estimators"]
        for estimator, (low, high) in bands.items():
            value = entry[estimator]["coverage"]
            if not (low - slack) <= value <= min(high + slack, 1.0):
                problems.append(
                    f"coverage[{scenario}][{estimator}] = {value} outside "
                    f"[{low - slack:.3f}, {min(high + slack, 1.0):.3f}]"
                )
    for scenario, limit in thresholds["max_abs_bias"].items():
        for estimator, bound in limit.items():
            value = abs(report["ate_accuracy"][scenario]["estimators"][estimator]["bias"])
            if value > bound:
                problems.append(f"|bias|[{scenario}][{estimator}] = {value} exceeds {bound}")
    for scenario, bound in thresholds["min_abs_bias"].items():
        for estimator, limit in bound.items():
            value = abs(report["ate_accuracy"][scenario]["estimators"][estimator]["bias"])
            if value < limit:
                problems.append(
                    f"|bias|[{scenario}][{estimator}] = {value} is below {limit}; the naive "
                    "estimator is supposed to fail visibly under confounding"
                )
    for scenario, bound in thresholds["gate_refusal_rate"].items():
        value = report["ate_accuracy"][scenario]["gate_refusal_rate"]
        if "min" in bound and value < bound["min"] - slack:
            problems.append(f"gate_refusal_rate[{scenario}] = {value} below {bound['min']}")
        if "max" in bound and value > bound["max"] + slack:
            problems.append(f"gate_refusal_rate[{scenario}] = {value} above {bound['max']}")
    for scenario, bound in thresholds["max_pehe"].items():
        value = report["cate_quality"][scenario]["pehe_dr_learner"]
        baseline = report["cate_quality"][scenario]["pehe_constant_ate"]
        if value > bound * baseline:
            problems.append(
                f"pehe_dr_learner[{scenario}] = {value} exceeds {bound} x the constant-effect "
                f"baseline ({baseline})"
            )
    for name, bound in thresholds["max_false_flag_rate"].items():
        value = report["refuters"]["false_flag_rate"].get(name, 0.0)
        if value > bound + slack:
            problems.append(f"false_flag_rate[{name}] = {value} exceeds {bound}")
    detection = report["refuters"]["hidden_confounder_detection_rate"]
    if detection < thresholds["min_hidden_confounder_detection"] - slack:
        problems.append(
            f"hidden_confounder_detection_rate = {detection} below "
            f"{thresholds['min_hidden_confounder_detection']}"
        )
    for key, bound in thresholds["targeting"].items():
        value = report["targeting"][key]
        if value < bound - slack:
            problems.append(f"targeting[{key}] = {value} below {bound}")
    return problems


def machine() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
    }


def _print_accuracy(report: dict[str, Any]) -> None:
    print("| scenario | estimator | bias | RMSE | empirical SD | mean SE | coverage |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for scenario, entry in report.items():
        for estimator in ATE_ESTIMATORS:
            stats = entry["estimators"][estimator]
            print(
                f"| {scenario} | {estimator} | {stats['bias']:+.3f} | {stats['rmse']:.3f} "
                f"| {stats['empirical_sd']:.3f} | {stats['mean_se']:.3f} "
                f"| {stats['coverage']:.3f} |"
            )
    print()
    print("| scenario | gate refusal rate |")
    print("|---|---:|")
    for scenario, entry in report.items():
        print(f"| {scenario} | {entry['gate_refusal_rate']:.3f} |")


def _print_rest(report: dict[str, Any]) -> None:
    print()
    print("| scenario | PEHE DR-learner | PEHE outcome regression | PEHE constant ATE | slope |")
    print("|---|---:|---:|---:|---:|")
    for scenario, entry in report["cate_quality"].items():
        print(
            f"| {scenario} | {entry['pehe_dr_learner']:.2f} "
            f"| {entry['pehe_outcome_regression']:.2f} | {entry['pehe_constant_ate']:.2f} "
            f"| {entry['calibration_slope']:.3f} |"
        )
    print()
    print("| refuter | false-flag rate (well-specified) |")
    print("|---|---:|")
    for name, value in report["refuters"]["false_flag_rate"].items():
        print(f"| {name} | {value:.3f} |")
    print(
        f"\nunmeasured-confounder detection rate: "
        f"{report['refuters']['hidden_confounder_detection_rate']:.3f} "
        f"(mean bias {report['refuters']['hidden_confounder_mean_bias']:+.1f})"
    )
    targeting = report["targeting"]
    print()
    print("| targeting at a 20% budget | value |")
    print("|---|---:|")
    print(f"| pilot CI covers the true policy value | {targeting['pilot_ci_coverage']:.3f} |")
    beats = targeting["uplift_beats_propensity_rate"]
    print(f"| uplift ranking beats propensity ranking | {beats:.3f} |")
    print(f"| mean true value, uplift ranking | {targeting['mean_true_value_uplift']:.3f} |")
    propensity_value = targeting["mean_true_value_propensity"]
    print(f"| mean true value, propensity ranking | {propensity_value:.3f} |")
    print(f"| mean true value, treat everyone | {targeting['mean_true_value_treat_all']:.3f} |")


def run(replications: int, rows: int) -> dict[str, Any]:
    secondary = max(replications // 5, 4)
    detection = max(replications // 2, 8)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "replications": replications,
        "rows": rows,
        "ate_accuracy": study_ate_accuracy(replications, rows),
        "cate_quality": study_cate_quality(secondary, rows * 4),
        "refuters": study_refuters(secondary, detection, rows),
        "targeting": study_targeting(secondary, rows * 2, 0.2),
    }
    report["seconds"] = round(time.perf_counter() - started, 1)
    report["machine"] = machine()
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Monte Carlo evaluation against ground truth")
    parser.add_argument("--replications", type=int, default=50)
    parser.add_argument("--rows", type=int, default=2_000)
    parser.add_argument("--check", action="store_true", help="enforce bench/mc_thresholds.json")
    args = parser.parse_args(argv)
    if args.replications < 4:
        parser.error("at least 4 replications are required")

    report = run(args.replications, args.rows)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        f"Monte Carlo: {args.replications} replications, n = {args.rows}, "
        f"{report['seconds']} s on {report['machine']['cpu_count']} CPUs\n"
    )
    _print_accuracy(report["ate_accuracy"])
    _print_rest(report)

    if args.check:
        problems = check_thresholds(report, args.replications)
        print()
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print(
            f"all Monte Carlo metrics within their tolerance bands "
            f"(allowance {allowance(args.replications):.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
