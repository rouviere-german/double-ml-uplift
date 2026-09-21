"""Micro-benchmarks with CI regression thresholds, plus an end-to-end scaling measurement.

Usage:
  uv run python bench/bench_baseline.py            # run, write bench/results/latest.json
  uv run python bench/bench_baseline.py --check    # exit 1 if bench/thresholds.json is exceeded
  uv run python bench/bench_baseline.py --rows N   # scaling run at N rows (default 200000)

Thresholds are deliberately loose (about five times the numbers measured on a laptop) because
shared CI runners are noisy; they catch order-of-magnitude regressions, not small drifts. The
scaling
section is reported but not thresholded: it measures a whole 5-fold DML pass end to end, where
the machine matters far more than the code.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from double_ml_uplift.causal.bundle import ScoringBundle
from double_ml_uplift.causal.engine import EstimationConfig, UpliftEngine
from double_ml_uplift.causal.folds import assign_folds, make_fold_plan
from double_ml_uplift.causal.graph import CausalGraph
from double_ml_uplift.causal.identification import identify_backdoor
from double_ml_uplift.causal.nuisance import NuisanceConfig, cross_fit
from double_ml_uplift.causal.scenarios import load_scenario
from double_ml_uplift.causal.scm import simulate_campaign
from double_ml_uplift.causal.spec import bundled_spec
from double_ml_uplift.ml.drift import DriftMonitor
from double_ml_uplift.service.audit import AuditLog, verify_file

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / "latest.json"
THRESHOLDS = ROOT / "bench" / "thresholds.json"


def _timed(function: Callable[[], object], runs: int, warmup: int = 5) -> dict[str, float]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        function()
        samples.append(time.perf_counter_ns() - start)
    micros = np.asarray(samples, dtype=np.float64) / 1_000.0
    return {
        "runs": runs,
        "p50_us": round(float(np.percentile(micros, 50)), 1),
        "p95_us": round(float(np.percentile(micros, 95)), 1),
    }


def _random_dag(n_nodes: int, seed: int) -> CausalGraph:
    rng = np.random.default_rng(seed)
    names = [f"v{index:02d}" for index in range(n_nodes)]
    edges = [
        (names[i], names[j])
        for i in range(n_nodes)
        for j in range(i + 1, n_nodes)
        if rng.random() < 0.25
    ]
    return CausalGraph.from_edges(names, edges)


def bench_d_separation() -> dict[str, float]:
    graph = _random_dag(24, seed=7)
    conditioned = ["v05", "v11", "v17"]
    return _timed(lambda: graph.is_d_separated(["v01"], ["v23"], conditioned), runs=2_000)


def bench_identification() -> dict[str, float]:
    spec = bundled_spec("retention_offer")
    return _timed(lambda: identify_backdoor(spec), runs=200, warmup=2)


def bench_fold_assignment() -> dict[str, float]:
    unit_ids = [f"cust_{index:07d}" for index in range(100_000)]
    return _timed(lambda: assign_folds(unit_ids, 5, salt="bench"), runs=10, warmup=1)


def bench_cross_fit(features: Any, treatment: Any, outcome: Any, unit_ids: Any) -> dict[str, float]:
    plan = make_fold_plan(unit_ids, 5, salt="bench")
    config = NuisanceConfig(kind="gbm")
    return _timed(lambda: cross_fit(features, treatment, outcome, plan, config), runs=5, warmup=1)


def bench_scoring(bundle: ScoringBundle, features: Any) -> dict[str, float]:
    return _timed(lambda: bundle.predict(features), runs=20, warmup=2)


def bench_drift() -> dict[str, float]:
    rng = np.random.default_rng(0)
    monitor = DriftMonitor({"x": rng.normal(size=10_000)})
    current = {"x": rng.normal(loc=0.1, size=10_000)}
    return _timed(lambda: monitor.check(current), runs=30, warmup=2)


def bench_audit_append(workdir: Path) -> dict[str, float]:
    log = AuditLog(workdir / "append.jsonl")
    counter = iter(range(10**9))
    return _timed(
        lambda: log.append("bench", "step", {"index": next(counter), "note": "x" * 100}), runs=100
    )


def bench_audit_verify(workdir: Path) -> dict[str, float]:
    path = workdir / "verify.jsonl"
    log = AuditLog(path)
    for index in range(500):
        log.append("bench", "step", {"index": index, "note": "x" * 100})
    return _timed(lambda: verify_file(path), runs=10, warmup=1)


def scaling(rows: int) -> dict[str, Any]:
    """One full 5-fold DML pass at ``rows`` rows, with both nuisance families."""
    spec = bundled_spec("retention_offer")
    campaign = simulate_campaign(load_scenario("linear", n_customers=rows, seed=99))
    report: dict[str, Any] = {"rows": rows}
    for kind in ("ridge", "gbm"):
        config = EstimationConfig(
            nuisance=NuisanceConfig(kind=kind),
            cate_learner=NuisanceConfig(kind=kind, num_leaves=4, n_estimators=80),
            run_refuters=False,
            enforce_gate=False,
        )
        engine = UpliftEngine(spec, config)
        start = time.perf_counter()
        prepared = engine.prepare(campaign.frame)
        prepare_seconds = time.perf_counter() - start
        start = time.perf_counter()
        result = engine.fit_prepared(prepared)
        report[kind] = {
            "prepare_seconds": round(prepare_seconds, 3),
            "fit_seconds": round(time.perf_counter() - start, 3),
            "ate": round(result.estimates["dml"].ate, 4),
            "se": round(result.estimates["dml"].se, 4),
        }
    report["true_ate"] = round(campaign.true_ate, 4)
    return report


def machine() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
    }


def run(rows: int) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    spec = bundled_spec("retention_offer")
    campaign = simulate_campaign(load_scenario("linear", n_customers=10_000, seed=11))
    engine = UpliftEngine(spec, EstimationConfig(run_refuters=False, enforce_gate=False))
    prepared = engine.prepare(campaign.frame)
    result = engine.fit_prepared(prepared)
    manifest, artifacts = engine.bundle(result)
    bundle = ScoringBundle.load(manifest, artifacts)

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        results = {
            "d_separation_24_node_dag": bench_d_separation(),
            "identify_backdoor_retention_spec": bench_identification(),
            "fold_assignment_100k_ids": bench_fold_assignment(),
            "cross_fit_10k_rows_5_folds_gbm": bench_cross_fit(
                prepared.features, prepared.treatment, prepared.outcome, prepared.unit_ids
            ),
            "score_10k_rows": bench_scoring(bundle, prepared.features),
            "drift_check_10k_vs_10k": bench_drift(),
            "audit_append_fsync": bench_audit_append(workdir),
            "audit_verify_500_entries": bench_audit_verify(workdir),
        }
    return results, scaling(rows)


def check_thresholds(results: dict[str, dict[str, float]]) -> list[str]:
    thresholds: dict[str, dict[str, float]] = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    problems = []
    for name, limits in thresholds.items():
        if name.startswith("_"):
            continue  # "_comment" and friends document the file; they are not benchmarks
        if name not in results:
            problems.append(f"{name}: benchmark missing")
            continue
        for metric, limit in limits.items():
            value = results[name][metric]
            if value > limit:
                problems.append(f"{name}.{metric} = {value} exceeds threshold {limit}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="micro-benchmarks and a scaling measurement")
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    parser.add_argument("--rows", type=int, default=200_000, help="rows for the scaling section")
    args = parser.parse_args(argv)

    results, scale = run(args.rows)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps({"machine": machine(), "results": results, "scaling": scale}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("| benchmark | runs | p50 (us) | p95 (us) |")
    print("|---|---:|---:|---:|")
    for name, stats in results.items():
        print(f"| {name} | {stats['runs']:.0f} | {stats['p50_us']} | {stats['p95_us']} |")
    print()
    print(f"| scaling ({scale['rows']} rows, 5-fold DML) | prepare (s) | fit (s) | ATE | SE |")
    print("|---|---:|---:|---:|---:|")
    for kind in ("ridge", "gbm"):
        entry = scale[kind]
        print(
            f"| {kind} nuisances | {entry['prepare_seconds']} | {entry['fit_seconds']} "
            f"| {entry['ate']} | {entry['se']} |"
        )
    print(f"\ntrue ATE for the scaling sample: {scale['true_ate']}")

    if args.check:
        problems = check_thresholds(results)
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all benchmarks within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
