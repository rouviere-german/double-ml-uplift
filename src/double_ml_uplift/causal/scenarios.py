"""Bundled simulation scenarios.

Each YAML file under ``configs/scenarios`` is a serialized :class:`~.scm.ScmConfig`. Keeping them
as data rather than as Python constants means the Monte Carlo bench, the demo, the tests and the
model card all draw from the same definitions, and a scenario's parameter hash changes visibly
when the file changes.
"""

from __future__ import annotations

from importlib import resources
from typing import Final

import yaml

from double_ml_uplift.causal.scm import ScmConfig

__all__ = ["MAX_SCENARIO_BYTES", "load_scenario", "scenario_names"]

_PACKAGE: Final = "double_ml_uplift.configs.scenarios"
MAX_SCENARIO_BYTES: Final = 16 * 1024


def scenario_names() -> tuple[str, ...]:
    """Names of the bundled scenarios, sorted."""
    return tuple(
        sorted(
            item.name.removesuffix(".yaml")
            for item in resources.files(_PACKAGE).iterdir()
            if item.name.endswith(".yaml")
        )
    )


def load_scenario(
    name: str, *, n_customers: int | None = None, seed: int | None = None
) -> ScmConfig:
    """Load a bundled scenario, optionally overriding the sample size and the seed."""
    if name not in scenario_names():
        raise ValueError(f"unknown scenario {name!r}; available: {', '.join(scenario_names())}")
    text = resources.files(_PACKAGE).joinpath(f"{name}.yaml").read_text(encoding="utf-8")
    if len(text.encode("utf-8")) > MAX_SCENARIO_BYTES:
        raise ValueError(f"scenario {name!r} exceeds {MAX_SCENARIO_BYTES} bytes")
    document = yaml.safe_load(text)
    if not isinstance(document, dict):
        raise ValueError(f"scenario {name!r} must be a YAML mapping")
    if n_customers is not None:
        document["n_customers"] = n_customers
    if seed is not None:
        document["seed"] = seed
    return ScmConfig.model_validate(document)
