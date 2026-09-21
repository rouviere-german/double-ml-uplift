"""The declared causal specification: treatment, outcome, DAG and forbidden columns.

A specification is data, not code: it is parsed with ``yaml.safe_load`` from a size-limited
document, validated by pydantic and then checked for structural sense (known endpoints, acyclic,
a directed path from treatment to outcome, a binary treatment). Everything downstream -- the data
contract, identification, the leakage guard and the reproducibility manifest -- is derived from
it, and its SHA-256 is recorded with every estimate so a stored bundle cannot be reused under a
different specification.

The DAG says what the analyst *believes*. This module never tries to discover it; see
MODEL_CARD.md for what that assumption buys and what it costs.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any, Final, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from double_ml_uplift.causal.graph import CausalGraph, GraphError
from double_ml_uplift.core.canonical import canonical_json, sha256_hex

__all__ = [
    "MAX_EDGES",
    "MAX_NODES",
    "MAX_SPEC_BYTES",
    "CausalSpec",
    "NodeSpec",
    "SpecError",
    "bundled_spec",
    "bundled_spec_names",
    "load_spec_file",
    "load_spec_text",
]

MAX_SPEC_BYTES: Final = 64 * 1024
MAX_NODES: Final = 200
MAX_EDGES: Final = 2_000
_NAME = r"^[a-z][a-z0-9_]{0,62}$"
_NAME_RE = re.compile(_NAME)
_SPEC_PACKAGE: Final = "double_ml_uplift.configs.specs"


class SpecError(ValueError):
    """The causal specification is missing, oversized, malformed or structurally invalid."""


class NodeSpec(BaseModel):
    """One variable in the declared DAG, with the range its data contract will enforce."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["continuous", "count", "binary"]
    role: Literal["confounder", "treatment", "outcome", "design", "post_treatment"] = "confounder"
    observed: bool = True
    minimum: float | None = None
    maximum: float | None = None
    description: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        if self.kind == "binary" and (self.minimum, self.maximum) not in {(None, None), (0.0, 1.0)}:
            raise ValueError("binary nodes take no explicit bounds")
        return self


class CausalSpec(BaseModel):
    """A validated causal specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1]
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    unit_id: str = Field(pattern=_NAME)
    treatment: str = Field(pattern=_NAME)
    outcome: str = Field(pattern=_NAME)
    pilot_column: str | None = Field(default=None, pattern=_NAME)
    protected_attribute: str | None = Field(default=None, pattern=_NAME)
    nodes: dict[str, NodeSpec] = Field(min_length=2, max_length=MAX_NODES)
    edges: tuple[tuple[str, str], ...] = Field(max_length=MAX_EDGES)
    forbidden_columns: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_structure(self) -> Self:
        self._check_names()
        self._check_endpoints()
        self._check_optional_columns()
        unknown = sorted(set(self.forbidden_columns) - set(self.nodes))
        if unknown:
            raise ValueError(f"forbidden_columns mention unknown nodes: {', '.join(unknown)}")
        try:
            graph = CausalGraph.from_edges(self.nodes, self.edges)
        except GraphError as exc:
            raise ValueError(str(exc)) from None
        if self.outcome not in graph.descendants([self.treatment], inclusive=False):
            raise ValueError("the DAG has no directed path from the treatment to the outcome")
        self._check_roles(graph)
        return self

    def _check_names(self) -> None:
        for name in self.nodes:
            if not isinstance(name, str) or _NAME_RE.fullmatch(name) is None:
                raise ValueError(f"invalid node name {name!r}")
        if self.unit_id in self.nodes:
            raise ValueError("unit_id identifies rows; it must not be a node of the DAG")

    def _check_endpoints(self) -> None:
        for label, name in (("treatment", self.treatment), ("outcome", self.outcome)):
            if name not in self.nodes:
                raise ValueError(f"{label} {name!r} is not declared as a node")
        if self.treatment == self.outcome:
            raise ValueError("treatment and outcome must differ")
        if self.nodes[self.treatment].kind != "binary":
            raise ValueError("this repository estimates effects of a binary treatment only")
        if not self.nodes[self.treatment].observed or not self.nodes[self.outcome].observed:
            raise ValueError("treatment and outcome must be observed")

    def _check_optional_columns(self) -> None:
        """``pilot_column`` and ``protected_attribute`` are optional, but if named must exist."""
        for label, name in (
            ("pilot_column", self.pilot_column),
            ("protected_attribute", self.protected_attribute),
        ):
            if name is None:
                continue
            if name not in self.nodes:
                raise ValueError(f"{label} {name!r} is not declared as a node")
            if self.nodes[name].kind != "binary":
                raise ValueError(f"{label} {name!r} must be a binary node")

    def _check_roles(self, graph: CausalGraph) -> None:
        """Declared roles must agree with the DAG, so leakage cannot hide behind a label."""
        if self.nodes[self.treatment].role != "treatment":
            raise ValueError(f"{self.treatment!r} is the treatment but is not declared as one")
        if self.nodes[self.outcome].role != "outcome":
            raise ValueError(f"{self.outcome!r} is the outcome but is not declared as one")
        downstream = graph.descendants([self.treatment]) | graph.descendants([self.outcome])
        downstream -= {self.treatment, self.outcome}
        for name, node in self.nodes.items():
            if name in {self.treatment, self.outcome}:
                continue
            if name in downstream and node.role != "post_treatment":
                raise ValueError(
                    f"{name!r} is downstream of the treatment or the outcome but is declared "
                    f"{node.role!r}; adjusting for it would leak post-treatment information"
                )
            if name not in downstream and node.role == "post_treatment":
                raise ValueError(
                    f"{name!r} is declared post_treatment but the DAG places it upstream of the "
                    "treatment; add the missing edge or correct the role"
                )

    # --- derived views -------------------------------------------------------------------

    def graph(self) -> CausalGraph:
        """The declared DAG. Cheap to rebuild; the engine keeps one instance per run."""
        return CausalGraph.from_edges(self.nodes, self.edges)

    def observed_nodes(self) -> tuple[str, ...]:
        return tuple(name for name, node in self.nodes.items() if node.observed)

    def unobserved_nodes(self) -> tuple[str, ...]:
        return tuple(name for name, node in self.nodes.items() if not node.observed)

    def frame_columns(self) -> tuple[str, ...]:
        """Columns a conforming data frame must contain: the unit id and every observed node."""
        return (self.unit_id, *self.observed_nodes())

    def sha256(self) -> str:
        """Content hash of the specification, recorded with every estimate and bundle."""
        return sha256_hex(canonical_json(self.model_dump(mode="json")))


def _normalise(document: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(document)
    edges = data.get("edges", [])
    if not isinstance(edges, list):
        raise SpecError("edges must be a list of [source, target] pairs")
    normalised: list[tuple[str, str]] = []
    for edge in edges:
        if not (isinstance(edge, list | tuple) and len(edge) == 2):
            raise SpecError(f"malformed edge {edge!r}: expected [source, target]")
        source, target = edge
        if not isinstance(source, str) or not isinstance(target, str):
            raise SpecError(f"malformed edge {edge!r}: endpoints must be strings")
        normalised.append((source, target))
    data["edges"] = normalised
    return data


def load_spec_text(text: str) -> CausalSpec:
    """Parse and validate a specification from YAML text.

    ``yaml.safe_load`` cannot construct arbitrary Python objects, and the document is rejected
    before parsing if it exceeds :data:`MAX_SPEC_BYTES`.
    """
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_SPEC_BYTES:
        raise SpecError(f"specification exceeds {MAX_SPEC_BYTES} bytes")
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SpecError(f"specification is not valid YAML: {type(exc).__name__}") from None
    if not isinstance(document, dict):
        raise SpecError("specification must be a YAML mapping")
    try:
        return CausalSpec.model_validate(_normalise(document))
    except SpecError:
        raise
    except ValueError as exc:
        raise SpecError(f"invalid causal specification: {exc}") from None


def load_spec_file(path: Path) -> CausalSpec:
    """Read a specification from disk, refusing anything larger than the size limit."""
    if path.stat().st_size > MAX_SPEC_BYTES:
        raise SpecError(f"specification exceeds {MAX_SPEC_BYTES} bytes")
    return load_spec_text(path.read_text(encoding="utf-8"))


def bundled_spec_names() -> tuple[str, ...]:
    """Names of the specifications shipped inside the package."""
    return tuple(
        sorted(
            item.name.removesuffix(".yaml")
            for item in resources.files(_SPEC_PACKAGE).iterdir()
            if item.name.endswith(".yaml")
        )
    )


def bundled_spec(name: str) -> CausalSpec:
    """Load a bundled specification by name (``retention_offer``, ``leaky``, ``unobserved``)."""
    if name not in bundled_spec_names():
        raise SpecError(f"unknown bundled specification {name!r}")
    return load_spec_text(
        resources.files(_SPEC_PACKAGE).joinpath(f"{name}.yaml").read_text(encoding="utf-8")
    )
