"""Backdoor identification and the post-treatment leakage guard.

Given the declared DAG, ``identify_backdoor`` looks for a set ``Z`` of observed variables that
satisfies Pearl's backdoor criterion (Pearl, *Causality*, 2009, Definition 3.3.1) for the
treatment-outcome pair:

1. no member of ``Z`` is a descendant of the treatment, and
2. ``Z`` blocks every path from treatment to outcome that starts with an arrow into the
   treatment.

Condition 2 is checked as d-separation of treatment and outcome in the graph with the treatment's
outgoing edges removed, which is equivalent and linear-time. The search starts from the *canonical*
adjustment set -- the allowed variables that are ancestors of the treatment or the outcome -- and
then greedily drops variables that are not needed. Starting there rather than from every
pre-treatment covariate is what makes the search complete; see :func:`_admissible_candidates`. The
answer is locally minimal (no single variable can be removed) and deterministic, because
candidates are considered in sorted order.

If no admissible set blocks the backdoor -- typically because a common cause is declared
unobserved -- the module refuses and names an open path instead of returning a number.

The leakage guard is deliberately stricter than the criterion: it also refuses descendants of the
*outcome*, because conditioning on them opens collider paths, and it refuses variables whose name
looks post-treatment but whose DAG position does not, which is almost always a forgotten edge.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Final

from double_ml_uplift.causal.graph import CausalGraph
from double_ml_uplift.causal.spec import CausalSpec

__all__ = [
    "POST_TREATMENT_NAME_PATTERN",
    "Identification",
    "LeakageError",
    "NotIdentifiedError",
    "format_path",
    "identify_backdoor",
    "leakage_guard",
    "verify_adjustment_set",
]

# Names that conventionally mark a value recorded after the intervention. A variable matching one
# of these must be declared post-treatment in the DAG, or the DAG is out of date.
POST_TREATMENT_NAME_PATTERN: Final = re.compile(
    r"(^|_)(post|after)(_|$)|_(at|ts|timestamp|time)$|_after_", re.IGNORECASE
)


class NotIdentifiedError(ValueError):
    """The declared DAG does not identify the effect from the observed variables."""

    def __init__(self, reasons: Sequence[str]) -> None:
        super().__init__("; ".join(reasons) or "the effect is not identified")
        self.reasons = tuple(reasons)


class LeakageError(ValueError):
    """A requested adjustment variable is post-treatment, a descendant of the outcome, or banned."""

    def __init__(self, offenders: Sequence[tuple[str, str]]) -> None:
        detail = "; ".join(f"{name}: {reason}" for name, reason in offenders)
        super().__init__(f"adjustment set rejected ({detail})")
        self.offenders = tuple(offenders)


@dataclass(frozen=True, slots=True)
class Identification:
    """The outcome of an identification query."""

    identified: bool
    estimand: str
    adjustment_set: tuple[str, ...]
    reasons: tuple[str, ...] = ()
    backdoor_paths: int = 0
    considered: tuple[str, ...] = ()

    def require(self) -> tuple[str, ...]:
        """Return the adjustment set, or fail closed with the refusal reasons."""
        if not self.identified:
            raise NotIdentifiedError(self.reasons)
        return self.adjustment_set


def format_path(graph: CausalGraph, path: Sequence[str]) -> str:
    """Render an undirected trail with arrowheads, e.g. ``t <- u -> y``."""
    parts = [path[0]]
    for left, right in pairwise(path):
        arrow = " -> " if right in graph.children(left) else " <- "
        parts.append(arrow)
        parts.append(right)
    return "".join(parts)


def _blocks_backdoor(mutilated: CausalGraph, spec: CausalSpec, adjustment: Iterable[str]) -> bool:
    return mutilated.is_d_separated([spec.treatment], [spec.outcome], adjustment)


def leakage_guard(spec: CausalSpec, requested: Iterable[str]) -> tuple[str, ...]:
    """Validate requested adjustment variables, raising :class:`LeakageError` on any offender.

    Returns the requested names in sorted order when all of them are admissible.
    """
    graph = spec.graph()
    names = sorted(set(requested))
    unknown = [name for name in names if name not in spec.nodes]
    if unknown:
        raise LeakageError([(name, "not declared in the causal specification") for name in unknown])

    post_treatment = graph.descendants([spec.treatment])
    post_outcome = graph.descendants([spec.outcome])
    offenders: list[tuple[str, str]] = []
    for name in names:
        if name == spec.treatment:
            offenders.append((name, "is the treatment"))
        elif name == spec.outcome:
            offenders.append((name, "is the outcome"))
        elif not spec.nodes[name].observed:
            offenders.append((name, "is declared unobserved"))
        elif name in spec.forbidden_columns:
            offenders.append((name, "is listed in forbidden_columns"))
        elif spec.nodes[name].role == "post_treatment":
            offenders.append((name, "is declared post-treatment in the causal specification"))
        # The outcome is downstream of the treatment, so every descendant of the outcome is also
        # a descendant of the treatment. Report the more specific reason first.
        elif name in post_outcome:
            offenders.append(
                (name, "is a descendant of the outcome (conditioning opens a collider)")
            )
        elif name in post_treatment:
            offenders.append((name, "is a descendant of the treatment (post-treatment leakage)"))
        elif POST_TREATMENT_NAME_PATTERN.search(name):
            offenders.append(
                (name, "name looks post-treatment but the DAG places it before the treatment")
            )
    if offenders:
        raise LeakageError(offenders)
    return tuple(names)


def _admissible_candidates(spec: CausalSpec, graph: CausalGraph) -> list[str]:
    """The canonical adjustment set: allowed variables that are ancestors of treatment or outcome.

    Restricting to ancestors is what makes the search *complete*. Adjusting for "every
    pre-treatment covariate" is the usual reflex and it is wrong: a pre-treatment collider
    (the M-bias structure ``t <- a -> m <- b -> y``) opens a path that was closed, and a search
    that starts from all covariates then refuses whenever it cannot repair the damage -- even
    though the empty set was admissible all along.

    A node that is not an ancestor of the treatment or the outcome cannot lie on an open backdoor
    trail as a non-collider, so it can never be needed; ``m`` above is exactly such a node. The
    canonical set ``An({T, Y}) & allowed \\ De(T)`` is therefore a valid backdoor adjustment set
    whenever *any* subset of the allowed variables is (van der Zander, Liskiewicz and Textor,
    "Constructing separators and adjustment sets in ancestral graphs", UAI 2014; Shpitser,
    VanderWeele and Robins, "On the validity of covariate adjustment", UAI 2010).

    ``tests/unit/test_identification.py::test_a_refusal_means_no_admissible_subset_would_have_worked``
    checks that completeness against brute-force enumeration of every subset on random DAGs.
    """
    excluded = set(graph.descendants([spec.treatment]))
    excluded |= set(graph.descendants([spec.outcome]))
    excluded |= set(spec.forbidden_columns)
    excluded |= set(spec.unobserved_nodes())
    excluded |= {name for name, node in spec.nodes.items() if node.role == "post_treatment"}
    excluded |= {spec.treatment, spec.outcome}
    relevant = graph.ancestors([spec.treatment, spec.outcome])
    candidates = []
    for name in sorted((set(spec.nodes) & relevant) - excluded):
        if POST_TREATMENT_NAME_PATTERN.search(name):
            continue  # the guard would reject it; never propose it
        candidates.append(name)
    return candidates


def identify_backdoor(spec: CausalSpec) -> Identification:
    """Find a minimal observed adjustment set, or refuse with the open backdoor paths."""
    graph = spec.graph()
    mutilated = graph.without_outgoing_edges(spec.treatment)
    candidates = _admissible_candidates(spec, graph)
    n_backdoor = len(graph.backdoor_paths(spec.treatment, spec.outcome))

    if not _blocks_backdoor(mutilated, spec, candidates):
        open_trails = mutilated.open_paths(spec.treatment, spec.outcome, candidates)
        reasons = [
            "no observed adjustment set satisfies the backdoor criterion for "
            f"{spec.treatment} -> {spec.outcome}"
        ]
        reasons += [f"open backdoor path: {format_path(graph, path)}" for path in open_trails[:3]]
        return Identification(
            identified=False,
            estimand="backdoor-adjustment",
            adjustment_set=(),
            reasons=tuple(reasons),
            backdoor_paths=n_backdoor,
            considered=tuple(candidates),
        )

    minimal = list(candidates)
    for name in candidates:
        trimmed = [item for item in minimal if item != name]
        if _blocks_backdoor(mutilated, spec, trimmed):
            minimal = trimmed
    return Identification(
        identified=True,
        estimand="backdoor-adjustment",
        adjustment_set=tuple(minimal),
        backdoor_paths=n_backdoor,
        considered=tuple(candidates),
    )


def verify_adjustment_set(spec: CausalSpec, requested: Iterable[str]) -> Identification:
    """Check a caller-supplied adjustment set: leakage guard first, then the backdoor criterion."""
    names = leakage_guard(spec, requested)
    graph = spec.graph()
    mutilated = graph.without_outgoing_edges(spec.treatment)
    n_backdoor = len(graph.backdoor_paths(spec.treatment, spec.outcome))
    if _blocks_backdoor(mutilated, spec, names):
        return Identification(
            identified=True,
            estimand="backdoor-adjustment",
            adjustment_set=names,
            backdoor_paths=n_backdoor,
            considered=names,
        )
    open_trails = mutilated.open_paths(spec.treatment, spec.outcome, names)
    reasons = [
        f"the requested adjustment set does not block every backdoor path ({len(names)} variables)"
    ]
    reasons += [f"open backdoor path: {format_path(graph, path)}" for path in open_trails[:3]]
    return Identification(
        identified=False,
        estimand="backdoor-adjustment",
        adjustment_set=(),
        reasons=tuple(reasons),
        backdoor_paths=n_backdoor,
        considered=names,
    )
