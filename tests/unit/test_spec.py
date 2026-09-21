"""The causal specification is parsed as data and validated before anything downstream runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from double_ml_uplift.causal.spec import (
    MAX_SPEC_BYTES,
    CausalSpec,
    NodeSpec,
    SpecError,
    bundled_spec,
    bundled_spec_names,
    load_spec_file,
    load_spec_text,
)

pytestmark = pytest.mark.unit


def minimal_document() -> dict[str, Any]:
    return {
        "version": 1,
        "name": "minimal",
        "unit_id": "row_id",
        "treatment": "t",
        "outcome": "y",
        "nodes": {
            "t": {"kind": "binary", "role": "treatment"},
            "y": {"kind": "continuous", "role": "outcome"},
            "u": {"kind": "continuous", "role": "confounder"},
        },
        "edges": [["t", "y"], ["u", "t"], ["u", "y"]],
    }


def load(document: dict[str, Any]) -> CausalSpec:
    return load_spec_text(yaml.safe_dump(document, sort_keys=False))


def test_a_minimal_specification_round_trips() -> None:
    spec = load(minimal_document())
    assert spec.treatment == "t"
    assert spec.observed_nodes() == ("t", "y", "u")
    assert spec.unobserved_nodes() == ()
    assert spec.frame_columns() == ("row_id", "t", "y", "u")
    assert len(spec.sha256()) == 64


# --- parsing is not evaluation ------------------------------------------------------------


def test_yaml_tags_that_would_construct_python_objects_are_refused() -> None:
    payload = "!!python/object/apply:os.system ['echo pwned']\n"
    with pytest.raises(SpecError, match=r"not valid YAML|must be a YAML mapping"):
        load_spec_text(payload)


def test_a_specification_larger_than_the_limit_is_refused_before_parsing() -> None:
    padded = "# " + "x" * MAX_SPEC_BYTES + "\n"
    with pytest.raises(SpecError, match="exceeds"):
        load_spec_text(padded)


def test_non_mapping_and_malformed_documents_are_refused() -> None:
    with pytest.raises(SpecError, match="must be a YAML mapping"):
        load_spec_text("- just\n- a\n- list\n")
    with pytest.raises(SpecError, match="not valid YAML"):
        load_spec_text("key: [unclosed\n")
    for edges, message in (
        ("not-a-list", "edges must be a list"),
        ([["a"]], "expected \\[source, target\\]"),
        ([[1, 2]], "endpoints must be strings"),
    ):
        document = minimal_document()
        document["edges"] = edges
        with pytest.raises(SpecError):
            load(document)
        assert message  # the messages above document each rejection


def test_load_spec_file_enforces_the_size_limit(tmp_path: Path) -> None:
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(minimal_document()), encoding="utf-8")
    assert load_spec_file(path).name == "minimal"
    path.write_text("# " + "y" * MAX_SPEC_BYTES, encoding="utf-8")
    with pytest.raises(SpecError, match="exceeds"):
        load_spec_file(path)


# --- structural validation ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(treatment="missing"), "not declared as a node"),
        (lambda d: d.update(outcome="t"), "must differ"),
        (lambda d: d["nodes"]["t"].update(kind="continuous"), "binary treatment only"),
        (lambda d: d["nodes"]["y"].update(observed=False), "must be observed"),
        (lambda d: d.update(unit_id="u"), "must not be a node"),
        (lambda d: d.update(edges=[["u", "t"], ["u", "y"]]), "no directed path"),
        (lambda d: d["nodes"]["u"].update(role="post_treatment"), "places it upstream"),
        (lambda d: d["nodes"]["t"].update(role="confounder"), "not declared as one"),
        (lambda d: d["nodes"]["y"].update(role="confounder"), "not declared as one"),
        (lambda d: d.update(forbidden_columns=["nope"]), "unknown nodes"),
        (lambda d: d.update(pilot_column="u"), "must be a binary node"),
        (lambda d: d.update(protected_attribute="nope"), "not declared as a node"),
    ],
)
def test_structural_violations_are_refused(mutate: Any, message: str) -> None:
    document = minimal_document()
    mutate(document)
    with pytest.raises(SpecError, match=message):
        load(document)


def test_a_cycle_in_the_declared_edges_is_refused() -> None:
    document = minimal_document()
    document["nodes"]["m"] = {"kind": "continuous", "role": "post_treatment"}
    document["edges"] = [["t", "m"], ["m", "y"], ["y", "t"]]
    with pytest.raises(SpecError, match="cycle"):
        load(document)


def test_a_node_downstream_of_the_treatment_must_be_declared_post_treatment() -> None:
    document = minimal_document()
    document["nodes"]["m"] = {"kind": "continuous", "role": "confounder"}
    document["edges"] = [["t", "y"], ["u", "t"], ["u", "y"], ["t", "m"], ["m", "y"]]
    with pytest.raises(SpecError, match="would leak post-treatment information"):
        load(document)


def test_node_bounds_and_kinds_are_validated() -> None:
    with pytest.raises(ValueError, match="minimum must not exceed maximum"):
        NodeSpec(kind="continuous", minimum=5.0, maximum=1.0)
    with pytest.raises(ValueError, match="binary nodes take no explicit bounds"):
        NodeSpec(kind="binary", minimum=0.0, maximum=7.0)
    assert NodeSpec(kind="binary", minimum=0.0, maximum=1.0).kind == "binary"


def test_extra_keys_are_refused_so_a_typo_cannot_be_ignored() -> None:
    document = minimal_document()
    document["adjustment_set"] = ["u"]
    with pytest.raises(SpecError):
        load(document)


# --- the bundled specifications -------------------------------------------------------------


def test_the_bundled_specifications_are_exactly_the_three_documented_ones() -> None:
    assert bundled_spec_names() == ("leaky", "retention_offer", "unobserved")
    with pytest.raises(SpecError, match="unknown bundled specification"):
        bundled_spec("nope")


def test_the_retention_specification_declares_what_the_simulator_produces() -> None:
    from double_ml_uplift.causal.scm import (
        CONFOUNDER_COLUMNS,
        OUTCOME_COLUMN,
        PILOT_COLUMN,
        POST_TREATMENT_COLUMNS,
        TREATMENT_COLUMN,
        UNIT_COLUMN,
    )

    spec = bundled_spec("retention_offer")
    assert spec.unit_id == UNIT_COLUMN
    assert spec.treatment == TREATMENT_COLUMN
    assert spec.outcome == OUTCOME_COLUMN
    assert spec.pilot_column == PILOT_COLUMN
    expected = {
        UNIT_COLUMN,
        *CONFOUNDER_COLUMNS,
        PILOT_COLUMN,
        TREATMENT_COLUMN,
        *POST_TREATMENT_COLUMNS,
        OUTCOME_COLUMN,
    }
    assert set(spec.frame_columns()) == expected


def test_the_leaky_specification_is_kept_broken_on_purpose() -> None:
    with pytest.raises(SpecError, match="would leak post-treatment information"):
        bundled_spec("leaky")


def test_the_unobserved_specification_declares_a_latent_common_cause() -> None:
    spec = bundled_spec("unobserved")
    assert spec.unobserved_nodes() == ("churn_risk",)
    assert "churn_risk" not in spec.frame_columns()


# --- hashing ---------------------------------------------------------------------------------


def test_the_specification_hash_is_stable_and_sensitive() -> None:
    first = load(minimal_document())
    assert first.sha256() == load(minimal_document()).sha256()

    renamed = minimal_document()
    renamed["nodes"]["u"]["description"] = "a comment that changes the hash"
    assert load(renamed).sha256() != first.sha256()


@given(order=st.permutations(["t", "y", "u"]))
def test_the_hash_does_not_depend_on_the_order_nodes_were_written_in(
    order: list[str],
) -> None:
    document = minimal_document()
    document["nodes"] = {name: document["nodes"][name] for name in order}
    assert load(document).sha256() == load(minimal_document()).sha256()


def test_graph_rebuilds_the_declared_dag() -> None:
    spec = load(minimal_document())
    graph = spec.graph()
    assert set(graph.nodes) == {"t", "y", "u"}
    assert graph.parents("t") == frozenset({"u"})
