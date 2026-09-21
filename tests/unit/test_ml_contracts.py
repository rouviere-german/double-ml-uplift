"""The data contract refuses leaked ground truth first, then validates shape and ranges."""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from double_ml_uplift.causal.scm import ScmConfig, simulate_campaign
from double_ml_uplift.causal.spec import bundled_spec
from double_ml_uplift.ml.contracts import (
    FORBIDDEN_COLUMNS,
    ContractLimits,
    DataContractError,
    PotentialOutcomeLeakError,
    build_frame_schema,
    validate_campaign_frame,
)

pytestmark = pytest.mark.unit

SPEC = bundled_spec("retention_offer")


def campaign(n: int = 200, *, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    simulated = simulate_campaign(ScmConfig(name="contract", n_customers=n, seed=seed))
    return simulated.frame, simulated.oracle


def failures_for(frame: pd.DataFrame) -> DataContractError:
    with pytest.raises(DataContractError) as info:
        validate_campaign_frame(frame, SPEC)
    return info.value


# --- a conforming frame ---------------------------------------------------------------------


def test_a_simulated_campaign_satisfies_its_own_contract() -> None:
    frame, _ = campaign()
    assert len(validate_campaign_frame(frame, SPEC)) == 200


def test_the_schema_is_derived_from_the_specification_not_hard_coded() -> None:
    schema = build_frame_schema(SPEC)
    assert set(schema.columns) == set(SPEC.frame_columns())
    assert schema.strict is True
    assert schema.name == SPEC.name
    assert schema.columns[SPEC.unit_id].unique is True


# --- oracle columns are refused before anything else ------------------------------------------


@pytest.mark.parametrize("column", sorted(FORBIDDEN_COLUMNS))
def test_every_oracle_column_is_refused_by_name(column: str) -> None:
    frame, _ = campaign()
    frame[column] = 1.0
    with pytest.raises(PotentialOutcomeLeakError) as info:
        validate_campaign_frame(frame, SPEC)
    assert info.value.columns == [column]


def test_the_leak_check_runs_before_the_size_checks() -> None:
    frame, oracle = campaign()
    frame["y1"] = oracle["y1"].to_numpy()
    with pytest.raises(PotentialOutcomeLeakError):
        validate_campaign_frame(frame, SPEC, ContractLimits(max_rows=1))


def test_several_leaked_columns_are_all_named() -> None:
    frame, oracle = campaign()
    frame["y0"] = oracle["y0"].to_numpy()
    frame["true_cate"] = oracle["true_cate"].to_numpy()
    with pytest.raises(PotentialOutcomeLeakError) as info:
        validate_campaign_frame(frame, SPEC)
    assert info.value.columns == ["true_cate", "y0"]
    assert len(info.value.failures) == 2


# --- size limits ------------------------------------------------------------------------------


def test_an_oversized_frame_is_refused_cheaply() -> None:
    frame, _ = campaign(60)
    error = failures_for_with_limits(frame, ContractLimits(max_rows=10))
    assert error.failures[0]["check"] == "max_rows(10)"
    assert error.failures[0]["count"] == 60


def failures_for_with_limits(frame: pd.DataFrame, limits: ContractLimits) -> DataContractError:
    with pytest.raises(DataContractError) as info:
        validate_campaign_frame(frame, SPEC, limits)
    return info.value


def test_too_many_columns_is_refused() -> None:
    frame, _ = campaign(60)
    error = failures_for_with_limits(frame, ContractLimits(max_columns=3))
    assert error.failures[0]["check"] == "max_columns(3)"


def test_an_empty_frame_is_refused() -> None:
    frame, _ = campaign(60)
    error = failures_for(frame.iloc[0:0])
    assert error.failures[0]["check"] == "non_empty"


def test_limits_must_be_positive() -> None:
    with pytest.raises(ValueError, match="limits must be positive"):
        ContractLimits(max_rows=0)
    with pytest.raises(ValueError, match="limits must be positive"):
        ContractLimits(max_columns=0)


# --- schema violations --------------------------------------------------------------------------


def test_an_undeclared_column_is_refused_rather_than_carried_through() -> None:
    frame, _ = campaign(60)
    frame["surprise"] = 1.0
    error = failures_for(frame)
    assert {"column": "surprise", "check": "column_in_schema", "count": 1} in error.failures


def test_a_missing_column_is_refused_and_named() -> None:
    frame, _ = campaign(60)
    error = failures_for(frame.drop(columns=["tenure_months"]))
    assert {
        "column": "tenure_months",
        "check": "column_in_dataframe",
        "count": 1,
    } in error.failures


def test_out_of_range_values_are_refused_per_column() -> None:
    frame, _ = campaign(60)
    frame.loc[0, "tenure_months"] = -5.0
    frame.loc[1, "engagement_score"] = 99.0
    error = failures_for(frame)
    columns = {str(item["column"]) for item in error.failures}
    assert {"tenure_months", "engagement_score"} <= columns


def test_a_non_binary_treatment_is_refused() -> None:
    frame, _ = campaign(60)
    frame.loc[0, "offer_sent"] = 2
    error = failures_for(frame)
    assert any(str(item["column"]) == "offer_sent" for item in error.failures)


def test_a_duplicate_customer_id_is_refused() -> None:
    frame, _ = campaign(60)
    frame.loc[1, "customer_id"] = frame.loc[0, "customer_id"]
    error = failures_for(frame)
    assert any(str(item["column"]) == "customer_id" for item in error.failures)


def test_a_customer_id_with_odd_characters_is_refused() -> None:
    frame, _ = campaign(60)
    frame.loc[0, "customer_id"] = "../../etc/passwd"
    error = failures_for(frame)
    assert any(str(item["column"]) == "customer_id" for item in error.failures)


def test_a_missing_value_is_refused() -> None:
    frame, _ = campaign(60)
    frame.loc[0, "monthly_spend"] = None
    error = failures_for(frame)
    assert any(str(item["column"]) == "monthly_spend" for item in error.failures)


def test_lazy_validation_reports_every_violation_in_one_pass() -> None:
    frame, _ = campaign(60)
    frame.loc[0, "tenure_months"] = -1.0
    frame.loc[1, "monthly_spend"] = -1.0
    frame.loc[2, "support_tickets"] = -1
    error = failures_for(frame)
    columns = {str(item["column"]) for item in error.failures}
    assert {"tenure_months", "monthly_spend", "support_tickets"} <= columns


# --- failure summaries never echo the data -----------------------------------------------------


def test_a_failure_summary_carries_counts_and_never_a_cell_value() -> None:
    frame, _ = campaign(60)
    secret = 987654.321
    frame.loc[0, "monthly_spend"] = secret
    error = failures_for(frame)
    rendered = str(error) + str(error.failures)
    assert "987654" not in rendered
    assert "failure(s) across" in str(error)
    for item in error.failures:
        assert set(item) == {"column", "check", "count"}
        assert isinstance(item["count"], int)


def test_a_customer_id_never_appears_in_the_summary_of_another_columns_failure() -> None:
    frame, _ = campaign(60)
    frame.loc[0, "customer_id"] = "cust_secret_identifier"
    frame.loc[0, "monthly_spend"] = -1.0
    error = failures_for(frame)
    rendered = str(error) + str(error.failures)
    assert any(str(item["column"]) == "monthly_spend" for item in error.failures)
    assert "cust_secret_identifier" not in rendered


def test_naming_a_column_never_extends_to_naming_a_value() -> None:
    """The schema-level checks report a column name; every data check reports only a count."""
    frame, _ = campaign(60)
    frame["surprise"] = 1.0
    frame.loc[0, "engagement_score"] = 42.5
    frame.loc[1, "engagement_score"] = 43.5
    error = failures_for(frame)
    rendered = str(error.failures)
    assert "surprise" in rendered
    assert "42.5" not in rendered
    assert "43.5" not in rendered
    engagement = [item for item in error.failures if str(item["column"]) == "engagement_score"]
    assert len(engagement) == 1
    assert engagement[0]["count"] == 2


# --- generated frames ---------------------------------------------------------------------------


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=50, max_value=300),
    seed=st.integers(min_value=0, max_value=5_000),
)
def test_every_simulated_frame_passes_and_every_oracle_column_fails(n: int, seed: int) -> None:
    frame, oracle = campaign(n, seed=seed)
    assert len(validate_campaign_frame(frame, SPEC)) == n
    leaked = frame.copy()
    leaked["true_propensity"] = oracle["true_propensity"].to_numpy()
    with pytest.raises(PotentialOutcomeLeakError):
        validate_campaign_frame(leaked, SPEC)
