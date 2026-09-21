"""The data contract for a campaign table, derived from the declared causal specification.

Two jobs, in this order:

1. **Refuse leaked ground truth.** A simulated campaign carries both potential outcomes, the true
   conditional effect and the hidden confounder. Those columns exist so the evaluation code can
   score the estimators; an estimation request that contains any of them is a bug or an attack,
   never a convenience, so it fails with its own exception type before anything else runs.
2. **Validate the shape and ranges** every declared node promises, with pandera, lazily, so one
   pass reports every violation. Failure summaries carry the column, the check and a count and
   never the offending values, because in a customer table the offending values are exactly the
   data worth protecting.

The schema is *built from the specification*, not hard-coded: adding a covariate to the DAG adds
it to the contract, and a frame with columns the specification does not declare is rejected
(``strict=True``) rather than silently carried into the estimator.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Final

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

from double_ml_uplift.causal.spec import CausalSpec

__all__ = [
    "DEFAULT_LIMITS",
    "FORBIDDEN_COLUMNS",
    "ContractLimits",
    "DataContractError",
    "PotentialOutcomeLeakError",
    "build_frame_schema",
    "validate_campaign_frame",
]

# Columns produced by the simulator for scoring only. Never acceptable in an estimation request.
FORBIDDEN_COLUMNS: Final = frozenset(
    {"y0", "y1", "true_cate", "true_propensity", "hidden_confounder", "potential_outcome"}
)
_UNIT_ID_PATTERN: Final = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$"
# Checks whose ``failure_case`` is a column name rather than a cell value.
_SCHEMA_LEVEL_CHECKS: Final = frozenset({"column_in_schema", "column_in_dataframe"})


@dataclass(frozen=True, slots=True)
class ContractLimits:
    """Caps applied before any parsing work, so an oversized upload is cheap to refuse."""

    max_rows: int = 2_000_000
    max_columns: int = 200

    def __post_init__(self) -> None:
        if self.max_rows < 1 or self.max_columns < 1:
            raise ValueError("contract limits must be positive")


DEFAULT_LIMITS: Final = ContractLimits()


class DataContractError(ValueError):
    """A frame violated its contract. ``failures`` holds (column, check, count) summaries."""

    def __init__(self, failures: list[dict[str, str | int]]) -> None:
        total = sum(int(item["count"]) for item in failures)
        super().__init__(
            f"data contract violated: {total} failure(s) across {len(failures)} check(s)"
        )
        self.failures = failures


class PotentialOutcomeLeakError(DataContractError):
    """The frame carries potential outcomes or another oracle column."""

    def __init__(self, columns: list[str]) -> None:
        super().__init__(
            [{"column": name, "check": "not_an_oracle_column", "count": 1} for name in columns]
        )
        self.columns = columns


def _column_for(kind: str, minimum: float | None, maximum: float | None) -> pa.Column:
    checks: list[pa.Check] = []
    if kind == "binary":
        return pa.Column("int64", checks=[pa.Check.isin([0, 1])], nullable=False)
    if minimum is not None:
        checks.append(pa.Check.ge(minimum))
    if maximum is not None:
        checks.append(pa.Check.le(maximum))
    dtype = "int64" if kind == "count" else "float64"
    return pa.Column(dtype, checks=checks, nullable=False)


def build_frame_schema(spec: CausalSpec) -> pa.DataFrameSchema:
    """Build the pandera schema this specification implies."""
    columns: dict[str, pa.Column] = {
        spec.unit_id: pa.Column(
            str,
            checks=[pa.Check.str_matches(_UNIT_ID_PATTERN)],
            unique=True,
            nullable=False,
        )
    }
    for name in spec.observed_nodes():
        node = spec.nodes[name]
        columns[name] = _column_for(node.kind, node.minimum, node.maximum)
    return pa.DataFrameSchema(columns, strict=True, coerce=False, name=spec.name)


def _summarize(exc: SchemaErrors) -> list[dict[str, str | int]]:
    """Reduce pandera's failure cases to (column, check, count), never to a cell value.

    For the two *schema-level* checks -- a column the specification does not declare, and a
    declared column the frame does not have -- pandera reports the schema's own name in
    ``column`` and the offending column's name in ``failure_case``. A column name describes the
    shape of the upload, not a customer, so it is safe to report and it is the only thing that
    makes the error actionable. Every other check puts an offending *cell value* in that field,
    which is exactly the data worth protecting, so it is dropped.
    """
    frame = exc.failure_cases
    columns = [str(value) for value in frame["column"].tolist()]
    checks = [str(value) for value in frame["check"].tolist()]
    cases = (
        [str(value) for value in frame["failure_case"].tolist()]
        if "failure_case" in frame.columns
        else list(columns)
    )
    counts: Counter[tuple[str, str]] = Counter(
        (case if check in _SCHEMA_LEVEL_CHECKS else column, check)
        for column, check, case in zip(columns, checks, cases, strict=True)
    )
    return [
        {"column": column, "check": check, "count": count}
        for (column, check), count in sorted(counts.items())
    ]


def validate_campaign_frame(
    frame: pd.DataFrame, spec: CausalSpec, limits: ContractLimits = DEFAULT_LIMITS
) -> pd.DataFrame:
    """Validate a campaign table against the contract implied by ``spec``.

    Raises :class:`PotentialOutcomeLeakError` for oracle columns and :class:`DataContractError`
    for size violations and schema violations. Never echoes a cell value.
    """
    leaked = sorted(set(frame.columns) & FORBIDDEN_COLUMNS)
    if leaked:
        raise PotentialOutcomeLeakError(leaked)
    if len(frame) > limits.max_rows:
        raise DataContractError(
            [{"column": "<frame>", "check": f"max_rows({limits.max_rows})", "count": len(frame)}]
        )
    if len(frame.columns) > limits.max_columns:
        raise DataContractError(
            [
                {
                    "column": "<frame>",
                    "check": f"max_columns({limits.max_columns})",
                    "count": len(frame.columns),
                }
            ]
        )
    if frame.empty:
        raise DataContractError([{"column": "<frame>", "check": "non_empty", "count": 0}])
    schema = build_frame_schema(spec)
    try:
        return schema.validate(frame, lazy=True)
    except SchemaErrors as exc:
        raise DataContractError(_summarize(exc)) from None
    except SchemaError as exc:  # pragma: no cover - lazy validation raises SchemaErrors
        raise DataContractError(
            [{"column": str(exc.schema.name), "check": "schema", "count": 1}]
        ) from None
