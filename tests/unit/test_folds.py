"""Cross-fitting folds are a pure function of ``(salt, unit id)``, and the digest proves it."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from double_ml_uplift.causal.folds import (
    MAX_FOLDS,
    assign_folds,
    fold_digest,
    make_fold_plan,
)

pytestmark = pytest.mark.unit

_ids = st.lists(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", min_size=1, max_size=24),
    min_size=1,
    max_size=200,
    unique=True,
)


def ids(n: int, *, prefix: str = "cust") -> list[str]:
    return [f"{prefix}_{index:06d}" for index in range(n)]


# --- purity ---------------------------------------------------------------------------------


@settings(max_examples=150)
@given(
    unit_ids=_ids,
    n_folds=st.integers(min_value=1, max_value=MAX_FOLDS),
    salt=st.text(max_size=20),
)
def test_a_units_fold_never_depends_on_the_other_units(
    unit_ids: list[str], n_folds: int, salt: str
) -> None:
    folds = assign_folds(unit_ids, n_folds, salt=salt)
    assert folds.shape == (len(unit_ids),)
    assert np.all((folds >= 0) & (folds < n_folds))

    # Reordering the input permutes the output the same way...
    order = np.argsort(unit_ids)
    reordered = [unit_ids[index] for index in order]
    assert np.array_equal(assign_folds(reordered, n_folds, salt=salt), folds[order])

    # ...and dropping rows leaves the survivors where they were.
    keep = list(range(0, len(unit_ids), 2))
    subset = [unit_ids[index] for index in keep]
    assert np.array_equal(assign_folds(subset, n_folds, salt=salt), folds[keep])


@settings(max_examples=100)
@given(unit_ids=_ids, n_folds=st.integers(min_value=2, max_value=MAX_FOLDS))
def test_assignment_is_deterministic_across_calls(unit_ids: list[str], n_folds: int) -> None:
    assert np.array_equal(
        assign_folds(unit_ids, n_folds, salt="a"), assign_folds(unit_ids, n_folds, salt="a")
    )


def test_a_different_salt_reassigns_a_meaningful_share_of_units() -> None:
    unit_ids = ids(4_000)
    first = assign_folds(unit_ids, 5, salt="one")
    second = assign_folds(unit_ids, 5, salt="two")
    moved = float(np.mean(first != second))
    # With five folds, an independent reassignment moves 80% of units. Anything far below that
    # would mean the salt is barely keying the hash.
    assert 0.7 < moved < 0.9


def test_folds_are_roughly_balanced_on_a_realistic_id_space() -> None:
    folds = assign_folds(ids(10_000), 5, salt="balance")
    counts = np.bincount(folds, minlength=5)
    assert counts.min() > 1_800
    assert counts.max() < 2_200


# --- guards ----------------------------------------------------------------------------------


@pytest.mark.parametrize("n_folds", [0, -1, MAX_FOLDS + 1])
def test_the_fold_count_is_bounded(n_folds: int) -> None:
    with pytest.raises(ValueError, match="n_folds must be between"):
        assign_folds(ids(10), n_folds)


def test_one_fold_disables_cross_fitting() -> None:
    plan = make_fold_plan(ids(9), 1, salt="s")
    assert plan.sizes() == (9,)
    train, test = plan.indices(0)
    assert np.array_equal(train, test)
    assert train.size == 9


def test_an_empty_fold_is_refused_rather_than_silently_skipped() -> None:
    with pytest.raises(ValueError, match="are empty"):
        make_fold_plan(ids(3), 20, salt="s")


def test_indices_outside_the_plan_are_refused() -> None:
    plan = make_fold_plan(ids(200), 4, salt="s")
    with pytest.raises(ValueError, match="outside"):
        plan.indices(4)
    with pytest.raises(ValueError, match="outside"):
        plan.indices(-1)


def test_train_and_test_partition_the_rows_exactly() -> None:
    plan = make_fold_plan(ids(500), 5, salt="s")
    seen: set[int] = set()
    for fold in range(plan.n_folds):
        train, test = plan.indices(fold)
        assert set(train.tolist()).isdisjoint(test.tolist())
        assert train.size + test.size == 500
        seen |= set(test.tolist())
    assert seen == set(range(500))
    assert sum(plan.sizes()) == 500


# --- the digest --------------------------------------------------------------------------------


def test_the_digest_commits_to_the_units_the_folds_the_salt_and_the_count() -> None:
    unit_ids = ids(64)
    folds = assign_folds(unit_ids, 4, salt="x")
    base = fold_digest(unit_ids, folds, 4, salt="x")

    assert base == fold_digest(unit_ids, folds, 4, salt="x")
    assert base != fold_digest(unit_ids, folds, 4, salt="y")
    assert base != fold_digest(unit_ids, folds, 5, salt="x")
    assert base != fold_digest(ids(64, prefix="other"), folds, 4, salt="x")
    moved = folds.copy()
    moved[0] = (moved[0] + 1) % 4
    assert base != fold_digest(unit_ids, moved, 4, salt="x")
    assert len(base) == 64


def test_the_digest_refuses_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        fold_digest(ids(4), assign_folds(ids(5), 2), 2)


def test_make_fold_plan_records_everything_needed_to_reproduce_it() -> None:
    plan = make_fold_plan(ids(300), 3, salt="reproducible")
    assert plan.n_folds == 3
    assert plan.salt == "reproducible"
    assert plan.digest == fold_digest(ids(300), plan.folds, 3, salt="reproducible")
    assert make_fold_plan(ids(300), 3, salt="reproducible").digest == plan.digest


def test_a_long_salt_is_truncated_to_a_bounded_key() -> None:
    unit_ids = ids(50)
    long_salt = "k" * 200
    assert np.array_equal(
        assign_folds(unit_ids, 5, salt=long_salt),
        assign_folds(unit_ids, 5, salt="k" * 64),
    )
