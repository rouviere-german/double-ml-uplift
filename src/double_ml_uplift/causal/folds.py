"""Reproducible cross-fitting folds derived from a hash of the unit identifier.

Cross-fitting needs the same customer to land in the same fold every time a run is repeated,
including when rows arrive in a different order, when the sample is a subset, or when the work is
split across processes. A shuffled ``KFold`` gives none of that, so folds here are a pure
function of ``(salt, unit_id)``:

    fold(unit) = int(blake2b(salt || 0x00 || unit, 8 bytes)) mod n_folds

The digest of the whole assignment goes into the run manifest, so two runs that claim to be the
same run can be compared without shipping the assignment itself.

The salt keys the hash, so an attacker who can choose customer ids cannot steer them into a
chosen fold *unless the salt is known*. The bundled default is the literal ``"retention-offer"``
(``EstimationConfig.fold_salt``): it is committed to this repository and its digest is returned
in every ``/v1/estimate`` response, so at the default setting the keying buys reproducibility,
not unpredictability. Where fold assignment is adversarially relevant, set ``APP_FOLD_SALT`` to a
per-environment secret; the service passes it through to the estimation configuration, and only
its hash ever leaves the process.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["MAX_FOLDS", "FoldPlan", "assign_folds", "fold_digest", "make_fold_plan"]

IntArray = npt.NDArray[np.int64]
MAX_FOLDS = 20
_DIGEST_BYTES = 8


def _fold_of(unit_id: str, salt_bytes: bytes, n_folds: int) -> int:
    digest = hashlib.blake2b(
        unit_id.encode("utf-8"), digest_size=_DIGEST_BYTES, key=salt_bytes
    ).digest()
    return int.from_bytes(digest, "big") % n_folds


def assign_folds(unit_ids: Iterable[str], n_folds: int, *, salt: str = "") -> IntArray:
    """Assign each unit id to a fold in ``[0, n_folds)``. Order-independent and deterministic."""
    if not 1 <= n_folds <= MAX_FOLDS:
        raise ValueError(f"n_folds must be between 1 and {MAX_FOLDS}")
    salt_bytes = salt.encode("utf-8")[:64]
    if n_folds == 1:
        return np.zeros(sum(1 for _ in unit_ids), dtype=np.int64)
    return np.fromiter(
        (_fold_of(unit_id, salt_bytes, n_folds) for unit_id in unit_ids),
        dtype=np.int64,
    )


def fold_digest(unit_ids: Sequence[str], folds: IntArray, n_folds: int, *, salt: str = "") -> str:
    """SHA-256 over the (unit, fold) pairs in the order given, plus the salt and fold count."""
    if len(unit_ids) != folds.size:
        raise ValueError("unit ids and fold assignment differ in length")
    hasher = hashlib.sha256()
    hasher.update(f"folds/v1|{salt}|{n_folds}\n".encode())
    for unit_id, fold in zip(unit_ids, folds.tolist(), strict=True):
        hasher.update(f"{unit_id}\x00{fold}\n".encode())
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class FoldPlan:
    """A fold assignment plus everything needed to reproduce and audit it."""

    n_folds: int
    salt: str
    folds: IntArray
    digest: str

    def indices(self, fold: int) -> tuple[IntArray, IntArray]:
        """``(train, test)`` row indices for one fold. With ``n_folds == 1`` both are all rows."""
        if not 0 <= fold < self.n_folds:
            raise ValueError(f"fold {fold} is outside [0, {self.n_folds})")
        test = np.flatnonzero(self.folds == fold).astype(np.int64)
        if self.n_folds == 1:
            return test, test
        train = np.flatnonzero(self.folds != fold).astype(np.int64)
        return train, test

    def sizes(self) -> tuple[int, ...]:
        return tuple(int(count) for count in np.bincount(self.folds, minlength=self.n_folds))


def make_fold_plan(unit_ids: Sequence[str], n_folds: int, *, salt: str = "") -> FoldPlan:
    """Build a :class:`FoldPlan`, refusing an assignment that leaves a fold empty."""
    folds = assign_folds(unit_ids, n_folds, salt=salt)
    counts = np.bincount(folds, minlength=n_folds)
    empty = [index for index, count in enumerate(counts.tolist()) if count == 0]
    if empty:
        raise ValueError(f"fold(s) {empty} are empty; use fewer folds or more rows")
    return FoldPlan(
        n_folds=n_folds,
        salt=salt,
        folds=folds,
        digest=fold_digest(unit_ids, folds, n_folds, salt=salt),
    )
