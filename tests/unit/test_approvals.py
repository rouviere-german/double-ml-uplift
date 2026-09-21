"""The approval store: what it commits to, who may act, and what happens when it fills up."""

from __future__ import annotations

from pathlib import Path

import pytest

from double_ml_uplift.service.approvals import (
    ApprovalError,
    ApprovalStore,
    ApprovalStoreFullError,
    ExportPayload,
    payload_digest,
)

pytestmark = pytest.mark.unit


class Clock:
    """A hand-wound clock, so expiry can be reached without waiting for it."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def payload(*, ranking: str = "uplift", size: int = 3) -> ExportPayload:
    return ExportPayload(
        spec_sha256="a" * 64,
        config_sha256="b" * 64,
        dataset_sha256="c" * 64,
        ranking=ranking,
        budget_fraction=0.2,
        cost_per_offer=1.0,
        pseudonyms=tuple(f"cid_{index:040x}" for index in range(size)),
    )


def store(tmp_path: Path, **kwargs: object) -> ApprovalStore:
    return ApprovalStore(tmp_path / "approvals.json", **kwargs)  # type: ignore[arg-type]


def test_an_approval_commits_to_one_exact_payload(tmp_path: Path) -> None:
    subject = store(tmp_path)
    record = subject.request("operator-a", payload())
    assert record.payload_sha256 == payload_digest(payload())
    subject.decide(record.request_id, "operator-b", approve=True)

    with pytest.raises(ApprovalError, match="does not match"):
        subject.consume(record.request_id, "operator-a", payload(ranking="propensity"))
    assert subject.consume(record.request_id, "operator-a", payload()).status == "consumed"


def test_an_unknown_request_id_is_refused_rather_than_returning_nothing(tmp_path: Path) -> None:
    subject = store(tmp_path)
    with pytest.raises(ApprovalError, match="unknown export request"):
        subject.get("exp_" + "0" * 32)


def test_a_flood_of_requests_cannot_evict_somebody_elses_pending_approval(
    tmp_path: Path,
) -> None:
    """The availability failure the cap could otherwise cause.

    Pruning by age alone would let any operator push another operator's pending or approved
    record out of a full store, and the release would then fail with "unknown export request".
    Refusing the new request keeps the existing one answerable.
    """
    subject = store(tmp_path, max_records=3)
    mine = subject.request("operator-a", payload())
    subject.decide(mine.request_id, "operator-b", approve=True)
    others = [subject.request("operator-c", payload(size=size)) for size in (4, 5)]

    with pytest.raises(ApprovalStoreFullError, match="3 live export requests"):
        subject.request("operator-c", payload(size=6))

    assert subject.get(mine.request_id).status == "approved"
    for record in others:
        assert subject.get(record.request_id).status == "pending"
    # And the approval it protected is still spendable.
    assert subject.consume(mine.request_id, "operator-a", payload()).status == "consumed"


def test_spent_records_are_pruned_before_live_ones(tmp_path: Path) -> None:
    """Consumed and denied records give up their slots first; they cannot be acted on again."""
    subject = store(tmp_path, max_records=3)
    denied = subject.request("operator-a", payload(size=1))
    subject.decide(denied.request_id, "operator-b", approve=False)

    live = [subject.request("operator-a", payload(size=size)) for size in (2, 3, 4)]

    with pytest.raises(ApprovalError, match="unknown export request"):
        subject.get(denied.request_id)
    for record in live:
        assert subject.get(record.request_id).status == "pending"


def test_an_expired_record_does_not_hold_a_slot(tmp_path: Path) -> None:
    clock = Clock()
    subject = store(tmp_path, max_records=2, ttl_seconds=60.0, clock=clock)
    stale = [subject.request("operator-a", payload(size=size)) for size in (1, 2)]

    with pytest.raises(ApprovalStoreFullError):
        subject.request("operator-a", payload(size=3))

    clock.advance(61.0)
    fresh = [subject.request("operator-a", payload(size=size)) for size in (3, 4)]

    # Two live slots, both now held by the fresh requests; the expired pair gave theirs up.
    for record in fresh:
        assert subject.get(record.request_id).status == "pending"
    for record in stale:
        with pytest.raises(ApprovalError, match="unknown export request"):
            subject.get(record.request_id)


def test_an_expired_record_that_is_still_stored_is_refused_on_its_own_terms(
    tmp_path: Path,
) -> None:
    """Expiry is checked by ``decide`` and ``consume``, independently of the cap."""
    clock = Clock()
    subject = store(tmp_path, max_records=10, ttl_seconds=60.0, clock=clock)
    record = subject.request("operator-a", payload())
    clock.advance(61.0)

    assert subject.get(record.request_id).status == "pending"
    with pytest.raises(ApprovalError, match="expired"):
        subject.decide(record.request_id, "operator-b", approve=True)


def test_the_cap_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_records must be positive"):
        store(tmp_path, max_records=0)
    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        store(tmp_path, ttl_seconds=0.0)


def test_an_oversized_store_converges_back_to_the_cap(tmp_path: Path) -> None:
    """A store restored from an oversized file shrinks instead of growing forever.

    ``request`` refuses before the live set can exceed the cap, so this is the recovery path
    rather than the normal one; it is here so the file has a bound under every history.
    """
    subject = store(tmp_path, max_records=6)
    records = [subject.request("operator-a", payload(size=size)) for size in range(1, 7)]
    shrunk = ApprovalStore(subject.path, max_records=2)

    # Any write re-applies the smaller cap.
    shrunk.decide(records[-1].request_id, "operator-b", approve=True)

    remaining = [record for record in records if _present(shrunk, record.request_id)]
    assert len(remaining) == 2
    assert records[-1].request_id in {record.request_id for record in remaining}


def _present(subject: ApprovalStore, request_id: str) -> bool:
    try:
        subject.get(request_id)
    except ApprovalError:
        return False
    return True
