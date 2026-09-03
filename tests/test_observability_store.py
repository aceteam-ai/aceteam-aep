"""Tests for SqliteEventStore."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from aceteam_aep.observability import FlaggedCall, ObservabilityEvent, SqliteEventStore


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteEventStore]:
    s = SqliteEventStore(str(tmp_path / "test.db"))
    await s.initialize()
    yield s
    await s.close()


def _make_event(**kwargs) -> ObservabilityEvent:
    defaults = dict(session_id="sess-1", type="call_end", call_id="call-1")
    return ObservabilityEvent(**(defaults | kwargs))


def _make_flagged(**kwargs) -> FlaggedCall:
    defaults = dict(
        call_id="call-1",
        session_id="sess-1",
        action="flag",
        input_messages=[{"role": "user", "content": "hello"}],
    )
    return FlaggedCall(**(defaults | kwargs))


async def test_record_and_query_event(store: SqliteEventStore) -> None:
    event = _make_event(model="gpt-4o", tokens_in=100, tokens_out=50)
    await store.record(event)

    results = await store.query_events(session_id="sess-1")
    assert len(results) == 1
    r = results[0]
    assert r.id == event.id
    assert r.session_id == "sess-1"
    assert r.type == "call_end"
    assert r.model == "gpt-4o"
    assert r.tokens_in == 100
    assert r.tokens_out == 50


async def test_query_events_by_type(store: SqliteEventStore) -> None:
    await store.record(_make_event(type="call_start"))
    await store.record(_make_event(type="call_end"))
    await store.record(_make_event(type="safety_signal"))

    results = await store.query_events(type="call_end")
    assert len(results) == 1
    assert results[0].type == "call_end"


async def test_query_events_by_call_id(store: SqliteEventStore) -> None:
    await store.record(_make_event(call_id="call-a", type="call_end"))
    await store.record(_make_event(call_id="call-b", type="call_end"))

    results = await store.query_events(call_id="call-a")
    assert len(results) == 1
    assert results[0].call_id == "call-a"


async def test_query_events_with_limit(store: SqliteEventStore) -> None:
    for _ in range(10):
        await store.record(_make_event())

    results = await store.query_events(limit=3)
    assert len(results) == 3


async def test_record_and_query_flagged_call(store: SqliteEventStore) -> None:
    msgs = [{"role": "user", "content": "do something risky"}]
    call = _make_flagged(
        call_id="call-99",
        session_id="sess-2",
        action="block",
        detector="pii",
        severity="high",
        input_messages=msgs,
        output_text="blocked",
    )
    await store.record_flagged_call(call)

    results = await store.query_flagged_calls(session_id="sess-2")
    assert len(results) == 1
    r = results[0]
    assert r.id == call.id
    assert r.call_id == "call-99"
    assert r.action == "block"
    assert r.detector == "pii"
    assert r.severity == "high"
    assert r.output_text == "blocked"
    assert r.input_messages == msgs  # deserialized from JSON


async def test_query_flagged_calls_unreviewed(store: SqliteEventStore) -> None:
    await store.record_flagged_call(_make_flagged(call_id="c1"))
    await store.record_flagged_call(_make_flagged(call_id="c2"))

    results = await store.query_flagged_calls(verdict="unreviewed")
    assert len(results) == 2
    assert all(r.verdict is None for r in results)


async def test_update_verdict(store: SqliteEventStore) -> None:
    call = _make_flagged(call_id="call-v")
    await store.record_flagged_call(call)

    await store.update_verdict("call-v", "confirmed", "reviewer@example.com", "looks bad")

    results = await store.query_flagged_calls(verdict="confirmed")
    assert len(results) == 1
    r = results[0]
    assert r.verdict == "confirmed"
    assert r.verdict_by == "reviewer@example.com"
    assert r.verdict_note == "looks bad"
    assert r.verdict_at is not None


async def test_record_failure_does_not_raise(store: SqliteEventStore) -> None:
    await store.close()
    # record() must log the error but never raise
    event = _make_event()
    await store.record(event)  # should not raise


async def test_db_auto_creates(tmp_path: Path) -> None:
    nested = tmp_path / "deeply" / "nested" / "subdir" / "obs.db"
    s = SqliteEventStore(str(nested))
    await s.initialize()
    try:
        assert nested.exists()
    finally:
        await s.close()
