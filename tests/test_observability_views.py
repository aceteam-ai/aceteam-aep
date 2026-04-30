"""Tests for observability view projections (timeline, traffic, topology)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aceteam_aep.observability import ObservabilityEvent, SqliteEventStore
from aceteam_aep.observability.views.timeline import get_timeline
from aceteam_aep.observability.views.topology import get_topology
from aceteam_aep.observability.views.traffic import get_traffic_stats


@pytest.fixture
async def store(tmp_path: Path) -> SqliteEventStore:
    s = SqliteEventStore(str(tmp_path / "views_test.db"))
    await s.initialize()

    # 5 OpenAI calls: call_0 through call_4
    for i in range(5):
        cid = f"call_{i}"
        # call_start
        await s.record(
            ObservabilityEvent(
                session_id="sess_1",
                type="call_start",
                call_id=cid,
                model="gpt-4o",
                provider="openai",
            )
        )
        # call_end
        await s.record(
            ObservabilityEvent(
                session_id="sess_1",
                type="call_end",
                call_id=cid,
                model="gpt-4o",
                provider="openai",
                latency_ms=500 + i * 100,
                tokens_in=10 + i,
                tokens_out=20 + i,
                cost_usd=0.001 * (i + 1),
            )
        )
        # enforcement: call_2 is flagged, rest pass
        if i == 2:
            await s.record(
                ObservabilityEvent(
                    session_id="sess_1",
                    type="safety_signal",
                    call_id=cid,
                    detector="pii",
                    severity="high",
                    reason="PII detected",
                )
            )
            await s.record(
                ObservabilityEvent(
                    session_id="sess_1",
                    type="enforcement",
                    call_id=cid,
                    action="flag",
                )
            )
        else:
            await s.record(
                ObservabilityEvent(
                    session_id="sess_1",
                    type="enforcement",
                    call_id=cid,
                    action="pass",
                )
            )

    # 1 Anthropic call
    await s.record(
        ObservabilityEvent(
            session_id="sess_1",
            type="call_start",
            call_id="call_anth",
            model="claude-3-haiku",
            provider="anthropic",
        )
    )
    await s.record(
        ObservabilityEvent(
            session_id="sess_1",
            type="call_end",
            call_id="call_anth",
            model="claude-3-haiku",
            provider="anthropic",
            latency_ms=300,
            tokens_in=15,
            tokens_out=25,
            cost_usd=0.002,
        )
    )
    await s.record(
        ObservabilityEvent(
            session_id="sess_1",
            type="enforcement",
            call_id="call_anth",
            action="pass",
        )
    )

    return s


async def test_timeline_returns_call_spans(store: SqliteEventStore) -> None:
    spans = await get_timeline(store, session_id="sess_1")
    assert len(spans) == 6
    for span in spans:
        assert "call_id" in span
        assert span["call_id"] is not None
        assert "started_at" in span
        assert span["started_at"] is not None


async def test_timeline_attaches_signals(store: SqliteEventStore) -> None:
    spans = await get_timeline(store, session_id="sess_1")
    flagged = [s for s in spans if s.get("action") == "flag"]
    assert len(flagged) == 1
    flagged_span = flagged[0]
    assert flagged_span["call_id"] == "call_2"
    assert len(flagged_span["signals"]) == 1
    assert flagged_span["signals"][0]["detector"] == "pii"


async def test_traffic_stats_per_model(store: SqliteEventStore) -> None:
    stats = await get_traffic_stats(store, session_id="sess_1")
    assert stats["total_calls"] == 6
    assert len(stats["by_model"]) == 2
    gpt_row = next(m for m in stats["by_model"] if m["model"] == "gpt-4o")
    assert gpt_row["calls"] == 5


async def test_traffic_stats_latency(store: SqliteEventStore) -> None:
    stats = await get_traffic_stats(store, session_id="sess_1")
    assert stats["latency_p50"] is not None
    assert stats["latency_p50"] > 0


async def test_topology_nodes(store: SqliteEventStore) -> None:
    topo = await get_topology(store, session_id="sess_1")
    provider_ids = {p["id"] for p in topo["providers"]}
    assert "openai" in provider_ids
    assert "anthropic" in provider_ids


async def test_topology_edges(store: SqliteEventStore) -> None:
    topo = await get_topology(store, session_id="sess_1")
    assert len(topo["edges"]) >= 2
