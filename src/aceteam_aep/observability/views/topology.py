"""Topology view: groups call_end events into agents, providers, and edges."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aceteam_aep.observability.store import EventStore


def _percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    idx = (pct / 100) * (len(sorted_values) - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sorted_values):
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def _health(p95_ms: float | None) -> str:
    if p95_ms is None:
        return "green"
    if p95_ms > 5000:
        return "red"
    if p95_ms > 2000:
        return "yellow"
    return "green"


async def get_topology(
    store: EventStore,
    *,
    session_id: str | None = None,
    since: str | None = None,
) -> dict[str, Any]:
    """Return topology graph: agents (sessions), providers, and edges between them."""
    events = await store.query_events(
        session_id=session_id, since=since, type="call_end", limit=10000
    )

    agent_calls: dict[str, int] = defaultdict(int)
    provider_calls: dict[str, int] = defaultdict(int)
    provider_cost: dict[str, float] = defaultdict(float)
    provider_latencies: dict[str, list[float]] = defaultdict(list)
    edge_calls: dict[tuple[str, str], int] = defaultdict(int)

    for ev in events:
        agent_id = ev.session_id
        provider_id = ev.provider or "unknown"

        agent_calls[agent_id] += 1
        provider_calls[provider_id] += 1

        if ev.cost_usd is not None:
            provider_cost[provider_id] += ev.cost_usd
        if ev.latency_ms is not None:
            provider_latencies[provider_id].append(ev.latency_ms)

        edge_calls[(agent_id, provider_id)] += 1

    agents = [
        {"id": sid, "name": sid, "calls": cnt}
        for sid, cnt in sorted(agent_calls.items(), key=lambda x: -x[1])
    ]

    providers: list[dict[str, Any]] = []
    for pid, cnt in sorted(provider_calls.items(), key=lambda x: -x[1]):
        lats = sorted(provider_latencies[pid])
        p95 = _percentile(lats, 95)
        providers.append(
            {
                "id": pid,
                "name": pid,
                "health": _health(p95),
                "calls": cnt,
                "total_cost_usd": provider_cost[pid],
                "p95_ms": p95,
            }
        )

    edges = [
        {"from": agent_id, "to": provider_id, "calls": cnt}
        for (agent_id, provider_id), cnt in sorted(edge_calls.items(), key=lambda x: -x[1])
    ]

    return {"agents": agents, "providers": providers, "edges": edges}
