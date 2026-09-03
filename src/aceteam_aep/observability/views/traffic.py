"""Traffic view: aggregates call_end events into summary stats."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aceteam_aep.observability.store import EventStore


def _percentile(sorted_values: list[float], pct: float) -> float | None:
    """Return the pct-th percentile of an already-sorted list (0-100 scale)."""
    if not sorted_values:
        return None
    idx = (pct / 100) * (len(sorted_values) - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sorted_values):
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


async def get_traffic_stats(
    store: EventStore,
    *,
    session_id: str | None = None,
    since: str | None = None,
) -> dict[str, Any]:
    """Return aggregated traffic statistics from call_end events."""
    events = await store.query_events(
        session_id=session_id, since=since, type="call_end", limit=10000
    )

    total_calls = 0
    total_cost_usd = 0.0
    total_tokens_in = 0
    total_tokens_out = 0
    all_latencies: list[float] = []

    by_model: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0, "latencies": []}
    )

    for ev in events:
        total_calls += 1
        if ev.cost_usd is not None:
            total_cost_usd += ev.cost_usd
        if ev.tokens_in is not None:
            total_tokens_in += ev.tokens_in
        if ev.tokens_out is not None:
            total_tokens_out += ev.tokens_out
        if ev.latency_ms is not None:
            all_latencies.append(ev.latency_ms)

        model_key = ev.model or "unknown"
        m = by_model[model_key]
        m["calls"] += 1
        if ev.cost_usd is not None:
            m["cost_usd"] += ev.cost_usd
        if ev.tokens_in is not None:
            m["tokens_in"] += ev.tokens_in
        if ev.tokens_out is not None:
            m["tokens_out"] += ev.tokens_out
        if ev.latency_ms is not None:
            m["latencies"].append(ev.latency_ms)

    all_latencies.sort()

    model_rows: list[dict[str, Any]] = []
    for model_name, m in by_model.items():
        lats = sorted(m.pop("latencies"))
        model_rows.append(
            {
                "model": model_name,
                "calls": m["calls"],
                "cost_usd": m["cost_usd"],
                "tokens_in": m["tokens_in"],
                "tokens_out": m["tokens_out"],
                "latency_p50": _percentile(lats, 50),
                "latency_p95": _percentile(lats, 95),
            }
        )

    return {
        "total_calls": total_calls,
        "total_cost_usd": total_cost_usd,
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "latency_p50": _percentile(all_latencies, 50),
        "latency_p95": _percentile(all_latencies, 95),
        "latency_p99": _percentile(all_latencies, 99),
        "by_model": model_rows,
    }
