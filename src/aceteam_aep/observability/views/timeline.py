"""Timeline view: joins call_start/call_end events into spans with signal overlays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aceteam_aep.observability.store import EventStore


async def get_timeline(
    store: EventStore,
    *,
    session_id: str | None = None,
    since: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Return call spans with attached safety signals, sorted by started_at ascending.

    Each span joins a call_start and call_end event by call_id and attaches any
    safety_signal and enforcement events as overlays.
    """
    events = await store.query_events(session_id=session_id, since=since, limit=5000)

    starts: dict[str, Any] = {}
    ends: dict[str, Any] = {}
    signals: dict[str, list[dict[str, Any]]] = {}
    actions: dict[str, str] = {}

    for ev in events:
        cid = ev.call_id
        if cid is None:
            continue
        if ev.type == "call_start":
            starts[cid] = ev
        elif ev.type == "call_end":
            ends[cid] = ev
        elif ev.type == "safety_signal":
            signals.setdefault(cid, []).append(
                {
                    "detector": ev.detector,
                    "severity": ev.severity,
                    "reason": ev.reason,
                }
            )
        elif ev.type == "enforcement" and ev.action is not None:
            actions[cid] = ev.action

    spans: list[dict[str, Any]] = []
    for cid, start_ev in starts.items():
        end_ev = ends.get(cid)
        span: dict[str, Any] = {
            "call_id": cid,
            "model": start_ev.model,
            "provider": start_ev.provider,
            "started_at": start_ev.timestamp,
            "ended_at": end_ev.timestamp if end_ev else None,
            "latency_ms": end_ev.latency_ms if end_ev else None,
            "tokens_in": end_ev.tokens_in if end_ev else None,
            "tokens_out": end_ev.tokens_out if end_ev else None,
            "cost_usd": end_ev.cost_usd if end_ev else None,
            "action": actions.get(cid),
            "signals": signals.get(cid, []),
        }
        spans.append(span)

    spans.sort(key=lambda s: s["started_at"] or "")
    return spans[:limit]
