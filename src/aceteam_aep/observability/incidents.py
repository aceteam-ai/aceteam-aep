"""Incident bundle export — builds a downloadable zip archive for a flagged call."""

from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from io import BytesIO

from .events import FlaggedCall, ObservabilityEvent
from .store import SqliteEventStore


async def build_incident_bundle(store: SqliteEventStore, call_id: str) -> bytes | None:
    """Return a zip archive for the flagged call, or None if call_id is not found.

    The archive contains:
      aep_incident_YYYYMMDD_HHMMSS/
        summary.md        -- human-readable incident report
        flagged_call.json -- full FlaggedCall model as JSON
        events.json       -- surrounding session events (up to 500)
        manifest.json     -- metadata (call_id, session_id, exported_at, event_count)
    """
    flagged_calls = await store.query_flagged_calls()
    fc: FlaggedCall | None = next((c for c in flagged_calls if c.call_id == call_id), None)
    if fc is None:
        return None

    events: list[ObservabilityEvent] = await store.query_events(
        session_id=fc.session_id, limit=500
    )

    exported_at = datetime.now(tz=UTC)
    dir_name = f"aep_incident_{exported_at.strftime('%Y%m%d_%H%M%S')}"

    summary = _build_summary(fc, events)
    flagged_call_json = fc.model_dump_json(indent=2)
    events_json = json.dumps([e.model_dump() for e in events], indent=2)
    manifest = json.dumps(
        {
            "call_id": fc.call_id,
            "session_id": fc.session_id,
            "exported_at": exported_at.isoformat(),
            "event_count": len(events),
        },
        indent=2,
    )

    buf = BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{dir_name}/summary.md", summary)
        zf.writestr(f"{dir_name}/flagged_call.json", flagged_call_json)
        zf.writestr(f"{dir_name}/events.json", events_json)
        zf.writestr(f"{dir_name}/manifest.json", manifest)

    return buf.getvalue()


def _build_summary(fc: FlaggedCall, events: list[ObservabilityEvent]) -> str:
    lines: list[str] = [
        "# AEP Incident Report",
        "",
        f"**Call ID:** {fc.call_id}",
        f"**Session ID:** {fc.session_id}",
        f"**Timestamp:** {fc.timestamp}",
        f"**Action:** {fc.action}",
        f"**Detector:** {fc.detector or 'N/A'}",
        f"**Severity:** {fc.severity or 'N/A'}",
        f"**Reason:** {fc.reason or 'N/A'}",
        f"**Model:** {fc.model or 'N/A'}",
        "",
        f"**Surrounding Events:** {len(events)}",
        "",
    ]

    if fc.output_text:
        preview = fc.output_text[:500]
        truncated = len(fc.output_text) > 500
        lines += [
            "## Output Preview",
            "",
            "```",
            preview + ("..." if truncated else ""),
            "```",
            "",
        ]

    return "\n".join(lines)
