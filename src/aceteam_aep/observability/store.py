"""SQLite-backed event store for proxy observability data."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import aiosqlite

from .events import FlaggedCall, ObservabilityEvent

logger = logging.getLogger(__name__)

_CREATE_EVENTS = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, session_id TEXT NOT NULL,
    type TEXT NOT NULL, call_id TEXT, model TEXT, provider TEXT,
    tokens_in INTEGER, tokens_out INTEGER, cost_usd REAL, latency_ms REAL,
    action TEXT, detector TEXT, severity TEXT, reason TEXT, metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
CREATE INDEX IF NOT EXISTS idx_events_call_id ON events(call_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
"""

_CREATE_FLAGGED_CALLS = """
CREATE TABLE IF NOT EXISTS flagged_calls (
    id TEXT PRIMARY KEY, call_id TEXT NOT NULL, session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL, action TEXT NOT NULL, detector TEXT, severity TEXT,
    reason TEXT, model TEXT, input_messages TEXT NOT NULL, output_text TEXT,
    verdict TEXT, verdict_by TEXT, verdict_at TEXT, verdict_note TEXT, metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_flagged_session ON flagged_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_flagged_verdict ON flagged_calls(verdict);
CREATE INDEX IF NOT EXISTS idx_flagged_detector ON flagged_calls(detector);
"""

_DEFAULT_DB_PATH = Path.home() / ".config" / "aceteam-aep" / "observability.db"


@runtime_checkable
class EventStore(Protocol):
    """Protocol for observability event persistence backends."""

    async def record(self, event: ObservabilityEvent) -> None: ...

    async def record_flagged_call(self, call: FlaggedCall) -> None: ...

    async def query_events(
        self,
        *,
        session_id: str | None = None,
        type: str | None = None,
        since: str | None = None,
        until: str | None = None,
        call_id: str | None = None,
        limit: int = 1000,
    ) -> list[ObservabilityEvent]: ...

    async def query_flagged_calls(
        self,
        *,
        session_id: str | None = None,
        verdict: str | None = None,
        limit: int = 100,
    ) -> list[FlaggedCall]: ...

    async def update_verdict(
        self,
        call_id: str,
        verdict: str,
        verdict_by: str,
        verdict_note: str | None = None,
    ) -> None: ...


class SqliteEventStore:
    """SQLite-backed implementation of EventStore using aiosqlite."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else _DEFAULT_DB_PATH
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create the DB file and tables if they do not exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_CREATE_EVENTS)
        await self._conn.executescript(_CREATE_FLAGGED_CALLS)
        await self._conn.commit()

    async def close(self) -> None:
        """Close the underlying database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def record(self, event: ObservabilityEvent) -> None:
        """Insert an ObservabilityEvent. Never raises — logs on failure."""
        try:
            assert self._conn is not None
            await self._conn.execute(
                """
                INSERT INTO events (
                    id, timestamp, session_id, type, call_id, model, provider,
                    tokens_in, tokens_out, cost_usd, latency_ms,
                    action, detector, severity, reason, metadata
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    event.id,
                    event.timestamp,
                    event.session_id,
                    event.type,
                    event.call_id,
                    event.model,
                    event.provider,
                    event.tokens_in,
                    event.tokens_out,
                    event.cost_usd,
                    event.latency_ms,
                    event.action,
                    event.detector,
                    event.severity,
                    event.reason,
                    json.dumps(event.metadata) if event.metadata is not None else None,
                ),
            )
            await self._conn.commit()
        except Exception:
            logger.warning("Failed to record observability event", exc_info=True)

    async def record_flagged_call(self, call: FlaggedCall) -> None:
        """Insert a FlaggedCall. Never raises — logs on failure."""
        try:
            assert self._conn is not None
            await self._conn.execute(
                """
                INSERT INTO flagged_calls (
                    id, call_id, session_id, timestamp, action, detector, severity,
                    reason, model, input_messages, output_text,
                    verdict, verdict_by, verdict_at, verdict_note, metadata
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    call.id,
                    call.call_id,
                    call.session_id,
                    call.timestamp,
                    call.action,
                    call.detector,
                    call.severity,
                    call.reason,
                    call.model,
                    json.dumps(call.input_messages),
                    call.output_text,
                    call.verdict,
                    call.verdict_by,
                    call.verdict_at,
                    call.verdict_note,
                    json.dumps(call.metadata) if call.metadata is not None else None,
                ),
            )
            await self._conn.commit()
        except Exception:
            logger.warning("Failed to record flagged call", exc_info=True)

    async def query_events(
        self,
        *,
        session_id: str | None = None,
        type: str | None = None,
        since: str | None = None,
        until: str | None = None,
        call_id: str | None = None,
        limit: int = 1000,
    ) -> list[ObservabilityEvent]:
        """Return events matching the given filters, newest first."""
        assert self._conn is not None
        clauses: list[str] = []
        params: list[Any] = []

        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if type is not None:
            clauses.append("type = ?")
            params.append(type)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            clauses.append("timestamp <= ?")
            params.append(until)
        if call_id is not None:
            clauses.append("call_id = ?")
            params.append(call_id)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        rows = await self._conn.execute_fetchall(
            f"SELECT * FROM events {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        )
        return [_row_to_event(dict(row)) for row in rows]

    async def query_flagged_calls(
        self,
        *,
        session_id: str | None = None,
        verdict: str | None = None,
        limit: int = 100,
    ) -> list[FlaggedCall]:
        """Return flagged calls matching the given filters."""
        assert self._conn is not None
        clauses: list[str] = []
        params: list[Any] = []

        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if verdict is not None:
            if verdict == "unreviewed":
                clauses.append("verdict IS NULL")
            else:
                clauses.append("verdict = ?")
                params.append(verdict)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        rows = await self._conn.execute_fetchall(
            f"SELECT * FROM flagged_calls {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        )
        return [_row_to_flagged_call(dict(row)) for row in rows]

    async def update_verdict(
        self,
        call_id: str,
        verdict: str,
        verdict_by: str,
        verdict_note: str | None = None,
    ) -> None:
        """Set the human verdict on a flagged call."""
        assert self._conn is not None
        verdict_at = datetime.now(tz=UTC).isoformat()
        await self._conn.execute(
            """
            UPDATE flagged_calls
            SET verdict = ?, verdict_by = ?, verdict_at = ?, verdict_note = ?
            WHERE call_id = ?
            """,
            (verdict, verdict_by, verdict_at, verdict_note, call_id),
        )
        await self._conn.commit()


# ---------------------------------------------------------------------------
# Row deserializers
# ---------------------------------------------------------------------------


def _row_to_event(row: dict[str, Any]) -> ObservabilityEvent:
    if row.get("metadata") is not None:
        row["metadata"] = json.loads(row["metadata"])
    return ObservabilityEvent.model_validate(row)


def _row_to_flagged_call(row: dict[str, Any]) -> FlaggedCall:
    row["input_messages"] = json.loads(row["input_messages"])
    if row.get("metadata") is not None:
        row["metadata"] = json.loads(row["metadata"])
    return FlaggedCall.model_validate(row)
