"""Pydantic models for proxy observability events."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def _default_id() -> str:
    return uuid4().hex


def _default_timestamp() -> str:
    return datetime.now(tz=UTC).isoformat()


class ObservabilityEvent(BaseModel):
    """A single event in the proxy request lifecycle."""

    id: str = Field(default_factory=_default_id)
    timestamp: str = Field(default_factory=_default_timestamp)
    session_id: str
    type: Literal["call_start", "call_end", "safety_signal", "enforcement", "cost"]
    call_id: str | None = None
    model: str | None = None
    provider: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    action: Literal["pass", "flag", "block"] | None = None
    detector: str | None = None
    severity: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] | None = None


class FlaggedCall(BaseModel):
    """Full payload captured on FLAG/BLOCK for human review."""

    id: str = Field(default_factory=_default_id)
    call_id: str
    session_id: str
    timestamp: str = Field(default_factory=_default_timestamp)
    action: Literal["pass", "flag", "block"] = "flag"
    detector: str | None = None
    severity: str | None = None
    reason: str | None = None
    model: str | None = None
    input_messages: list[dict[str, Any]] = Field(default_factory=list)
    output_text: str | None = None
    verdict: Literal["confirmed", "dismissed"] | None = None
    verdict_by: str | None = None
    verdict_at: str | None = None
    verdict_note: str | None = None
    metadata: dict[str, Any] | None = None
