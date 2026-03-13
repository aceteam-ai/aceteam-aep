"""AEP span tracking - creates and manages execution spans."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .costs import CostNode


@dataclass
class Span:
    """An AEP execution span."""

    span_id: str
    executor_type: str
    executor_id: str
    parent_span_id: str | None = None
    status: Literal["OK", "ERROR", "CANCELLED", "BUDGET_EXCEEDED"] = "OK"
    started_at: str = ""
    ended_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    cost: CostNode | None = None

    @property
    def duration_ms(self) -> float | None:
        """Calculate duration in milliseconds."""
        if not self.ended_at or not self.started_at:
            return None
        start = datetime.fromisoformat(self.started_at)
        end = datetime.fromisoformat(self.ended_at)
        return (end - start).total_seconds() * 1000


class SpanTracker:
    """Tracks AEP spans during execution.

    Creates a tree of spans representing the execution hierarchy:
    - Root span for the agent loop
    - Child spans for each LLM call
    - Child spans for each tool invocation
    """

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex
        self._spans: dict[str, Span] = {}

    def start_span(
        self,
        executor_type: str,
        executor_id: str,
        parent_span_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Span:
        """Start a new span."""
        span = Span(
            span_id=uuid.uuid4().hex,
            executor_type=executor_type,
            executor_id=executor_id,
            parent_span_id=parent_span_id,
            started_at=datetime.now(UTC).isoformat(),
            metadata=metadata or {},
        )
        self._spans[span.span_id] = span
        return span

    def end_span(
        self,
        span_id: str,
        status: Literal["OK", "ERROR", "CANCELLED", "BUDGET_EXCEEDED"] = "OK",
        metadata: dict[str, Any] | None = None,
    ) -> Span:
        """End a span."""
        span = self._spans[span_id]
        span.ended_at = datetime.now(UTC).isoformat()
        span.status = status
        if metadata:
            span.metadata.update(metadata)
        return span

    def get_spans(self) -> list[Span]:
        """Get all spans."""
        return list(self._spans.values())

    def get_span(self, span_id: str) -> Span | None:
        """Get a specific span by ID."""
        return self._spans.get(span_id)


__all__ = ["Span", "SpanTracker"]
