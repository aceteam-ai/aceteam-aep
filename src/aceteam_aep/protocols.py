"""AEP protocol interfaces for integration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from aceteam_aep.envelope import Citation


@runtime_checkable
class HasCitations(Protocol):
    """Objects that carry AEP citations."""

    citations: list[Citation] | None


@runtime_checkable
class UsageCollector(Protocol):
    """Async interface for capturing LLM usage data during execution.

    Implementers may accept the optional ``channel_id`` and ``pricing_snapshot``
    kwargs to attribute the call to a specific routing channel and snapshot the
    pricing in effect at call time. Both default to ``None`` so legacy
    callers and implementers continue to work without change.
    """

    async def record_usage(
        self,
        node_id: str,
        model_name: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        execution_time_ms: int,
        *,
        channel_id: str | None = None,
        pricing_snapshot: Mapping[str, Any] | None = None,
    ) -> None: ...


__all__ = ["HasCitations", "UsageCollector"]
