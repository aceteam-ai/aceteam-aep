"""AEP protocol interfaces for integration."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aceteam_aep.envelope import Citation


@runtime_checkable
class HasCitations(Protocol):
    """Objects that carry AEP citations."""

    citations: list[Citation] | None


@runtime_checkable
class UsageCollector(Protocol):
    """Async interface for capturing LLM usage data during execution."""

    async def record_usage(
        self,
        node_id: str,
        model_name: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        execution_time_ms: int,
    ) -> None: ...


__all__ = ["HasCitations", "UsageCollector"]
