"""AEP/JQS stream event types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .costs import CostNode
from .types import Usage

StreamEventType = Literal[
    "span_start",
    "span_end",
    "chunk",
    "tool_call_start",
    "tool_call_end",
    "cost",
    "budget_warning",
    "error",
    "end",
]


@dataclass
class StreamEvent:
    """A stream event emitted during agent execution."""

    type: StreamEventType
    data: dict[str, Any] = field(default_factory=dict)


def span_start_event(
    span_id: str,
    executor_type: str,
    executor_id: str,
    parent_span_id: str | None = None,
) -> StreamEvent:
    return StreamEvent(
        type="span_start",
        data={
            "span_id": span_id,
            "executor_type": executor_type,
            "executor_id": executor_id,
            "parent_span_id": parent_span_id,
        },
    )


def span_end_event(span_id: str, status: str = "OK") -> StreamEvent:
    return StreamEvent(
        type="span_end",
        data={"span_id": span_id, "status": status},
    )


def chunk_event(text: str) -> StreamEvent:
    return StreamEvent(type="chunk", data={"text": text})


def tool_call_start_event(tool_call_id: str, name: str, arguments: dict[str, Any]) -> StreamEvent:
    return StreamEvent(
        type="tool_call_start",
        data={
            "tool_call_id": tool_call_id,
            "name": name,
            "arguments": arguments,
        },
    )


def tool_call_end_event(tool_call_id: str, output: Any, error: str | None = None) -> StreamEvent:
    return StreamEvent(
        type="tool_call_end",
        data={
            "tool_call_id": tool_call_id,
            "output": output,
            "error": error,
        },
    )


def cost_event(cost_node: CostNode) -> StreamEvent:
    return StreamEvent(
        type="cost",
        data={
            "id": cost_node.id,
            "category": cost_node.category,
            "compute_cost": str(cost_node.compute_cost),
            "model": cost_node.metadata.get("model"),
        },
    )


def error_event(code: str, message: str) -> StreamEvent:
    return StreamEvent(type="error", data={"code": code, "message": message})


def end_event(usage: Usage | None = None) -> StreamEvent:
    data: dict[str, Any] = {}
    if usage:
        data["usage"] = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    return StreamEvent(type="end", data=data)


__all__ = [
    "StreamEvent",
    "StreamEventType",
    "chunk_event",
    "cost_event",
    "end_event",
    "error_event",
    "span_end_event",
    "span_start_event",
    "tool_call_end_event",
    "tool_call_start_event",
]
