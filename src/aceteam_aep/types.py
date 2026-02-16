"""Core types for aceteam-aep."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ContentBlock:
    """A block of content in a message (text, image, etc.)."""

    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: str | None = None
    detail: str | None = None  # For image_url: "auto", "low", "high"


@dataclass
class ToolCallRequest:
    """A tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatMessage:
    """Universal chat message replacing all LangChain message types."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentBlock]
    tool_calls: list[ToolCallRequest] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @property
    def text(self) -> str:
        """Extract text content from message."""
        if isinstance(self.content, str):
            return self.content
        return "".join(block.text or "" for block in self.content if block.type == "text")


@dataclass
class Usage:
    """Token usage statistics from an LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass
class ChatResponse:
    """Response from a non-streaming chat call."""

    message: ChatMessage
    usage: Usage
    model: str
    finish_reason: str | None = None


@dataclass
class StreamChunk:
    """A chunk from a streaming chat response."""

    delta_text: str = ""
    delta_tool_calls: list[ToolCallRequest] | None = None
    usage: Usage | None = None
    finish_reason: str | None = None
    model: str | None = None


@dataclass
class AgentResult:
    """Result of a complete agent loop execution."""

    messages: list[ChatMessage]
    usage: Usage = field(default_factory=Usage)
    iterations: int = 0


__all__ = [
    "AgentResult",
    "ChatMessage",
    "ChatResponse",
    "ContentBlock",
    "StreamChunk",
    "ToolCallRequest",
    "Usage",
]
