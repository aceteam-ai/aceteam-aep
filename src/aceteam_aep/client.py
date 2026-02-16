"""ChatClient protocol - the core abstraction replacing BaseChatModel."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from .types import ChatMessage, ChatResponse, StreamChunk


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for LLM chat clients.

    Each provider implements this protocol to provide a uniform interface
    for both streaming and non-streaming chat completions.
    """

    @property
    def model_name(self) -> str:
        """The model identifier."""
        ...

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Send a chat completion request.

        Args:
            messages: The conversation messages.
            tools: JSON Schema tool definitions for function calling.
            temperature: Sampling temperature override.
            max_tokens: Max output tokens override.
            response_format: Structured output format (JSON schema).

        Returns:
            ChatResponse with the model's reply and usage stats.
        """
        ...

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion response.

        Args:
            messages: The conversation messages.
            tools: JSON Schema tool definitions for function calling.
            temperature: Sampling temperature override.
            max_tokens: Max output tokens override.

        Yields:
            StreamChunk with delta text, tool calls, and usage.
        """
        ...


__all__ = ["ChatClient"]
