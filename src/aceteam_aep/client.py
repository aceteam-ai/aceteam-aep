"""ChatClient protocol - the core abstraction replacing BaseChatModel."""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from .types import AepRequestContext, AepResponseMetadata, ChatMessage, ChatResponse, StreamChunk

_tool_origin: ContextVar[AepResponseMetadata | None] = ContextVar("aep_tool_origin", default=None)
_logger = logging.getLogger(__name__)


async def _run_cleanup_preserving_error(
    cleanup: Callable[[], Awaitable[Any]],
    primary: BaseException | None,
) -> None:
    try:
        await cleanup()
    except BaseException:
        if primary is None:
            raise
        _logger.warning("Stream cleanup also failed; preserving the primary error", exc_info=True)


@asynccontextmanager
async def _closing_preserving_error(close: Callable[[], Awaitable[Any]]) -> AsyncIterator[None]:
    try:
        yield
    finally:
        await _run_cleanup_preserving_error(close, sys.exception())


@asynccontextmanager
async def _response_context_preserving_error(context: Any) -> AsyncIterator[Any]:
    raw = await context.__aenter__()
    try:
        yield raw
    finally:
        primary = sys.exception()

        async def exit_response() -> None:
            await context.__aexit__(
                type(primary) if primary is not None else None,
                primary,
                primary.__traceback__ if primary is not None else None,
            )

        await _run_cleanup_preserving_error(exit_response, primary)


def current_tool_origin() -> AepResponseMetadata | None:
    """Gateway response that proposed the tool currently being invoked, if any."""
    return _tool_origin.get()


class GatewayResponseError(Exception):
    """A gateway body failed after its headers were received.

    ``response_metadata`` retains the exact allowlisted header values;
    ``response_complete`` is always false. The original error is chained.
    """

    def __init__(self, cause: Exception, metadata: AepResponseMetadata) -> None:
        super().__init__(str(cause))
        self.response_metadata = metadata
        self.response_complete = False


def _valid_header(value: str | None) -> str | None:
    if value and len(value) <= 128 and all(33 <= ord(char) <= 126 for char in value):
        return value
    return None


def _gateway_metadata(headers: Any) -> AepResponseMetadata:
    """Read only the three public AEP headers from a trusted raw response."""
    return AepResponseMetadata(
        call_id=_valid_header(headers.get("x-aep-call-id")),
        trace_id=_valid_header(headers.get("x-aep-trace-id")),
        entity=_valid_header(headers.get("x-aep-entity")),
    )


def _context_headers(context: AepRequestContext | None) -> dict[str, str]:
    if context is None:
        return {}
    headers = {}
    for name, value in (("X-AEP-Trace-ID", context.trace_id), ("X-AEP-Entity", context.entity)):
        if value is not None:
            if _valid_header(value) != value:
                raise ValueError(f"Invalid {name} request context")
            headers[name] = value
    return headers


def _validate_gateway_url(url: str) -> str:
    if any(ord(char) <= 32 or ord(char) == 127 for char in url):
        raise ValueError("trusted_gateway_url must be an HTTP(S) endpoint URL")
    parsed = urlsplit(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("trusted_gateway_url must be an HTTP(S) endpoint URL")
    return url


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

    def chat_stream(
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


@runtime_checkable
class ContextChatClient(ChatClient, Protocol):
    """Optional request context capability implemented by gateway clients."""

    async def chat_with_context(
        self,
        messages: list[ChatMessage],
        *,
        request_context: AepRequestContext,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse: ...

    def chat_stream_with_context(
        self,
        messages: list[ChatMessage],
        *,
        request_context: AepRequestContext,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]: ...


__all__ = ["ChatClient", "ContextChatClient", "GatewayResponseError", "current_tool_origin"]
