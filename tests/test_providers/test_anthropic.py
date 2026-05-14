"""Tests for the Anthropic provider — format conversion and stream guards."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from aceteam_aep.providers import ProviderResponseError
from aceteam_aep.providers.anthropic import AnthropicClient


class _FakeAsyncStream:
    """Minimal stand-in for ``anthropic.AsyncMessageStream``.

    The real SDK returns an async context manager that yields raw stream
    events. We only need to drive ``chat_stream`` with a scripted event
    list, so we re-implement the protocol manually.
    """

    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def __aenter__(self) -> _FakeAsyncStream:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[Any]:
        async def _gen() -> AsyncIterator[Any]:
            for event in self._events:
                yield event

        return _gen()


def _make_client(events: list[Any]) -> AnthropicClient:
    client = AnthropicClient(api_key="test", model="claude-test")
    client._client = MagicMock()
    client._client.messages.stream = MagicMock(return_value=_FakeAsyncStream(events))
    return client


def _text_event(text: str) -> MagicMock:
    event = MagicMock()
    event.type = "content_block_delta"
    event.delta = MagicMock()
    event.delta.text = text
    # The provider uses ``hasattr(event.delta, "partial_json")`` to
    # discriminate text vs. tool-input deltas — ensure the attribute
    # really is absent here.
    del event.delta.partial_json
    return event


def _finish_event(stop_reason: str = "end_turn") -> MagicMock:
    event = MagicMock()
    event.type = "message_delta"
    event.delta = MagicMock()
    event.delta.stop_reason = stop_reason
    event.usage = MagicMock()
    event.usage.output_tokens = 12
    return event


def _tool_use_start_event(tool_id: str, tool_name: str) -> MagicMock:
    event = MagicMock()
    event.type = "content_block_start"
    event.content_block = MagicMock()
    event.content_block.type = "tool_use"
    event.content_block.id = tool_id
    event.content_block.name = tool_name
    return event


def _tool_input_delta_event(partial_json: str) -> MagicMock:
    event = MagicMock()
    event.type = "content_block_delta"
    event.delta = MagicMock()
    # Mirror the real Anthropic SDK: a tool-input delta has
    # ``partial_json`` and no ``text``. The provider distinguishes
    # via hasattr on each, so explicitly delete the wrong-shape attrs.
    event.delta.partial_json = partial_json
    del event.delta.text
    return event


def _content_block_stop_event() -> MagicMock:
    event = MagicMock()
    event.type = "content_block_stop"
    return event


@pytest.mark.asyncio
async def test_chat_stream_raises_on_empty_stream() -> None:
    """Stream with zero events == upstream silent rejection."""
    client = _make_client(events=[])

    with pytest.raises(ProviderResponseError) as exc_info:
        async for _ in client.chat_stream(messages=[]):
            pass

    assert exc_info.value.provider == "anthropic"
    assert "claude-test" in str(exc_info.value)
    assert exc_info.value.user_message  # always present


@pytest.mark.asyncio
async def test_chat_stream_does_not_raise_on_text() -> None:
    """A normal text response must not trip the empty-stream guard."""
    client = _make_client(events=[_text_event("hello"), _finish_event()])

    chunks: list[Any] = []
    async for chunk in client.chat_stream(messages=[]):
        chunks.append(chunk)

    # delta_text + finish_reason chunks
    assert any(c.delta_text == "hello" for c in chunks)
    assert any(c.finish_reason == "end_turn" for c in chunks)


@pytest.mark.asyncio
async def test_chat_stream_does_not_raise_when_only_finish_reason() -> None:
    """A finish-reason without text is rare but legal (e.g., refusal). Don't raise."""
    client = _make_client(events=[_finish_event(stop_reason="end_turn")])

    chunks: list[Any] = []
    async for chunk in client.chat_stream(messages=[]):
        chunks.append(chunk)

    assert any(c.finish_reason == "end_turn" for c in chunks)


@pytest.mark.asyncio
async def test_chat_stream_does_not_raise_on_tool_only_response() -> None:
    """A tool-only response (no text) is a normal path when tools are
    enabled. The empty-stream guard must not misfire here.
    """
    client = _make_client(
        events=[
            _tool_use_start_event(tool_id="toolu_1", tool_name="search"),
            _tool_input_delta_event(partial_json='{"query": '),
            _tool_input_delta_event(partial_json='"hello"}'),
            _content_block_stop_event(),
            _finish_event(stop_reason="tool_use"),
        ]
    )

    chunks: list[Any] = []
    async for chunk in client.chat_stream(messages=[]):
        chunks.append(chunk)

    # No text deltas — but the tool call must be emitted as a StreamChunk
    # with parsed arguments, and the run must not raise.
    tool_chunks = [c for c in chunks if c.delta_tool_calls]
    assert len(tool_chunks) == 1
    tool_call = tool_chunks[0].delta_tool_calls[0]
    assert tool_call.id == "toolu_1"
    assert tool_call.name == "search"
    assert tool_call.arguments == {"query": "hello"}
    assert any(c.finish_reason == "tool_use" for c in chunks)


def test_provider_response_error_carries_user_message() -> None:
    err = ProviderResponseError("upstream rejected", provider="anthropic")
    assert err.provider == "anthropic"
    assert err.user_message
    assert "API key" in err.user_message

    err2 = ProviderResponseError(
        "x",
        provider="anthropic",
        user_message="custom user-facing copy",
    )
    assert err2.user_message == "custom user-facing copy"
