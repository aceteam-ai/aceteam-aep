"""Tests for Google provider - schema helpers and stream guards."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from aceteam_aep.providers import StreamFailedError
from aceteam_aep.providers.google import GoogleClient, _strip_additional_properties


def test_strip_top_level():
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    result = _strip_additional_properties(schema)
    assert "additionalProperties" not in result
    assert result["type"] == "object"


def test_strip_nested_objects():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "address": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"city": {"type": "string"}},
            },
        },
    }
    result = _strip_additional_properties(schema)
    assert "additionalProperties" not in result
    assert "additionalProperties" not in result["properties"]["address"]
    assert result["properties"]["address"]["properties"]["city"]["type"] == "string"


def test_strip_array_items():
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"id": {"type": "integer"}},
        },
    }
    result = _strip_additional_properties(schema)
    assert "additionalProperties" not in result["items"]
    assert result["items"]["properties"]["id"]["type"] == "integer"


def test_no_additionalProperties_unchanged():
    schema = {"type": "object", "properties": {"x": {"type": "number"}}}
    result = _strip_additional_properties(schema)
    assert result == schema


def test_preserves_list_values():
    schema = {"required": ["a", "b"], "type": "object", "additionalProperties": False}
    result = _strip_additional_properties(schema)
    assert result["required"] == ["a", "b"]


class _AsyncChunkIterator:
    """Stand-in for ``aio.models.generate_content_stream`` results."""

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> AsyncIterator[Any]:
        async def _gen() -> AsyncIterator[Any]:
            for chunk in self._chunks:
                yield chunk

        return _gen()


def _make_stream_client(chunks: list[Any]) -> GoogleClient:
    """Return a GoogleClient whose stream call yields ``chunks``.

    ``generate_content_stream`` is itself an awaitable that returns the
    async iterator, so the patched callable returns the awaitable.
    """
    client = GoogleClient(api_key="test", model="gemini-test")
    client._client = MagicMock()

    async def _fake_stream(**_kwargs: Any) -> _AsyncChunkIterator:
        return _AsyncChunkIterator(chunks)

    client._client.aio.models.generate_content_stream = _fake_stream
    return client


def _text_chunk(text: str) -> MagicMock:
    chunk = MagicMock()
    chunk.usage_metadata = None
    part = MagicMock()
    part.text = text
    part.function_call = None
    content = MagicMock()
    content.parts = [part]
    candidate = MagicMock()
    candidate.content = content
    candidate.finish_reason = None
    chunk.candidates = [candidate]
    return chunk


def _function_call_chunk(name: str, args: dict[str, Any]) -> MagicMock:
    chunk = MagicMock()
    chunk.usage_metadata = None
    part = MagicMock()
    part.text = None
    part.function_call = MagicMock()
    part.function_call.name = name
    part.function_call.args = args
    content = MagicMock()
    content.parts = [part]
    candidate = MagicMock()
    candidate.content = content
    candidate.finish_reason = None
    chunk.candidates = [candidate]
    return chunk


def _finish_chunk(*, finish_reason: str = "STOP", include_usage: bool = True) -> MagicMock:
    chunk = MagicMock()
    if include_usage:
        chunk.usage_metadata = MagicMock()
        chunk.usage_metadata.prompt_token_count = 5
        chunk.usage_metadata.candidates_token_count = 1
        chunk.usage_metadata.total_token_count = 6
    else:
        chunk.usage_metadata = None
    content = MagicMock()
    content.parts = []
    candidate = MagicMock()
    candidate.content = content
    candidate.finish_reason = finish_reason
    chunk.candidates = [candidate]
    return chunk


@pytest.mark.asyncio
async def test_chat_stream_raises_on_empty_stream() -> None:
    """Zero chunks == silent rejection."""
    client = _make_stream_client(chunks=[])

    with pytest.raises(StreamFailedError) as exc_info:
        async for _ in client.chat_stream(messages=[]):
            pass

    assert exc_info.value.provider == "google"
    assert "gemini-test" in str(exc_info.value)


@pytest.mark.asyncio
async def test_chat_stream_does_not_raise_on_text() -> None:
    """A normal text + usage stream must not trip the guard."""
    client = _make_stream_client(chunks=[_text_chunk("hello"), _finish_chunk(finish_reason="STOP")])

    chunks: list[Any] = []
    async for chunk in client.chat_stream(messages=[]):
        chunks.append(chunk)

    assert any(c.delta_text == "hello" for c in chunks)
    assert any(c.usage is not None for c in chunks)


@pytest.mark.asyncio
async def test_chat_stream_does_not_raise_when_only_finish_reason() -> None:
    """A finish-reason without text or usage (e.g., safety stop with no
    body) still counts — the candidate-level finish_reason proves the
    upstream actually responded.
    """
    client = _make_stream_client(
        chunks=[_finish_chunk(finish_reason="SAFETY", include_usage=False)]
    )

    chunks: list[Any] = []
    async for chunk in client.chat_stream(messages=[]):
        chunks.append(chunk)

    assert chunks  # at least one chunk yielded, no raise


@pytest.mark.asyncio
async def test_chat_stream_does_not_raise_on_tool_only_response() -> None:
    """A function-call-only response (no text) must not misfire the guard."""
    client = _make_stream_client(
        chunks=[
            _function_call_chunk("search", {"query": "hi"}),
            _finish_chunk(finish_reason="STOP"),
        ]
    )

    chunks: list[Any] = []
    async for chunk in client.chat_stream(messages=[]):
        chunks.append(chunk)

    tool_chunks = [c for c in chunks if c.delta_tool_calls]
    assert len(tool_chunks) == 1
    call = tool_chunks[0].delta_tool_calls[0]
    assert call.name == "search"
    assert call.arguments == {"query": "hi"}


def test_stream_failed_error_carries_google_slug() -> None:
    err = StreamFailedError("upstream rejected", provider="google")
    assert err.provider == "google"
