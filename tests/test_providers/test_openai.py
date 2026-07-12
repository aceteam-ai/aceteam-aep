"""Tests for OpenAI provider - format conversion and stream guards."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from aceteam_aep.models import MODEL_REGISTRY, get_model_info
from aceteam_aep.providers import StreamFailedError
from aceteam_aep.providers.openai import (
    OpenAIClient,
    _format_messages,
    _supports_temperature,
    _uses_max_completion_tokens,
)
from aceteam_aep.types import ChatMessage, ContentBlock, ToolCallRequest


def test_format_simple_messages():
    messages = [
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="Hello"),
    ]
    result = _format_messages(messages)
    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are helpful."
    assert result[1]["role"] == "user"


def test_format_multimodal_message():
    messages = [
        ChatMessage(
            role="user",
            content=[
                ContentBlock(type="text", text="What's in this image?"),
                ContentBlock(type="image_url", image_url="https://example.com/img.png"),
            ],
        )
    ]
    result = _format_messages(messages)
    assert len(result) == 1
    assert isinstance(result[0]["content"], list)
    assert result[0]["content"][0]["type"] == "text"
    assert result[0]["content"][1]["type"] == "image_url"


def test_format_tool_calls():
    messages = [
        ChatMessage(
            role="assistant",
            content="Let me search.",
            tool_calls=[ToolCallRequest(id="call_1", name="search", arguments={"query": "test"})],
        )
    ]
    result = _format_messages(messages)
    assert "tool_calls" in result[0]
    assert result[0]["tool_calls"][0]["function"]["name"] == "search"


def test_format_tool_result():
    messages = [
        ChatMessage(
            role="tool",
            content='{"result": "found"}',
            tool_call_id="call_1",
            name="search",
        )
    ]
    result = _format_messages(messages)
    assert result[0]["role"] == "tool"
    assert result[0]["tool_call_id"] == "call_1"


def test_openai_client_model_name():
    client = OpenAIClient(api_key="test", model="gpt-4o")
    assert client.model_name == "gpt-4o"


def test_uses_max_completion_tokens():
    # Models that require max_completion_tokens (registry-driven)
    mct_models = (
        "o1",
        "o1-mini",
        "o1-preview",
        "o3",
        "o3-mini",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
    )
    for model in mct_models:
        assert _uses_max_completion_tokens(model), f"{model} should use max_completion_tokens"

    # Models that use max_tokens
    for model in ("gpt-4o", "gpt-4o-mini", "gpt-4-turbo"):
        assert not _uses_max_completion_tokens(model), f"{model} should use max_tokens"


def test_uses_max_completion_tokens_dotted_gpt5_variants():
    """Dotted point-release gpt-5.x names (not yet in the registry) must
    still hit the max_completion_tokens fallback via prefix matching.

    Regression test for aceteam-ai/aceteam#5566: OpenAI's dotted names
    (gpt-5.6-terra, gpt-5.4, ...) previously fell through the dash-only
    prefix check and 400'd with "max_tokens is not supported".
    """
    dotted_mct_models = (
        "gpt-5.6-terra",
        "gpt-5.6-sol",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.1",
        "gpt-5",
        "o3-mini",
    )
    for model in dotted_mct_models:
        assert _uses_max_completion_tokens(model), f"{model} should use max_completion_tokens"

    for model in ("gpt-4o", "gpt-4.5-preview"):
        assert not _uses_max_completion_tokens(model), f"{model} should use max_tokens"


def test_supports_temperature_dotted_gpt5_variants():
    """The parallel fallback for temperature support must not drift from
    _uses_max_completion_tokens - both consult the same _matches_family
    helper so a dotted gpt-5.x model (not yet in the registry) is treated
    as no-temperature too."""
    for model in ("gpt-5.6-terra", "gpt-5.4"):
        assert not _supports_temperature(model), f"{model} should not support temperature"

    assert _supports_temperature("gpt-4o")

    # Exact "gpt-5" is a registry hit, not a fallback case: the registry
    # entry for gpt-5 doesn't set supports_temperature=False, so this
    # fallback fix does not change (and must not be asserted about) its
    # temperature behavior. See MODEL_REGISTRY in models.py.
    assert _supports_temperature("gpt-5")


def test_registry_drives_max_completion_tokens():
    """_uses_max_completion_tokens must agree with the registry for all known models."""
    for model, info in MODEL_REGISTRY.items():
        if info.provider == "openai" and not info.is_embedding:
            assert _uses_max_completion_tokens(model) == info.uses_max_completion_tokens


def test_registry_model_info():
    info = get_model_info("gpt-4o")
    assert info is not None
    assert info.provider == "openai"
    assert info.supports_vision
    assert not info.uses_max_completion_tokens

    o1 = get_model_info("o1")
    assert o1 is not None
    assert o1.uses_max_completion_tokens
    assert not o1.supports_temperature

    assert get_model_info("not-a-real-model") is None


class _AsyncChunkIterator:
    """Stand-in for the OpenAI SDK's async stream of completion chunks."""

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> AsyncIterator[Any]:
        async def _gen() -> AsyncIterator[Any]:
            for chunk in self._chunks:
                yield chunk

        return _gen()


def _make_stream_client(chunks: list[Any]) -> OpenAIClient:
    """Return an OpenAIClient whose underlying SDK call yields ``chunks``."""
    client = OpenAIClient(api_key="test", model="gpt-test")
    client._client = MagicMock()

    async def _fake_create(**_kwargs: Any) -> _AsyncChunkIterator:
        return _AsyncChunkIterator(chunks)

    client._client.chat.completions.create = _fake_create
    return client


def _make_completion_response(model: str = "gpt-test") -> MagicMock:
    """Stand-in for the OpenAI SDK's non-streaming ChatCompletion object."""
    response = MagicMock()
    response.model = model
    response.usage = None
    choice = MagicMock()
    choice.message = MagicMock()
    choice.message.content = "ok"
    choice.message.tool_calls = None
    choice.finish_reason = "stop"
    response.choices = [choice]
    return response


def _make_capturing_client(
    model: str = "gpt-test", **client_kwargs: Any
) -> tuple[OpenAIClient, dict[str, Any]]:
    """Return an OpenAIClient plus a dict that captures the kwargs passed to
    the underlying SDK ``create`` call (for both chat() and chat_stream())."""
    client = OpenAIClient(api_key="test", model=model, **client_kwargs)
    client._client = MagicMock()
    captured: dict[str, Any] = {}

    async def _fake_create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        if kwargs.get("stream"):
            return _AsyncChunkIterator([_make_choice_chunk(content="hi", finish_reason="stop")])
        return _make_completion_response(model=model)

    client._client.chat.completions.create = _fake_create
    return client, captured


@pytest.mark.asyncio
async def test_chat_uses_max_completion_tokens_override_true() -> None:
    """A model that would normally use max_tokens (gpt-4o) must send
    max_completion_tokens when the override forces it True."""
    client, captured = _make_capturing_client(model="gpt-4o", uses_max_completion_tokens=True)
    await client.chat(messages=[])
    assert "max_completion_tokens" in captured
    assert "max_tokens" not in captured


@pytest.mark.asyncio
async def test_chat_uses_max_completion_tokens_override_false() -> None:
    """A model that would normally use max_completion_tokens (gpt-5) must
    send max_tokens when the override forces it False."""
    client, captured = _make_capturing_client(model="gpt-5", uses_max_completion_tokens=False)
    await client.chat(messages=[])
    assert "max_tokens" in captured
    assert "max_completion_tokens" not in captured


@pytest.mark.asyncio
async def test_chat_uses_max_completion_tokens_default_follows_heuristic() -> None:
    """With no override (the default), behavior is unchanged: the dotted
    gpt-5.x fallback heuristic decides."""
    client, captured = _make_capturing_client(model="gpt-5.6-terra")
    await client.chat(messages=[])
    assert "max_completion_tokens" in captured
    assert "max_tokens" not in captured


@pytest.mark.asyncio
async def test_chat_stream_uses_max_completion_tokens_override() -> None:
    """The same override must also apply to chat_stream()'s token param."""
    client, captured = _make_capturing_client(model="gpt-4o", uses_max_completion_tokens=True)
    async for _ in client.chat_stream(messages=[]):
        pass
    assert "max_completion_tokens" in captured
    assert "max_tokens" not in captured


def _make_choice_chunk(
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    tool_calls: list[Any] | None = None,
    model: str = "gpt-test",
) -> MagicMock:
    chunk = MagicMock()
    chunk.model = model
    chunk.usage = None
    choice = MagicMock()
    choice.delta = MagicMock()
    choice.delta.content = content
    choice.delta.tool_calls = tool_calls
    choice.finish_reason = finish_reason
    chunk.choices = [choice]
    return chunk


def _make_usage_only_chunk() -> MagicMock:
    """A terminal chunk that carries usage but no choices.

    Some OpenAI-compatible aggregators emit this even when the upstream
    request was rejected — the guard must not treat it as ``produced``.
    """
    chunk = MagicMock()
    chunk.choices = []
    chunk.usage = MagicMock()
    chunk.usage.prompt_tokens = 4
    chunk.usage.completion_tokens = 0
    chunk.usage.total_tokens = 4
    return chunk


def _make_tool_call_delta(
    *, index: int, id_: str | None, name: str | None, arguments: str | None
) -> MagicMock:
    tc = MagicMock()
    tc.index = index
    tc.id = id_
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


@pytest.mark.asyncio
async def test_chat_stream_raises_on_empty_stream() -> None:
    """Zero choice-bearing chunks == upstream silent rejection."""
    client = _make_stream_client(chunks=[])

    with pytest.raises(StreamFailedError) as exc_info:
        async for _ in client.chat_stream(messages=[]):
            pass

    assert exc_info.value.provider == "openai"
    assert "gpt-test" in str(exc_info.value)


@pytest.mark.asyncio
async def test_chat_stream_does_not_raise_on_text() -> None:
    """A normal text response must not trip the empty-stream guard."""
    client = _make_stream_client(
        chunks=[
            _make_choice_chunk(content="hello"),
            _make_choice_chunk(content=" world"),
            _make_choice_chunk(finish_reason="stop"),
        ]
    )

    chunks: list[Any] = []
    async for chunk in client.chat_stream(messages=[]):
        chunks.append(chunk)

    assert any(c.delta_text == "hello" for c in chunks)
    assert any(c.finish_reason == "stop" for c in chunks)


@pytest.mark.asyncio
async def test_chat_stream_does_not_raise_when_only_finish_reason() -> None:
    """A finish-reason without text (refusal, content filter) still counts."""
    client = _make_stream_client(chunks=[_make_choice_chunk(finish_reason="content_filter")])

    chunks: list[Any] = []
    async for chunk in client.chat_stream(messages=[]):
        chunks.append(chunk)

    assert any(c.finish_reason == "content_filter" for c in chunks)


@pytest.mark.asyncio
async def test_chat_stream_does_not_raise_on_tool_only_response() -> None:
    """A tool-only response is a normal path when tools are enabled."""
    client = _make_stream_client(
        chunks=[
            _make_choice_chunk(
                tool_calls=[
                    _make_tool_call_delta(index=0, id_="call_1", name="search", arguments=None)
                ],
            ),
            _make_choice_chunk(
                tool_calls=[
                    _make_tool_call_delta(index=0, id_=None, name=None, arguments='{"query": ')
                ],
            ),
            _make_choice_chunk(
                tool_calls=[_make_tool_call_delta(index=0, id_=None, name=None, arguments='"hi"}')],
            ),
            _make_choice_chunk(finish_reason="tool_calls"),
        ]
    )

    chunks: list[Any] = []
    async for chunk in client.chat_stream(messages=[]):
        chunks.append(chunk)

    tool_chunks = [c for c in chunks if c.delta_tool_calls]
    assert len(tool_chunks) == 1
    call = tool_chunks[0].delta_tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "search"
    assert call.arguments == {"query": "hi"}
    assert any(c.finish_reason == "tool_calls" for c in chunks)


@pytest.mark.asyncio
async def test_chat_stream_usage_only_chunk_does_not_satisfy_guard() -> None:
    """An aggregator that only emits a usage frame for a rejected request
    must still trip the empty-stream guard. Usage alone is not proof the
    upstream produced any content.
    """
    client = _make_stream_client(chunks=[_make_usage_only_chunk()])

    with pytest.raises(StreamFailedError) as exc_info:
        async for _ in client.chat_stream(messages=[]):
            pass

    assert exc_info.value.provider == "openai"
