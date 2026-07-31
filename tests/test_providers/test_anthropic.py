"""Tests for the Anthropic provider — format conversion and stream guards."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aceteam_aep import create_client
from aceteam_aep.providers import StreamFailedError
from aceteam_aep.providers.anthropic import AnthropicClient
from aceteam_aep.types import ChatMessage, Usage


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


def _message_start_event(
    input_tokens: int = 3,
    cache_read: int | None = 0,
    cache_creation: int | None = 0,
) -> MagicMock:
    event = MagicMock()
    event.type = "message_start"
    event.message = MagicMock()
    event.message.usage = MagicMock()
    event.message.usage.input_tokens = input_tokens
    event.message.usage.cache_read_input_tokens = cache_read
    event.message.usage.cache_creation_input_tokens = cache_creation
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

    with pytest.raises(StreamFailedError) as exc_info:
        async for _ in client.chat_stream(messages=[]):
            pass

    assert exc_info.value.provider == "anthropic"
    assert "claude-test" in str(exc_info.value)


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


def test_stream_failed_error_carries_provider_slug() -> None:
    err = StreamFailedError("upstream rejected", provider="anthropic")
    assert err.provider == "anthropic"
    assert "upstream rejected" in str(err)


# ---------------------------------------------------------------------------
# supports_temperature — newer Anthropic models (claude-opus-4-8,
# claude-sonnet-5) 400 if ``temperature`` is sent at all. The caller drives
# omission via ``create_client(..., supports_temperature=False)``, which must
# keep the key out of the kwargs passed to the underlying SDK entirely.
# ---------------------------------------------------------------------------


def _make_nonstream_response() -> MagicMock:
    """A minimal ``messages.create`` return value that ``chat()`` can parse.

    ``content`` must be a real (empty) list so the block-iteration loop
    doesn't blow up, and usage/model/stop_reason must be real scalars.
    """
    response = MagicMock()
    response.content = []
    response.usage = MagicMock()
    response.usage.input_tokens = 3
    response.usage.output_tokens = 5
    response.usage.cache_read_input_tokens = 0
    response.usage.cache_creation_input_tokens = 0
    response.model = "claude-opus-4-8"
    response.stop_reason = "end_turn"
    return response


def _make_factory_client(*, supports_temperature: bool) -> AnthropicClient:
    """Build the client through the public factory, then swap in a mock SDK."""
    client = create_client(
        model="claude-opus-4-8",
        api_key="test",
        provider="anthropic",
        supports_temperature=supports_temperature,
    )
    assert isinstance(client, AnthropicClient)
    client._client = MagicMock()
    return client


@pytest.mark.asyncio
async def test_chat_omits_temperature_when_unsupported() -> None:
    client = _make_factory_client(supports_temperature=False)
    create_mock = AsyncMock(return_value=_make_nonstream_response())
    client._client.messages.create = create_mock

    await client.chat(messages=[])

    assert "temperature" not in create_mock.call_args.kwargs


@pytest.mark.asyncio
async def test_chat_includes_temperature_by_default() -> None:
    client = _make_factory_client(supports_temperature=True)
    create_mock = AsyncMock(return_value=_make_nonstream_response())
    client._client.messages.create = create_mock

    await client.chat(messages=[])

    assert "temperature" in create_mock.call_args.kwargs


@pytest.mark.asyncio
async def test_chat_stream_omits_temperature_when_unsupported() -> None:
    client = _make_factory_client(supports_temperature=False)
    stream_mock = MagicMock(return_value=_FakeAsyncStream([_finish_event()]))
    client._client.messages.stream = stream_mock

    async for _ in client.chat_stream(messages=[]):
        pass

    assert "temperature" not in stream_mock.call_args.kwargs


@pytest.mark.asyncio
async def test_chat_stream_includes_temperature_by_default() -> None:
    client = _make_factory_client(supports_temperature=True)
    stream_mock = MagicMock(return_value=_FakeAsyncStream([_finish_event()]))
    client._client.messages.stream = stream_mock

    async for _ in client.chat_stream(messages=[]):
        pass

    assert "temperature" in stream_mock.call_args.kwargs


# ---------------------------------------------------------------------------
# Prompt caching: request shaping. cache_control breakpoints must land on the
# static prefix (tool definitions + system prompt) so Anthropic serves the
# repeated ~26k-token prefix from cache. Behavior-neutral: the text is
# byte-identical whether or not the breakpoints are present.
# ---------------------------------------------------------------------------

_OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "search the web",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "do math",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _make_chat_client(*, prompt_caching: bool | None) -> tuple[AnthropicClient, AsyncMock]:
    client = AnthropicClient(api_key="test", model="claude-opus-4-8", prompt_caching=prompt_caching)
    create_mock = AsyncMock(return_value=_make_nonstream_response())
    client._client = MagicMock()
    client._client.messages.create = create_mock
    return client, create_mock


@pytest.mark.asyncio
async def test_chat_caches_system_and_last_tool_when_enabled() -> None:
    client, create_mock = _make_chat_client(prompt_caching=True)

    await client.chat(
        messages=[ChatMessage(role="system", content="You are helpful."),
                  ChatMessage(role="user", content="hi")],
        tools=_OPENAI_TOOLS,
    )

    kwargs = create_mock.call_args.kwargs
    # System becomes a block list with a cache_control breakpoint on the block.
    system = kwargs["system"]
    assert isinstance(system, list)
    assert system[0]["text"] == "You are helpful."
    assert system[-1]["cache_control"] == {"type": "ephemeral"}
    # Only the last tool carries a breakpoint; earlier tools do not.
    tools = kwargs["tools"]
    assert "cache_control" not in tools[0]
    assert tools[-1]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_chat_no_cache_control_when_disabled() -> None:
    client, create_mock = _make_chat_client(prompt_caching=False)

    await client.chat(
        messages=[ChatMessage(role="system", content="You are helpful."),
                  ChatMessage(role="user", content="hi")],
        tools=_OPENAI_TOOLS,
    )

    kwargs = create_mock.call_args.kwargs
    # Disabled path is a byte-identical revert: plain string system, no markers.
    assert kwargs["system"] == "You are helpful."
    assert all("cache_control" not in tool for tool in kwargs["tools"])


@pytest.mark.asyncio
async def test_chat_caches_last_tool_without_system_prompt() -> None:
    client, create_mock = _make_chat_client(prompt_caching=True)

    await client.chat(messages=[ChatMessage(role="user", content="hi")], tools=_OPENAI_TOOLS)

    kwargs = create_mock.call_args.kwargs
    # No system prompt: the tools are the last static block, so they carry the
    # breakpoint and no system param is sent.
    assert "system" not in kwargs
    assert kwargs["tools"][-1]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_chat_cache_control_wraps_full_system_with_schema() -> None:
    client, create_mock = _make_chat_client(prompt_caching=True)

    await client.chat(
        messages=[ChatMessage(role="system", content="Base prompt."),
                  ChatMessage(role="user", content="hi")],
        response_format={
            "type": "json_schema",
            "json_schema": {"schema": {"type": "object", "properties": {"x": {"type": "string"}}}},
        },
    )

    system = create_mock.call_args.kwargs["system"]
    # A single block holds base prompt + schema instruction, and the breakpoint
    # sits on that fully-assembled block (not before the schema append).
    assert isinstance(system, list)
    assert len(system) == 1
    assert system[0]["text"].startswith("Base prompt.")
    assert "valid JSON object" in system[0]["text"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_env_toggle_disables_caching(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEP_PROMPT_CACHING", "0")
    # prompt_caching=None → env toggle decides.
    client, create_mock = _make_chat_client(prompt_caching=None)

    await client.chat(
        messages=[ChatMessage(role="system", content="You are helpful.")],
        tools=_OPENAI_TOOLS,
    )

    assert create_mock.call_args.kwargs["system"] == "You are helpful."
    assert all("cache_control" not in tool for tool in create_mock.call_args.kwargs["tools"])


@pytest.mark.asyncio
async def test_chat_stream_shapes_cache_control_when_enabled() -> None:
    client = AnthropicClient(api_key="test", model="claude-opus-4-8", prompt_caching=True)
    stream_mock = MagicMock(
        return_value=_FakeAsyncStream([_message_start_event(), _finish_event()])
    )
    client._client = MagicMock()
    client._client.messages.stream = stream_mock

    async for _ in client.chat_stream(
        messages=[ChatMessage(role="system", content="You are helpful.")],
        tools=_OPENAI_TOOLS,
    ):
        pass

    kwargs = stream_mock.call_args.kwargs
    assert kwargs["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["tools"][-1]["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# Prompt caching: usage propagation. cache_read/cache_creation must surface
# on Usage so the host app can log and bill them.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_propagates_cache_usage() -> None:
    client, create_mock = _make_chat_client(prompt_caching=True)
    response = _make_nonstream_response()
    response.usage.cache_read_input_tokens = 24000
    response.usage.cache_creation_input_tokens = 100
    create_mock.return_value = response

    result = await client.chat(messages=[ChatMessage(role="user", content="hi")])

    assert result.usage.cache_read_input_tokens == 24000
    assert result.usage.cache_creation_input_tokens == 100
    # prompt_tokens keeps its prior meaning (the uncached remainder Anthropic
    # reports as input_tokens); cache fields are additive.
    assert result.usage.prompt_tokens == 3


@pytest.mark.asyncio
async def test_chat_cache_usage_defaults_zero_when_absent() -> None:
    client, create_mock = _make_chat_client(prompt_caching=True)
    response = _make_nonstream_response()
    # SDK reports None when caching is inactive; must coerce to 0.
    response.usage.cache_read_input_tokens = None
    response.usage.cache_creation_input_tokens = None
    create_mock.return_value = response

    result = await client.chat(messages=[ChatMessage(role="user", content="hi")])

    assert result.usage.cache_read_input_tokens == 0
    assert result.usage.cache_creation_input_tokens == 0


@pytest.mark.asyncio
async def test_chat_stream_propagates_cache_usage() -> None:
    client = AnthropicClient(api_key="test", model="claude-opus-4-8", prompt_caching=True)
    stream_mock = MagicMock(
        return_value=_FakeAsyncStream(
            [_message_start_event(cache_read=24000, cache_creation=100), _finish_event()]
        )
    )
    client._client = MagicMock()
    client._client.messages.stream = stream_mock

    usages = [c.usage async for c in client.chat_stream(messages=[]) if c.usage]

    assert usages, "expected a usage-bearing chunk"
    assert usages[-1].cache_read_input_tokens == 24000
    assert usages[-1].cache_creation_input_tokens == 100


def test_usage_add_sums_cache_fields() -> None:
    # Aggregation across an agent loop (agent.py does total += call_usage) must
    # carry the cache fields, not drop them.
    total = Usage() + Usage(cache_read_input_tokens=10, cache_creation_input_tokens=2)
    total = total + Usage(cache_read_input_tokens=5, cache_creation_input_tokens=3)
    assert total.cache_read_input_tokens == 15
    assert total.cache_creation_input_tokens == 5
