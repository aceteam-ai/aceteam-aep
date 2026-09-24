"""Public gateway metadata contract, exercised at the raw HTTP boundary."""

from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import openai
import pytest

from aceteam_aep import (
    AepRequestContext,
    ChatMessage,
    GatewayResponseError,
    create_client,
    current_tool_origin,
    run_agent_loop,
    run_agent_loop_stream,
    tool,
)
from aceteam_aep.providers.anthropic import AnthropicClient
from aceteam_aep.providers.openai import OpenAIClient
from aceteam_aep.types import AepResponseMetadata, ChatResponse, StreamChunk, ToolCallRequest


def _openai_body(*, call_tool: bool = False) -> bytes:
    message: dict[str, Any] = {"role": "assistant", "content": "answer"}
    if call_tool:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool-distinct",
                    "type": "function",
                    "function": {"name": "record", "arguments": "{}"},
                }
            ],
        }
    return json.dumps(
        {
            "id": "provider-distinct",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if call_tool else "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode()


def _anthropic_body(*, call_tool: bool = False) -> bytes:
    return json.dumps(
        {
            "id": "provider-distinct",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": (
                [{"type": "tool_use", "id": "tool-distinct", "name": "record", "input": {}}]
                if call_tool
                else [{"type": "text", "text": "answer"}]
            ),
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    ).encode()


def _openai_sse() -> bytes:
    frames = [
        {
            "id": "provider-distinct",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-test",
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        },
        {
            "id": "provider-distinct",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-test",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    return (
        "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames) + "data: [DONE]\n\n"
    ).encode()


def _anthropic_sse() -> bytes:
    frames = [
        ("message_start", {"type": "message_start", "message": json.loads(_anthropic_body())}),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hi"},
            },
        ),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(
        f"event: {kind}\ndata: {json.dumps(frame)}\n\n" for kind, frame in frames
    ).encode()


def _tool_sse(protocol: str) -> bytes:
    if protocol == "openai":
        frames = [
            {
                "id": "provider-distinct",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-test",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "tool-distinct",
                                    "type": "function",
                                    "function": {"name": "record", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        ]
        return (
            "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames) + "data: [DONE]\n\n"
        ).encode()
    frames = [
        ("message_start", {"type": "message_start", "message": json.loads(_anthropic_body())}),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "tool-distinct",
                    "name": "record",
                    "input": {},
                },
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "{}"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 1},
            },
        ),
    ]
    return "".join(
        f"event: {kind}\ndata: {json.dumps(frame)}\n\n" for kind, frame in frames
    ).encode()


def _client(
    protocol: str,
    handler: Any,
    *,
    trusted: bool = True,
    closed: list[str] | None = None,
    close_error: BaseException | None = None,
    stream_error: BaseException | None = None,
) -> OpenAIClient | AnthropicClient:
    url = "https://gateway.example/v1" if trusted else None
    if protocol == "openai":
        client = OpenAIClient("key", "gpt-test", trusted_gateway_url=url)
    else:
        client = AnthropicClient("key", "claude-test", trusted_gateway_url=url)

    def as_object(value: Any) -> Any:
        if isinstance(value, dict):
            return SimpleNamespace(**{key: as_object(item) for key, item in value.items()})
        if isinstance(value, list):
            return [as_object(item) for item in value]
        return value

    async def response_for(kwargs: dict[str, Any]) -> httpx.Response:
        request = httpx.Request(
            "POST", url or "https://provider.example/v1", headers=kwargs.get("extra_headers", {})
        )
        response = await handler(request)
        response.request = request
        return response

    class Raw:
        def __init__(self, response: httpx.Response, streaming: bool) -> None:
            self.headers = response.headers
            self._response = response
            self._streaming = streaming

        def parse(self) -> Any:
            if not self._streaming:
                if protocol == "openai":
                    return openai.types.chat.ChatCompletion.model_validate_json(
                        self._response.content
                    )
                return anthropic.types.Message.model_validate_json(self._response.content)

            async def events() -> Any:
                if stream_error is not None:
                    raise stream_error
                for frame in self._response.text.split("\n\n"):
                    if not frame:
                        continue
                    data = next(
                        (line[6:] for line in frame.splitlines() if line.startswith("data: ")), ""
                    )
                    if not data or data == "[DONE]":
                        continue
                    if protocol == "openai":
                        yield openai.types.chat.ChatCompletionChunk.model_validate_json(data)
                    else:
                        yield as_object(json.loads(data))

            return events()

    class StreamingRaw(Raw):
        async def parse(self) -> Any:
            return super().parse()

    class RawContext:
        def __init__(self, kwargs: dict[str, Any]) -> None:
            self.kwargs = kwargs

        async def __aenter__(self) -> StreamingRaw:
            return StreamingRaw(await response_for(self.kwargs), True)

        async def __aexit__(self, *_args: Any) -> None:
            if closed is not None:
                closed.append("raw response closed")
            if close_error is not None:
                raise close_error
            return None

    async def raw_create(**kwargs: Any) -> Raw:
        return Raw(await response_for(kwargs), False)

    async def direct_create(**kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return Raw(await response_for(kwargs), True).parse()
        return (await raw_create(**kwargs)).parse()

    endpoint = SimpleNamespace(
        create=direct_create,
        with_raw_response=SimpleNamespace(create=raw_create),
        with_streaming_response=SimpleNamespace(create=lambda **kwargs: RawContext(kwargs)),
    )
    if protocol == "openai":
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
    else:
        client._client = SimpleNamespace(messages=endpoint)
    return client


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
async def test_gateway_nonstream_exact_header_and_request_context(protocol: str) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "gateway.example"
        assert request.headers["x-aep-trace-id"] == "trace-caller"
        assert request.headers["x-aep-entity"] == "org:caller"
        return httpx.Response(
            200,
            headers={
                "x-AeP-CaLl-Id": "opaque.gateway-42",
                "X-AEP-Trace-ID": "trace-returned",
                "x-secret": "never-exposed",
            },
            content=_openai_body() if protocol == "openai" else _anthropic_body(),
        )

    client = _client(protocol, handler)
    response = await client.chat_with_context(
        [ChatMessage(role="user", content="hi")],
        request_context=AepRequestContext(trace_id="trace-caller", entity="org:caller"),
    )
    assert response.response_metadata == AepResponseMetadata(
        call_id="opaque.gateway-42", trace_id="trace-returned"
    )
    assert response.response_metadata.call_id != "provider-distinct"
    assert not hasattr(response.response_metadata, "x_secret")
    with pytest.raises(FrozenInstanceError):
        response.response_metadata.call_id = "mutated"


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
async def test_gateway_stream_captures_headers_before_sse(protocol: str) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"X-AEP-Call-ID": "stream.opaque"},
            content=_openai_sse() if protocol == "openai" else _anthropic_sse(),
        )

    client = _client(protocol, handler)
    chunks = [chunk async for chunk in client.chat_stream([])]
    assert chunks[0].response_metadata == AepResponseMetadata(call_id="stream.opaque")
    assert chunks[0].delta_text == ""
    assert any(chunk.delta_text == "hi" for chunk in chunks)
    assert chunks[-1].response_complete is True


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
async def test_stream_tool_proposal_has_gateway_origin(protocol: str) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"X-AEP-Call-ID": "gateway.tool"}, content=_tool_sse(protocol)
        )

    chunks = [chunk async for chunk in _client(protocol, handler).chat_stream([])]
    proposals = [proposal for chunk in chunks for proposal in (chunk.delta_tool_calls or [])]
    assert len(proposals) == 1
    assert proposals[0].id == "tool-distinct"
    assert proposals[0].origin.call_id == "gateway.tool"


@pytest.mark.parametrize("value", [None, "", "bad id", "a" * 129, "bad\x7fvalue"])
async def test_missing_or_malformed_call_id_is_absent(value: str | None) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        headers = {"X-AEP-Call-ID": value} if value is not None else {}
        return httpx.Response(200, headers=headers, content=_openai_body())

    response = await _client("openai", handler).chat([])
    assert response.response_metadata is not None
    assert response.response_metadata.call_id is None


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
async def test_direct_provider_never_trusts_or_sends_aep_headers(protocol: str) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert not any(key.startswith("x-aep-") for key in request.headers)
        return httpx.Response(
            200,
            headers={"X-AEP-Call-ID": "spoofed"},
            content=_openai_body() if protocol == "openai" else _anthropic_body(),
        )

    client = _client(protocol, handler, trusted=False)
    response = await client.chat([])
    assert response.response_metadata is None
    with pytest.raises(ValueError, match="trusted gateway"):
        await client.chat_with_context([], request_context=AepRequestContext(entity="org:caller"))


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
async def test_body_failure_retains_gateway_header(protocol: str) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"X-AEP-Call-ID": "before.failure"}, content=b"not json")

    with pytest.raises(GatewayResponseError) as error:
        await _client(protocol, handler).chat([])
    assert error.value.response_metadata.call_id == "before.failure"
    assert error.value.response_complete is False


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
async def test_failed_attempt_does_not_replace_next_attempt_metadata(protocol: str) -> None:
    attempts = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            200,
            headers={"X-AEP-Call-ID": f"attempt.{attempts}"},
            content=b"not json"
            if attempts == 1
            else (_openai_body() if protocol == "openai" else _anthropic_body()),
        )

    client = _client(protocol, handler)
    with pytest.raises(GatewayResponseError) as error:
        await client.chat([])
    result = await client.chat([])
    assert error.value.response_metadata.call_id == "attempt.1"
    assert result.response_metadata.call_id == "attempt.2"


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
async def test_cancelled_stream_keeps_incomplete_header_snapshot(protocol: str) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"X-AEP-Call-ID": "cancel.opaque"},
            content=_openai_sse() if protocol == "openai" else _anthropic_sse(),
        )

    stream = _client(protocol, handler).chat_stream([])
    first = await anext(stream)
    await stream.aclose()
    assert first.response_metadata.call_id == "cancel.opaque"
    assert first.response_complete is False


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("with_context", [False, True])
async def test_early_close_releases_raw_response(protocol: str, with_context: bool) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"X-AEP-Call-ID": "close.opaque"},
            content=_openai_sse() if protocol == "openai" else _anthropic_sse(),
        )

    closed: list[str] = []
    client = _client(protocol, handler, closed=closed)
    stream = (
        client.chat_stream_with_context([], request_context=AepRequestContext(trace_id="trace"))
        if with_context
        else client.chat_stream([])
    )
    first = await anext(stream)
    assert first.response_metadata.call_id == "close.opaque"
    assert first.response_complete is False
    assert closed == []
    await stream.aclose()
    assert closed == ["raw response closed"]


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
async def test_early_agent_close_releases_raw_response(protocol: str) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"X-AEP-Call-ID": "agent.close"},
            content=_openai_sse() if protocol == "openai" else _anthropic_sse(),
        )

    closed: list[str] = []
    client = _client(protocol, handler, closed=closed)
    stream = run_agent_loop_stream(client, [ChatMessage(role="user", content="hi")])
    first = await anext(stream)
    assert first.type == "response_metadata"
    assert first.response_metadata.call_id == "agent.close"
    assert first.data == {"complete": False}
    assert closed == []
    await stream.aclose()
    assert closed == ["raw response closed"]


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
async def test_stream_parse_failure_closes_raw_response_once(protocol: str) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"X-AEP-Call-ID": "parse.failure"},
            content=b"data: not-json\n\n",
        )

    closed: list[str] = []
    stream = _client(protocol, handler, closed=closed).chat_stream([])
    first = await anext(stream)
    assert first.response_metadata.call_id == "parse.failure"
    with pytest.raises(GatewayResponseError) as error:
        await anext(stream)
    assert error.value.response_metadata.call_id == "parse.failure"
    assert error.value.response_complete is False
    assert closed == ["raw response closed"]


async def test_early_close_releases_direct_openai_sdk_stream() -> None:
    closed: list[str] = []

    class DirectStream:
        def __init__(self) -> None:
            self.sent = False

        def __aiter__(self) -> DirectStream:
            return self

        async def __anext__(self) -> Any:
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="hi", tool_calls=None), finish_reason=None
                    )
                ],
                model="gpt-test",
            )

        async def close(self) -> None:
            closed.append("sdk stream closed")

    async def create(**_kwargs: Any) -> DirectStream:
        return DirectStream()

    client = OpenAIClient("key", "gpt-test")
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    stream = client.chat_stream([])
    first = await anext(stream)
    assert first.delta_text == "hi"
    await stream.aclose()
    assert closed == ["sdk stream closed"]


@pytest.mark.parametrize("primary", [ValueError("iteration failed"), asyncio.CancelledError()])
async def test_direct_openai_primary_survives_close_failure(
    primary: BaseException,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class DirectStream:
        def __aiter__(self) -> DirectStream:
            return self

        async def __anext__(self) -> Any:
            raise primary

        async def close(self) -> None:
            raise RuntimeError("close failed")

    async def create(**_kwargs: Any) -> DirectStream:
        return DirectStream()

    client = OpenAIClient("key", "gpt-test")
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    with pytest.raises(type(primary)) as error:
        await anext(client.chat_stream([]))
    assert error.value is primary
    assert "Stream cleanup also failed" in caplog.text
    assert "close failed" in caplog.text


async def test_direct_openai_cleanup_only_failure_surfaces() -> None:
    class DirectStream:
        def __aiter__(self) -> DirectStream:
            return self

        async def __anext__(self) -> Any:
            raise StopAsyncIteration

        async def close(self) -> None:
            raise RuntimeError("close only failed")

    async def create(**_kwargs: Any) -> DirectStream:
        return DirectStream()

    client = OpenAIClient("key", "gpt-test")
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    with pytest.raises(RuntimeError, match="close only failed"):
        await anext(client.chat_stream([]))


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
async def test_gateway_primary_retains_metadata_when_raw_exit_fails(
    protocol: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"X-AEP-Call-ID": "dual.failure"}, content=b"data: not-json\n\n"
        )

    closed: list[str] = []
    stream = _client(
        protocol, handler, closed=closed, close_error=RuntimeError("raw exit failed")
    ).chat_stream([])
    first = await anext(stream)
    assert first.response_metadata.call_id == "dual.failure"
    with pytest.raises(GatewayResponseError) as error:
        await anext(stream)
    assert error.value.response_metadata.call_id == "dual.failure"
    assert error.value.response_complete is False
    assert error.value.__cause__ is not None
    assert "raw exit failed" not in str(error.value.__cause__)
    assert closed == ["raw response closed"]
    assert "Stream cleanup also failed" in caplog.text
    assert "raw exit failed" in caplog.text


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
async def test_gateway_cancellation_retains_metadata_when_raw_exit_fails(
    protocol: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"X-AEP-Call-ID": "cancel.dual"}, content=b"")

    cancelled = asyncio.CancelledError("iteration cancelled")
    stream = _client(
        protocol, handler, close_error=RuntimeError("raw exit failed"), stream_error=cancelled
    ).chat_stream([])
    await anext(stream)
    with pytest.raises(asyncio.CancelledError) as error:
        await anext(stream)
    assert error.value is cancelled
    assert error.value.response_metadata.call_id == "cancel.dual"
    assert error.value.response_complete is False
    assert "Stream cleanup also failed" in caplog.text


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
async def test_gateway_cleanup_only_failure_retains_metadata(protocol: str) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"X-AEP-Call-ID": "cleanup.only"},
            content=_openai_sse() if protocol == "openai" else _anthropic_sse(),
        )

    stream = _client(protocol, handler, close_error=RuntimeError("raw exit failed")).chat_stream([])
    with pytest.raises(GatewayResponseError) as error:
        async for _ in stream:
            pass
    assert error.value.response_metadata.call_id == "cleanup.only"
    assert isinstance(error.value.__cause__, RuntimeError)


async def test_two_model_iterations_bind_each_tool_to_own_response() -> None:
    attempts = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            200,
            headers={"X-AEP-Call-ID": f"iteration.{attempts}"},
            content=_openai_body(call_tool=attempts <= 2),
        )

    seen: list[str | None] = []

    @tool
    def record() -> str:
        origin = current_tool_origin()
        seen.append(origin.call_id if origin else None)
        return "ok"

    result = await run_agent_loop(
        _client("openai", handler),
        [ChatMessage(role="user", content="hi")],
        tools=[record],
    )
    assert seen == ["iteration.1", "iteration.2"]
    assert [metadata.call_id for metadata in result.response_metadata] == [
        "iteration.1",
        "iteration.2",
        "iteration.3",
    ]
    proposals = [message.tool_calls[0] for message in result.messages if message.tool_calls]
    assert [proposal.origin.call_id for proposal in proposals] == ["iteration.1", "iteration.2"]
    assert current_tool_origin() is None


async def test_anthropic_tool_proposal_carries_gateway_origin() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"X-AEP-Call-ID": "anthropic.gateway"},
            content=_anthropic_body(call_tool=True),
        )

    response = await _client("anthropic", handler).chat([])
    assert response.message.tool_calls[0].id == "tool-distinct"
    assert response.message.tool_calls[0].origin.call_id == "anthropic.gateway"


async def test_parallel_calls_and_agent_tool_origin_do_not_leak() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        trace = request.headers.get("x-aep-trace-id", "none")
        await asyncio.sleep(0.01 if trace == "a" else 0)
        return httpx.Response(
            200, headers={"X-AEP-Call-ID": f"gateway.{trace}"}, content=_openai_body(call_tool=True)
        )

    client = _client("openai", handler)
    seen: list[str | None] = []

    @tool
    async def record() -> str:
        origin = current_tool_origin()
        seen.append(origin.call_id if origin else None)
        await asyncio.sleep(0.001)
        return "ok"

    async def run(trace: str) -> Any:
        return await run_agent_loop(
            client,
            [ChatMessage(role="user", content="hi")],
            tools=[record],
            request_context=AepRequestContext(trace_id=trace),
            max_iterations=1,
        )

    a, b = await asyncio.gather(run("a"), run("b"))
    assert {a.response_metadata[0].call_id, b.response_metadata[0].call_id} == {
        "gateway.a",
        "gateway.b",
    }
    assert set(seen) == {"gateway.a", "gateway.b"}
    assert current_tool_origin() is None


async def test_stream_agent_exposes_origin_and_resets_after_tool() -> None:
    class ScriptedClient:
        model_name = "test"

        async def chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
            raise AssertionError("stream only")

        async def chat_stream(self, messages: list[ChatMessage], **kwargs: Any) -> Any:
            metadata = AepResponseMetadata(call_id="first" if len(messages) == 1 else "second")
            yield StreamChunk(response_metadata=metadata)
            if len(messages) == 1:
                yield StreamChunk(
                    delta_tool_calls=[ToolCallRequest("tool", "observe", {})],
                    response_metadata=metadata,
                    finish_reason="tool_calls",
                )
            else:
                yield StreamChunk(
                    delta_text="done", response_metadata=metadata, finish_reason="stop"
                )

    seen: list[str | None] = []

    @tool
    def observe() -> str:
        origin = current_tool_origin()
        seen.append(origin.call_id if origin else None)
        return "ok"

    events = [
        event
        async for event in run_agent_loop_stream(
            ScriptedClient(), [ChatMessage(role="user", content="hi")], tools=[observe]
        )
    ]
    assert seen == ["first"]
    assert current_tool_origin() is None
    assert [
        event.response_metadata.call_id for event in events if event.type == "response_metadata"
    ] == ["first", "first", "second", "second"]


def test_factory_requires_explicit_gateway_setting() -> None:
    direct = create_client("gpt-test", "key", provider="openai", base_url="https://gateway.example")
    assert isinstance(direct, OpenAIClient)
    assert direct._trusted_gateway is False
    gateway = create_client(
        "gpt-test", "key", provider="openai", trusted_gateway_url="https://gateway.example/v1"
    )
    assert isinstance(gateway, OpenAIClient)
    assert gateway._trusted_gateway is True
    with pytest.raises(ValueError):
        create_client(
            "gpt-test",
            "key",
            base_url="https://provider.example",
            trusted_gateway_url="https://gateway.example",
        )
    for invalid in ("", "gateway.example", "https://gateway.example\n", "file:///tmp/gateway"):
        with pytest.raises(ValueError, match="trusted_gateway_url"):
            create_client("gpt-test", "key", trusted_gateway_url=invalid)
