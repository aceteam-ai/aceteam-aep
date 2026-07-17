"""Tests for SSE streaming through the AEP proxy."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from starlette.responses import StreamingResponse

from aceteam_aep.enforcement import EnforcementPolicy
from aceteam_aep.proxy.app import _ensure_openai_stream_usage
from aceteam_aep.proxy.streaming import (
    _accumulate_stream_chunks,
    _parse_sse_line,
    handle_streaming_request,
)
from aceteam_aep.safety.base import DetectorRegistry


def test_parse_sse_data_line() -> None:
    result = _parse_sse_line('data: {"choices": [{"delta": {"content": "Hi"}}]}')
    assert result is not None
    assert result["choices"][0]["delta"]["content"] == "Hi"


def test_parse_sse_done() -> None:
    result = _parse_sse_line("data: [DONE]")
    assert result is None


def test_parse_sse_empty_line() -> None:
    assert _parse_sse_line("") is None
    assert _parse_sse_line("\n") is None


def test_parse_sse_non_data_line() -> None:
    assert _parse_sse_line("event: message") is None


def test_accumulate_chunks_text() -> None:
    chunks = [
        {"model": "gpt-4o", "choices": [{"delta": {"content": "Hello"}}]},
        {"model": "gpt-4o", "choices": [{"delta": {"content": " world"}}]},
        {
            "model": "gpt-4o",
            "choices": [{"delta": {}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    ]
    model, text, inp, out = _accumulate_stream_chunks(chunks)
    assert model == "gpt-4o"
    assert text == "Hello world"
    assert inp == 10
    assert out == 5


def test_accumulate_empty_chunks() -> None:
    model, text, inp, out = _accumulate_stream_chunks([])
    assert model == "unknown"
    assert text == ""
    assert inp == 0
    assert out == 0


def test_accumulate_no_usage() -> None:
    chunks = [
        {"model": "gpt-4o", "choices": [{"delta": {"content": "Hi"}}]},
    ]
    model, text, inp, out = _accumulate_stream_chunks(chunks)
    assert text == "Hi"
    assert inp == 0  # no usage chunk
    assert out == 0


def test_ensure_openai_stream_usage_sets_include_usage() -> None:
    body: dict = {"model": "gpt-4o", "messages": [], "stream": True}
    _ensure_openai_stream_usage(body, "/v1/chat/completions")
    assert body["stream_options"]["include_usage"] is True


def test_ensure_openai_stream_usage_skips_non_chat_path() -> None:
    body: dict = {"stream": True}
    _ensure_openai_stream_usage(body, "/v1/embeddings")
    assert "stream_options" not in body


def test_ensure_openai_stream_usage_skips_non_stream() -> None:
    body: dict = {"stream": False}
    _ensure_openai_stream_usage(body, "/v1/chat/completions")
    assert "stream_options" not in body


# --- handle_streaming_request: upstream error surfacing (#5988) ---


def _patch_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Any,
) -> None:
    """Route the handler's internal httpx.AsyncClient through a MockTransport."""
    real_client = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


async def _collect_body(resp: StreamingResponse) -> str:
    parts: list[str] = []
    async for chunk in resp.body_iterator:
        assert isinstance(chunk, str)
        parts.append(chunk)
    return "".join(parts)


async def _call_handler(on_complete: Any = None) -> Any:
    return await handle_streaming_request(
        target_url="https://upstream.test/v1/chat/completions",
        body_bytes=b'{"stream": true}',
        headers={"Content-Type": "application/json"},
        call_id="call-test",
        input_text="hi",
        registry=DetectorRegistry(),
        policy=EnforcementPolicy(),
        on_complete=on_complete,
    )


async def test_upstream_400_openai_body_passed_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-2xx before any bytes stream -> upstream status + JSON body, no SSE."""
    error = {
        "error": {
            "message": "Unsupported parameter: 'context_management'",
            "type": "invalid_request_error",
            "param": "context_management",
            "code": None,
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            content=json.dumps(error).encode(),
            headers={"content-type": "application/json"},
        )

    _patch_transport(monkeypatch, handler)
    completions: list[dict[str, Any]] = []
    resp = await _call_handler(on_complete=lambda **kw: completions.append(kw))

    assert not isinstance(resp, StreamingResponse)
    assert resp.status_code == 400
    assert json.loads(resp.body) == error
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.headers["x-aep-call-id"] == "call-test"
    assert completions == []


async def test_upstream_400_anthropic_body_passed_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anthropic-wire error shape is preserved verbatim."""
    error = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "context-management-2025-06-27 requires the beta header",
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            content=json.dumps(error).encode(),
            headers={"content-type": "application/json"},
        )

    _patch_transport(monkeypatch, handler)
    resp = await _call_handler()

    assert not isinstance(resp, StreamingResponse)
    assert resp.status_code == 400
    assert json.loads(resp.body) == error


async def test_upstream_200_happy_path_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2xx upstream still streams through and fires on_complete metering."""
    sse_body = (
        b'data: {"model": "gpt-4o", "choices": [{"delta": {"content": "Hello"}}]}\n\n'
        b'data: {"model": "gpt-4o", "choices": [{"delta": {}}],'
        b' "usage": {"prompt_tokens": 3, "completion_tokens": 2}}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse_body,
            headers={"content-type": "text/event-stream"},
        )

    _patch_transport(monkeypatch, handler)
    completions: list[dict[str, Any]] = []
    resp = await _call_handler(on_complete=lambda **kw: completions.append(kw))

    assert isinstance(resp, StreamingResponse)
    assert resp.status_code == 200
    out = await _collect_body(resp)
    assert "Hello" in out
    assert "data: [DONE]" in out
    assert "event: error" not in out
    assert len(completions) == 1
    assert completions[0]["model"] == "gpt-4o"
    assert completions[0]["input_tokens"] == 3
    assert completions[0]["output_tokens"] == 2


async def test_mid_stream_failure_emits_error_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upstream dropping mid-stream -> final SSE error event, no hang."""

    class FailingStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield (b'data: {"model": "gpt-4o", "choices": [{"delta": {"content": "Hi"}}]}\n\n')
            raise httpx.ReadError("connection reset by upstream")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=FailingStream(),
            headers={"content-type": "text/event-stream"},
        )

    _patch_transport(monkeypatch, handler)
    completions: list[dict[str, Any]] = []
    resp = await _call_handler(on_complete=lambda **kw: completions.append(kw))

    assert isinstance(resp, StreamingResponse)
    out = await _collect_body(resp)
    assert "Hi" in out

    # Stream must terminate with an Anthropic-format error event
    assert "event: error\n" in out
    last_data = out.rstrip().splitlines()[-1]
    assert last_data.startswith("data: ")
    payload = json.loads(last_data[6:])
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "api_error"
    assert "connection reset by upstream" in payload["error"]["message"]

    # Interrupted streams are not metered as successful completions
    assert completions == []
