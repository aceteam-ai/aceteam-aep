"""Tests for SSE streaming through the AEP proxy."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from starlette.responses import StreamingResponse

from aceteam_aep.enforcement import EnforcementDecision, EnforcementPolicy
from aceteam_aep.proxy.app import _ensure_openai_stream_usage
from aceteam_aep.proxy.streaming import (
    EvaluationResult,
    TerminalOutcome,
    _accumulate_stream_chunks,
    _parse_sse_line,
    handle_streaming_request,
)
from aceteam_aep.safety.base import DetectorRegistry, SafetySignal
from aceteam_aep.safety.pipeline import LayerResult, RegexLayer, SafetyPipeline

_REAL_ASYNC_CLIENT = httpx.AsyncClient


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
) -> list[httpx.AsyncClient]:
    """Route the handler's internal httpx.AsyncClient through a MockTransport."""
    clients: list[httpx.AsyncClient] = []

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.setdefault("transport", httpx.MockTransport(handler))
        client = _REAL_ASYNC_CLIENT(*args, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return clients


async def _collect_body(resp: StreamingResponse) -> str:
    parts: list[str] = []
    async for chunk in resp.body_iterator:
        assert isinstance(chunk, str)
        parts.append(chunk)
    return "".join(parts)


async def _call_handler(
    on_complete: Any = None,
    *,
    on_terminal: Any = None,
    registry: DetectorRegistry | None = None,
    call_id: str = "call-test",
    evaluation_enabled: bool = True,
    evaluation_runner: Any = None,
    pipeline: Any = None,
) -> Any:
    return await handle_streaming_request(
        target_url="https://upstream.test/v1/chat/completions",
        body_bytes=b'{"stream": true}',
        headers={"Content-Type": "application/json"},
        call_id=call_id,
        input_text="hi",
        registry=registry or DetectorRegistry(),
        policy=EnforcementPolicy(),
        pipeline=pipeline,
        on_complete=on_complete,
        on_terminal=on_terminal,
        evaluation_enabled=evaluation_enabled,
        evaluation_runner=evaluation_runner,
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
    assert out == sse_body.decode()
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


_COMPLETE_SSE = (
    b'data: {"model": "gpt-4o", "choices": [{"delta": {"content": "Hello"}}]}\n\ndata: [DONE]\n\n'
)


class _Detector:
    name = "test_detector"

    def __init__(self, severity: str | None = None, *, fails: bool = False) -> None:
        self.severity = severity
        self.fails = fails
        self.calls = 0

    async def check(
        self, *, input_text: str, output_text: str, call_id: str, **kwargs: Any
    ) -> list[SafetySignal]:
        self.calls += 1
        if self.fails:
            raise RuntimeError("secret detector failure")
        if self.severity is None:
            return []
        return [
            SafetySignal(
                signal_type="test_type",
                severity=self.severity,
                call_id=call_id,
                detail="sensitive output text",
            )
        ]


@pytest.mark.parametrize(
    ("severity", "action"),
    [(None, "pass"), ("medium", "flag"), ("high", "block")],
)
async def test_terminal_completed_decisions_and_exact_wire(
    monkeypatch: pytest.MonkeyPatch, severity: str | None, action: str
) -> None:
    clients = _patch_transport(monkeypatch, lambda _: httpx.Response(200, content=_COMPLETE_SSE))
    registry = DetectorRegistry()
    detector = _Detector(severity)
    registry.add(detector)
    outcomes: list[TerminalOutcome] = []
    completions: list[dict[str, Any]] = []
    call_id = "gateway/opaque:001"
    response = await _call_handler(
        on_complete=lambda **kw: completions.append(kw),
        on_terminal=outcomes.append,
        registry=registry,
        call_id=call_id,
    )
    assert response.headers["x-aep-call-id"] == call_id
    body = await _collect_body(response)
    assert body.startswith(_COMPLETE_SSE.decode())
    assert ("aep_safety_block" in body) == (action == "block")
    if action != "block":
        assert body == _COMPLETE_SSE.decode()
    assert len(completions) == len(outcomes) == detector.calls == 1
    assert completions[0]["decision"].action == action
    assert outcomes[0].call_id == call_id
    assert outcomes[0].transport == "completed"
    assert outcomes[0].evaluation == "completed"
    assert outcomes[0].action == action
    assert outcomes[0].scope == "observed_output"
    assert outcomes[0].http_status == 200
    assert "sensitive output text" not in repr(outcomes[0])
    assert all(client.is_closed for client in clients)


@pytest.mark.parametrize("enabled", [False, True])
async def test_terminal_not_evaluated_for_disabled_or_zero_detectors(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    _patch_transport(monkeypatch, lambda _: httpx.Response(200, content=_COMPLETE_SSE))
    registry = DetectorRegistry()
    detector = _Detector("high")
    if not enabled:
        registry.add(detector)
    outcomes: list[TerminalOutcome] = []
    response = await _call_handler(
        on_terminal=outcomes.append, registry=registry, evaluation_enabled=enabled
    )
    assert await _collect_body(response) == _COMPLETE_SSE.decode()
    assert len(outcomes) == 1
    assert outcomes[0].evaluation == "not_evaluated"
    assert outcomes[0].action is None
    assert outcomes[0].scope == "observed_output"
    assert detector.calls == 0


async def test_terminal_detector_failure_is_unavailable_even_with_empty_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transport(monkeypatch, lambda _: httpx.Response(200, content=_COMPLETE_SSE))
    registry = DetectorRegistry()
    registry.add(_Detector(fails=True))
    outcomes: list[TerminalOutcome] = []
    response = await _call_handler(on_terminal=outcomes.append, registry=registry)
    assert await _collect_body(response) == _COMPLETE_SSE.decode()
    assert len(outcomes) == 1
    assert outcomes[0].evaluation == "unavailable"
    assert outcomes[0].action is None
    assert outcomes[0].signals == ()
    assert "secret detector failure" not in repr(outcomes[0])


async def test_terminal_policy_aware_runner_and_missing_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transport(monkeypatch, lambda _: httpx.Response(200, content=_COMPLETE_SSE))
    outcomes: list[TerminalOutcome] = []

    async def runner(*, input_text: str, output_text: str, call_id: str) -> EvaluationResult:
        assert (input_text, output_text, call_id) == ("hi", "Hello", "call-test")
        return EvaluationResult([], EnforcementDecision(action="flag"), "completed")

    response = await _call_handler(on_terminal=outcomes.append, evaluation_runner=runner)
    assert await _collect_body(response) == _COMPLETE_SSE.decode()
    assert outcomes[0].action == "flag"

    outcomes.clear()

    async def no_decision(*, input_text: str, output_text: str, call_id: str) -> EvaluationResult:
        return EvaluationResult([], None, "completed")

    response = await _call_handler(on_terminal=outcomes.append, evaluation_runner=no_decision)
    await _collect_body(response)
    assert outcomes[0].evaluation == "unavailable"
    assert outcomes[0].action is None

    outcomes.clear()

    async def swallowed_failure(
        *, input_text: str, output_text: str, call_id: str
    ) -> EvaluationResult:
        return EvaluationResult([], EnforcementDecision(action="pass"), "unavailable")

    response = await _call_handler(on_terminal=outcomes.append, evaluation_runner=swallowed_failure)
    await _collect_body(response)
    assert outcomes[0].evaluation == "unavailable"
    assert outcomes[0].action is None


async def test_terminal_callback_exception_does_not_change_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transport(monkeypatch, lambda _: httpx.Response(200, content=_COMPLETE_SSE))
    registry = DetectorRegistry()
    registry.add(_Detector())
    outcomes: list[TerminalOutcome] = []

    def callback(outcome: TerminalOutcome) -> None:
        outcomes.append(outcome)
        raise RuntimeError("callback failure")

    response = await _call_handler(on_terminal=callback, registry=registry)
    assert await _collect_body(response) == _COMPLETE_SSE.decode()
    assert len(outcomes) == 1
    assert outcomes[0].action == "pass"


async def test_terminal_pipeline_swallowed_layer_failure_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transport(monkeypatch, lambda _: httpx.Response(200, content=_COMPLETE_SSE))

    class FailedLayer:
        name = "failed"
        prior_p_safe = 0.95

        async def score(self, **kwargs: Any) -> LayerResult:
            raise RuntimeError("hidden layer failure")

    class PassedLayer:
        name = "passed"
        prior_p_safe = 0.95

        async def score(self, **kwargs: Any) -> LayerResult:
            return LayerResult(layer_name=self.name, p_safe=1.0)

    pipeline = SafetyPipeline([FailedLayer(), PassedLayer()])
    outcomes: list[TerminalOutcome] = []
    response = await _call_handler(on_terminal=outcomes.append, pipeline=pipeline)
    await _collect_body(response)
    assert len(outcomes) == 1
    assert outcomes[0].evaluation == "unavailable"
    assert outcomes[0].action is None


async def test_terminal_pipeline_nested_detector_failure_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transport(monkeypatch, lambda _: httpx.Response(200, content=_COMPLETE_SSE))
    detector = _Detector(fails=True)
    pipeline = SafetyPipeline([RegexLayer([detector])])
    outcomes: list[TerminalOutcome] = []
    response = await _call_handler(on_terminal=outcomes.append, pipeline=pipeline)
    assert await _collect_body(response) == _COMPLETE_SSE.decode()
    assert detector.calls == 1
    assert len(outcomes) == 1
    assert outcomes[0].transport == "completed"
    assert outcomes[0].evaluation == "unavailable"
    assert outcomes[0].action is None
    assert outcomes[0].signals == ()


async def test_terminal_runner_exception_preserves_original_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transport(monkeypatch, lambda _: httpx.Response(200, content=_COMPLETE_SSE))
    outcomes: list[TerminalOutcome] = []
    completions: list[dict[str, Any]] = []

    async def runner(*, input_text: str, output_text: str, call_id: str) -> EvaluationResult:
        raise ValueError("detector failed")

    response = await _call_handler(
        on_terminal=outcomes.append,
        on_complete=lambda **kw: completions.append(kw),
        evaluation_runner=runner,
    )
    with pytest.raises(ValueError, match="detector failed"):
        await _collect_body(response)
    assert len(outcomes) == 1
    assert outcomes[0].transport == "completed"
    assert outcomes[0].evaluation == "unavailable"
    assert outcomes[0].action is None
    assert completions == []


async def test_terminal_prestream_http_error_and_callback_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients = _patch_transport(monkeypatch, lambda _: httpx.Response(429, content=b"private error"))
    outcomes: list[TerminalOutcome] = []

    def callback(outcome: TerminalOutcome) -> None:
        outcomes.append(outcome)
        raise RuntimeError("callback failed")

    response = await _call_handler(on_terminal=callback, call_id="caller-issued-id")
    assert response.status_code == 429
    assert response.body == b"private error"
    assert response.headers["x-aep-call-id"] == "caller-issued-id"
    assert len(outcomes) == 1
    assert outcomes[0].transport == "http_error"
    assert outcomes[0].evaluation == "not_evaluated"
    assert outcomes[0].action is None
    assert outcomes[0].scope == "not_applicable"
    assert "private error" not in repr(outcomes[0])
    assert all(client.is_closed for client in clients)


async def test_terminal_prestream_network_error_preserves_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private network detail")

    clients = _patch_transport(monkeypatch, handler)
    outcomes: list[TerminalOutcome] = []
    with pytest.raises(httpx.ConnectError, match="private network detail"):
        await _call_handler(on_terminal=outcomes.append)
    assert len(outcomes) == 1
    assert outcomes[0].transport == "network_error"
    assert outcomes[0].action is None
    assert outcomes[0].scope == "not_applicable"
    assert "private network detail" not in repr(outcomes[0])
    assert all(client.is_closed for client in clients)


async def test_terminal_midstream_error_and_truncated_eof_never_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n'
            raise httpx.ReadError("private upstream detail")

    for stream, expected in [(FailingStream(), "stream_error"), (None, "interrupted")]:
        content = b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n'
        clients = _patch_transport(
            monkeypatch,
            lambda _, stream=stream, content=content: httpx.Response(
                200, stream=stream if stream is not None else httpx.ByteStream(content)
            ),
        )
        registry = DetectorRegistry()
        registry.add(_Detector())
        outcomes: list[TerminalOutcome] = []
        response = await _call_handler(on_terminal=outcomes.append, registry=registry)
        body = await _collect_body(response)
        assert len(outcomes) == 1
        assert outcomes[0].transport == expected
        assert outcomes[0].evaluation == "interrupted"
        assert outcomes[0].action is None
        assert "private upstream detail" not in repr(outcomes[0])
        assert all(client.is_closed for client in clients)
        assert ("event: error" in body) == (expected == "stream_error")


async def test_terminal_consumer_close_and_task_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waiting = asyncio.Event()
    started = asyncio.Event()

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n'
            started.set()
            await waiting.wait()
            yield b"data: [DONE]\n\n"

    for cancel_task in (False, True):
        waiting.clear()
        started.clear()
        clients = _patch_transport(monkeypatch, lambda _: httpx.Response(200, stream=SlowStream()))
        outcomes: list[TerminalOutcome] = []
        response = await _call_handler(on_terminal=outcomes.append)
        iterator = response.body_iterator
        assert "Hi" in await anext(iterator)
        if cancel_task:
            assert await anext(iterator) == "\n"
            task = asyncio.create_task(anext(iterator))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await iterator.aclose()
        assert len(outcomes) == 1
        assert outcomes[0].transport == "interrupted"
        assert outcomes[0].evaluation == "interrupted"
        assert outcomes[0].action is None
        assert all(client.is_closed for client in clients)


async def test_terminal_close_before_first_byte_cleans_up_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TrackingStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield _COMPLETE_SSE

        async def aclose(self) -> None:
            self.closed = True

    upstream_stream = TrackingStream()
    clients = _patch_transport(monkeypatch, lambda _: httpx.Response(200, stream=upstream_stream))
    outcomes: list[TerminalOutcome] = []
    response = await _call_handler(on_terminal=outcomes.append, call_id="opaque-before-first-byte")
    assert response.headers["x-aep-call-id"] == "opaque-before-first-byte"
    await response.body_iterator.aclose()
    await response.body_iterator.aclose()
    assert upstream_stream.closed
    assert all(client.is_closed for client in clients)
    assert len(outcomes) == 1
    assert outcomes[0].call_id == "opaque-before-first-byte"
    assert outcomes[0].transport == "interrupted"
    assert outcomes[0].evaluation == "interrupted"
    assert outcomes[0].action is None


async def test_app_streaming_400_passes_through_upstream_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full proxy route: stream=true + upstream 400 -> 400 JSON, not empty 200 SSE."""
    from aceteam_aep.proxy.app import create_proxy_app

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
    monkeypatch.setattr("aceteam_aep.proxy.app.default_custom_policies", lambda: ())
    app = create_proxy_app(detectors=[])
    async with _REAL_ASYNC_CLIENT(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            headers={"Authorization": "Bearer sk-test"},
        )

    assert resp.status_code == 400
    assert resp.json() == error
    assert resp.headers["content-type"].startswith("application/json")
