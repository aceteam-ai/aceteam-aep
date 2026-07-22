"""Integration tests — verify the proxy emits observability events to the event store."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
from starlette.testclient import TestClient

from aceteam_aep.observability.events import FlaggedCall, ObservabilityEvent
from aceteam_aep.observability.store import EventStore
from aceteam_aep.proxy.app import _detect_provider, create_proxy_app
from aceteam_aep.safety.base import SafetyDetector, SafetySignal

# ---------------------------------------------------------------------------
# In-memory event store for testing (avoids async SQLite + event loop issues)
# ---------------------------------------------------------------------------


class InMemoryEventStore:
    """Minimal in-memory EventStore that satisfies the protocol."""

    def __init__(self) -> None:
        self.events: list[ObservabilityEvent] = []
        self.flagged_calls: list[FlaggedCall] = []

    async def record(self, event: ObservabilityEvent) -> None:
        self.events.append(event)

    async def record_flagged_call(self, call: FlaggedCall) -> None:
        self.flagged_calls.append(call)

    async def query_events(self, **kwargs: Any) -> list[ObservabilityEvent]:
        return list(self.events)

    async def query_flagged_calls(self, **kwargs: Any) -> list[FlaggedCall]:
        return list(self.flagged_calls)

    async def update_verdict(self, **kwargs: Any) -> None:
        pass


# Verify it satisfies the protocol at import time
assert isinstance(InMemoryEventStore(), EventStore)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chat_response(
    content: str = "Hello!",
    model: str = "gpt-4o",
    input_tokens: int = 50,
    output_tokens: int = 100,
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-obs",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def _mock_upstream(data: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        content=json.dumps(data).encode(),
        headers={"content-type": "application/json"},
    )


def _make_call(
    client: TestClient,
    messages: list[dict[str, Any]] | None = None,
    model: str = "gpt-4o",
    response: dict[str, Any] | None = None,
) -> httpx.Response:
    """Make a proxied call with mocked upstream."""
    if messages is None:
        messages = [{"role": "user", "content": "Hello"}]
    body = {"model": model, "messages": messages}
    upstream = response or _chat_response(model=model)

    with patch("aceteam_aep.proxy.app.httpx.AsyncClient") as mock_cls:
        mock = AsyncMock()
        mock.__aenter__ = AsyncMock(return_value=mock)
        mock.__aexit__ = AsyncMock(return_value=False)
        mock.request = AsyncMock(return_value=_mock_upstream(upstream))
        mock_cls.return_value = mock

        return client.post(
            "/v1/chat/completions",
            json=body,
            headers={"Authorization": "Bearer sk-test"},
        )


class _ConfigurableDetector(SafetyDetector):
    """Detector whose mode can be switched between pass / flag / block."""

    name = "configurable"

    def __init__(self) -> None:
        self.mode: str = "pass"

    async def check(self, **kwargs: Any) -> Sequence[SafetySignal]:
        call_id = kwargs.get("call_id", "")
        if self.mode == "block":
            return [
                SafetySignal(
                    signal_type="agent_threat",
                    severity="high",
                    call_id=call_id,
                    detail="reverse shell detected",
                    score=1.0,
                )
            ]
        if self.mode == "flag":
            return [
                SafetySignal(
                    signal_type="content_safety",
                    severity="medium",
                    call_id=call_id,
                    detail="mildly toxic output",
                    score=0.75,
                )
            ]
        return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestObservabilityProxyIntegration:
    """Verify that proxy handler emits call_start, enforcement, and call_end events."""

    def test_successful_call_emits_events(self) -> None:
        """A normal PASS call should produce call_start, enforcement, and call_end."""
        store = InMemoryEventStore()
        det = _ConfigurableDetector()
        app = create_proxy_app(detectors=[det], dashboard=False, event_store=store)
        client = TestClient(app)

        resp = _make_call(client)
        assert resp.status_code == 200

        types = [e.type for e in store.events]
        assert "call_start" in types
        assert "enforcement" in types
        assert "call_end" in types

        # call_start has model + provider
        start_ev = next(e for e in store.events if e.type == "call_start")
        assert start_ev.model == "gpt-4o"
        assert start_ev.provider == "openai"

        # enforcement records pass
        enf_ev = next(e for e in store.events if e.type == "enforcement")
        assert enf_ev.action == "pass"

        # call_end has tokens + cost
        end_ev = next(e for e in store.events if e.type == "call_end")
        assert end_ev.tokens_in is not None and end_ev.tokens_in > 0
        assert end_ev.tokens_out is not None and end_ev.tokens_out > 0
        assert end_ev.cost_usd is not None and end_ev.cost_usd >= 0
        assert end_ev.provider == "openai"

        # All events share the same session_id and call_id
        session_ids = {e.session_id for e in store.events}
        assert len(session_ids) == 1
        call_ids = {e.call_id for e in store.events if e.call_id}
        assert len(call_ids) == 1

    def test_input_blocked_emits_events(self) -> None:
        """An input-blocked call should emit safety_signal, enforcement, flagged_call, call_end."""
        store = InMemoryEventStore()
        det = _ConfigurableDetector()
        det.mode = "block"
        app = create_proxy_app(detectors=[det], dashboard=False, event_store=store)
        client = TestClient(app)

        resp = _make_call(client)
        assert resp.status_code == 400

        types = [e.type for e in store.events]
        assert "call_start" in types
        assert "safety_signal" in types
        assert "enforcement" in types
        assert "call_end" in types

        # enforcement is block
        enf_ev = next(e for e in store.events if e.type == "enforcement")
        assert enf_ev.action == "block"

        # call_end has zero tokens (blocked before LLM)
        end_ev = next(e for e in store.events if e.type == "call_end")
        assert end_ev.tokens_in == 0
        assert end_ev.tokens_out == 0
        assert end_ev.latency_ms is None

        # Flagged call was recorded
        assert len(store.flagged_calls) == 1
        fc = store.flagged_calls[0]
        assert fc.action == "block"
        assert fc.output_text is None  # blocked before LLM

    def test_flagged_call_emits_events(self) -> None:
        """A flagged call should record safety_signal, enforcement, and a flagged_call entry."""
        store = InMemoryEventStore()
        det = _ConfigurableDetector()
        det.mode = "flag"
        app = create_proxy_app(detectors=[det], dashboard=False, event_store=store)
        client = TestClient(app)

        resp = _make_call(client)
        assert resp.status_code == 200

        types = [e.type for e in store.events]
        assert "safety_signal" in types
        assert "enforcement" in types

        enf_ev = next(e for e in store.events if e.type == "enforcement")
        assert enf_ev.action == "flag"

        # Flagged call was recorded with output_text
        assert len(store.flagged_calls) == 1
        fc = store.flagged_calls[0]
        assert fc.action == "flag"
        assert fc.output_text is not None

    def test_no_event_store_does_not_break(self) -> None:
        """When event_store is None, the proxy works normally without emitting events."""
        det = _ConfigurableDetector()
        app = create_proxy_app(detectors=[det], dashboard=False, event_store=None)
        client = TestClient(app)

        resp = _make_call(client)
        assert resp.status_code == 200

    def test_multiple_calls_share_session(self) -> None:
        """Multiple calls through the same proxy share the same session_id."""
        store = InMemoryEventStore()
        det = _ConfigurableDetector()
        app = create_proxy_app(detectors=[det], dashboard=False, event_store=store)
        client = TestClient(app)

        _make_call(client)
        _make_call(client)

        session_ids = {e.session_id for e in store.events}
        assert len(session_ids) == 1

        starts = [e for e in store.events if e.type == "call_start"]
        ends = [e for e in store.events if e.type == "call_end"]
        assert len(starts) == 2
        assert len(ends) == 2

        # Different call_ids
        call_ids = {e.call_id for e in starts}
        assert len(call_ids) == 2

    def test_enforcement_includes_policy_metadata(self) -> None:
        """Enforcement events should contain the active policy in metadata."""
        store = InMemoryEventStore()
        det = _ConfigurableDetector()
        app = create_proxy_app(detectors=[det], dashboard=False, event_store=store)
        client = TestClient(app)

        _make_call(client)

        enf_ev = next(e for e in store.events if e.type == "enforcement")
        assert enf_ev.metadata is not None
        assert "policy" in enf_ev.metadata
        assert "block_on" in enf_ev.metadata["policy"]
        assert "flag_on" in enf_ev.metadata["policy"]


class TestDetectProvider:
    """Unit tests for _detect_provider helper."""

    def test_openai(self) -> None:
        assert _detect_provider("https://api.openai.com") == "openai"

    def test_anthropic(self) -> None:
        assert _detect_provider("https://api.anthropic.com") == "anthropic"

    def test_ollama_port(self) -> None:
        assert _detect_provider("http://localhost:11434") == "ollama"

    def test_ollama_name(self) -> None:
        assert _detect_provider("http://ollama.local:8080") == "ollama"

    def test_google(self) -> None:
        assert _detect_provider("https://generativelanguage.googleapis.com") == "google"

    def test_custom_hostname(self) -> None:
        assert _detect_provider("https://custom-llm.example.com") == "custom-llm.example.com"
