"""End-to-end proxy verification — mimics OpenClaw-style agent call patterns.

Exercises the full proxy pipeline: multi-turn conversations, tool use
responses, governance headers, streaming flag, mixed safety outcomes,
and verifies all data surfaces correctly in the dashboard API.

This replaces manual E2E verification (#18).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
from starlette.testclient import TestClient

from aceteam_aep.proxy.app import create_proxy_app
from aceteam_aep.safety.base import SafetyDetector, SafetySignal

# ---------------------------------------------------------------------------
# Mock response builders
# ---------------------------------------------------------------------------


def _chat_response(
    content: str = "Hello!",
    model: str = "gpt-4o",
    input_tokens: int = 50,
    output_tokens: int = 100,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """OpenAI chat completion response."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-e2e",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def _tool_call_response(model: str = "gpt-4o") -> dict[str, Any]:
    """Response with a tool call (no text content)."""
    return {
        "id": "chatcmpl-tool",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"query": "AEP safety"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
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
    headers: dict[str, str] | None = None,
    stream: bool = False,
) -> httpx.Response:
    """Make a proxied call with mocked upstream."""
    if messages is None:
        messages = [{"role": "user", "content": "Hello"}]
    body: dict[str, Any] = {"model": model, "messages": messages}
    if stream:
        body["stream"] = True

    req_headers = {"Authorization": "Bearer sk-test"}
    if headers:
        req_headers.update(headers)

    upstream = response or _chat_response()

    with patch("aceteam_aep.proxy.app.httpx.AsyncClient") as mock_cls:
        mock = AsyncMock()
        mock.__aenter__ = AsyncMock(return_value=mock)
        mock.__aexit__ = AsyncMock(return_value=False)
        mock.request = AsyncMock(return_value=_mock_upstream(upstream))
        mock_cls.return_value = mock

        return client.post(
            "/v1/chat/completions",
            json=body,
            headers=req_headers,
        )


# ---------------------------------------------------------------------------
# Configurable detector for controlling test outcomes
# ---------------------------------------------------------------------------


class ScenarioDetector(SafetyDetector):
    """Detector that can be switched between scenarios per-call."""

    name = "scenario"

    def __init__(self) -> None:
        self.mode: str = "pass"  # "pass", "flag", "block"

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
# E2E Tests
# ---------------------------------------------------------------------------


class TestOpenClawStyleE2E:
    """Simulate an OpenClaw-like agent making multiple LLM calls through the proxy."""

    def test_multi_turn_conversation(self) -> None:
        """Multi-turn conversation: all calls tracked, costs accumulate."""
        det = ScenarioDetector()
        app = create_proxy_app(detectors=[det], dashboard=True)
        client = TestClient(app)

        # Turn 1: user asks a question
        _make_call(client, messages=[{"role": "user", "content": "What is AEP?"}])

        # Turn 2: follow-up with history
        _make_call(
            client,
            messages=[
                {"role": "user", "content": "What is AEP?"},
                {"role": "assistant", "content": "AEP is..."},
                {"role": "user", "content": "Tell me more about safety"},
            ],
        )

        # Turn 3: another follow-up
        _make_call(
            client,
            messages=[
                {"role": "user", "content": "What is AEP?"},
                {"role": "assistant", "content": "AEP is..."},
                {"role": "user", "content": "Tell me more about safety"},
                {"role": "assistant", "content": "Safety includes..."},
                {"role": "user", "content": "How does cost tracking work?"},
            ],
        )

        state = client.get("/aep/api/state").json()
        assert state["calls"] == 3
        assert state["cost"] > 0
        assert len(state["spans"]) == 3
        # Each span has a unique call_id
        call_ids = {s["call_id"] for s in state["spans"]}
        assert len(call_ids) == 3

    def test_tool_use_response(self) -> None:
        """Tool call responses (no text content) should not crash safety checks."""
        det = ScenarioDetector()
        app = create_proxy_app(detectors=[det], dashboard=True)
        client = TestClient(app)

        # Call that returns a tool call
        resp = _make_call(
            client,
            messages=[{"role": "user", "content": "Search for AEP docs"}],
            response=_tool_call_response(),
        )
        assert resp.status_code == 200

        # Follow-up with tool result
        _make_call(
            client,
            messages=[
                {"role": "user", "content": "Search for AEP docs"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {"name": "search", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "content": "Found 3 results about AEP safety.",
                },
            ],
        )

        state = client.get("/aep/api/state").json()
        assert state["calls"] == 2
        assert len(state["spans"]) == 2

    def test_mixed_safety_outcomes(self) -> None:
        """Agent session with PASS, FLAG, and BLOCK calls."""
        det = ScenarioDetector()
        app = create_proxy_app(detectors=[det], dashboard=True)
        client = TestClient(app)

        # Call 1: safe
        det.mode = "pass"
        _make_call(client, messages=[{"role": "user", "content": "Hello"}])

        # Call 2: flagged (mildly toxic)
        det.mode = "flag"
        resp2 = _make_call(client, messages=[{"role": "user", "content": "edgy request"}])
        assert resp2.status_code == 200  # flagged, not blocked

        # Call 3: blocked (threat)
        det.mode = "block"
        resp3 = _make_call(client, messages=[{"role": "user", "content": "run nmap scan"}])
        assert resp3.status_code == 400

        # Call 4: back to safe
        det.mode = "pass"
        _make_call(client, messages=[{"role": "user", "content": "Summarize"}])

        state = client.get("/aep/api/state").json()
        assert state["calls"] == 4
        assert state["action"] == "pass"  # latest call was safe

        # Verify signals from flagged + blocked calls
        types = {s["type"] for s in state["signals"]}
        assert "content_safety" in types
        assert "agent_threat" in types

        # Verify scores are present
        scored = [s for s in state["signals"] if s["score"] is not None]
        assert len(scored) >= 2

    def test_governance_headers_flow_through(self) -> None:
        """Governance headers from different entities track correctly."""
        det = ScenarioDetector()
        app = create_proxy_app(detectors=[det], dashboard=True)
        client = TestClient(app)

        # Simulate calls from different orgs/teams
        _make_call(
            client,
            messages=[{"role": "user", "content": "Q1 revenue summary"}],
            headers={
                "X-AEP-Entity": "org:finance",
                "X-AEP-Classification": "confidential",
            },
        )
        _make_call(
            client,
            messages=[{"role": "user", "content": "Draft marketing email"}],
            headers={
                "X-AEP-Entity": "org:marketing",
                "X-AEP-Classification": "internal",
            },
        )
        _make_call(
            client,
            messages=[{"role": "user", "content": "Analyze customer churn"}],
            headers={
                "X-AEP-Entity": "org:finance",
                "X-AEP-Classification": "restricted",
            },
        )

        state = client.get("/aep/api/state").json()
        assert state["calls"] == 3

        # Governance contexts
        gov = state["governance"]
        assert len(gov) == 3
        finance = [g for g in gov if g["entity"] == "org:finance"]
        marketing = [g for g in gov if g["entity"] == "org:marketing"]
        assert len(finance) == 2
        assert len(marketing) == 1
        assert finance[0]["classification"] == "confidential"
        assert finance[1]["classification"] == "restricted"

        # Cross-reference: spans have cost for entity breakdown
        for span in state["spans"]:
            assert "cost" in span
            assert span["cost"] >= 0

    def test_different_models(self) -> None:
        """Calls to different models are tracked separately."""
        det = ScenarioDetector()
        app = create_proxy_app(detectors=[det], dashboard=True)
        client = TestClient(app)

        _make_call(
            client,
            model="gpt-4o",
            response=_chat_response(model="gpt-4o"),
        )
        _make_call(
            client,
            model="gpt-4o-mini",
            response=_chat_response(model="gpt-4o-mini", input_tokens=20, output_tokens=40),
        )

        state = client.get("/aep/api/state").json()
        assert state["calls"] == 2
        models = {s["executor"] for s in state["spans"]}
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models

    def test_upstream_error_handling(self) -> None:
        """Upstream 500 errors should pass through without crashing."""
        det = ScenarioDetector()
        app = create_proxy_app(detectors=[det], dashboard=True)
        client = TestClient(app)

        error_resp = {"error": {"message": "Internal server error", "type": "server_error"}}
        resp = _make_call(
            client,
            messages=[{"role": "user", "content": "Hello"}],
            response=error_resp,
        )
        # The mock returns 200 status, but let's test a real error
        with patch("aceteam_aep.proxy.app.httpx.AsyncClient") as mock_cls:
            mock = AsyncMock()
            mock.__aenter__ = AsyncMock(return_value=mock)
            mock.__aexit__ = AsyncMock(return_value=False)
            mock.request = AsyncMock(return_value=_mock_upstream(error_resp, status=500))
            mock_cls.return_value = mock

            resp = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": "Bearer sk-test"},
            )

        assert resp.status_code == 500
        # Dashboard should still work
        state = client.get("/aep/api/state").json()
        assert "cost" in state

    def test_full_session_state_contract(self) -> None:
        """Verify the complete /aep/api/state schema for dashboard consumption."""
        det = ScenarioDetector()
        app = create_proxy_app(detectors=[det], dashboard=True)
        client = TestClient(app)

        det.mode = "pass"
        _make_call(
            client,
            headers={"X-AEP-Entity": "org:demo"},
        )

        state = client.get("/aep/api/state").json()

        # Top-level fields
        assert isinstance(state["cost"], (int, float))
        assert isinstance(state["calls"], int)
        assert state["action"] in ("pass", "flag", "block")
        assert isinstance(state["reason"], str)
        assert "session_started" in state
        assert "T" in state["session_started"]

        # Signals schema
        assert isinstance(state["signals"], list)

        # Spans schema
        assert isinstance(state["spans"], list)
        assert len(state["spans"]) >= 1
        span = state["spans"][0]
        assert "id" in span
        assert "call_id" in span
        assert "executor" in span
        assert "status" in span
        assert "duration_ms" in span
        assert "started_at" in span
        assert "cost" in span

        # Governance schema
        assert isinstance(state["governance"], list)
        assert len(state["governance"]) >= 1
        gov = state["governance"][0]
        assert "call_id" in gov
        assert "entity" in gov
        assert "classification" in gov

    def test_high_volume_stability(self) -> None:
        """50 rapid calls should not cause state corruption."""
        det = ScenarioDetector()
        app = create_proxy_app(detectors=[det], dashboard=True)
        client = TestClient(app)

        for i in range(50):
            det.mode = "flag" if i % 10 == 0 else "pass"
            _make_call(client, messages=[{"role": "user", "content": f"Call {i}"}])

        state = client.get("/aep/api/state").json()
        assert state["calls"] == 50
        assert len(state["spans"]) == 50
        assert state["cost"] > 0
        # 5 flagged calls (i=0,10,20,30,40), 2 signals each (input+output check)
        flagged = [s for s in state["signals"] if s["severity"] == "medium"]
        assert len(flagged) == 10
