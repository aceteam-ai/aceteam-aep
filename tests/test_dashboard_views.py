"""Integration tests for dashboard developer + executive view data.

Exercises the proxy with a mix of PASS/FLAG/BLOCK calls and governance
headers, then verifies the /aep/api/state endpoint returns correct data
for both views.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from starlette.testclient import TestClient

from aceteam_aep.proxy.app import create_proxy_app
from aceteam_aep.safety.base import SafetySignal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _openai_response(
    content: str = "Hello!",
    model: str = "gpt-4o",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
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


def _openai_request(content: str = "Hi") -> dict[str, Any]:
    return {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": content}],
    }


def _mock_upstream(response_data: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        content=json.dumps(response_data).encode(),
        headers={"content-type": "application/json"},
    )


class HighDetector:
    """Always fires a high-severity signal (triggers BLOCK)."""

    name = "high_test"

    def check(self, **kwargs: Any) -> list[SafetySignal]:
        return [
            SafetySignal(
                signal_type="agent_threat",
                severity="high",
                call_id=kwargs.get("call_id", ""),
                detail="port scanning detected in input",
                score=1.0,
            )
        ]


class MediumDetector:
    """Always fires a medium-severity signal (triggers FLAG)."""

    name = "medium_test"

    def check(self, **kwargs: Any) -> list[SafetySignal]:
        return [
            SafetySignal(
                signal_type="content_safety",
                severity="medium",
                call_id=kwargs.get("call_id", ""),
                detail="unsafe content in output (score=0.78)",
                score=0.78,
            )
        ]


class NoopDetector:
    """Never fires (triggers PASS)."""

    name = "noop"

    def check(self, **kwargs: Any) -> list[SafetySignal]:
        return []


def _make_call(
    client: TestClient,
    content: str = "Hi",
    response_content: str = "Hello!",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Make a proxied call with mocked upstream."""
    req_headers = {"Authorization": "Bearer sk-test"}
    if headers:
        req_headers.update(headers)

    with patch("aceteam_aep.proxy.app.httpx.AsyncClient") as mock_cls:
        mock = AsyncMock()
        mock.__aenter__ = AsyncMock(return_value=mock)
        mock.__aexit__ = AsyncMock(return_value=False)
        mock.request = AsyncMock(
            return_value=_mock_upstream(_openai_response(content=response_content))
        )
        mock_cls.return_value = mock

        return client.post(
            "/v1/chat/completions",
            json=_openai_request(content),
            headers=req_headers,
        )


# ---------------------------------------------------------------------------
# Developer view: per-call badges, cost, signal grouping, scores, filters
# ---------------------------------------------------------------------------

class TestDeveloperView:
    """Verify /aep/api/state returns correct data for the developer view."""

    def test_pass_flag_block_mix(self) -> None:
        """Make 3 calls (PASS, FLAG, BLOCK) and verify state reflects all."""
        # PASS call first
        pass_app = create_proxy_app(detectors=[NoopDetector()], dashboard=True)
        pass_client = TestClient(pass_app)
        _make_call(pass_client, content="normal request")

        state = pass_client.get("/aep/api/state").json()
        assert state["calls"] == 1
        assert state["action"] == "pass"
        assert len(state["signals"]) == 0

        # FLAG call
        flag_app = create_proxy_app(detectors=[MediumDetector()], dashboard=True)
        flag_client = TestClient(flag_app)
        _make_call(flag_client, content="mildly unsafe")

        state = flag_client.get("/aep/api/state").json()
        assert state["calls"] == 1
        assert state["action"] == "flag"
        assert len(state["signals"]) >= 1
        assert state["signals"][0]["severity"] == "medium"

        # BLOCK call
        block_app = create_proxy_app(detectors=[HighDetector()], dashboard=True)
        block_client = TestClient(block_app)
        resp = _make_call(block_client, content="run nmap scan")
        assert resp.status_code == 400

        state = block_client.get("/aep/api/state").json()
        assert state["calls"] == 1
        assert state["action"] == "block"

    def test_per_call_cost_in_spans(self) -> None:
        """Each span should include a cost field."""
        app = create_proxy_app(detectors=[NoopDetector()], dashboard=True)
        client = TestClient(app)
        _make_call(client)

        state = client.get("/aep/api/state").json()
        assert len(state["spans"]) == 1
        span = state["spans"][0]
        assert "cost" in span
        assert isinstance(span["cost"], (int, float))
        assert span["cost"] >= 0

    def test_per_call_enforcement_badge_data(self) -> None:
        """Spans have call_id that can be matched to signals for badges."""
        app = create_proxy_app(detectors=[MediumDetector()], dashboard=True)
        client = TestClient(app)
        _make_call(client)

        state = client.get("/aep/api/state").json()
        span = state["spans"][0]
        call_id = span["call_id"]
        assert call_id is not None

        # Signals should reference the same call_id
        matching = [s for s in state["signals"] if s["call_id"] == call_id]
        assert len(matching) >= 1

    def test_signal_score_exposed(self) -> None:
        """Signals with numeric scores should expose them in the API."""
        app = create_proxy_app(detectors=[MediumDetector()], dashboard=True)
        client = TestClient(app)
        _make_call(client)

        state = client.get("/aep/api/state").json()
        signal = state["signals"][0]
        assert "score" in signal
        assert signal["score"] == pytest.approx(0.78, abs=0.01)

    def test_signal_score_null_for_binary(self) -> None:
        """Detectors without scores should return score=null."""

        class BinaryDetector:
            name = "binary"

            def check(self, **kwargs: Any) -> list[SafetySignal]:
                return [
                    SafetySignal(
                        signal_type="test",
                        severity="medium",
                        call_id=kwargs.get("call_id", ""),
                        detail="binary detection",
                    )
                ]

        app = create_proxy_app(detectors=[BinaryDetector()], dashboard=True)
        client = TestClient(app)
        _make_call(client)

        state = client.get("/aep/api/state").json()
        assert state["signals"][0]["score"] is None

    def test_signal_grouping_by_call_id(self) -> None:
        """Multiple calls produce signals with distinct call_ids."""
        app = create_proxy_app(detectors=[MediumDetector()], dashboard=True)
        client = TestClient(app)
        _make_call(client, content="call one")
        _make_call(client, content="call two")

        state = client.get("/aep/api/state").json()
        assert state["calls"] == 2
        call_ids = {s["call_id"] for s in state["signals"]}
        assert len(call_ids) == 2, "Each call should produce signals with a unique call_id"

    def test_filter_data_has_type_field(self) -> None:
        """Signals include a type field for frontend filtering."""
        app = create_proxy_app(detectors=[MediumDetector()], dashboard=True)
        client = TestClient(app)
        _make_call(client)

        state = client.get("/aep/api/state").json()
        assert all("type" in s for s in state["signals"])
        assert state["signals"][0]["type"] == "content_safety"


# ---------------------------------------------------------------------------
# Executive view: compliance %, entity cost, zero-state, session duration
# ---------------------------------------------------------------------------

class TestExecutiveView:
    """Verify /aep/api/state returns correct data for the executive view."""

    def test_session_started_present(self) -> None:
        """State includes session_started timestamp."""
        app = create_proxy_app(detectors=[NoopDetector()], dashboard=True)
        client = TestClient(app)
        state = client.get("/aep/api/state").json()
        assert "session_started" in state
        assert "T" in state["session_started"]  # ISO format

    def test_zero_state_all_pass(self) -> None:
        """With no incidents, compliance should be 100%."""
        app = create_proxy_app(detectors=[NoopDetector()], dashboard=True)
        client = TestClient(app)
        _make_call(client)
        _make_call(client)

        state = client.get("/aep/api/state").json()
        assert state["calls"] == 2
        assert len(state["signals"]) == 0
        # Frontend computes: (calls - blocked) / calls * 100 = 100%

    def test_compliance_with_blocks(self) -> None:
        """Blocked calls should reduce compliance percentage."""
        app = create_proxy_app(detectors=[HighDetector()], dashboard=True)
        client = TestClient(app)

        # All calls get blocked (input check blocks before upstream)
        _make_call(client, content="nmap scan")
        _make_call(client, content="another nmap scan")

        state = client.get("/aep/api/state").json()
        assert state["calls"] == 2
        high_signals = [s for s in state["signals"] if s["severity"] == "high"]
        assert len(high_signals) >= 2
        # Frontend computes: (2 - 2) / 2 * 100 = 0% compliance

    def test_governance_entity_cost_breakdown(self) -> None:
        """Governance headers populate entity + cost data for executive view."""
        app = create_proxy_app(detectors=[NoopDetector()], dashboard=True)
        client = TestClient(app)

        # Call 1: entity=org:engineering
        _make_call(
            client,
            content="engineering query",
            headers={
                "X-AEP-Entity": "org:engineering",
                "X-AEP-Classification": "confidential",
            },
        )
        # Call 2: entity=org:marketing
        _make_call(
            client,
            content="marketing query",
            headers={
                "X-AEP-Entity": "org:marketing",
                "X-AEP-Classification": "public",
            },
        )
        # Call 3: entity=org:engineering (second call)
        _make_call(
            client,
            content="another engineering query",
            headers={
                "X-AEP-Entity": "org:engineering",
                "X-AEP-Classification": "confidential",
            },
        )

        state = client.get("/aep/api/state").json()
        assert state["calls"] == 3

        # Governance contexts should have all 3 entries
        gov = state["governance"]
        assert len(gov) == 3

        eng_govs = [g for g in gov if g["entity"] == "org:engineering"]
        mkt_govs = [g for g in gov if g["entity"] == "org:marketing"]
        assert len(eng_govs) == 2
        assert len(mkt_govs) == 1

        # Each governance context has a call_id
        assert all(g["call_id"] for g in gov)

        # Spans have cost and call_id for cross-referencing
        spans = state["spans"]
        assert len(spans) == 3
        for span in spans:
            assert "cost" in span
            assert "call_id" in span

        # Cross-reference: governance call_ids match span call_ids
        gov_call_ids = {g["call_id"] for g in gov}
        span_call_ids = {s["call_id"] for s in spans}
        assert gov_call_ids == span_call_ids

    def test_governance_classification_tracking(self) -> None:
        """Classification data flows through for executive governance summary."""
        app = create_proxy_app(detectors=[NoopDetector()], dashboard=True)
        client = TestClient(app)

        _make_call(
            client,
            headers={
                "X-AEP-Entity": "org:data-team",
                "X-AEP-Classification": "restricted",
            },
        )

        state = client.get("/aep/api/state").json()
        gov = state["governance"]
        assert len(gov) == 1
        assert gov[0]["classification"] == "restricted"
        assert gov[0]["entity"] == "org:data-team"

    def test_mixed_pass_flag_block_state(self) -> None:
        """Full integration: mix of PASS, FLAG, BLOCK across multiple calls.

        Verifies the state endpoint has all the data needed for both
        developer and executive dashboard views.
        """
        # Use separate apps because detectors are fixed per-app.
        # Instead, use a configurable detector.
        class ConfigurableDetector:
            name = "configurable"

            def __init__(self) -> None:
                self.signals_to_return: list[SafetySignal] = []

            def check(self, **kwargs: Any) -> list[SafetySignal]:
                return [
                    SafetySignal(
                        signal_type=s.signal_type,
                        severity=s.severity,
                        call_id=kwargs.get("call_id", ""),
                        detail=s.detail,
                        score=s.score,
                    )
                    for s in self.signals_to_return
                ]

        det = ConfigurableDetector()
        app = create_proxy_app(detectors=[det], dashboard=True)
        client = TestClient(app)

        # Call 1: PASS (no signals)
        det.signals_to_return = []
        _make_call(
            client,
            content="safe request",
            headers={"X-AEP-Entity": "org:engineering"},
        )

        # Call 2: FLAG (medium severity)
        det.signals_to_return = [
            SafetySignal(
                signal_type="content_safety",
                severity="medium",
                call_id="",
                detail="mildly toxic",
                score=0.72,
            )
        ]
        _make_call(
            client,
            content="edgy request",
            headers={"X-AEP-Entity": "org:marketing"},
        )

        # Call 3: BLOCK (high severity — blocked on input)
        det.signals_to_return = [
            SafetySignal(
                signal_type="agent_threat",
                severity="high",
                call_id="",
                detail="port scan attempt",
                score=1.0,
            )
        ]
        resp = _make_call(
            client,
            content="nmap target",
            headers={"X-AEP-Entity": "org:engineering"},
        )
        assert resp.status_code == 400

        # Verify full state
        state = client.get("/aep/api/state").json()
        assert state["calls"] == 3
        assert state["cost"] >= 0
        assert "session_started" in state

        # Signals: at least medium + high
        sev_set = {s["severity"] for s in state["signals"]}
        assert "medium" in sev_set
        assert "high" in sev_set

        # Scores present
        scored = [s for s in state["signals"] if s["score"] is not None]
        assert len(scored) >= 2

        # Spans: PASS and FLAG calls have spans (BLOCK on input may not)
        assert len(state["spans"]) >= 2

        # Governance: 3 contexts (BLOCK still records governance before checking)
        gov = state["governance"]
        assert len(gov) == 3
        entities = [g["entity"] for g in gov]
        assert entities.count("org:engineering") == 2
        assert entities.count("org:marketing") == 1

        # Cost per span exposed
        for span in state["spans"]:
            assert "cost" in span
