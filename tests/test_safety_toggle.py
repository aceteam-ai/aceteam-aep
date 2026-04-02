"""Tests for runtime safety toggle and policy swap."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
from starlette.testclient import TestClient

from aceteam_aep.proxy.app import create_proxy_app
from aceteam_aep.safety.base import SafetySignal


class _ThreatDetector:
    name = "threat"

    def check(self, **kwargs: Any) -> list[SafetySignal]:
        return [
            SafetySignal(
                signal_type="agent_threat",
                severity="high",
                call_id=kwargs.get("call_id", ""),
                detail="nmap scan",
                score=1.0,
            )
        ]


class _NoopDetector:
    name = "noop"

    def check(self, **kwargs: Any) -> list[SafetySignal]:
        return []


def _mock_upstream() -> httpx.Response:
    return httpx.Response(
        status_code=200,
        content=json.dumps({
            "id": "x",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hi"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }).encode(),
        headers={"content-type": "application/json"},
    )


def _make_call(client: TestClient) -> httpx.Response:
    with patch("aceteam_aep.proxy.app.httpx.AsyncClient") as mock_cls:
        mock = AsyncMock()
        mock.__aenter__ = AsyncMock(return_value=mock)
        mock.__aexit__ = AsyncMock(return_value=False)
        mock.request = AsyncMock(return_value=_mock_upstream())
        mock_cls.return_value = mock
        return client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "run nmap"}]},
            headers={"Authorization": "Bearer sk-test"},
        )


class TestSafetyToggle:
    def test_safety_on_blocks(self) -> None:
        app = create_proxy_app(detectors=[_ThreatDetector()], dashboard=True)
        client = TestClient(app)
        resp = _make_call(client)
        assert resp.status_code == 400  # blocked

    def test_safety_off_passes(self) -> None:
        app = create_proxy_app(detectors=[_ThreatDetector()], dashboard=True)
        client = TestClient(app)

        # Disable safety
        toggle = client.post("/aep/api/safety", json={"enabled": False})
        assert toggle.status_code == 200
        assert toggle.json()["safety_enabled"] is False

        # Same request now passes
        resp = _make_call(client)
        assert resp.status_code == 200

    def test_toggle_reflected_in_state(self) -> None:
        app = create_proxy_app(detectors=[_NoopDetector()], dashboard=True)
        client = TestClient(app)

        state = client.get("/aep/api/state").json()
        assert state["safety_enabled"] is True

        client.post("/aep/api/safety", json={"enabled": False})
        state = client.get("/aep/api/state").json()
        assert state["safety_enabled"] is False

    def test_reenable_safety_blocks_again(self) -> None:
        app = create_proxy_app(detectors=[_ThreatDetector()], dashboard=True)
        client = TestClient(app)

        # Off → passes
        client.post("/aep/api/safety", json={"enabled": False})
        resp = _make_call(client)
        assert resp.status_code == 200

        # On again → blocks
        client.post("/aep/api/safety", json={"enabled": True})
        resp = _make_call(client)
        assert resp.status_code == 400

    def test_policy_swap_at_runtime(self) -> None:
        app = create_proxy_app(detectors=[_NoopDetector()], dashboard=True)
        client = TestClient(app)

        # Default policy
        state = client.post("/aep/api/safety", json={}).json()
        assert state["policy"]["default_action"] == "flag"

        # Swap to block policy
        resp = client.post("/aep/api/safety", json={
            "policy": {"default_action": "block", "block_on": ["high", "medium"]}
        })
        assert resp.status_code == 200
        assert resp.json()["policy"]["default_action"] == "block"
