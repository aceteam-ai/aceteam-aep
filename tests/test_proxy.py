"""Tests for the AEP reverse proxy."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
from starlette.testclient import TestClient

from aceteam_aep.proxy.app import ProxyState, create_proxy_app
from aceteam_aep.safety.base import SafetyDetector, SafetySignal
from aceteam_aep.safety.cost_anomaly import CostAnomalyDetector


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
    """Create a mock httpx Response (not AsyncMock — httpx.Response is sync)."""
    return httpx.Response(
        status_code=status,
        content=json.dumps(response_data).encode(),
        headers={"content-type": "application/json"},
    )


class TestProxyForwarding:
    """Test that the proxy forwards requests and records cost."""

    def test_forwards_and_records_cost(self) -> None:
        app = create_proxy_app(detectors=[CostAnomalyDetector()], dashboard=False)
        client = TestClient(app)

        with patch("aceteam_aep.proxy.app.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.request = AsyncMock(return_value=_mock_upstream(_openai_response()))
            mock_client_cls.return_value = mock_client

            resp = client.post(
                "/v1/chat/completions",
                json=_openai_request(),
                headers={"Authorization": "Bearer sk-test"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello!"
        assert "X-AEP-Cost" in resp.headers
        assert "X-AEP-Enforcement" in resp.headers
        assert resp.headers["X-AEP-Enforcement"] == "pass"

    def test_passes_auth_header(self) -> None:
        app = create_proxy_app(detectors=[CostAnomalyDetector()], dashboard=False)
        client = TestClient(app)

        with patch("aceteam_aep.proxy.app.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.request = AsyncMock(return_value=_mock_upstream(_openai_response()))
            mock_client_cls.return_value = mock_client

            client.post(
                "/v1/chat/completions",
                json=_openai_request(),
                headers={"Authorization": "Bearer sk-real-key"},
            )

            # Verify the auth header was forwarded
            call_kwargs = mock_client.request.call_args
            assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer sk-real-key"


class TestProxySafetyBlocking:
    """Test that the proxy blocks unsafe content."""

    def test_blocks_pii_in_output(self) -> None:
        from aceteam_aep.safety.pii import PiiDetector

        app = create_proxy_app(
            detectors=[PiiDetector(model_name="nonexistent/force-regex")],
            dashboard=False,
        )
        client = TestClient(app)

        with patch("aceteam_aep.proxy.app.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            # Response contains PII
            mock_client.request = AsyncMock(
                return_value=_mock_upstream(_openai_response(content="His SSN is 123-45-6789"))
            )
            mock_client_cls.return_value = mock_client

            resp = client.post(
                "/v1/chat/completions",
                json=_openai_request("What is John's SSN?"),
                headers={"Authorization": "Bearer sk-test"},
            )

        assert resp.status_code == 400
        data = resp.json()
        assert data["error"]["code"] == "safety_block"
        assert "aep" in data["error"]["type"].lower()

    def test_flags_medium_severity(self) -> None:
        """Medium severity signals should flag (pass through with header)."""

        class MediumDetector(SafetyDetector):
            name = "medium_test"

            async def check(self, **kwargs) -> Sequence[SafetySignal]:
                return [
                    SafetySignal(
                        signal_type="test",
                        severity="medium",
                        call_id=kwargs.get("call_id", ""),
                        detail="medium signal",
                    )
                ]

        app = create_proxy_app(detectors=[MediumDetector()], dashboard=False)
        client = TestClient(app)

        with patch("aceteam_aep.proxy.app.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.request = AsyncMock(return_value=_mock_upstream(_openai_response()))
            mock_client_cls.return_value = mock_client

            resp = client.post(
                "/v1/chat/completions",
                json=_openai_request(),
                headers={"Authorization": "Bearer sk-test"},
            )

        # Flagged but not blocked — response passes through
        assert resp.status_code == 200
        assert resp.headers["X-AEP-Enforcement"] == "flag"
        assert "X-AEP-Flag-Reason" in resp.headers


class TestProxyState:
    """Test the proxy state tracking."""

    def test_state_accumulates(self) -> None:
        state = ProxyState(detectors=[CostAnomalyDetector()])
        assert state.cost_usd == 0
        assert state.call_count == 0
        assert state.latest_enforcement.action == "pass"

    def test_state_to_dict(self) -> None:
        state = ProxyState(detectors=[CostAnomalyDetector()])
        d = state.to_dict()
        assert d["cost"] == 0.0
        assert d["calls"] == 0
        assert d["action"] == "pass"
        assert isinstance(d["signals"], list)
        assert isinstance(d["spans"], list)


class TestProxyDashboard:
    """Test that the dashboard is mounted on the proxy."""

    def test_dashboard_at_aep_path(self) -> None:
        app = create_proxy_app(detectors=[CostAnomalyDetector()], dashboard=True)
        client = TestClient(app)
        resp = client.get("/aep/")
        assert resp.status_code == 200
        assert "AEP Dashboard" in resp.text

    def test_dashboard_redirects_aep_without_trailing_slash(self) -> None:
        app = create_proxy_app(detectors=[CostAnomalyDetector()], dashboard=True)
        client = TestClient(app, follow_redirects=False)
        resp = client.get("/aep")
        assert resp.status_code == 307
        assert resp.headers.get("location") == "/aep/"

    def test_dashboard_api_at_aep_path(self) -> None:
        app = create_proxy_app(detectors=[CostAnomalyDetector()], dashboard=True)
        client = TestClient(app)
        resp = client.get("/aep/api/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "cost" in data
        assert "calls" in data
