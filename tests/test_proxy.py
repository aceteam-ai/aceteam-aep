"""Tests for the AEP reverse proxy."""

from __future__ import annotations

import json
from collections.abc import Sequence
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
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

    def test_uses_dashboard_byok_key_when_header_missing(self) -> None:
        """When a /v1/* request has no Authorization header, the proxy should
        fall back to the BYOK key set via POST /dashboard/api/api-key."""
        app = create_proxy_app(detectors=[CostAnomalyDetector()], dashboard=True)
        client = TestClient(app)

        client.post("/dashboard/api/api-key", json={"api_key": "sk-byok-from-dashboard"})

        with patch("aceteam_aep.proxy.app.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.request = AsyncMock(return_value=_mock_upstream(_openai_response()))
            mock_client_cls.return_value = mock_client

            client.post("/v1/chat/completions", json=_openai_request())

            call_kwargs = mock_client.request.call_args
            assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer sk-byok-from-dashboard"

    def test_byok_overrides_caller_auth(self) -> None:
        """When the proxy has a managed key (env or dashboard BYOK), it
        substitutes its own key for whatever Authorization the caller sent.
        This lets OpenClaw-style clients send a placeholder sentinel and have
        the proxy inject the real credential."""
        app = create_proxy_app(detectors=[CostAnomalyDetector()], dashboard=True)
        client = TestClient(app)

        client.post("/dashboard/api/api-key", json={"api_key": "sk-proxy-managed"})

        with patch("aceteam_aep.proxy.app.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.request = AsyncMock(return_value=_mock_upstream(_openai_response()))
            mock_client_cls.return_value = mock_client

            client.post(
                "/v1/chat/completions",
                json=_openai_request(),
                headers={"Authorization": "Bearer aep-proxy-managed"},
            )

            call_kwargs = mock_client.request.call_args
            assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer sk-proxy-managed"

    def test_proxy_state_inits_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ProxyState picks up OPENAI_API_KEY / ANTHROPIC_API_KEY from env at
        startup, matching the provider implied by target_base_url."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-openai-env")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert ProxyState().api_key == "sk-from-openai-env"

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-anthropic-env")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert (
            ProxyState(target_base_url="https://api.anthropic.com").api_key
            == "sk-from-anthropic-env"
        )

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert ProxyState().api_key is None


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

    def test_dashboard_at_dashboard_path(self) -> None:
        app = create_proxy_app(detectors=[CostAnomalyDetector()], dashboard=True)
        client = TestClient(app)
        resp = client.get("/dashboard/")
        assert resp.status_code == 200
        assert "Agent Safety Net Dashboard" in resp.text

    def test_dashboard_redirects_without_trailing_slash(self) -> None:
        app = create_proxy_app(detectors=[CostAnomalyDetector()], dashboard=True)
        client = TestClient(app, follow_redirects=False)
        resp = client.get("/dashboard")
        assert resp.status_code == 307
        assert resp.headers.get("location") == "/dashboard/"

    def test_dashboard_api_at_dashboard_path(self) -> None:
        app = create_proxy_app(detectors=[CostAnomalyDetector()], dashboard=True)
        client = TestClient(app)
        resp = client.get("/dashboard/api/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "cost" in data
        assert "calls" in data


class TestCostAggregation:
    """Per-span cost totals must include fees and stay in Decimal while summing."""

    def _state_with_two_nodes(self):
        from aceteam_aep.proxy.app import ProxyState
        from aceteam_aep.types import Usage

        state = ProxyState()
        for _ in range(2):
            node = state.cost_tracker.record_llm_cost(
                span_id="span-1",
                model="gpt-4o",
                usage=Usage(prompt_tokens=1000, completion_tokens=1000, total_tokens=2000),
            )
            node.value_added_fee = Decimal("0.01")
            node.platform_fee = Decimal("0.02")
        return state

    def test_span_cost_includes_fees_and_matches_decimal_sum(self) -> None:
        state = self._state_with_two_nodes()
        nodes = state.cost_tracker.get_cost_tree()
        expected = sum((n.total_cost() for n in nodes), Decimal("0"))
        compute_only = sum((n.compute_cost for n in nodes), Decimal("0"))

        lookup = state._cost_by_span_id()

        # Exact: the sum happens in Decimal, with a single conversion at the end.
        assert lookup["span-1"] == float(expected)
        # compute_cost alone would omit the 0.06 of fees across both nodes.
        assert lookup["span-1"] != float(compute_only)
