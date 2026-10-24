"""Provider-specific upstream routing and dashboard alias contracts."""

from __future__ import annotations

import argparse
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from starlette.testclient import TestClient

from aceteam_aep.proxy.app import ProxyState, create_proxy_app
from aceteam_aep.proxy.cli import _run_proxy
from aceteam_aep.proxy.upstreams import (
    UpstreamTargets,
    load_provider_targets,
    normalize_upstream_url,
    request_provider,
)
from aceteam_aep.safety.cost_anomaly import CostAnomalyDetector


def _response(model: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": model,
            "choices": [{"message": {"content": "OK"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        },
        headers={"content-type": "application/json"},
    )


def test_upstream_env_contract_and_path_prefixes() -> None:
    targets = load_provider_targets(
        listen_port=8899,
        env={
            "OPENAI_BASE_URL": "https://openai.example.test/tenant/v1/",
            "ANTHROPIC_BASE_URL": "https://anthropic.example.test/region/v1",
        },
    )
    assert targets == UpstreamTargets(
        openai="https://openai.example.test/tenant",
        anthropic="https://anthropic.example.test/region",
    )

    explicit = load_provider_targets(
        listen_port=8899,
        env={
            "UPSTREAM_OPENAI_BASE_URL": "https://upstream.example.test/v1",
            "OPENAI_BASE_URL": "http://aep-proxy:8899/v1",
        },
    )
    assert explicit.openai == "https://upstream.example.test"
    assert normalize_upstream_url("https://node.example.test/api/V1", listen_port=8899) == (
        "https://node.example.test/api"
    )
    assert normalize_upstream_url("http://host.docker.internal:9000/v1", listen_port=8899) == (
        "http://host.docker.internal:9000"
    )


def test_client_alias_pointing_to_proxy_is_ignored_but_explicit_upstream_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    targets = load_provider_targets(
        listen_port=8899,
        env={"OPENAI_BASE_URL": "http://aep-proxy:8899/v1"},
    )
    assert targets.openai is None
    assert "Ignoring OPENAI_BASE_URL" in caplog.text
    with pytest.raises(ValueError, match="points back"):
        load_provider_targets(
            listen_port=8899,
            env={"UPSTREAM_OPENAI_BASE_URL": "http://aep-proxy:8899/v1"},
        )


def test_proxy_cli_loads_env_targets_without_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEP_TARGET", "https://fallback.example.test/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example.test/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.example.test/v1")
    args = argparse.Namespace(
        config=None,
        debug=False,
        target=None,
        port=8899,
        host="127.0.0.1",
        no_dashboard=True,
        no_safety=True,
        detector=None,
        policy=None,
    )

    with (
        patch("aceteam_aep.proxy.app.create_proxy_app") as create_app,
        patch("aceteam_aep.proxy.cli._load_sign_key", return_value=(None, None)),
        patch("uvicorn.Server") as server_cls,
    ):
        server_cls.return_value.serve = AsyncMock()
        _run_proxy(args)

    kwargs = create_app.call_args.kwargs
    assert kwargs["target_base_url"] == "https://fallback.example.test"
    assert kwargs["provider_targets"] == UpstreamTargets(
        openai="https://openai.example.test",
        anthropic="https://anthropic.example.test",
    )


@pytest.mark.parametrize(
    "value",
    [
        "http://localhost:8899/v1",
        "http://localhost:9000/v1",
        "http://127.0.0.1:8899/v1",
        "http://127.7.8.9:8899/v1",
        "http://[::1]:8899/v1",
        "http://aep-proxy:8899/v1",
        "http://safeclaw-proxy:8899/v1",
        "http://host.docker.internal:8899/v1",
        "http://user:password@example.test/v1",
        "https://example.test/v1?token=secret",
        "ftp://example.test/v1",
        "example.test/v1",
    ],
)
def test_rejects_proxy_loops_and_invalid_upstreams(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_upstream_url(value, listen_port=8899)


def test_anthropic_count_tokens_uses_anthropic_upstream() -> None:
    assert request_provider("/v1/messages/count_tokens", {}) == "anthropic"


def test_custom_fallback_does_not_inject_vendor_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "vendor-secret")
    monkeypatch.delenv("AEP_TARGET_API_KEY", raising=False)
    assert ProxyState(target_base_url="https://sovereign.example.test").api_key is None
    app = create_proxy_app(
        target_base_url="https://sovereign.example.test",
        detectors=[CostAnomalyDetector()],
        dashboard=False,
    )

    with patch("aceteam_aep.proxy.app.httpx.AsyncClient") as client_cls:
        upstream = AsyncMock()
        upstream.__aenter__ = AsyncMock(return_value=upstream)
        upstream.__aexit__ = AsyncMock(return_value=False)
        upstream.request = AsyncMock(return_value=_response("custom-model"))
        client_cls.return_value = upstream
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={"model": "custom-model", "messages": []},
            headers={"Authorization": "Bearer caller-key"},
        )

    assert response.status_code == 200
    assert upstream.request.call_args.kwargs["headers"]["Authorization"] == "Bearer caller-key"

    monkeypatch.setenv("AEP_TARGET_API_KEY", "custom-key")
    assert ProxyState(target_base_url="https://sovereign.example.test").api_key == "custom-key"


def test_custom_single_target_uses_explicit_key_for_anthropic_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AEP_TARGET_API_KEY", "custom-key")
    app = create_proxy_app(
        target_base_url="https://node.example.test",
        detectors=[CostAnomalyDetector()],
        dashboard=False,
    )

    with patch("aceteam_aep.proxy.app.httpx.AsyncClient") as client_cls:
        upstream = AsyncMock()
        upstream.__aenter__ = AsyncMock(return_value=upstream)
        upstream.__aexit__ = AsyncMock(return_value=False)
        upstream.request = AsyncMock(return_value=_response("custom-model"))
        client_cls.return_value = upstream
        response = TestClient(app).post(
            "/v1/messages",
            json={"model": "custom-model", "messages": []},
            headers={"x-api-key": "placeholder"},
        )

    assert response.status_code == 200
    headers = upstream.request.call_args.kwargs["headers"]
    assert headers["x-api-key"] == "custom-key"
    assert "Authorization" not in headers


def test_routes_each_provider_with_its_own_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-vendor-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-vendor-secret")
    monkeypatch.setenv("UPSTREAM_OPENAI_API_KEY", "openai-managed")
    monkeypatch.setenv("UPSTREAM_ANTHROPIC_API_KEY", "anthropic-managed")
    targets = UpstreamTargets(
        openai="https://sovereign.example.test/openai",
        anthropic="https://onprem.example.test/anthropic",
    )
    event_store = MagicMock()
    event_store.record = AsyncMock()
    app = create_proxy_app(
        detectors=[CostAnomalyDetector()],
        dashboard=False,
        provider_targets=targets,
        event_store=event_store,
    )
    client = TestClient(app)

    with patch("aceteam_aep.proxy.app.httpx.AsyncClient") as client_cls:
        upstream = AsyncMock()
        upstream.__aenter__ = AsyncMock(return_value=upstream)
        upstream.__aexit__ = AsyncMock(return_value=False)
        upstream.request = AsyncMock(
            side_effect=[_response("custom-model"), _response("custom-model")]
        )
        client_cls.return_value = upstream

        openai_response = client.post(
            "/v1/chat/completions",
            json={"model": "custom-model", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer placeholder"},
        )
        anthropic_response = client.post(
            "/v1/messages",
            json={"model": "custom-model", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"x-api-key": "placeholder", "anthropic-version": "2023-06-01"},
        )

    assert openai_response.status_code == 200
    assert anthropic_response.status_code == 200
    calls = upstream.request.call_args_list
    assert calls[0].kwargs["url"] == "https://sovereign.example.test/openai/v1/chat/completions"
    assert calls[0].kwargs["headers"]["Authorization"] == "Bearer openai-managed"
    assert "x-api-key" not in calls[0].kwargs["headers"]
    assert calls[1].kwargs["url"] == "https://onprem.example.test/anthropic/v1/messages"
    assert calls[1].kwargs["headers"]["x-api-key"] == "anthropic-managed"
    assert "Authorization" not in calls[1].kwargs["headers"]

    # A custom endpoint is not automatically a vendor or a zero-cost local model.
    assert float(openai_response.headers["X-AEP-Cost"]) > 0
    assert float(anthropic_response.headers["X-AEP-Cost"]) > 0
    call_end_providers = [
        call.args[0].provider
        for call in event_store.record.call_args_list
        if call.args[0].type == "call_end"
    ]
    assert call_end_providers == ["sovereign.example.test", "onprem.example.test"]


def test_custom_provider_without_managed_key_keeps_caller_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "default-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-vendor-secret")
    monkeypatch.delenv("UPSTREAM_ANTHROPIC_API_KEY", raising=False)
    app = create_proxy_app(
        detectors=[CostAnomalyDetector()],
        dashboard=False,
        provider_targets=UpstreamTargets(anthropic="https://anthropic.example.test"),
    )

    with patch("aceteam_aep.proxy.app.httpx.AsyncClient") as client_cls:
        upstream = AsyncMock()
        upstream.__aenter__ = AsyncMock(return_value=upstream)
        upstream.__aexit__ = AsyncMock(return_value=False)
        upstream.request = AsyncMock(return_value=_response("custom-model"))
        client_cls.return_value = upstream
        response = TestClient(app).post(
            "/v1/messages",
            json={"model": "custom-model", "messages": []},
            headers={"x-api-key": "caller-key", "anthropic-version": "2023-06-01"},
        )

    assert response.status_code == 200
    assert upstream.request.call_args.kwargs["headers"]["x-api-key"] == "caller-key"


def test_custom_openai_upstream_does_not_inject_vendor_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "vendor-secret")
    monkeypatch.delenv("UPSTREAM_OPENAI_API_KEY", raising=False)
    app = create_proxy_app(
        detectors=[CostAnomalyDetector()],
        dashboard=False,
        provider_targets=UpstreamTargets(openai="https://sovereign.example.test"),
    )

    with patch("aceteam_aep.proxy.app.httpx.AsyncClient") as client_cls:
        upstream = AsyncMock()
        upstream.__aenter__ = AsyncMock(return_value=upstream)
        upstream.__aexit__ = AsyncMock(return_value=False)
        upstream.request = AsyncMock(return_value=_response("custom-model"))
        client_cls.return_value = upstream
        response = TestClient(app).post(
            "/v1/chat/completions",
            json={"model": "custom-model", "messages": []},
            headers={"Authorization": "Bearer caller-key"},
        )

    assert response.status_code == 200
    assert upstream.request.call_args.kwargs["headers"]["Authorization"] == "Bearer caller-key"


@pytest.mark.parametrize(
    ("source", "destination"),
    [
        ("/aep", "/dashboard/"),
        ("/aep/", "/dashboard/"),
        ("/aep/api/state?view=full", "/dashboard/api/state?view=full"),
    ],
)
def test_aep_alias_redirects_to_dashboard(source: str, destination: str) -> None:
    app = create_proxy_app(detectors=[CostAnomalyDetector()], dashboard=True)
    client = TestClient(app, follow_redirects=False)
    response = client.get(source)
    assert response.status_code == 307
    assert response.headers["location"] == destination


def test_aep_alias_is_absent_when_dashboard_is_disabled() -> None:
    app = create_proxy_app(detectors=[CostAnomalyDetector()], dashboard=False)
    assert TestClient(app).get("/aep/").status_code == 404


def test_aep_alias_preserves_api_post_method() -> None:
    app = create_proxy_app(detectors=[CostAnomalyDetector()], dashboard=True)
    response = TestClient(app, follow_redirects=False).post(
        "/aep/api/safety", json={"enabled": True}
    )
    assert response.status_code == 307
    assert response.headers["location"] == "/dashboard/api/safety"
