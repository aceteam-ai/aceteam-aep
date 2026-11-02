"""Tests for the openclaw.json auto-sync from /v1/models."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aceteam_aep.proxy.openclaw_sync import (
    _splice_into_existing,
    _transform_models_to_providers,
    get_configured_path,
    refresh_openclaw_config,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("AEP_OPENCLAW_CONFIG_PATH", raising=False)
    yield


def _gateway_models_response(models: list[dict]) -> dict:
    return {"object": "list", "data": models}


def _make_resp(status_code: int, body: dict | None = None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=body or {})
    return resp


def _patch_httpx(resp_or_exc):
    """Patch httpx.AsyncClient to return resp on .get(), or raise an exception."""
    mock_client = AsyncMock()
    if isinstance(resp_or_exc, Exception):
        mock_client.get = AsyncMock(side_effect=resp_or_exc)
    else:
        mock_client.get = AsyncMock(return_value=resp_or_exc)

    async def _aenter(self):
        return mock_client

    async def _aexit(self, *_):
        return None

    return (
        patch("httpx.AsyncClient.__aenter__", _aenter),
        patch("httpx.AsyncClient.__aexit__", _aexit),
        mock_client,
    )


class TestGetConfiguredPath:
    def test_returns_none_when_unset(self) -> None:
        assert get_configured_path() is None

    def test_returns_path_when_set(self, monkeypatch) -> None:
        monkeypatch.setenv("AEP_OPENCLAW_CONFIG_PATH", "/openclaw-config/openclaw.json")
        assert get_configured_path() == "/openclaw-config/openclaw.json"

    def test_strips_whitespace(self, monkeypatch) -> None:
        monkeypatch.setenv("AEP_OPENCLAW_CONFIG_PATH", "  /a/b.json  ")
        assert get_configured_path() == "/a/b.json"


class TestTransformModelsToProviders:
    def test_groups_by_owned_by(self) -> None:
        out = _transform_models_to_providers(
            [
                {
                    "id": "gpt-4o",
                    "owned_by": "openai",
                    "context_window": 128000,
                    "max_output_tokens": 16384,
                    "modalities": ["text", "image"],
                    "cost_per_million_tokens": {
                        "input": 2.5,
                        "output": 10,
                        "cache_read": 1.25,
                        "cache_write": 0,
                    },
                },
                {
                    "id": "gpt-4o-mini",
                    "owned_by": "openai",
                    "context_window": 128000,
                    "max_output_tokens": 16384,
                    "modalities": ["text"],
                    "cost_per_million_tokens": {
                        "input": 0.15,
                        "output": 0.6,
                        "cache_read": 0.075,
                        "cache_write": 0,
                    },
                },
                {
                    "id": "tr-deepseek-v4-pro",
                    "owned_by": "deepseek",
                    "context_window": 128000,
                    "max_output_tokens": 16384,
                    "modalities": ["text"],
                    "cost_per_million_tokens": {
                        "input": 1.74,
                        "output": 3.48,
                        "cache_read": 0.145,
                        "cache_write": 0,
                    },
                },
            ]
        )
        assert set(out.keys()) == {"openai", "deepseek"}
        assert len(out["openai"]["models"]) == 2
        assert {m["id"] for m in out["openai"]["models"]} == {"gpt-4o", "gpt-4o-mini"}
        assert out["deepseek"]["models"][0]["id"] == "tr-deepseek-v4-pro"
        assert out["openai"]["baseUrl"] == "http://aep-proxy:8899/v1"
        assert out["openai"]["apiKey"] == "aep-proxy-managed"

    def test_preserves_modalities_and_cost_shape(self) -> None:
        out = _transform_models_to_providers(
            [
                {
                    "id": "x",
                    "owned_by": "openai",
                    "context_window": 200000,
                    "max_output_tokens": 8192,
                    "modalities": ["text", "image", "audio"],
                    "cost_per_million_tokens": {
                        "input": 1,
                        "output": 2,
                        "cache_read": 0.1,
                        "cache_write": 0.5,
                    },
                }
            ]
        )
        m = out["openai"]["models"][0]
        assert m["input"] == ["text", "image", "audio"]
        assert m["cost"] == {
            "input": 1,
            "output": 2,
            "cacheRead": 0.1,
            "cacheWrite": 0.5,
        }
        assert m["contextWindow"] == 200000
        assert m["maxTokens"] == 8192

    def test_handles_missing_optional_fields(self) -> None:
        """Models with no modalities / no cost fields shouldn't blow up."""
        out = _transform_models_to_providers(
            [
                {
                    "id": "minimal",
                    # owned_by absent → "unknown"
                    # modalities absent → ["text"]
                    # cost_per_million_tokens absent → zeros
                }
            ]
        )
        m = out["unknown"]["models"][0]
        assert m["input"] == ["text"]
        assert m["cost"]["input"] == 0
        assert m["contextWindow"] == 128000


class TestSpliceIntoExisting:
    def test_preserves_unrelated_sections(self) -> None:
        existing = {
            "gateway": {"mode": "local"},
            "agents": {"defaults": {"model": {"primary": "openai/gpt-4o"}}},
            "plugins": {"entries": {"openai": {"enabled": True}}},
            "models": {"providers": {"oldstuff": "should be replaced"}},
            "meta": {"keep": "this"},
        }
        out = _splice_into_existing(existing, {"openai": {"baseUrl": "x", "models": []}})
        assert out["gateway"] == existing["gateway"]
        assert out["agents"] == existing["agents"]
        assert out["plugins"] == existing["plugins"]
        # models.providers is replaced wholesale
        assert out["models"]["providers"] == {"openai": {"baseUrl": "x", "models": []}}
        # mode forced to "replace" so OpenClaw doesn't merge bundled defaults
        assert out["models"]["mode"] == "replace"
        # meta.keep retained, lastTouchedAt added
        assert out["meta"]["keep"] == "this"
        assert "lastTouchedAt" in out["meta"]

    def test_handles_missing_models_section(self) -> None:
        existing = {"gateway": {"mode": "local"}}
        out = _splice_into_existing(existing, {"openai": {"models": []}})
        assert out["models"]["mode"] == "replace"
        assert out["models"]["providers"] == {"openai": {"models": []}}


class TestRefreshOpenclawConfig:
    @pytest.mark.asyncio
    async def test_writes_file_on_success(self, tmp_path) -> None:
        target = tmp_path / "openclaw.json"
        target.write_text(json.dumps({"gateway": {"mode": "local"}}))

        resp = _make_resp(
            200,
            _gateway_models_response(
                [
                    {
                        "id": "tr-deepseek-v4-pro",
                        "owned_by": "deepseek",
                        "context_window": 128000,
                        "max_output_tokens": 16384,
                        "modalities": ["text"],
                        "cost_per_million_tokens": {
                            "input": 1.74,
                            "output": 3.48,
                            "cache_read": 0.145,
                            "cache_write": 0,
                        },
                    }
                ]
            ),
        )
        p1, p2, mock_client = _patch_httpx(resp)
        with p1, p2:
            status = await refresh_openclaw_config(
                api_key="act_xxxxxxxxxxxxxxxx",
                target_base_url="https://aceteam.ai/api/gateway",
                config_path=str(target),
            )

        assert status["ok"] is True
        assert status["model_count"] == 1
        assert status["restart_required"] is True

        written = json.loads(target.read_text())
        assert written["models"]["mode"] == "replace"
        assert "deepseek" in written["models"]["providers"]
        assert written["gateway"] == {"mode": "local"}  # preserved

        # File mode is world-writable so a host user (jason) can edit even
        # when aep-proxy wrote it as root inside its container.
        assert (target.stat().st_mode & 0o777) == 0o666

        # Confirm the gateway URL was constructed correctly.
        assert mock_client.get.await_args.args[0] == "https://aceteam.ai/api/gateway/v1/models"

    @pytest.mark.asyncio
    async def test_skips_for_non_act_keys(self, tmp_path) -> None:
        target = tmp_path / "openclaw.json"
        original = {"gateway": {"mode": "local"}}
        target.write_text(json.dumps(original))

        status = await refresh_openclaw_config(
            api_key="sk-openai-xyz",
            target_base_url="https://aceteam.ai/api/gateway",
            config_path=str(target),
        )
        assert status["ok"] is False
        assert "non-aceteam" in status["error"].lower()
        # File untouched
        assert json.loads(target.read_text()) == original

    @pytest.mark.asyncio
    async def test_does_not_clobber_when_catalog_empty(self, tmp_path) -> None:
        """Gateway returning 0 models = soft-fail; preserve the existing config."""
        target = tmp_path / "openclaw.json"
        original = {
            "gateway": {"mode": "local"},
            "models": {
                "mode": "replace",
                "providers": {"openai": {"models": [{"id": "gpt-4o"}]}},
            },
        }
        target.write_text(json.dumps(original))

        resp = _make_resp(200, _gateway_models_response([]))
        p1, p2, _client = _patch_httpx(resp)
        with p1, p2:
            status = await refresh_openclaw_config(
                api_key="act_xxxxxxxxxxxxxxxx",
                target_base_url="https://aceteam.ai/api/gateway",
                config_path=str(target),
            )

        assert status["ok"] is False
        assert "empty" in status["error"].lower()
        assert json.loads(target.read_text()) == original

    @pytest.mark.asyncio
    async def test_handles_gateway_5xx(self, tmp_path) -> None:
        target = tmp_path / "openclaw.json"
        target.write_text("{}")
        resp = _make_resp(503)
        p1, p2, _ = _patch_httpx(resp)
        with p1, p2:
            status = await refresh_openclaw_config(
                api_key="act_xxxxxxxxxxxxxxxx",
                target_base_url="https://aceteam.ai/api/gateway",
                config_path=str(target),
            )
        assert status["ok"] is False
        assert "503" in status["error"]

    @pytest.mark.asyncio
    async def test_handles_network_failure(self, tmp_path) -> None:
        target = tmp_path / "openclaw.json"
        target.write_text("{}")
        p1, p2, _ = _patch_httpx(ConnectionError("dns refused"))
        with p1, p2:
            status = await refresh_openclaw_config(
                api_key="act_xxxxxxxxxxxxxxxx",
                target_base_url="https://aceteam.ai/api/gateway",
                config_path=str(target),
            )
        assert status["ok"] is False
        assert "unreachable" in status["error"].lower()

    @pytest.mark.asyncio
    async def test_handles_invalid_target_base_url(self, tmp_path) -> None:
        target = tmp_path / "openclaw.json"
        target.write_text("{}")
        status = await refresh_openclaw_config(
            api_key="act_xxxxxxxxxxxxxxxx",
            target_base_url="not-a-url",
            config_path=str(target),
        )
        assert status["ok"] is False
        assert "invalid" in status["error"].lower()

    @pytest.mark.asyncio
    async def test_creates_file_when_absent(self, tmp_path) -> None:
        """Fresh install: no openclaw.json yet — should still write a valid one."""
        target = tmp_path / "subdir" / "openclaw.json"
        target.parent.mkdir()  # parent must exist for the rename
        # File does not yet exist.

        resp = _make_resp(
            200,
            _gateway_models_response(
                [
                    {
                        "id": "x",
                        "owned_by": "openai",
                        "context_window": 1,
                        "max_output_tokens": 1,
                        "modalities": ["text"],
                        "cost_per_million_tokens": {
                            "input": 0,
                            "output": 0,
                            "cache_read": 0,
                            "cache_write": 0,
                        },
                    }
                ]
            ),
        )
        p1, p2, _ = _patch_httpx(resp)
        with p1, p2:
            status = await refresh_openclaw_config(
                api_key="act_xxxxxxxxxxxxxxxx",
                target_base_url="https://aceteam.ai/api/gateway",
                config_path=str(target),
            )

        assert status["ok"] is True
        assert target.exists()
        written = json.loads(target.read_text())
        # Default scaffold added
        assert written["gateway"] == {"mode": "local"}
        assert "openai" in written["models"]["providers"]


class TestApiKeyHandlerWiring:
    """Verify the api-key handler triggers sync iff AEP_OPENCLAW_CONFIG_PATH is set."""

    def _build_app(self):
        from starlette.testclient import TestClient

        from aceteam_aep.proxy.app import create_proxy_app
        from aceteam_aep.safety.base import SafetyDetector

        class _Noop(SafetyDetector):
            name = "noop"

            async def check(self, **kwargs):
                return ()

        app = create_proxy_app(detectors=[_Noop()], dashboard=True)
        return app, TestClient(app)

    def test_status_endpoint_reports_disabled_when_env_unset(self) -> None:
        _app, client = self._build_app()
        body = client.get("/dashboard/api/openclaw-config-status").json()
        assert body["enabled"] is False
        assert body["status"] is None

    def test_status_endpoint_reports_enabled_when_env_set(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("AEP_OPENCLAW_CONFIG_PATH", str(tmp_path / "openclaw.json"))
        _app, client = self._build_app()
        body = client.get("/dashboard/api/openclaw-config-status").json()
        assert body["enabled"] is True
        assert body["config_path"] == str(tmp_path / "openclaw.json")
        assert body["status"] is None  # no sync attempt yet

    def test_post_api_key_triggers_sync_when_env_set(self, monkeypatch, tmp_path) -> None:
        target = tmp_path / "openclaw.json"
        target.write_text(json.dumps({"gateway": {"mode": "local"}}))
        monkeypatch.setenv("AEP_OPENCLAW_CONFIG_PATH", str(target))

        _app, client = self._build_app()

        # Mock both the whoami refresh and the /v1/models fetch — the api-key
        # handler triggers _refresh_connected_account first, then sync.
        whoami_resp = _make_resp(
            200,
            {
                "auth_type": "api_key",
                "email": "u@example.com",
                "organization_id": "org_xyz",
                "organization_name": "Acme",
            },
        )
        models_resp = _make_resp(
            200,
            _gateway_models_response(
                [
                    {
                        "id": "tr-deepseek-v4-pro",
                        "owned_by": "deepseek",
                        "context_window": 128000,
                        "max_output_tokens": 16384,
                        "modalities": ["text"],
                        "cost_per_million_tokens": {
                            "input": 1.74,
                            "output": 3.48,
                            "cache_read": 0.145,
                            "cache_write": 0,
                        },
                    }
                ]
            ),
        )
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[whoami_resp, models_resp])

        async def _aenter(self):
            return mock_client

        async def _aexit(self, *_):
            return None

        with (
            patch("httpx.AsyncClient.__aenter__", _aenter),
            patch("httpx.AsyncClient.__aexit__", _aexit),
        ):
            saved = client.post(
                "/dashboard/api/api-key",
                json={
                    "api_key": "act_xxxxxxxxxxxxxxxx",
                    "base_url": "http://aceteam.local/api/gateway",
                },
            ).json()

        assert saved["set"] is True

        # File got rewritten with the synced catalog.
        written = json.loads(target.read_text())
        assert "deepseek" in written["models"]["providers"]
        assert written["models"]["mode"] == "replace"

        # Status endpoint surfaces success
        status = client.get("/dashboard/api/openclaw-config-status").json()
        assert status["enabled"] is True
        assert status["status"]["ok"] is True
        assert status["status"]["model_count"] == 1
        assert status["status"]["restart_required"] is True

    def test_post_api_key_skips_sync_when_env_unset(self, tmp_path) -> None:
        """Without AEP_OPENCLAW_CONFIG_PATH, no sync attempt — status stays None."""
        _app, client = self._build_app()
        # No env set, no httpx mocking — sync path must be skipped before hitting network.
        client.post(
            "/dashboard/api/api-key",
            json={"api_key": "sk-byok-only"},
        )
        status = client.get("/dashboard/api/openclaw-config-status").json()
        assert status["enabled"] is False
        assert status["status"] is None
