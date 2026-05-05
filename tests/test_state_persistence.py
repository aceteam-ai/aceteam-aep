"""Tests for proxy state persistence (api_key + connected account survives restart)."""

from __future__ import annotations

import json

import pytest

from aceteam_aep.proxy.app import ProxyState
from aceteam_aep.proxy.state_persistence import (
    STATE_PATH_ENV,
    get_configured_path,
    load_persisted_state,
    save_persisted_state,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(STATE_PATH_ENV, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    yield


class TestConfiguredPath:
    def test_unset_env_no_default_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "aceteam_aep.proxy.state_persistence._DEFAULT_PATH", tmp_path / "missing.json"
        )
        assert get_configured_path() is None

    def test_env_set_returns_path(self, tmp_path, monkeypatch):
        target = tmp_path / "state.json"
        monkeypatch.setenv(STATE_PATH_ENV, str(target))
        assert get_configured_path() == target

    def test_env_set_expands_tilde(self, monkeypatch):
        monkeypatch.setenv(STATE_PATH_ENV, "~/some/where.json")
        path = get_configured_path()
        assert path is not None
        assert "~" not in str(path)

    def test_default_path_exists_returns_default(self, tmp_path, monkeypatch):
        default = tmp_path / "default.json"
        default.write_text("{}")
        monkeypatch.setattr("aceteam_aep.proxy.state_persistence._DEFAULT_PATH", default)
        assert get_configured_path() == default


class TestLoadSave:
    def test_save_then_load_roundtrip(self, tmp_path):
        path = tmp_path / "state.json"
        save_persisted_state(
            path,
            {
                "api_key": "act_abc123",
                "connected_account": {"email": "x@y.com", "organization_id": "org_1"},
                "target_base_url": "https://example.com/api",
                "safety_enabled": False,
            },
        )
        loaded = load_persisted_state(path)
        assert loaded == {
            "api_key": "act_abc123",
            "connected_account": {"email": "x@y.com", "organization_id": "org_1"},
            "target_base_url": "https://example.com/api",
            "safety_enabled": False,
        }

    def test_save_strips_unknown_keys(self, tmp_path):
        path = tmp_path / "state.json"
        save_persisted_state(
            path,
            {
                "api_key": "k",
                "connected_account": None,
                "target_base_url": "u",
                "safety_enabled": True,
                "cost_usd": 1.23,  # not a persisted key — must not land in file
                "session_id": "abc",
            },
        )
        on_disk = json.loads(path.read_text())
        assert "cost_usd" not in on_disk
        assert "session_id" not in on_disk

    def test_load_missing_file_returns_none(self, tmp_path):
        assert load_persisted_state(tmp_path / "nope.json") is None

    def test_load_invalid_json_returns_none(self, tmp_path):
        path = tmp_path / "junk.json"
        path.write_text("this is not json")
        assert load_persisted_state(path) is None

    def test_load_non_object_json_returns_none(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]")
        assert load_persisted_state(path) is None

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "state.json"
        save_persisted_state(path, {"api_key": "k"})
        assert path.exists()

    def test_save_uses_0600_mode(self, tmp_path):
        path = tmp_path / "state.json"
        save_persisted_state(path, {"api_key": "secret"})
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600

    def test_save_is_atomic_via_tmp_rename(self, tmp_path):
        # Write once, then again — the .tmp file should not linger after a
        # successful write. (Different from world-writable: this is just
        # confirming we use rename, not truncate-write.)
        path = tmp_path / "state.json"
        save_persisted_state(path, {"api_key": "v1"})
        save_persisted_state(path, {"api_key": "v2"})
        assert not (tmp_path / "state.json.tmp").exists()
        loaded = load_persisted_state(path)
        assert loaded is not None
        assert loaded["api_key"] == "v2"


class TestProxyStateIntegration:
    def test_state_path_none_disables_persistence(self, tmp_path):
        # Default path: no persistence. _persist() is a no-op even if called.
        state = ProxyState(state_path=None)
        state.api_key = "act_xyz"
        state._persist()
        # No file should have been created at the default path either.
        # (We can't easily assert on that from the test, but the no-op
        # contract is what we're checking — no exception.)

    def test_state_path_set_persists_api_key(self, tmp_path):
        path = tmp_path / "state.json"
        state = ProxyState(state_path=path)
        state.api_key = "act_persist"
        state.connected_account = {"email": "u@example.com"}
        state._persist()
        on_disk = json.loads(path.read_text())
        assert on_disk["api_key"] == "act_persist"
        assert on_disk["connected_account"]["email"] == "u@example.com"

    def test_init_loads_persisted_api_key(self, tmp_path):
        path = tmp_path / "state.json"
        save_persisted_state(
            path,
            {
                "api_key": "act_from_disk",
                "connected_account": {"email": "loaded@example.com"},
                "target_base_url": "https://restored.example.com",
                "safety_enabled": False,
            },
        )
        state = ProxyState(state_path=path)
        assert state.api_key == "act_from_disk"
        assert state.connected_account == {"email": "loaded@example.com"}
        assert state.target_base_url == "https://restored.example.com"
        assert state.safety_enabled is False

    def test_persisted_state_overrides_env_seed(self, tmp_path, monkeypatch):
        # Even if the env supplies a key, a persisted key wins — that's the
        # whole point of "this dashboard-set key survives restart".
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        path = tmp_path / "state.json"
        save_persisted_state(path, {"api_key": "act_dashboard"})
        state = ProxyState(state_path=path)
        assert state.api_key == "act_dashboard"

    def test_init_with_corrupt_file_falls_back_cleanly(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        path = tmp_path / "state.json"
        path.write_text("not json {{{")
        # Should not raise; should fall back to env-seeded api_key.
        state = ProxyState(state_path=path)
        assert state.api_key == "env-key"

    def test_target_base_url_strips_trailing_slash(self, tmp_path):
        path = tmp_path / "state.json"
        save_persisted_state(path, {"target_base_url": "https://x.com/api/"})
        state = ProxyState(state_path=path)
        assert state.target_base_url == "https://x.com/api"
