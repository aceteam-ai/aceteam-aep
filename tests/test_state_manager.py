"""Tests for the per-entity ProxyStateManager (multi-tenant proxy support)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aceteam_aep.proxy.app import ProxyState
from aceteam_aep.proxy.state_manager import (
    ENTITY_HEADER,
    MAX_ENTITIES_ENV,
    MULTI_TENANT_ENV,
    ProxyStateManager,
    entity_for_request,
    get_max_entities,
    is_multi_tenant_enabled,
    per_entity_state_path,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(MULTI_TENANT_ENV, raising=False)
    monkeypatch.delenv(MAX_ENTITIES_ENV, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    yield


def _request_with_headers(**headers) -> MagicMock:
    """Build a minimal request stand-in with case-insensitive header lookup."""
    req = MagicMock()
    # Starlette's request.headers is case-insensitive; mirror that.
    norm = {k.lower(): v for k, v in headers.items()}
    req.headers = MagicMock()
    req.headers.get = lambda name, default="": norm.get(name.lower(), default)
    return req


class TestIsMultiTenantEnabled:
    def test_unset_returns_false(self):
        assert is_multi_tenant_enabled() is False

    @pytest.mark.parametrize("v", ["1", "true", "TRUE", "yes", "on", "True"])
    def test_truthy_values(self, v, monkeypatch):
        monkeypatch.setenv(MULTI_TENANT_ENV, v)
        assert is_multi_tenant_enabled() is True

    @pytest.mark.parametrize("v", ["0", "false", "no", "off", "garbage", ""])
    def test_falsy_values(self, v, monkeypatch):
        monkeypatch.setenv(MULTI_TENANT_ENV, v)
        assert is_multi_tenant_enabled() is False


class TestGetMaxEntities:
    def test_unset_returns_default(self):
        assert get_max_entities() == 64

    def test_explicit_value(self, monkeypatch):
        monkeypatch.setenv(MAX_ENTITIES_ENV, "10")
        assert get_max_entities() == 10

    def test_invalid_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(MAX_ENTITIES_ENV, "not-a-number")
        assert get_max_entities() == 64

    def test_zero_clamped_to_one(self, monkeypatch):
        monkeypatch.setenv(MAX_ENTITIES_ENV, "0")
        assert get_max_entities() == 1


class TestEntityForRequest:
    def test_explicit_header_wins(self):
        req = _request_with_headers(
            **{
                ENTITY_HEADER: "org:acme",
                "Authorization": "Bearer act_someothertoken",
            }
        )
        assert entity_for_request(req) == "org:acme"

    def test_header_is_stripped(self):
        req = _request_with_headers(**{ENTITY_HEADER: "  org:acme  "})
        assert entity_for_request(req) == "org:acme"

    def test_falls_back_to_authorization_prefix(self):
        req = _request_with_headers(Authorization="Bearer act_8b3abc123def4567")
        assert entity_for_request(req) == "key:act_8b3abc12"

    def test_authorization_lowercase_bearer_is_accepted(self):
        # HTTP scheme matching is case-insensitive.
        req = _request_with_headers(Authorization="bearer act_xyz_long_enough")
        assert entity_for_request(req) == "key:act_xyz_long"

    def test_missing_header_returns_default(self):
        req = _request_with_headers()
        assert entity_for_request(req) == "default"

    def test_authorization_without_bearer_returns_default(self):
        req = _request_with_headers(Authorization="Basic foobar")
        assert entity_for_request(req) == "default"

    def test_empty_bearer_returns_default(self):
        req = _request_with_headers(Authorization="Bearer ")
        assert entity_for_request(req) == "default"


class TestPerEntityStatePath:
    def test_none_base_path_returns_none(self):
        assert per_entity_state_path(None, "any") is None

    def test_default_entity_returns_base_path(self, tmp_path):
        base = tmp_path / "state.json"
        assert per_entity_state_path(base, "default") == base

    def test_non_default_lands_in_states_subdir(self, tmp_path):
        base = tmp_path / "state.json"
        result = per_entity_state_path(base, "org:acme")
        assert result is not None
        assert result.parent == tmp_path / "states"
        assert result.name == "org_acme.json"

    def test_unsafe_chars_sanitized(self, tmp_path):
        base = tmp_path / "state.json"
        result = per_entity_state_path(base, "../../etc/passwd")
        assert result is not None
        # "/" and "." (when leading) are scrubbed to underscores; the path
        # MUST still land inside the states/ subdir.
        assert result.parent == tmp_path / "states"
        assert ".." not in result.name


class TestProxyStateManagerSingleTenant:
    def test_singleton_mode_always_returns_default(self):
        default = ProxyState()
        manager = ProxyStateManager(
            multi_tenant=False,
            max_entities=64,
            default=default,
            factory=lambda eid: ProxyState(),
        )
        assert manager.get_for_entity("org:a") is default
        assert manager.get_for_entity("org:b") is default
        assert manager.get_for_entity("default") is default

    def test_singleton_factory_never_called(self):
        default = ProxyState()
        factory_calls = []
        manager = ProxyStateManager(
            multi_tenant=False,
            max_entities=64,
            default=default,
            factory=lambda eid: factory_calls.append(eid) or ProxyState(),
        )
        manager.get_for_entity("a")
        manager.get_for_entity("b")
        assert factory_calls == []


class TestProxyStateManagerMultiTenant:
    def test_distinct_entities_get_distinct_states(self):
        default = ProxyState()
        manager = ProxyStateManager(
            multi_tenant=True,
            max_entities=64,
            default=default,
            factory=lambda eid: ProxyState(),
        )
        a = manager.get_for_entity("org:a")
        b = manager.get_for_entity("org:b")
        assert a is not b
        assert a is not default

    def test_same_entity_returns_same_state(self):
        default = ProxyState()
        manager = ProxyStateManager(
            multi_tenant=True,
            max_entities=64,
            default=default,
            factory=lambda eid: ProxyState(),
        )
        first = manager.get_for_entity("org:a")
        second = manager.get_for_entity("org:a")
        assert first is second

    def test_default_entity_returns_default_state(self):
        default = ProxyState()
        manager = ProxyStateManager(
            multi_tenant=True,
            max_entities=64,
            default=default,
            factory=lambda eid: ProxyState(),
        )
        assert manager.get_for_entity("default") is default

    def test_lru_evicts_oldest_when_cap_hit(self):
        default = ProxyState()
        manager = ProxyStateManager(
            multi_tenant=True,
            max_entities=3,  # default + 2 entities = full
            default=default,
            factory=lambda eid: ProxyState(),
        )
        a = manager.get_for_entity("a")
        b = manager.get_for_entity("b")
        # Cap hit when we ask for c — a is the oldest non-default, evicted.
        c = manager.get_for_entity("c")
        # b and c still resident (and default of course).
        assert manager.get_for_entity("b") is b
        assert manager.get_for_entity("c") is c
        # Asking for "a" again creates a fresh state — old `a` is gone.
        a2 = manager.get_for_entity("a")
        assert a2 is not a

    def test_default_protected_from_eviction(self):
        default = ProxyState()
        manager = ProxyStateManager(
            multi_tenant=True,
            max_entities=2,  # default + 1 entity
            default=default,
            factory=lambda eid: ProxyState(),
        )
        manager.get_for_entity("a")
        manager.get_for_entity("b")  # would evict — but default must survive
        manager.get_for_entity("c")
        assert manager.get_for_entity("default") is default

    def test_recently_accessed_not_evicted(self):
        default = ProxyState()
        manager = ProxyStateManager(
            multi_tenant=True,
            max_entities=3,
            default=default,
            factory=lambda eid: ProxyState(),
        )
        a = manager.get_for_entity("a")
        manager.get_for_entity("b")
        # Re-access a (move to end of LRU)
        manager.get_for_entity("a")
        # Add c — b should be evicted (older), not a.
        manager.get_for_entity("c")
        assert manager.get_for_entity("a") is a

    def test_get_for_request_uses_header(self):
        default = ProxyState()
        manager = ProxyStateManager(
            multi_tenant=True,
            max_entities=64,
            default=default,
            factory=lambda eid: ProxyState(),
        )
        req_a = _request_with_headers(**{ENTITY_HEADER: "tenant_a"})
        req_b = _request_with_headers(**{ENTITY_HEADER: "tenant_b"})
        a = manager.get_for_request(req_a)
        b = manager.get_for_request(req_b)
        assert a is not b
        assert manager.get_for_request(req_a) is a

    def test_factory_receives_entity_id(self):
        default = ProxyState()
        seen = []
        manager = ProxyStateManager(
            multi_tenant=True,
            max_entities=64,
            default=default,
            factory=lambda eid: seen.append(eid) or ProxyState(),
        )
        manager.get_for_entity("org:acme")
        manager.get_for_entity("org:beta")
        assert seen == ["org:acme", "org:beta"]

    def test_all_entities_lists_resident(self):
        default = ProxyState()
        manager = ProxyStateManager(
            multi_tenant=True,
            max_entities=64,
            default=default,
            factory=lambda eid: ProxyState(),
        )
        manager.get_for_entity("a")
        manager.get_for_entity("b")
        assert sorted(manager.all_entities()) == ["a", "b", "default"]


class TestPerEntityPersistence:
    def test_each_entity_gets_own_state_file(self, tmp_path):
        base = tmp_path / "state.json"
        path_a = per_entity_state_path(base, "org:a")
        path_b = per_entity_state_path(base, "org:b")
        assert path_a != path_b

        state_a = ProxyState(state_path=path_a)
        state_b = ProxyState(state_path=path_b)
        state_a.api_key = "key_for_a"
        state_b.api_key = "key_for_b"
        state_a._persist()
        state_b._persist()

        # Each entity reads back its own key.
        restored_a = ProxyState(state_path=path_a)
        restored_b = ProxyState(state_path=path_b)
        assert restored_a.api_key == "key_for_a"
        assert restored_b.api_key == "key_for_b"
