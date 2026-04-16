"""Tests for custom policy CRUD on the proxy."""

from __future__ import annotations

from typing import Any

from starlette.testclient import TestClient

from aceteam_aep.proxy.app import create_proxy_app
from aceteam_aep.safety.base import SafetySignal


class _NoopDetector:
    name = "noop"

    def check(self, **kwargs: Any) -> list[SafetySignal]:
        return []


class TestCustomPoliciesAPI:
    def test_crud_flow(self) -> None:
        app = create_proxy_app(detectors=[_NoopDetector()], dashboard=True)
        client = TestClient(app)

        listed = client.get("/aep/api/custom-policies").json()
        assert listed == {"policies": []}

        create = client.post(
            "/aep/api/custom-policies",
            json={"name": "Block foo", "rule": "no foo", "enabled": True},
        )
        assert create.status_code == 201
        body = create.json()
        policy_id = body["id"]
        assert body["name"] == "Block foo"
        assert body["rule"] == "no foo"
        assert body["enabled"] is True

        listed = client.get("/aep/api/custom-policies").json()
        assert len(listed["policies"]) == 1
        assert listed["policies"][0]["id"] == policy_id

        one = client.get(f"/aep/api/custom-policies/{policy_id}")
        assert one.status_code == 200
        assert one.json() == body

        put = client.put(
            f"/aep/api/custom-policies/{policy_id}",
            json={"name": "Block foo", "rule": "no foo", "enabled": False},
        )
        assert put.status_code == 200
        assert put.json()["enabled"] is False

        put2 = client.put(
            f"/aep/api/custom-policies/{policy_id}",
            json={"name": "Renamed", "rule": "updated rule", "enabled": True},
        )
        assert put2.status_code == 200
        assert put2.json()["name"] == "Renamed"
        assert put2.json()["rule"] == "updated rule"
        assert put2.json()["enabled"] is True

        deleted = client.delete(f"/aep/api/custom-policies/{policy_id}")
        assert deleted.status_code == 204

        assert client.get(f"/aep/api/custom-policies/{policy_id}").status_code == 404

    def test_put_requires_all_fields(self) -> None:
        app = create_proxy_app(detectors=[_NoopDetector()], dashboard=True)
        client = TestClient(app)
        r = client.post(
            "/aep/api/custom-policies",
            json={"name": "x", "rule": "y", "enabled": True},
        )
        pid = r.json()["id"]
        bad = client.put(f"/aep/api/custom-policies/{pid}", json={"enabled": False})
        assert bad.status_code == 400

    def test_invalid_uuid(self) -> None:
        app = create_proxy_app(detectors=[_NoopDetector()], dashboard=True)
        client = TestClient(app)
        assert client.get("/aep/api/custom-policies/not-a-uuid").status_code == 400

    def test_create_rejects_client_id(self) -> None:
        """POST ignores any client-supplied id; server issues UUID via CustomPolicy."""
        app = create_proxy_app(detectors=[_NoopDetector()], dashboard=True)
        client = TestClient(app)
        r = client.post(
            "/aep/api/custom-policies",
            json={
                "id": "00000000-0000-0000-0000-000000000001",
                "name": "x",
                "rule": "y",
                "enabled": True,
            },
        )
        assert r.status_code == 201
        assert r.json()["id"] != "00000000-0000-0000-0000-000000000001"
