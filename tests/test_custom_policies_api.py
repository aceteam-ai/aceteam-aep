"""Tests for custom policy CRUD on the proxy."""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from aceteam_aep.proxy.app import create_proxy_app
from aceteam_aep.safety.base import SafetyDetector, SafetySignal
from aceteam_aep.safety.custom import CustomPolicy, CustomPolicyStore, CustomSafetyDetector


class _NoopDetector(SafetyDetector):
    name = "noop"

    async def check(self, **kwargs) -> Sequence[SafetySignal]:
        return ()


class TestCustomPoliciesAPI:
    def test_crud_flow(self) -> None:
        app = create_proxy_app(detectors=[_NoopDetector()], dashboard=True)
        client = TestClient(app)

        listed = client.get("/aep/api/custom-policies").json()
        names = {p["name"] for p in listed["policies"]}
        assert names == {"English only", "US SSN pattern (input)"}
        assert all(p["enabled"] is False for p in listed["policies"])

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
        assert body["applies_to"] == "both"
        assert body["severity"] == "high"

        listed = client.get("/aep/api/custom-policies").json()
        assert len(listed["policies"]) == 4
        assert any(p["id"] == policy_id for p in listed["policies"])

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

    def test_create_with_scope_and_severity(self) -> None:
        app = create_proxy_app(detectors=[_NoopDetector()], dashboard=True)
        client = TestClient(app)
        r = client.post(
            "/aep/api/custom-policies",
            json={
                "name": "Out only",
                "rule": "no secrets",
                "enabled": True,
                "applies_to": "output",
                "severity": "low",
            },
        )
        assert r.status_code == 201
        b = r.json()
        assert b["applies_to"] == "output"
        assert b["severity"] == "low"

    def test_put_preserves_scope_and_severity_when_omitted(self) -> None:
        """PUT without applies_to/severity must not reset those fields."""
        app = create_proxy_app(detectors=[_NoopDetector()], dashboard=True)
        client = TestClient(app)
        r = client.post(
            "/aep/api/custom-policies",
            json={
                "name": "Scoped",
                "rule": "r",
                "enabled": True,
                "applies_to": "input",
                "severity": "medium",
            },
        )
        pid = r.json()["id"]
        put = client.put(
            f"/aep/api/custom-policies/{pid}",
            json={"name": "Renamed", "rule": "new", "enabled": False},
        )
        assert put.status_code == 200
        b = put.json()
        assert b["applies_to"] == "input"
        assert b["severity"] == "medium"

    def test_rejects_multiple_custom_safety_detectors(self) -> None:
        store = CustomPolicyStore()
        with pytest.raises(ValueError, match="At most one CustomSafetyDetector"):
            create_proxy_app(
                detectors=[
                    CustomSafetyDetector(store),
                    CustomSafetyDetector(CustomPolicyStore()),
                ],
                dashboard=False,
            )


class TestCustomSafetyDetector:
    async def test_applies_to_input_only_uses_severity(self) -> None:
        store = CustomPolicyStore()
        store.upsert(
            CustomPolicy(
                name="In",
                rule="r",
                enabled=True,
                applies_to="input",
                severity="low",
            )
        )
        det = CustomSafetyDetector(store)

        async def violation(_self: CustomPolicy, text: str) -> bool:
            return text != "bad"

        with patch.object(CustomPolicy, "__call__", violation):
            sigs = await det.check(
                input_text="bad",
                output_text="ok",
                call_id="c1",
            )
        assert len(sigs) == 1
        assert sigs[0].severity == "low"
        assert "input" in sigs[0].detail

    async def test_applies_to_input_skips_output_text(self) -> None:
        store = CustomPolicyStore()
        store.upsert(
            CustomPolicy(
                name="In",
                rule="r",
                enabled=True,
                applies_to="input",
                severity="medium",
            )
        )
        det = CustomSafetyDetector(store)

        async def violation(_self: CustomPolicy, text: str) -> bool:
            return text != "bad"

        with patch.object(CustomPolicy, "__call__", violation):
            sigs = await det.check(
                input_text="fine",
                output_text="bad",
                call_id="c2",
            )
        assert sigs == []

    async def test_both_checks_output_then_input(self) -> None:
        store = CustomPolicyStore()
        store.upsert(
            CustomPolicy(name="Both", rule="r", enabled=True, applies_to="both", severity="high")
        )
        det = CustomSafetyDetector(store)
        checked: list[str] = []

        async def track(_self: CustomPolicy, text: str) -> bool:
            checked.append(text)
            return True

        with patch.object(CustomPolicy, "__call__", track):
            await det.check(input_text="in", output_text="out", call_id="c3")
        assert checked == ["out", "in"]
