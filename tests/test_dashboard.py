"""Tests for the AEP local web dashboard."""

from __future__ import annotations

from decimal import Decimal

from starlette.testclient import TestClient

from aceteam_aep.dashboard.app import create_app


def _mock_state() -> dict:
    return {
        "cost": Decimal("0.0042"),
        "calls": 3,
        "action": "pass",
        "reason": "",
        "signals": [],
        "spans": [
            {
                "id": "abc",
                "executor": "gpt-4o",
                "status": "OK",
                "duration_ms": 120.5,
                "started_at": "2026-03-21T00:00:00+00:00",
            }
        ],
    }


def test_dashboard_returns_200() -> None:
    app = create_app(get_state=_mock_state)
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "AEP Dashboard" in resp.text


def test_dashboard_api_json() -> None:
    app = create_app(get_state=_mock_state)
    client = TestClient(app)
    resp = client.get("/api/state")
    assert resp.status_code == 200
    data = resp.json()
    assert data["calls"] == 3
    assert data["action"] == "pass"
    assert len(data["spans"]) == 1


def test_dashboard_api_with_signals() -> None:
    def state_with_signal() -> dict:
        return {
            "cost": Decimal("0.01"),
            "calls": 1,
            "action": "block",
            "reason": "pii detected",
            "signals": [
                {
                    "type": "pii",
                    "severity": "high",
                    "detail": "SSN found",
                    "call_id": "x",
                    "detector": "pii",
                    "timestamp": "2026-03-21T00:00:00+00:00",
                }
            ],
            "spans": [],
        }

    app = create_app(get_state=state_with_signal)
    client = TestClient(app)
    resp = client.get("/api/state")
    data = resp.json()
    assert data["action"] == "block"
    assert len(data["signals"]) == 1
    assert data["signals"][0]["type"] == "pii"
