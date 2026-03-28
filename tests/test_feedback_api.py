"""Tests for the feedback API endpoints on the proxy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from starlette.testclient import TestClient

from aceteam_aep.enforcement import EnforcementPolicy
from aceteam_aep.proxy.app import create_proxy_app
from aceteam_aep.safety.base import SafetySignal


class _FlagDetector:
    name = "flag_test"

    def check(self, **kwargs: Any) -> list[SafetySignal]:
        return [
            SafetySignal(
                signal_type="pii",
                severity="medium",
                call_id=kwargs.get("call_id", ""),
                detail="SSN detected",
                score=0.72,
            )
        ]


class _NoopDetector:
    name = "noop"

    def check(self, **kwargs: Any) -> list[SafetySignal]:
        return []


class TestFeedbackAPI:
    def test_submit_verdict(self, tmp_path: Path) -> None:
        app = create_proxy_app(detectors=[_NoopDetector()], dashboard=True)
        client = TestClient(app)

        feedback_path = str(tmp_path / "feedback.jsonl")
        resp = client.post(
            "/aep/api/feedback",
            json={
                "signal_type": "pii",
                "score": 0.72,
                "verdict": "dismissed",
                "detail": "not real PII",
                "feedback_path": feedback_path,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["verdict"]["verdict"] == "dismissed"

        # Verify file written
        lines = Path(feedback_path).read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["signal_type"] == "pii"
        assert entry["score"] == 0.72

    def test_invalid_verdict_rejected(self) -> None:
        app = create_proxy_app(detectors=[_NoopDetector()], dashboard=True)
        client = TestClient(app)

        resp = client.post(
            "/aep/api/feedback",
            json={"signal_type": "pii", "verdict": "maybe"},
        )
        assert resp.status_code == 400

    def test_missing_fields_rejected(self) -> None:
        app = create_proxy_app(detectors=[_NoopDetector()], dashboard=True)
        client = TestClient(app)

        resp = client.post("/aep/api/feedback", json={"signal_type": "pii"})
        assert resp.status_code == 400

    def test_feedback_summary(self, tmp_path: Path) -> None:
        policy = EnforcementPolicy.from_dict(
            {"detectors": {"pii": {"action": "block", "threshold": 0.5}}}
        )
        app = create_proxy_app(detectors=[_NoopDetector()], dashboard=True, policy=policy)
        client = TestClient(app)

        feedback_path = str(tmp_path / "feedback.jsonl")

        # Submit enough verdicts
        for score in [0.3, 0.35, 0.4, 0.45, 0.5]:
            client.post(
                "/aep/api/feedback",
                json={
                    "signal_type": "pii",
                    "score": score,
                    "verdict": "dismissed",
                    "feedback_path": feedback_path,
                },
            )
        for score in [0.8, 0.9]:
            client.post(
                "/aep/api/feedback",
                json={
                    "signal_type": "pii",
                    "score": score,
                    "verdict": "confirmed",
                    "feedback_path": feedback_path,
                },
            )

        # Get summary
        resp = client.get(
            f"/aep/api/feedback/summary?feedback_path={feedback_path}&min_verdicts=5"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_verdicts"] == 7
        assert "pii" in data["recommendations"]
        rec = data["recommendations"]["pii"]
        assert rec["dismissed"] == 5
        assert rec["confirmed"] == 2
        assert rec["false_positive_rate"] > 0
