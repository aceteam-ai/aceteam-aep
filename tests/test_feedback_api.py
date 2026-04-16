"""Tests for the feedback API endpoints on the proxy."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from starlette.testclient import TestClient

from aceteam_aep.enforcement import EnforcementPolicy
from aceteam_aep.feedback import FeedbackStore
from aceteam_aep.proxy.app import create_proxy_app
from aceteam_aep.safety.base import SafetySignal


class _NoopDetector:
    name = "noop"

    def check(self, **kwargs: Any) -> list[SafetySignal]:
        return []


class TestFeedbackAPI:
    def test_submit_verdict(self, tmp_path: Path) -> None:
        store = FeedbackStore(tmp_path / "feedback.jsonl")
        with patch("aceteam_aep.feedback.FeedbackStore", return_value=store):
            app = create_proxy_app(detectors=[_NoopDetector()], dashboard=True)
        client = TestClient(app)

        resp = client.post(
            "/aep/api/feedback",
            json={
                "signal_type": "pii",
                "score": 0.72,
                "verdict": "dismissed",
                "detail": "not real PII",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["verdict"]["verdict"] == "dismissed"

        verdicts = store.load()
        assert len(verdicts) == 1
        assert verdicts[0].signal_type == "pii"
        assert verdicts[0].score == 0.72

    def test_invalid_verdict_rejected(self, tmp_path: Path) -> None:
        store = FeedbackStore(tmp_path / "feedback.jsonl")
        with patch("aceteam_aep.feedback.FeedbackStore", return_value=store):
            app = create_proxy_app(detectors=[_NoopDetector()], dashboard=True)
        client = TestClient(app)

        resp = client.post(
            "/aep/api/feedback",
            json={"signal_type": "pii", "verdict": "maybe"},
        )
        assert resp.status_code == 400

    def test_missing_fields_rejected(self, tmp_path: Path) -> None:
        store = FeedbackStore(tmp_path / "feedback.jsonl")
        with patch("aceteam_aep.feedback.FeedbackStore", return_value=store):
            app = create_proxy_app(detectors=[_NoopDetector()], dashboard=True)
        client = TestClient(app)

        resp = client.post("/aep/api/feedback", json={"signal_type": "pii"})
        assert resp.status_code == 400

    def test_feedback_summary(self, tmp_path: Path) -> None:
        store = FeedbackStore(tmp_path / "feedback.jsonl")
        policy = EnforcementPolicy.from_dict(
            {"detectors": {"pii": {"action": "block", "threshold": 0.5}}}
        )
        with patch("aceteam_aep.feedback.FeedbackStore", return_value=store):
            app = create_proxy_app(detectors=[_NoopDetector()], dashboard=True, policy=policy)
        client = TestClient(app)

        for score in [0.3, 0.35, 0.4, 0.45, 0.5]:
            client.post(
                "/aep/api/feedback",
                json={"signal_type": "pii", "score": score, "verdict": "dismissed"},
            )
        for score in [0.8, 0.9]:
            client.post(
                "/aep/api/feedback",
                json={"signal_type": "pii", "score": score, "verdict": "confirmed"},
            )

        resp = client.get("/aep/api/feedback/summary?min_verdicts=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_verdicts"] == 7
        assert "pii" in data["recommendations"]
        rec = data["recommendations"]["pii"]
        assert rec["dismissed"] == 5
        assert rec["confirmed"] == 2
        assert rec["false_positive_rate"] > 0
