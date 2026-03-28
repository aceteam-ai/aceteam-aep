"""Tests for the signal feedback loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from aceteam_aep.feedback import (
    FeedbackStore,
    apply_recommendations,
    recommend_thresholds,
)


@pytest.fixture()
def store(tmp_path: Path) -> FeedbackStore:
    return FeedbackStore(tmp_path / "feedback.jsonl")


class TestFeedbackStore:
    def test_record_and_load(self, store: FeedbackStore) -> None:
        store.record("pii", score=0.62, verdict="dismissed")
        store.record("pii", score=0.91, verdict="confirmed")

        verdicts = store.load()
        assert len(verdicts) == 2
        assert verdicts[0].signal_type == "pii"
        assert verdicts[0].score == 0.62
        assert verdicts[0].verdict == "dismissed"
        assert verdicts[1].verdict == "confirmed"

    def test_invalid_verdict_raises(self, store: FeedbackStore) -> None:
        with pytest.raises(ValueError, match="confirmed.*dismissed"):
            store.record("pii", score=0.5, verdict="maybe")

    def test_verdicts_for_filters(self, store: FeedbackStore) -> None:
        store.record("pii", score=0.5, verdict="dismissed")
        store.record("agent_threat", score=1.0, verdict="confirmed")
        store.record("pii", score=0.9, verdict="confirmed")

        pii = store.verdicts_for("pii")
        assert len(pii) == 2
        threat = store.verdicts_for("agent_threat")
        assert len(threat) == 1

    def test_empty_store(self, store: FeedbackStore) -> None:
        assert store.load() == []


class TestRecommendThresholds:
    def test_not_enough_verdicts(self, store: FeedbackStore) -> None:
        store.record("pii", score=0.5, verdict="dismissed")
        store.record("pii", score=0.6, verdict="dismissed")

        summary = recommend_thresholds(store, min_verdicts=5)
        rec = summary.recommendations["pii"]
        assert rec.suggested_threshold is None
        assert "Need 3 more" in rec.reason

    def test_recommends_higher_threshold(self, store: FeedbackStore) -> None:
        """False positives at low scores → recommend higher threshold."""
        # Dismissed (false positives) at low scores
        for score in [0.3, 0.4, 0.45, 0.5, 0.55]:
            store.record("pii", score=score, verdict="dismissed")
        # Confirmed (true positives) at high scores
        for score in [0.8, 0.85, 0.9, 0.95]:
            store.record("pii", score=score, verdict="confirmed")

        summary = recommend_thresholds(
            store,
            current_thresholds={"pii": 0.3},
            min_verdicts=5,
        )
        rec = summary.recommendations["pii"]
        assert rec.suggested_threshold is not None
        assert rec.suggested_threshold > 0.3
        # Should be below the lowest confirmed score
        assert rec.suggested_threshold < 0.8
        assert rec.false_positive_rate > 0

    def test_no_false_positives(self, store: FeedbackStore) -> None:
        """All confirmed → threshold is working well."""
        for score in [0.7, 0.8, 0.9, 0.95, 0.99]:
            store.record("pii", score=score, verdict="confirmed")

        summary = recommend_thresholds(store, min_verdicts=5)
        rec = summary.recommendations["pii"]
        assert rec.suggested_threshold is None
        assert "No false positives" in rec.reason

    def test_current_threshold_already_optimal(self, store: FeedbackStore) -> None:
        """If current threshold would already filter FPs, don't suggest lower."""
        for score in [0.3, 0.35, 0.4]:
            store.record("pii", score=score, verdict="dismissed")
        for score in [0.8, 0.9]:
            store.record("pii", score=score, verdict="confirmed")

        summary = recommend_thresholds(
            store,
            current_thresholds={"pii": 0.7},
            min_verdicts=5,
        )
        rec = summary.recommendations["pii"]
        assert rec.suggested_threshold is None
        assert "already optimal" in rec.reason

    def test_multiple_detectors(self, store: FeedbackStore) -> None:
        for _ in range(5):
            store.record("pii", score=0.4, verdict="dismissed")
        for _ in range(5):
            store.record("agent_threat", score=1.0, verdict="confirmed")

        summary = recommend_thresholds(store, min_verdicts=5)
        assert "pii" in summary.recommendations
        assert "agent_threat" in summary.recommendations
        assert summary.total_verdicts == 10


class TestApplyRecommendations:
    def test_updates_threshold_in_yaml(self, store: FeedbackStore, tmp_path: Path) -> None:
        # Create a policy file
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text(
            "default_action: block\n"
            "detectors:\n"
            "  pii:\n"
            "    action: block\n"
            "    threshold: 0.3\n"
            "  agent_threat:\n"
            "    action: block\n"
        )

        # Generate verdicts
        for score in [0.3, 0.35, 0.4, 0.45, 0.5]:
            store.record("pii", score=score, verdict="dismissed")
        for score in [0.8, 0.85, 0.9]:
            store.record("pii", score=score, verdict="confirmed")

        summary = recommend_thresholds(
            store,
            current_thresholds={"pii": 0.3},
            min_verdicts=5,
        )

        output = tmp_path / "updated.yaml"
        result = apply_recommendations(summary, policy_file, output)

        assert "Updated by AEP feedback loop" in result
        assert output.exists()

        # Parse and verify
        import yaml

        data = yaml.safe_load(output.read_text())
        new_thresh = data["detectors"]["pii"]["threshold"]
        assert new_thresh > 0.3
        # Agent threat should be untouched
        assert "threshold" not in data["detectors"]["agent_threat"]
