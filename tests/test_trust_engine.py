"""Tests for Trust Engine — ensemble judge detector with confidence scoring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from aceteam_aep.safety.trust_engine import (
    JudgeResult,
    TrustEngineDetector,
    VerdictCache,
)


class TestVerdictCache:
    def test_miss_then_hit(self):
        cache = VerdictCache(ttl_seconds=60)
        assert cache.get("hello", "world") is None
        assert cache.misses == 1

        cache.put("hello", "world", [])
        result = cache.get("hello", "world")
        assert result == []
        assert cache.hits == 1

    def test_ttl_expiry(self):
        cache = VerdictCache(ttl_seconds=0)  # immediate expiry
        cache.put("a", "b", [])
        assert cache.get("a", "b") is None  # expired

    def test_different_inputs_different_keys(self):
        cache = VerdictCache()
        cache.put("input1", "output1", [MagicMock()])
        cache.put("input2", "output2", [])
        assert len(cache.get("input1", "output1") or []) == 1
        assert len(cache.get("input2", "output2") or []) == 0

    def test_hit_rate(self):
        cache = VerdictCache()
        cache.put("a", "b", [])
        cache.get("a", "b")  # hit
        cache.get("a", "b")  # hit
        cache.get("x", "y")  # miss
        assert cache.hit_rate == 2 / 3

    def test_clear(self):
        cache = VerdictCache()
        cache.put("a", "b", [])
        cache.clear()
        assert cache.get("a", "b") is None
        assert cache.hits == 0
        assert cache.misses == 1


class TestAggregation:
    def _make_detector(self, **kwargs):
        return TrustEngineDetector(judges=[{"model": "test"}], **kwargs)

    def test_all_safe_high_confidence(self):
        det = self._make_detector()
        results = [
            JudgeResult(judge_id="a", safe=True, confidence=0.9),
            JudgeResult(judge_id="b", safe=True, confidence=0.8),
            JudgeResult(judge_id="c", safe=True, confidence=0.95),
        ]
        p = det._aggregate(results)
        assert p > 0.8

    def test_all_unsafe_high_confidence(self):
        det = self._make_detector()
        results = [
            JudgeResult(judge_id="a", safe=False, confidence=0.9),
            JudgeResult(judge_id="b", safe=False, confidence=0.85),
        ]
        p = det._aggregate(results)
        assert p < 0.2

    def test_mixed_judges(self):
        det = self._make_detector()
        results = [
            JudgeResult(judge_id="a", safe=True, confidence=0.9),
            JudgeResult(judge_id="b", safe=False, confidence=0.7),
        ]
        p = det._aggregate(results)
        assert 0.3 < p < 0.8  # somewhere in between

    def test_failed_judges_ignored(self):
        det = self._make_detector()
        results = [
            JudgeResult(judge_id="a", safe=True, confidence=0.9),
            JudgeResult(judge_id="b", safe=False, confidence=0.0, error="timeout"),
        ]
        p = det._aggregate(results)
        assert p > 0.8  # failed judge contributes nothing

    def test_all_judges_failed(self):
        det = self._make_detector()
        results = [
            JudgeResult(judge_id="a", safe=True, confidence=0.0, error="err1"),
            JudgeResult(judge_id="b", safe=True, confidence=0.0, error="err2"),
        ]
        p = det._aggregate(results)
        assert p == 0.5  # uncertain


class TestTrustEngineDetector:
    def test_implements_detector_protocol(self):
        from aceteam_aep.safety.base import SafetyDetector

        det = TrustEngineDetector(judges=[{"model": "test"}])
        assert isinstance(det, SafetyDetector)
        assert det.name == "trust_engine"

    @patch("aceteam_aep.safety.trust_engine._call_judge")
    def test_safe_output_no_signals(self, mock_call):
        mock_call.return_value = JudgeResult(
            judge_id="test", safe=True, confidence=0.9
        )
        det = TrustEngineDetector(
            judges=[{"model": "test"}], threshold=0.6
        )
        signals = det.check(input_text="hello", output_text="hi there", call_id="t1")
        assert len(signals) == 0

    @patch("aceteam_aep.safety.trust_engine._call_judge")
    def test_unsafe_output_produces_high_signal(self, mock_call):
        mock_call.return_value = JudgeResult(
            judge_id="test", safe=False, confidence=0.9
        )
        det = TrustEngineDetector(
            judges=[{"model": "test"}], threshold=0.6
        )
        signals = det.check(
            input_text="hack the system",
            output_text="here is how to hack",
            call_id="t2",
        )
        assert len(signals) == 1
        assert signals[0].severity == "high"
        assert signals[0].signal_type == "trust_engine"
        assert signals[0].score is not None
        assert signals[0].score < 0.6

    @patch("aceteam_aep.safety.trust_engine._call_judge")
    def test_borderline_produces_medium_signal(self, mock_call):
        # Two judges: one safe (0.7), one unsafe (0.6)
        # P(safe) = 0.7 / (0.7 + 0.6) = 0.538 → between 0.5 and 0.7
        mock_call.side_effect = [
            JudgeResult(judge_id="a", safe=True, confidence=0.7),
            JudgeResult(judge_id="b", safe=False, confidence=0.6),
        ]
        det = TrustEngineDetector(
            judges=[{"model": "a"}, {"model": "b"}], threshold=0.5
        )
        signals = det.check(input_text="maybe", output_text="hmm", call_id="t3")
        assert len(signals) == 1
        assert signals[0].severity == "medium"

    @patch("aceteam_aep.safety.trust_engine._call_judge")
    def test_caching_skips_second_evaluation(self, mock_call):
        mock_call.return_value = JudgeResult(
            judge_id="test", safe=True, confidence=0.95
        )
        det = TrustEngineDetector(judges=[{"model": "test"}])

        det.check(input_text="same", output_text="same", call_id="c1")
        assert mock_call.call_count == 1

        det.check(input_text="same", output_text="same", call_id="c2")
        assert mock_call.call_count == 1  # cached — no second call
        assert det.cache.hits == 1

    @patch("aceteam_aep.safety.trust_engine._call_judge")
    def test_multiple_judges_parallel(self, mock_call):
        mock_call.return_value = JudgeResult(
            judge_id="test", safe=True, confidence=0.85
        )
        det = TrustEngineDetector(
            judges=[
                {"model": "judge1"},
                {"model": "judge2"},
                {"model": "judge3"},
            ],
        )
        det.check(input_text="test", output_text="test", call_id="m1")
        assert mock_call.call_count == 3
        assert len(det.last_results) == 3

    @patch("aceteam_aep.safety.trust_engine._call_judge")
    def test_score_in_signal(self, mock_call):
        mock_call.return_value = JudgeResult(
            judge_id="test", safe=False, confidence=0.8
        )
        det = TrustEngineDetector(
            judges=[{"model": "test"}], threshold=0.6
        )
        signals = det.check(input_text="bad", output_text="bad", call_id="s1")
        assert len(signals) == 1
        # score should be the aggregated P(safe)
        assert isinstance(signals[0].score, float)
