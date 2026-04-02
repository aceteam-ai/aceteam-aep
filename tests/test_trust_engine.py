"""Tests for Trust Engine — multi-perspective and ensemble modes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from aceteam_aep.safety.trust_engine import (
    DEFAULT_DIMENSIONS,
    DimensionResult,
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
        cache = VerdictCache(ttl_seconds=0)
        cache.put("a", "b", [])
        assert cache.get("a", "b") is None

    def test_different_inputs(self):
        cache = VerdictCache()
        cache.put("input1", "output1", [MagicMock()])
        cache.put("input2", "output2", [])
        assert len(cache.get("input1", "output1") or []) == 1
        assert len(cache.get("input2", "output2") or []) == 0

    def test_hit_rate(self):
        cache = VerdictCache()
        cache.put("a", "b", [])
        cache.get("a", "b")
        cache.get("a", "b")
        cache.get("x", "y")
        assert cache.hit_rate == 2 / 3

    def test_clear(self):
        cache = VerdictCache()
        cache.put("a", "b", [])
        cache.clear()
        assert cache.get("a", "b") is None


class TestDimensionAggregation:
    def _det(self, **kw):
        return TrustEngineDetector(model="test", **kw)

    def test_all_safe_high_confidence(self):
        det = self._det()
        results = [
            DimensionResult(name="pii", safe=True, confidence=0.9),
            DimensionResult(name="toxicity", safe=True, confidence=0.85),
        ]
        p = det._aggregate_dimensions(results)
        assert p > 0.8

    def test_all_unsafe(self):
        det = self._det()
        results = [
            DimensionResult(name="pii", safe=False, confidence=0.9),
            DimensionResult(name="toxicity", safe=False, confidence=0.8),
        ]
        p = det._aggregate_dimensions(results)
        assert p < 0.2

    def test_mixed_dimensions(self):
        det = self._det()
        results = [
            DimensionResult(name="pii", safe=True, confidence=0.9),
            DimensionResult(name="agent_threat", safe=False, confidence=0.7),
        ]
        p = det._aggregate_dimensions(results)
        assert 0.3 < p < 0.8

    def test_weighted_dimensions(self):
        det = self._det(dimension_weights={"pii": 2.0, "toxicity": 1.0})
        # PII (safe, conf 0.8, weight 2.0) → 1.6 safe weight
        # Toxicity (unsafe, conf 0.8, weight 1.0) → 0.8 total weight
        results = [
            DimensionResult(name="pii", safe=True, confidence=0.8),
            DimensionResult(name="toxicity", safe=False, confidence=0.8),
        ]
        p = det._aggregate_dimensions(results)
        # weighted_safe = 2.0 * 0.8 = 1.6
        # total_weight = (2.0 * 0.8) + (1.0 * 0.8) = 2.4
        # p = 1.6 / 2.4 = 0.667
        assert 0.6 < p < 0.7

    def test_zero_confidence_ignored(self):
        det = self._det()
        results = [
            DimensionResult(name="pii", safe=True, confidence=0.9),
            DimensionResult(name="toxicity", safe=False, confidence=0.0),
        ]
        p = det._aggregate_dimensions(results)
        assert p > 0.8  # zero-confidence dim contributes nothing

    def test_all_zero_confidence(self):
        det = self._det()
        results = [
            DimensionResult(name="pii", safe=True, confidence=0.0),
        ]
        p = det._aggregate_dimensions(results)
        assert p == 0.5  # uncertain


class TestEnsembleAggregation:
    def _det(self, **kw):
        return TrustEngineDetector(mode="ensemble", judges=[{"model": "test"}], **kw)

    def test_all_safe(self):
        det = self._det()
        results = [
            JudgeResult(judge_id="a", safe=True, confidence=0.9),
            JudgeResult(judge_id="b", safe=True, confidence=0.8),
        ]
        p = det._aggregate_judges(results)
        assert p > 0.8

    def test_failed_judges_ignored(self):
        det = self._det()
        results = [
            JudgeResult(judge_id="a", safe=True, confidence=0.9),
            JudgeResult(judge_id="b", safe=False, confidence=0.0, error="timeout"),
        ]
        p = det._aggregate_judges(results)
        assert p > 0.8


class TestTrustEngineDetector:
    def test_protocol_compliance(self):
        from aceteam_aep.safety.base import SafetyDetector

        det = TrustEngineDetector(model="test")
        assert isinstance(det, SafetyDetector)
        assert det.name == "trust_engine"

    def test_default_dimensions(self):
        det = TrustEngineDetector(model="test")
        assert "pii" in det.dimensions
        assert "toxicity" in det.dimensions
        assert "agent_threat" in det.dimensions
        assert len(det.dimensions) == len(DEFAULT_DIMENSIONS)

    def test_custom_dimensions_list(self):
        det = TrustEngineDetector(model="test", dimensions=["pii", "toxicity"])
        assert len(det.dimensions) == 2
        assert "pii" in det.dimensions

    def test_custom_dimensions_dict(self):
        det = TrustEngineDetector(
            model="test",
            dimensions={"hipaa": "Check HIPAA compliance", "sox": "Check SOX compliance"},
        )
        assert "hipaa" in det.dimensions
        assert "sox" in det.dimensions

    @patch("aceteam_aep.safety.trust_engine._call_multi_perspective")
    def test_safe_output_no_signals(self, mock_call):
        mock_call.return_value = (
            [DimensionResult(name="pii", safe=True, confidence=0.9)],
            100,
        )
        det = TrustEngineDetector(model="test", threshold=0.6)
        signals = det.check(input_text="hello", output_text="hi", call_id="t1")
        assert len(signals) == 0

    @patch("aceteam_aep.safety.trust_engine._call_multi_perspective")
    def test_unsafe_output_high_signal(self, mock_call):
        mock_call.return_value = (
            [
                DimensionResult(name="pii", safe=False, confidence=0.9),
                DimensionResult(name="toxicity", safe=False, confidence=0.8),
            ],
            150,
        )
        det = TrustEngineDetector(model="test", threshold=0.6)
        signals = det.check(input_text="bad", output_text="bad", call_id="t2")
        assert len(signals) == 1
        assert signals[0].severity == "high"
        assert signals[0].score is not None
        assert signals[0].score < 0.6
        assert "multi-perspective" in signals[0].detail

    @patch("aceteam_aep.safety.trust_engine._call_multi_perspective")
    def test_caching(self, mock_call):
        mock_call.return_value = (
            [DimensionResult(name="pii", safe=True, confidence=0.95)],
            50,
        )
        det = TrustEngineDetector(model="test")
        det.check(input_text="same", output_text="same", call_id="c1")
        assert mock_call.call_count == 1

        det.check(input_text="same", output_text="same", call_id="c2")
        assert mock_call.call_count == 1  # cached
        assert det.cache.hits == 1

    @patch("aceteam_aep.safety.trust_engine._call_judge")
    def test_ensemble_mode(self, mock_call):
        mock_call.return_value = JudgeResult(judge_id="test", safe=True, confidence=0.85)
        det = TrustEngineDetector(
            mode="ensemble",
            judges=[{"model": "a"}, {"model": "b"}],
        )
        signals = det.check(input_text="test", output_text="test", call_id="e1")
        assert len(signals) == 0
        assert mock_call.call_count == 2

    @patch("aceteam_aep.safety.trust_engine._call_multi_perspective")
    def test_dimension_results_stored(self, mock_call):
        mock_call.return_value = (
            [
                DimensionResult(name="pii", safe=True, confidence=0.9),
                DimensionResult(name="toxicity", safe=True, confidence=0.85),
            ],
            200,
        )
        det = TrustEngineDetector(model="test")
        det.check(input_text="test", output_text="test", call_id="d1")
        assert len(det.last_dimension_results) == 2
        assert det.last_dimension_results[0].name == "pii"


class TestDomainDimensions:
    """Tests for R-Judge domain category dimensions."""

    def test_domain_dimensions_defined(self):
        from aceteam_aep.safety.trust_engine import DOMAIN_DIMENSIONS
        expected = {"finance", "iot", "software", "web", "program"}
        assert set(DOMAIN_DIMENSIONS.keys()) == expected

    def test_domain_dimensions_have_descriptions(self):
        from aceteam_aep.safety.trust_engine import DOMAIN_DIMENSIONS
        for name, desc in DOMAIN_DIMENSIONS.items():
            assert isinstance(desc, str) and len(desc) > 20, f"{name} missing description"

    def test_detector_with_domain_dimensions(self):
        from aceteam_aep.safety.trust_engine import TrustEngineDetector, DOMAIN_DIMENSIONS
        detector = TrustEngineDetector(
            dimensions=DOMAIN_DIMENSIONS,
            mode="multi-perspective",
        )
        assert set(detector.dimensions.keys()) == set(DOMAIN_DIMENSIONS.keys())

    def test_combined_default_and_domain(self):
        from aceteam_aep.safety.trust_engine import DEFAULT_DIMENSIONS, DOMAIN_DIMENSIONS
        combined = {**DEFAULT_DIMENSIONS, **DOMAIN_DIMENSIONS}
        assert len(combined) == 10
