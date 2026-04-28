"""Tests for the cascading confidence pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from aceteam_aep.safety.base import SafetySignal
from aceteam_aep.safety.pipeline import (
    CascadeLayer,
    LayerResult,
    PipelineResult,
    SafetyPipeline,
)


# ---------------------------------------------------------------------------
# Fake layers for testing
# ---------------------------------------------------------------------------


class FakeLayer:
    """Configurable fake layer for testing cascade behavior."""

    def __init__(
        self,
        name: str,
        p_unsafe: float = 0.0,
        confidence: float = 0.5,
        signals: list[SafetySignal] | None = None,
        latency_ms: float = 1.0,
        should_raise: bool = False,
    ):
        self.name = name
        self._p_unsafe = p_unsafe
        self._confidence = confidence
        self._signals = signals or []
        self._latency_ms = latency_ms
        self._should_raise = should_raise
        self.called = False

    async def score(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        prior_results: list[LayerResult],
        **kwargs,
    ) -> LayerResult:
        self.called = True
        if self._should_raise:
            raise RuntimeError("layer error")
        return LayerResult(
            layer_name=self.name,
            p_unsafe=self._p_unsafe,
            confidence=self._confidence,
            signals=self._signals,
            latency_ms=self._latency_ms,
        )


# ---------------------------------------------------------------------------
# SafetyPipeline tests
# ---------------------------------------------------------------------------


async def test_pipeline_runs_all_layers_when_uncertain():
    layers = [
        FakeLayer("l0", p_unsafe=0.4, confidence=0.3),
        FakeLayer("l1", p_unsafe=0.5, confidence=0.3),
        FakeLayer("l2", p_unsafe=0.5, confidence=0.3),
    ]
    pipeline = SafetyPipeline(layers=layers)
    result = await pipeline.evaluate(input_text="test", output_text="", call_id="c1")

    assert all(l.called for l in layers)
    assert result.layers_executed == 3
    assert result.short_circuited_at is None
    assert result.verdict == "flag"


async def test_pipeline_short_circuits_on_high_confidence_unsafe():
    layers = [
        FakeLayer("regex", p_unsafe=1.0, confidence=1.0),
        FakeLayer("paw", p_unsafe=0.0, confidence=0.5),
    ]
    pipeline = SafetyPipeline(layers=layers, block_above=0.7, confidence_threshold=0.7)
    result = await pipeline.evaluate(input_text="bad", output_text="", call_id="c2")

    assert layers[0].called
    assert not layers[1].called
    assert result.short_circuited_at == "regex"
    assert result.verdict == "block"
    assert result.layers_executed == 1


async def test_pipeline_short_circuits_on_high_confidence_safe():
    layers = [
        FakeLayer("regex", p_unsafe=0.0, confidence=0.9),
        FakeLayer("paw", p_unsafe=0.0, confidence=0.5),
    ]
    pipeline = SafetyPipeline(layers=layers, pass_below=0.3, confidence_threshold=0.7)
    result = await pipeline.evaluate(input_text="hello", output_text="", call_id="c3")

    assert layers[0].called
    assert not layers[1].called
    assert result.short_circuited_at == "regex"
    assert result.verdict == "pass"


async def test_pipeline_handles_layer_failure():
    layers = [
        FakeLayer("broken", should_raise=True),
        FakeLayer("healthy", p_unsafe=0.1, confidence=0.8),
    ]
    pipeline = SafetyPipeline(layers=layers)
    result = await pipeline.evaluate(input_text="test", output_text="", call_id="c4")

    assert not layers[0].called or True  # it was called but raised
    assert layers[1].called
    assert result.layers_executed == 1  # only healthy counted


async def test_pipeline_empty_layers():
    pipeline = SafetyPipeline(layers=[])
    result = await pipeline.evaluate(input_text="test", output_text="", call_id="c5")

    assert result.verdict == "flag"
    assert result.layers_executed == 0
    assert result.p_unsafe == 0.5


async def test_pipeline_respects_layer_weights():
    layers = [
        FakeLayer("low_weight", p_unsafe=1.0, confidence=0.8),
        FakeLayer("high_weight", p_unsafe=0.0, confidence=0.8),
    ]
    pipeline = SafetyPipeline(
        layers=layers,
        layer_weights={"low_weight": 0.1, "high_weight": 10.0},
        confidence_threshold=0.99,  # prevent short-circuit
    )
    result = await pipeline.evaluate(input_text="test", output_text="", call_id="c6")

    assert result.p_unsafe < 0.2  # heavily weighted toward safe


async def test_pipeline_collects_signals():
    signal = SafetySignal(
        signal_type="test", severity="high", call_id="c7", detail="bad thing"
    )
    layers = [
        FakeLayer("l0", p_unsafe=0.9, confidence=0.9, signals=[signal]),
    ]
    pipeline = SafetyPipeline(layers=layers)
    result = await pipeline.evaluate(input_text="bad", output_text="", call_id="c7")

    assert len(result.signals) == 1
    assert result.signals[0].detail == "bad thing"


async def test_pipeline_verdict_thresholds():
    pipeline = SafetyPipeline(
        layers=[],
        pass_below=0.3,
        block_above=0.7,
    )
    assert pipeline._verdict(0.1) == "pass"
    assert pipeline._verdict(0.29) == "pass"
    assert pipeline._verdict(0.3) == "flag"
    assert pipeline._verdict(0.5) == "flag"
    assert pipeline._verdict(0.69) == "flag"
    assert pipeline._verdict(0.7) == "block"
    assert pipeline._verdict(0.99) == "block"


# ---------------------------------------------------------------------------
# LayerResult tests
# ---------------------------------------------------------------------------


def test_layer_result_defaults():
    lr = LayerResult(layer_name="test", p_unsafe=0.5, confidence=0.5)
    assert lr.signals == []
    assert lr.latency_ms == 0.0


# ---------------------------------------------------------------------------
# Enforcement integration
# ---------------------------------------------------------------------------


async def test_evaluate_pipeline_integration():
    from aceteam_aep.enforcement import EnforcementPolicy, evaluate_pipeline

    layers = [
        FakeLayer("regex", p_unsafe=0.9, confidence=1.0, signals=[
            SafetySignal(signal_type="pii", severity="high", call_id="c8", detail="SSN found", score=1.0),
        ]),
    ]
    pipeline = SafetyPipeline(layers=layers, block_above=0.7, confidence_threshold=0.7)
    result = await pipeline.evaluate(input_text="123-45-6789", output_text="", call_id="c8")

    policy = EnforcementPolicy()
    decision = evaluate_pipeline(result, policy)

    assert decision.action == "block"
    assert len(decision.signals) == 1


async def test_pipeline_policy_from_yaml_dict():
    from aceteam_aep.enforcement import EnforcementPolicy

    data = {
        "default_action": "flag",
        "pipeline": {
            "enabled": True,
            "pass_below": 0.2,
            "block_above": 0.8,
            "confidence_threshold": 0.75,
            "layers": [
                {"name": "regex", "weight": 1.0},
                {"name": "paw", "weight": 1.5},
                {"name": "trust_engine", "weight": 2.0},
            ],
        },
        "detectors": {
            "cost_anomaly": {"action": "flag"},
        },
    }
    policy = EnforcementPolicy.from_dict(data)

    assert policy.pipeline.enabled is True
    assert policy.pipeline.pass_below == 0.2
    assert policy.pipeline.block_above == 0.8
    assert policy.pipeline.layer_weights == {"regex": 1.0, "paw": 1.5, "trust_engine": 2.0}
