"""Tests for the cascading confidence pipeline (product-based combination)."""

from __future__ import annotations

from collections.abc import Sequence

from aceteam_aep.safety.base import SafetySignal
from aceteam_aep.safety.pipeline import (
    LayerResult,
    SafetyPipeline,
)


class FakeLayer:
    def __init__(
        self,
        name: str,
        p_safe: float = 0.9,
        prior_p_safe: float = 0.5,
        signals: list[SafetySignal] | None = None,
        should_raise: bool = False,
    ):
        self.name = name
        self._p_safe = p_safe
        self.prior_p_safe = prior_p_safe
        self._signals = signals or []
        self._should_raise = should_raise
        self.called = False

    async def score(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        prior_results: Sequence[LayerResult],
        **kwargs,
    ) -> LayerResult:
        self.called = True
        if self._should_raise:
            raise RuntimeError("layer error")
        return LayerResult(
            layer_name=self.name,
            p_safe=self._p_safe,
            signals=self._signals,
            latency_ms=1.0,
        )


async def test_product_of_safe_layers():
    """P(safe) = 0.9 * 0.8 = 0.72."""
    layers = [
        FakeLayer("l0", p_safe=0.9, prior_p_safe=0.5),
        FakeLayer("l1", p_safe=0.8, prior_p_safe=0.5),
    ]
    pipeline = SafetyPipeline(layers=layers, pass_above=0.8, block_below=0.2)
    result = await pipeline.evaluate(input_text="test", output_text="", call_id="c1")

    assert abs(result.p_safe - 0.72) < 0.001
    assert result.verdict == "flag"


async def test_product_with_unsafe_layer():
    """One unsafe layer: 0.9 * 0.1 = 0.09 → block."""
    layers = [
        FakeLayer("safe", p_safe=0.9, prior_p_safe=0.5),
        FakeLayer("unsafe", p_safe=0.1, prior_p_safe=0.5),
    ]
    pipeline = SafetyPipeline(layers=layers, block_below=0.2)
    result = await pipeline.evaluate(input_text="test", output_text="", call_id="c2")

    assert abs(result.p_safe - 0.09) < 0.001
    assert result.verdict == "block"


async def test_unrun_layers_contribute_prior():
    """Short-circuited layers contribute their prior."""
    layers = [
        FakeLayer("blocker", p_safe=0.05, prior_p_safe=0.5),
        FakeLayer("skipped", p_safe=0.99, prior_p_safe=0.5),
    ]
    pipeline = SafetyPipeline(layers=layers, block_below=0.1)
    result = await pipeline.evaluate(input_text="bad", output_text="", call_id="c3")

    assert not layers[1].called
    assert result.short_circuited_at == "blocker"
    assert result.verdict == "block"
    # 0.05 * 0.5 (prior of skipped) = 0.025
    assert abs(result.p_safe - 0.025) < 0.001


async def test_short_circuits_on_safe():
    layers = [
        FakeLayer("safe", p_safe=0.99, prior_p_safe=0.5),
        FakeLayer("expensive", p_safe=0.5, prior_p_safe=0.95),
    ]
    # After layer 0: 0.99 * 0.95 (prior) = 0.9405 > 0.9 → pass
    pipeline = SafetyPipeline(layers=layers, pass_above=0.9, block_below=0.1)
    result = await pipeline.evaluate(input_text="ok", output_text="", call_id="c4")

    assert layers[0].called
    assert not layers[1].called
    assert result.short_circuited_at == "safe"
    assert result.verdict == "pass"


async def test_runs_all_layers_when_uncertain():
    layers = [
        FakeLayer("l0", p_safe=0.8, prior_p_safe=0.8),
        FakeLayer("l1", p_safe=0.9, prior_p_safe=0.8),
        FakeLayer("l2", p_safe=0.85, prior_p_safe=0.8),
    ]
    # After l0: 0.8 * 0.8 * 0.8 = 0.512 (flag zone)
    # After l1: 0.8 * 0.9 * 0.8 = 0.576 (flag zone)
    # After l2: 0.8 * 0.9 * 0.85 = 0.612 (flag zone)
    pipeline = SafetyPipeline(layers=layers, pass_above=0.7, block_below=0.2)
    result = await pipeline.evaluate(input_text="test", output_text="", call_id="c5")

    assert all(l.called for l in layers)
    assert result.layers_executed == 3
    assert result.short_circuited_at is None
    assert result.verdict == "flag"


async def test_handles_layer_failure():
    layers = [
        FakeLayer("broken", should_raise=True, prior_p_safe=0.5),
        FakeLayer("healthy", p_safe=0.8, prior_p_safe=0.5),
    ]
    pipeline = SafetyPipeline(layers=layers, pass_above=0.9, block_below=0.1)
    result = await pipeline.evaluate(input_text="test", output_text="", call_id="c6")

    assert layers[1].called
    assert result.layers_executed == 1
    # broken contributes prior 0.5, healthy ran: 0.5 * 0.8 = 0.4
    assert abs(result.p_safe - 0.4) < 0.001


async def test_empty_layers():
    pipeline = SafetyPipeline(layers=[])
    result = await pipeline.evaluate(input_text="test", output_text="", call_id="c7")

    assert result.p_safe == 1.0
    assert result.verdict == "pass"
    assert result.layers_executed == 0


async def test_collects_signals():
    signal = SafetySignal(
        signal_type="test", severity="high", call_id="c8", detail="bad thing"
    )
    layers = [FakeLayer("l0", p_safe=0.1, prior_p_safe=0.5, signals=[signal])]
    pipeline = SafetyPipeline(layers=layers, block_below=0.2)
    result = await pipeline.evaluate(input_text="bad", output_text="", call_id="c8")

    assert len(result.signals) == 1
    assert result.signals[0].detail == "bad thing"


async def test_verdict_thresholds():
    pipeline = SafetyPipeline(layers=[], pass_above=0.7, block_below=0.3)
    assert pipeline._verdict(0.1) == "block"
    assert pipeline._verdict(0.3) == "block"
    assert pipeline._verdict(0.31) == "flag"
    assert pipeline._verdict(0.5) == "flag"
    assert pipeline._verdict(0.69) == "flag"
    assert pipeline._verdict(0.7) == "pass"
    assert pipeline._verdict(0.99) == "pass"


async def test_p_unsafe_property():
    layers = [FakeLayer("l0", p_safe=0.3, prior_p_safe=1.0)]
    pipeline = SafetyPipeline(layers=layers, block_below=0.1)
    result = await pipeline.evaluate(input_text="test", output_text="", call_id="c9")

    assert abs(result.p_unsafe - 0.7) < 0.01


async def test_conservative_product():
    """Product is conservative: many mild risks compound."""
    layers = [
        FakeLayer(f"l{i}", p_safe=0.9, prior_p_safe=0.9)
        for i in range(5)
    ]
    pipeline = SafetyPipeline(layers=layers, pass_above=0.7, block_below=0.3)
    result = await pipeline.evaluate(input_text="test", output_text="", call_id="c10")

    # 0.9^5 = 0.59049
    assert abs(result.p_safe - 0.59049) < 0.001
    assert result.verdict == "flag"


async def test_evaluate_pipeline_integration():
    from aceteam_aep.enforcement import EnforcementPolicy, evaluate_pipeline

    layers = [
        FakeLayer("regex", p_safe=0.0, prior_p_safe=0.95, signals=[
            SafetySignal(signal_type="pii", severity="high", call_id="c11", detail="SSN found", score=1.0),
        ]),
    ]
    pipeline = SafetyPipeline(layers=layers, block_below=0.3)
    result = await pipeline.evaluate(input_text="123-45-6789", output_text="", call_id="c11")

    policy = EnforcementPolicy()
    decision = evaluate_pipeline(result, policy)

    assert decision.action == "block"
    assert len(decision.signals) == 1
    assert "P(safe)" in decision.reason


async def test_pipeline_policy_from_yaml_dict():
    from aceteam_aep.enforcement import EnforcementPolicy

    data = {
        "default_action": "flag",
        "pipeline": {
            "enabled": True,
            "pass_below": 0.2,
            "block_above": 0.8,
            "layers": [
                {"name": "regex", "weight": 1.0},
                {"name": "paw", "weight": 1.5},
            ],
        },
    }
    policy = EnforcementPolicy.from_dict(data)

    assert policy.pipeline.enabled is True
    assert policy.pipeline.pass_below == 0.2
    assert policy.pipeline.block_above == 0.8


def test_layer_result_defaults():
    lr = LayerResult(layer_name="test", p_safe=0.9)
    assert lr.signals == []
    assert lr.latency_ms == 0.0
