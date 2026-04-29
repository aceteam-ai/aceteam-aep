"""Cascading confidence pipeline — sequential safety evaluation with short-circuit.

Layers run in order. Each produces a probability (not a boolean). The pipeline
short-circuits when combined confidence is high enough to decide PASS or BLOCK
without running more expensive downstream layers.

    Layer 0: Deterministic (regex)     — free, <1ms
    Layer 1: PAW local classifier      — free, ~50ms, needs logprobs
    Layer 2: Content model (toxicity)   — free, ~100ms, local transformer
    Layer 3: TrustEngine API           — ~$0.001, ~500ms, structured reasoning

Usage::

    from aceteam_aep.safety.pipeline import SafetyPipeline, RegexLayer, PawLayer

    pipeline = SafetyPipeline(layers=[
        RegexLayer(pii=pii_detector, agent_threat=threat_detector),
        PawLayer(custom_detector),
        TrustEngineLayer(trust_detector),
    ])
    result = await pipeline.evaluate(input_text="...", output_text="...", call_id="x")
    # result.p_unsafe, result.confidence, result.verdict, result.layers_executed
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .base import SafetySignal

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer protocol
# ---------------------------------------------------------------------------


@dataclass
class LayerResult:
    """Output from a single cascade layer."""

    layer_name: str
    p_unsafe: float  # 0.0 = certainly safe, 1.0 = certainly unsafe
    confidence: float  # 0.0 = no opinion, 1.0 = fully certain
    signals: list[SafetySignal] = field(default_factory=list)
    latency_ms: float = 0.0


@runtime_checkable
class CascadeLayer(Protocol):
    """Protocol for layers in the safety cascade."""

    name: str

    async def score(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        prior_results: list[LayerResult],
        **kwargs,
    ) -> LayerResult: ...


# ---------------------------------------------------------------------------
# Layer adapters — wrap existing detectors
# ---------------------------------------------------------------------------


class RegexLayer:
    """Layer 0: Deterministic regex detectors (PII, agent threat, FERPA).

    On hit → p_unsafe=1.0, confidence=1.0 (certain).
    On miss → p_unsafe=0.0, confidence=0.1 (regex miss doesn't mean safe).
    """

    name = "regex"

    def __init__(self, detectors: list) -> None:
        self._detectors = detectors

    async def score(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        prior_results: list[LayerResult],
        **kwargs,
    ) -> LayerResult:
        start = time.monotonic()
        all_signals: list[SafetySignal] = []

        for detector in self._detectors:
            try:
                signals = await detector.check(
                    input_text=input_text,
                    output_text=output_text,
                    call_id=call_id,
                    **kwargs,
                )
                for s in signals:
                    s.detector = detector.name
                all_signals.extend(signals)
            except Exception:
                log.warning("Regex layer detector %s failed", detector.name, exc_info=True)

        latency = (time.monotonic() - start) * 1000

        if all_signals:
            max_score = max((s.score or 1.0) for s in all_signals)
            return LayerResult(
                layer_name=self.name,
                p_unsafe=max_score,
                confidence=1.0,
                signals=all_signals,
                latency_ms=latency,
            )

        return LayerResult(
            layer_name=self.name,
            p_unsafe=0.0,
            confidence=0.1,
            signals=[],
            latency_ms=latency,
        )


class PawLayer:
    """Layer 1: PAW local classifier with logprob confidence.

    Uses CustomSafetyDetector with the new confidence-aware path.
    p_unsafe comes from 1 - p_compliant (logprob softmax).
    confidence is the distance from 0.5 (how decisive the model was).
    """

    name = "paw"

    def __init__(self, detector) -> None:
        from .custom import CustomSafetyDetector

        if not isinstance(detector, CustomSafetyDetector):
            raise TypeError(f"Expected CustomSafetyDetector, got {type(detector).__name__}")
        self._detector = detector

    async def score(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        prior_results: list[LayerResult],
        **kwargs,
    ) -> LayerResult:
        start = time.monotonic()

        signals = list(
            await self._detector.check(
                input_text=input_text,
                output_text=output_text,
                call_id=call_id,
                **kwargs,
            )
        )

        latency = (time.monotonic() - start) * 1000

        if signals:
            scores = [s.score for s in signals if s.score is not None]
            p_unsafe = max(scores) if scores else 0.8
            confidence = abs(p_unsafe - 0.5) * 2 if scores else 0.5
            return LayerResult(
                layer_name=self.name,
                p_unsafe=p_unsafe,
                confidence=confidence,
                signals=signals,
                latency_ms=latency,
            )

        return LayerResult(
            layer_name=self.name,
            p_unsafe=0.0,
            confidence=0.6,
            signals=[],
            latency_ms=latency,
        )


class ContentModelLayer:
    """Layer 2: Local toxicity classifier.

    Wraps ContentSafetyDetector. Score maps directly to p_unsafe.
    """

    name = "content_model"

    def __init__(self, detector) -> None:
        from .content import ContentSafetyDetector

        if not isinstance(detector, ContentSafetyDetector):
            raise TypeError(f"Expected ContentSafetyDetector, got {type(detector).__name__}")
        self._detector = detector

    async def score(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        prior_results: list[LayerResult],
        **kwargs,
    ) -> LayerResult:
        start = time.monotonic()

        signals = list(
            await self._detector.check(
                input_text=input_text,
                output_text=output_text,
                call_id=call_id,
                **kwargs,
            )
        )

        latency = (time.monotonic() - start) * 1000

        if signals:
            max_score = max((s.score or 0.8) for s in signals)
            return LayerResult(
                layer_name=self.name,
                p_unsafe=max_score,
                confidence=max_score,
                signals=signals,
                latency_ms=latency,
            )

        return LayerResult(
            layer_name=self.name,
            p_unsafe=0.0,
            confidence=0.5,
            signals=[],
            latency_ms=latency,
        )


class TrustEngineLayer:
    """Layer 3: LLM-based multi-perspective evaluation via API.

    Wraps TrustEngineDetector. Already produces structured P(safe) with
    per-dimension confidence and reasoning.
    """

    name = "trust_engine"

    def __init__(self, detector) -> None:
        from .trust_engine import TrustEngineDetector

        if not isinstance(detector, TrustEngineDetector):
            raise TypeError(f"Expected TrustEngineDetector, got {type(detector).__name__}")
        self._detector = detector

    async def score(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        prior_results: list[LayerResult],
        **kwargs,
    ) -> LayerResult:
        start = time.monotonic()

        signals = list(
            await self._detector.check(
                input_text=input_text,
                output_text=output_text,
                call_id=call_id,
                **kwargs,
            )
        )

        latency = (time.monotonic() - start) * 1000

        if signals:
            scores = [s.score for s in signals if s.score is not None]
            p_unsafe = 1.0 - min(scores) if scores else 0.7
            return LayerResult(
                layer_name=self.name,
                p_unsafe=p_unsafe,
                confidence=0.9,
                signals=signals,
                latency_ms=latency,
            )

        return LayerResult(
            layer_name=self.name,
            p_unsafe=0.0,
            confidence=0.9,
            signals=[],
            latency_ms=latency,
        )


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Aggregate result from the full cascade."""

    p_unsafe: float
    confidence: float
    verdict: str  # "pass", "flag", "block"
    signals: list[SafetySignal] = field(default_factory=list)
    layer_results: list[LayerResult] = field(default_factory=list)
    layers_executed: int = 0
    short_circuited_at: str | None = None
    total_latency_ms: float = 0.0


class SafetyPipeline:
    """Sequential cascade of safety layers with short-circuit logic.

    Runs layers in order. After each layer, checks whether the combined
    confidence is high enough to make a decision without running the
    remaining (more expensive) layers.
    """

    def __init__(
        self,
        layers: list[CascadeLayer],
        *,
        pass_below: float = 0.3,
        block_above: float = 0.7,
        confidence_threshold: float = 0.7,
        layer_weights: dict[str, float] | None = None,
    ) -> None:
        self._layers = layers
        self._pass_below = pass_below
        self._block_above = block_above
        self._confidence_threshold = confidence_threshold
        self._layer_weights = layer_weights or {}

    def _combine(self, results: list[LayerResult]) -> tuple[float, float]:
        """Weighted average of layer results → (p_unsafe, confidence)."""
        total_weight = 0.0
        weighted_p = 0.0
        weighted_c = 0.0

        for r in results:
            w = self._layer_weights.get(r.layer_name, 1.0) * r.confidence
            if w <= 0:
                continue
            weighted_p += r.p_unsafe * w
            weighted_c += r.confidence * w
            total_weight += w

        if total_weight == 0:
            return 0.5, 0.0

        return weighted_p / total_weight, weighted_c / total_weight

    def _verdict(self, p_unsafe: float) -> str:
        if p_unsafe >= self._block_above:
            return "block"
        if p_unsafe < self._pass_below:
            return "pass"
        return "flag"

    async def evaluate(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        **kwargs,
    ) -> PipelineResult:
        pipeline_start = time.monotonic()
        layer_results: list[LayerResult] = []
        all_signals: list[SafetySignal] = []
        short_circuited_at: str | None = None

        for layer in self._layers:
            try:
                result = await layer.score(
                    input_text=input_text,
                    output_text=output_text,
                    call_id=call_id,
                    prior_results=layer_results,
                    **kwargs,
                )
            except Exception:
                log.warning("Pipeline layer %s failed, skipping", layer.name, exc_info=True)
                continue

            layer_results.append(result)
            all_signals.extend(result.signals)

            p_unsafe, confidence = self._combine(layer_results)

            if confidence >= self._confidence_threshold:
                if p_unsafe >= self._block_above or p_unsafe < self._pass_below:
                    short_circuited_at = layer.name
                    break

        p_unsafe, confidence = self._combine(layer_results)
        total_latency = (time.monotonic() - pipeline_start) * 1000

        return PipelineResult(
            p_unsafe=round(p_unsafe, 4),
            confidence=round(confidence, 4),
            verdict=self._verdict(p_unsafe),
            signals=all_signals,
            layer_results=layer_results,
            layers_executed=len(layer_results),
            short_circuited_at=short_circuited_at,
            total_latency_ms=round(total_latency, 1),
        )


__all__ = [
    "CascadeLayer",
    "ContentModelLayer",
    "LayerResult",
    "PawLayer",
    "PipelineResult",
    "RegexLayer",
    "SafetyPipeline",
    "TrustEngineLayer",
]
