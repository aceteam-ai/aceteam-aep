"""Cascading confidence pipeline — sequential safety evaluation with short-circuit.

Each layer evaluates one safety criterion and returns P(safe) for that criterion.
The overall P(safe) is the product of all per-criterion P(safe_i), under a
conditional independence assumption (analogous to naive Bayes).

Layers that haven't run yet contribute their prior_p_safe to the product.
Running a layer replaces its prior with the computed posterior. The cascade
short-circuits when the product crosses the block or pass threshold — no need
to run expensive downstream layers.

The independence assumption can distort in both directions:
- Positively correlated criteria → over-penalizes (conservative, safe default)
- Negatively correlated criteria → under-penalizes (only with contradictory rules)
This is a known limitation; the research track explores probabilistic graphical
models (DAGs) for modeling inter-criterion dependencies (Bishop Ch. 8).

Alternative combination approaches considered but not implemented:
- max(p_unsafe): simple but ignores AND semantics
- Bayesian update: principled for overlapping criteria but requires calibrated
  likelihoods (the research gap Gustavo's linear probes address)
- Graphical model factorization: chain rule with LLM-generated dependency graph;
  principled but combinatorially expensive

Usage::

    pipeline = SafetyPipeline(layers=[
        RegexLayer(detectors=[pii, threat]),
        PawLayer(custom_detector),
        TrustEngineLayer(trust_detector),
    ])
    result = await pipeline.evaluate(input_text="...", output_text="...", call_id="x")
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
    p_safe: float
    signals: list[SafetySignal] = field(default_factory=list)
    latency_ms: float = 0.0


@runtime_checkable
class CascadeLayer(Protocol):
    """Protocol for layers in the safety cascade."""

    name: str
    prior_p_safe: float

    async def score(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        prior_results: Sequence[LayerResult],
        **kwargs,
    ) -> LayerResult: ...


# ---------------------------------------------------------------------------
# Layer adapters
# ---------------------------------------------------------------------------


class RegexLayer(CascadeLayer):
    """Deterministic regex detectors (PII, agent threat, FERPA, secrets).

    Hit → p_safe near 0 (certain violation). Miss → prior unchanged.
    """

    name = "regex"
    prior_p_safe = 0.95

    def __init__(self, detectors: list, *, prior_p_safe: float = 0.95) -> None:
        self._detectors = detectors
        self.prior_p_safe = prior_p_safe

    async def score(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        prior_results: Sequence[LayerResult],
        **kwargs,
    ) -> LayerResult:
        start = time.monotonic_ns()
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

        latency = (time.monotonic_ns() - start) / 1_000_000

        if all_signals:
            max_score = max((s.score or 1.0) for s in all_signals)
            return LayerResult(
                layer_name=self.name,
                p_safe=1.0 - max_score,
                signals=all_signals,
                latency_ms=latency,
            )

        return LayerResult(
            layer_name=self.name,
            p_safe=self.prior_p_safe,
            signals=[],
            latency_ms=latency,
        )


class PawLayer(CascadeLayer):
    """PAW local classifier with logprob confidence.

    p_safe comes directly from the logprob softmax P(Y) extracted from
    the llama.cpp inference. Falls back to prior when no signals.
    """

    name = "paw"
    prior_p_safe = 0.5

    def __init__(self, detector, *, prior_p_safe: float = 0.5) -> None:
        from .custom import CustomSafetyDetector

        if not isinstance(detector, CustomSafetyDetector):
            raise TypeError(f"Expected CustomSafetyDetector, got {type(detector).__name__}")
        self._detector = detector
        self.prior_p_safe = prior_p_safe

    async def score(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        prior_results: Sequence[LayerResult],
        **kwargs,
    ) -> LayerResult:
        start = time.monotonic_ns()

        signals = list(
            await self._detector.check(
                input_text=input_text,
                output_text=output_text,
                call_id=call_id,
                **kwargs,
            )
        )

        latency = (time.monotonic_ns() - start) / 1_000_000

        if signals:
            scores = [s.score for s in signals if s.score is not None]
            if scores:
                p_safe = 1.0 - max(scores)
            else:
                p_safe = 0.2
            return LayerResult(
                layer_name=self.name,
                p_safe=p_safe,
                signals=signals,
                latency_ms=latency,
            )

        return LayerResult(
            layer_name=self.name,
            p_safe=self.prior_p_safe,
            signals=[],
            latency_ms=latency,
        )


class ContentModelLayer(CascadeLayer):
    """Local toxicity classifier. Score maps directly to p_unsafe."""

    name = "content_model"
    prior_p_safe = 0.9

    def __init__(self, detector, *, prior_p_safe: float = 0.9) -> None:
        from .content import ContentSafetyDetector

        if not isinstance(detector, ContentSafetyDetector):
            raise TypeError(f"Expected ContentSafetyDetector, got {type(detector).__name__}")
        self._detector = detector
        self.prior_p_safe = prior_p_safe

    async def score(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        prior_results: Sequence[LayerResult],
        **kwargs,
    ) -> LayerResult:
        start = time.monotonic_ns()

        signals = list(
            await self._detector.check(
                input_text=input_text,
                output_text=output_text,
                call_id=call_id,
                **kwargs,
            )
        )

        latency = (time.monotonic_ns() - start) / 1_000_000

        if signals:
            max_score = max((s.score or 0.8) for s in signals)
            return LayerResult(
                layer_name=self.name,
                p_safe=1.0 - max_score,
                signals=signals,
                latency_ms=latency,
            )

        return LayerResult(
            layer_name=self.name,
            p_safe=self.prior_p_safe,
            signals=[],
            latency_ms=latency,
        )


class TrustEngineLayer(CascadeLayer):
    """LLM-based multi-perspective evaluation via API.

    TrustEngineDetector already produces P(safe) with per-dimension
    confidence and reasoning. Signals carry score = P(safe).
    """

    name = "trust_engine"
    prior_p_safe = 0.5

    def __init__(self, detector, *, prior_p_safe: float = 0.5) -> None:
        from .trust_engine import TrustEngineDetector

        if not isinstance(detector, TrustEngineDetector):
            raise TypeError(f"Expected TrustEngineDetector, got {type(detector).__name__}")
        self._detector = detector
        self.prior_p_safe = prior_p_safe

    async def score(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        prior_results: Sequence[LayerResult],
        **kwargs,
    ) -> LayerResult:
        start = time.monotonic_ns()

        signals = list(
            await self._detector.check(
                input_text=input_text,
                output_text=output_text,
                call_id=call_id,
                **kwargs,
            )
        )

        latency = (time.monotonic_ns() - start) / 1_000_000

        if signals:
            scores = [s.score for s in signals if s.score is not None]
            p_safe = min(scores) if scores else 0.3
            return LayerResult(
                layer_name=self.name,
                p_safe=p_safe,
                signals=signals,
                latency_ms=latency,
            )

        return LayerResult(
            layer_name=self.name,
            p_safe=self.prior_p_safe,
            signals=[],
            latency_ms=latency,
        )


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Aggregate result from the full cascade."""

    p_safe: float
    verdict: str  # "pass", "flag", "block"
    signals: list[SafetySignal] = field(default_factory=list)
    layer_results: list[LayerResult] = field(default_factory=list)
    layers_executed: int = 0
    short_circuited_at: str | None = None
    total_latency_ms: float = 0.0

    @property
    def p_unsafe(self) -> float:
        return round(1.0 - self.p_safe, 4)

    @property
    def confidence(self) -> float:
        return round(abs(self.p_safe - 0.5) * 2, 4)


class SafetyPipeline:
    """Sequential cascade with product-based combination.

    P(safe) = product of P(safe_i) across all criteria. Layers that haven't
    run contribute their prior_p_safe; running a layer replaces its prior
    with the computed posterior. Short-circuits when P(safe) crosses a threshold.
    """

    def __init__(
        self,
        layers: Sequence[CascadeLayer],
        *,
        pass_above: float = 0.7,
        block_below: float = 0.3,
    ) -> None:
        self._layers = list(layers)
        self._pass_above = pass_above
        self._block_below = block_below

    def _compute_p_safe(
        self,
        layer_results: dict[str, LayerResult],
    ) -> float:
        """Product of per-layer P(safe). Unrun layers contribute their prior."""
        p = 1.0
        for layer in self._layers:
            result = layer_results.get(layer.name)
            p *= result.p_safe if result else layer.prior_p_safe
        return p

    def _verdict(self, p_safe: float) -> str:
        if p_safe <= self._block_below:
            return "block"
        if p_safe >= self._pass_above:
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
        start = time.monotonic_ns()
        results_map: dict[str, LayerResult] = {}
        ordered_results: list[LayerResult] = []
        all_signals: list[SafetySignal] = []
        short_circuited_at: str | None = None

        for layer in self._layers:
            try:
                result = await layer.score(
                    input_text=input_text,
                    output_text=output_text,
                    call_id=call_id,
                    prior_results=ordered_results,
                    **kwargs,
                )
            except Exception:
                log.warning("Pipeline layer %s failed, skipping", layer.name, exc_info=True)
                continue

            results_map[layer.name] = result
            ordered_results.append(result)
            all_signals.extend(result.signals)

            p_safe = self._compute_p_safe(results_map)
            if p_safe <= self._block_below or p_safe >= self._pass_above:
                short_circuited_at = layer.name
                break

        p_safe = self._compute_p_safe(results_map)
        total_latency = (time.monotonic_ns() - start) / 1_000_000

        return PipelineResult(
            p_safe=round(p_safe, 6),
            verdict=self._verdict(p_safe),
            signals=all_signals,
            layer_results=ordered_results,
            layers_executed=len(ordered_results),
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
