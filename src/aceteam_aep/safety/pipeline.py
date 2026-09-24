"""Sequential safety evaluation with a completed-score product heuristic.

Each layer evaluates one safety criterion and returns P(safe) for that criterion.
The configured thresholds apply to the product of scores from completed layers.
This is a heuristic, not a calibrated probability. Every configured layer runs
before a verdict; an incomplete required evaluation blocks without a score.

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
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from .base import EvaluationUnavailableError, SafetySignal

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer protocol
# ---------------------------------------------------------------------------


EvaluationStatus = Literal["evaluated", "skipped", "unavailable", "failed"]


def _valid_score(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _signal_score(signal: SafetySignal, default: float) -> float:
    score = default if signal.score is None else signal.score
    if not _valid_score(score):
        raise ValueError("detector returned an invalid score")
    return float(score)


@dataclass
class LayerResult:
    """Output from a single cascade layer."""

    layer_name: str
    p_safe: float | None
    signals: list[SafetySignal] = field(default_factory=list)
    latency_ms: float = 0.0
    status: EvaluationStatus = "evaluated"
    reason: str | None = None


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
            signals = await detector.check(
                input_text=input_text,
                output_text=output_text,
                call_id=call_id,
                strict=True,
                **kwargs,
            )
            for signal in signals:
                signal.detector = detector.name
                _signal_score(signal, 1.0)
            all_signals.extend(signals)

        latency = (time.monotonic_ns() - start) / 1_000_000

        if all_signals:
            max_score = max(_signal_score(s, 1.0) for s in all_signals)
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
                strict=True,
                **kwargs,
            )
        )

        latency = (time.monotonic_ns() - start) / 1_000_000

        if signals:
            for signal in signals:
                _signal_score(signal, 0.0)
            scores = [s.score for s in signals if s.score is not None]
            p_safe = 1.0 - max(scores) if scores else 0.2
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
                strict=True,
                **kwargs,
            )
        )

        latency = (time.monotonic_ns() - start) / 1_000_000

        if signals:
            max_score = max(_signal_score(s, 0.8) for s in signals)
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
                strict=True,
                **kwargs,
            )
        )

        latency = (time.monotonic_ns() - start) / 1_000_000

        if signals:
            for signal in signals:
                _signal_score(signal, 0.0)
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

    p_safe: float | None
    verdict: str  # "pass", "flag", "block"
    signals: list[SafetySignal] = field(default_factory=list)
    layer_results: list[LayerResult] = field(default_factory=list)
    layers_executed: int = 0
    short_circuited_at: str | None = None
    total_latency_ms: float = 0.0
    evaluation_unavailable_reason: str | None = None

    @property
    def p_unsafe(self) -> float | None:
        return round(1.0 - self.p_safe, 4) if self.p_safe is not None else None

    @property
    def confidence(self) -> float | None:
        return round(abs(self.p_safe - 0.5) * 2, 4) if self.p_safe is not None else None


class SafetyPipeline:
    """Evaluate every configured layer, then combine completed scores."""

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

    def _compute_p_safe(self, layer_results: Sequence[LayerResult]) -> float:
        """Product of fully evaluated layer scores only."""
        p = 1.0
        for result in layer_results:
            if result.status != "evaluated" or result.p_safe is None:
                raise ValueError("cannot score an incomplete evaluation")
            p *= result.p_safe
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
        ordered_results: list[LayerResult] = []
        all_signals: list[SafetySignal] = []

        for layer in self._layers:
            try:
                result = await layer.score(
                    input_text=input_text,
                    output_text=output_text,
                    call_id=call_id,
                    prior_results=ordered_results,
                    **kwargs,
                )
            except EvaluationUnavailableError:
                log.warning("Pipeline layer %s unavailable", layer.name, exc_info=True)
                result = LayerResult(
                    layer_name=layer.name,
                    p_safe=None,
                    status="unavailable",
                    reason="evaluator unavailable",
                )
            except Exception:
                log.warning("Pipeline layer %s failed", layer.name, exc_info=True)
                result = LayerResult(
                    layer_name=layer.name,
                    p_safe=None,
                    status="failed",
                    reason="evaluator raised an exception",
                )

            if not isinstance(result, LayerResult) or result.layer_name != layer.name:
                result = LayerResult(
                    layer_name=layer.name,
                    p_safe=None,
                    status="failed",
                    reason="evaluator returned an invalid result",
                )
            elif result.status not in ("evaluated", "skipped", "unavailable", "failed"):
                result = LayerResult(
                    layer_name=layer.name,
                    p_safe=None,
                    status="failed",
                    reason="evaluator returned an invalid status",
                )
            elif not isinstance(result.signals, list) or any(
                not isinstance(signal, SafetySignal) for signal in result.signals
            ):
                result = LayerResult(
                    layer_name=layer.name,
                    p_safe=None,
                    status="failed",
                    reason="evaluator returned invalid signals",
                )
            elif result.status == "evaluated" and (
                not _valid_score(result.p_safe)
                or any(s.score is not None and not _valid_score(s.score) for s in result.signals)
            ):
                result = LayerResult(
                    layer_name=layer.name,
                    p_safe=None,
                    status="failed",
                    reason="evaluator returned an invalid score",
                )
            elif result.status != "evaluated":
                result.p_safe = None
                result.signals = []

            ordered_results.append(result)
            if result.status == "evaluated":
                all_signals.extend(result.signals)

        incomplete = [r for r in ordered_results if r.status != "evaluated"]
        unavailable_reason = (
            "required evaluation unavailable: "
            + ", ".join(f"{r.layer_name} ({r.status})" for r in incomplete)
            if incomplete
            else None
        )
        p_safe = None if incomplete else self._compute_p_safe(ordered_results)
        total_latency = (time.monotonic_ns() - start) / 1_000_000

        return PipelineResult(
            p_safe=round(p_safe, 6) if p_safe is not None else None,
            verdict="block" if p_safe is None else self._verdict(p_safe),
            signals=all_signals,
            layer_results=ordered_results,
            layers_executed=len(ordered_results),
            short_circuited_at=None,
            total_latency_ms=round(total_latency, 1),
            evaluation_unavailable_reason=unavailable_reason,
        )


__all__ = [
    "CascadeLayer",
    "ContentModelLayer",
    "EvaluationStatus",
    "LayerResult",
    "PawLayer",
    "PipelineResult",
    "RegexLayer",
    "SafetyPipeline",
    "TrustEngineLayer",
]
