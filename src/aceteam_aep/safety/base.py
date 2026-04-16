"""Safety detector protocol and registry — pluggable T&S detection for AEP."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)


@dataclass
class SafetySignal:
    """A single T&S flag raised during a session."""

    signal_type: str  # "pii", "content_safety", "cost_anomaly", "prompt_injection"
    severity: str  # "low", "medium", "high"
    call_id: str
    detail: str
    detector: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    score: float | None = None  # numeric confidence (0–1); None = always above threshold


@runtime_checkable
class SafetyDetector(Protocol):
    """Protocol for pluggable safety detectors."""

    name: str

    def check(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        **kwargs,
    ) -> Sequence[SafetySignal]: ...


class DetectorRegistry:
    """Runs all registered detectors and collects safety signals.

    Individual detector failures are caught and logged — never crashes the caller.
    """

    def __init__(self) -> None:
        self._detectors: list[SafetyDetector] = []

    def add(self, detector: SafetyDetector) -> None:
        self._detectors.append(detector)

    def run_all(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        **kwargs,
    ) -> Sequence[SafetySignal]:
        signals: list[SafetySignal] = []
        for det in self._detectors:
            try:
                results = det.check(
                    input_text=input_text, output_text=output_text, call_id=call_id, **kwargs
                )
                for s in results:
                    s.detector = det.name
                signals.extend(results)
            except Exception:
                log.warning(
                    "Detector %s failed, skipping",
                    getattr(det, "name", "unknown"),
                    exc_info=True,
                )
        return signals


__all__ = ["DetectorRegistry", "SafetyDetector", "SafetySignal"]
