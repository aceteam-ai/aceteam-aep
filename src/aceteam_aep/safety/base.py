"""Safety detector protocol and registry — pluggable T&S detection for AEP."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

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

    async def check(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        **kwargs,
    ) -> Sequence[SafetySignal]: ...


@dataclass(frozen=True)
class DetectorCheckResult:
    """Signals and whether a detector skipped any check after a failure."""

    signals: Sequence[SafetySignal]
    had_failure: bool = False


async def check_detector_with_status(
    detector: SafetyDetector,
    *,
    input_text: str,
    output_text: str,
    call_id: str,
    **kwargs: Any,
) -> DetectorCheckResult:
    """Use a built-in status-aware check when available; keep older detectors valid."""

    check_with_status = getattr(detector, "check_with_status", None)
    classes = type(detector).__mro__
    check_owner = next(
        (i for i, cls in enumerate(classes) if "check" in cls.__dict__), len(classes)
    )
    status_owner = next(
        (i for i, cls in enumerate(classes) if "check_with_status" in cls.__dict__),
        len(classes),
    )
    if check_with_status is not None and status_owner <= check_owner:
        return await check_with_status(
            input_text=input_text,
            output_text=output_text,
            call_id=call_id,
            **kwargs,
        )
    return DetectorCheckResult(
        await detector.check(
            input_text=input_text,
            output_text=output_text,
            call_id=call_id,
            **kwargs,
        )
    )


class DetectorRegistry:
    """Runs all registered detectors and collects safety signals.

    Individual detector failures are caught and logged — never crashes the caller.
    """

    def __init__(self) -> None:
        self._detectors: list[SafetyDetector] = []

    def add(self, detector: SafetyDetector) -> None:
        self._detectors.append(detector)

    async def run_all(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        **kwargs,
    ) -> Sequence[SafetySignal]:
        async def _run_one(detector: SafetyDetector) -> Sequence[SafetySignal]:
            try:
                results = await detector.check(
                    input_text=input_text,
                    output_text=output_text,
                    call_id=call_id,
                    **kwargs,
                )
                for s in results:
                    s.detector = detector.name
                return list(results)
            except Exception:
                log.warning(
                    "Detector %s failed, skipping",
                    getattr(detector, "name", "unknown"),
                    exc_info=True,
                )
                return []

        chunks = await asyncio.gather(*(_run_one(detector) for detector in self._detectors))
        signals: list[SafetySignal] = []
        for chunk in chunks:
            signals.extend(chunk)
        return signals


__all__ = [
    "DetectorCheckResult",
    "DetectorRegistry",
    "SafetyDetector",
    "SafetySignal",
    "check_detector_with_status",
]
