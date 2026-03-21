"""Data classifier — determines sensitivity level of text.

Uses safety signals (PII detection) as sub-signals to classify:
- PII present → at least "confidential"
- Financial/health/legal keywords → "internal"
- No indicators → "public"

This is a SafetyDetector that plugs into the DetectorRegistry.
"""

from __future__ import annotations

import re

from ..safety.base import SafetySignal

_FINANCIAL_PATTERNS = re.compile(
    r"\b(ssn|social security|bank account|routing number|"
    r"credit card|tax id|ein|itin)\b",
    re.IGNORECASE,
)
_HEALTH_PATTERNS = re.compile(
    r"\b(diagnosis|patient|medical record|hipaa|"
    r"prescription|treatment plan|health insurance)\b",
    re.IGNORECASE,
)
_LEGAL_PATTERNS = re.compile(
    r"\b(attorney[- ]client|privileged|confidential settlement|"
    r"under seal|nda|non-disclosure)\b",
    re.IGNORECASE,
)


class DataClassifier:
    """Classifies text sensitivity for governance enforcement.

    Returns a SafetySignal with signal_type="data_classification" and
    detail containing the classification level.
    """

    name = "data_classification"

    def check(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        pii_detected: bool = False,
        **kwargs: object,
    ) -> list[SafetySignal]:
        combined = f"{input_text} {output_text}"
        level = self._classify(combined, pii_detected=pii_detected)

        if level == "public":
            return []

        severity_map = {
            "internal": "low",
            "confidential": "medium",
            "restricted": "high",
        }
        return [
            SafetySignal(
                signal_type="data_classification",
                severity=severity_map.get(level, "low"),
                call_id=call_id,
                detail=f"Data classified as {level}",
            )
        ]

    def _classify(self, text: str, *, pii_detected: bool = False) -> str:
        if pii_detected or _FINANCIAL_PATTERNS.search(text):
            return "confidential"
        if _HEALTH_PATTERNS.search(text) or _LEGAL_PATTERNS.search(text):
            return "confidential"
        if _LEGAL_PATTERNS.search(text):
            return "internal"
        return "public"

    def classify_text(self, text: str, *, pii_detected: bool = False) -> str:
        """Classify text and return the level string directly."""
        return self._classify(text, pii_detected=pii_detected)


__all__ = ["DataClassifier"]
