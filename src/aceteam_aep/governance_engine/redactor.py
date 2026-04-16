"""Redactor — replace detected PII/sensitive content with [REDACTED] markers.

Used as a governance enforcement action: instead of blocking the entire response,
redact only the sensitive parts and let the rest through.
"""

from __future__ import annotations

import re

# Same patterns as PII detector regex fallback
_REDACTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SSN", re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    (
        "PHONE",
        re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    ),
]


class Redactor:
    """Redact sensitive content from text."""

    def __init__(self, marker: str = "[REDACTED]") -> None:
        self._marker = marker

    def redact(self, text: str) -> tuple[str, list[str]]:
        """Redact sensitive patterns from text.

        Returns:
            (redacted_text, list of redaction descriptions)
        """
        redactions: list[str] = []
        result = text

        for pii_type, pattern in _REDACTION_PATTERNS:
            matches = pattern.findall(result)
            if matches:
                result = pattern.sub(f"{self._marker}", result)
                redactions.append(f"{pii_type}: {len(matches)} instance(s) redacted")

        return result, redactions

    def redact_if_needed(self, text: str, classification: str) -> tuple[str, list[str]]:
        """Redact only if classification warrants it.

        Only redacts for confidential and restricted data.
        """
        if classification in ("confidential", "restricted"):
            return self.redact(text)
        return text, []


__all__ = ["Redactor"]
