"""PII detector — uses a small local transformer NER model to detect personal information."""

from __future__ import annotations

import logging
import re

from .base import SafetySignal

log = logging.getLogger(__name__)

_PII_ENTITIES = {"SSN", "EMAIL", "PHONE", "CREDIT_CARD", "IP_ADDRESS", "PERSON", "ID_NUM"}

_REGEX_PATTERNS: dict[str, re.Pattern[str]] = {
    "SSN": re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "PHONE": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
}


class PiiDetector:
    """Detects PII in text using a small local transformer NER model.

    Model is lazy-loaded on first check() call. Falls back to regex if
    transformers is not installed or model fails to load.
    """

    name = "pii"

    def __init__(self, model_name: str = "iiiorg/piiranha-v1-detect-personal-information") -> None:
        self._model_name = model_name
        self._pipeline: object | None = None
        self._fallback = False
        self._load_attempted = False

    def _load(self) -> None:
        self._load_attempted = True
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "token-classification",
                model=self._model_name,
                aggregation_strategy="simple",
                device=-1,
            )
        except ImportError:
            log.warning("transformers not installed, PII detector falling back to regex")
            self._fallback = True
        except Exception:
            log.warning(
                "Failed to load PII model %s, falling back to regex",
                self._model_name,
                exc_info=True,
            )
            self._fallback = True

    def check(
        self, *, input_text: str, output_text: str, call_id: str, **kwargs: object
    ) -> list[SafetySignal]:
        if not self._load_attempted:
            self._load()

        if self._fallback:
            return self._check_regex(output_text, call_id)
        return self._check_model(output_text, call_id)

    def _check_model(self, text: str, call_id: str) -> list[SafetySignal]:
        assert self._pipeline is not None
        # Truncate to avoid OOM
        results = self._pipeline(text[:2048])  # type: ignore[operator]
        signals: list[SafetySignal] = []
        seen_types: set[str] = set()
        for entity in results:
            ent_type = entity.get("entity_group", "").upper()
            if ent_type in _PII_ENTITIES and ent_type not in seen_types:
                seen_types.add(ent_type)
                signals.append(
                    SafetySignal(
                        signal_type="pii",
                        severity="high",
                        call_id=call_id,
                        detail=f"PII detected: {ent_type} (score={entity.get('score', 0):.2f})",
                    )
                )
        return signals

    def _check_regex(self, text: str, call_id: str) -> list[SafetySignal]:
        """Fallback regex detection when transformers unavailable."""
        signals: list[SafetySignal] = []
        for pii_type, pattern in _REGEX_PATTERNS.items():
            if pattern.search(text):
                signals.append(
                    SafetySignal(
                        signal_type="pii",
                        severity="high",
                        call_id=call_id,
                        detail=f"PII pattern detected: {pii_type} (regex fallback)",
                    )
                )
        return signals


__all__ = ["PiiDetector"]
