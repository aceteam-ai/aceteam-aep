"""PII detector — uses a small local transformer NER model to detect personal information."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .base import SafetyDetector, SafetySignal

log = logging.getLogger(__name__)

_PII_ENTITIES = {
    "SSN",
    "EMAIL",
    "PHONE",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "PERSON",
    "ID_NUM",
}


def _normalize_entity_label(raw: str) -> str:
    """Strip B-/I- BIO prefixes from token-classification labels."""
    u = raw.upper().strip()
    if u.startswith("B-") or u.startswith("I-"):
        return u[2:]
    return u


_REGEX_PATTERNS: dict[str, re.Pattern[str]] = {
    "SSN": re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    # "EMAIL": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    # "PHONE": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
}


@dataclass(kw_only=True)
class PiiSafetySignal(SafetySignal):
    """A single PII signal."""

    entity_type: str
    span_start: int
    span_end: int


def _merge_span_signals(
    signals: Iterable[PiiSafetySignal],
) -> Sequence[PiiSafetySignal]:
    """Same ``(entity_type, span_start, span_end)`` → keep the signal with the higher score."""
    best: dict[tuple[str, int, int], PiiSafetySignal] = {}
    for s in signals:
        key = (s.entity_type, s.span_start, s.span_end)
        prev = best.get(key)
        s_score = s.score if s.score is not None else 0.0
        if prev is None:
            best[key] = s
            continue
        prev_score = prev.score if prev.score is not None else 0.0
        if s_score > prev_score:
            best[key] = s
    return sorted(
        best.values(),
        key=lambda x: (x.span_start, x.span_end, x.entity_type),
    )


class PiiDetector(SafetyDetector):
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

    async def check(
        self,
        *,
        input_text: str,
        output_text: str,
        call_id: str,
        **kwargs,
    ) -> Sequence[PiiSafetySignal]:
        if not self._load_attempted:
            self._load()

        signals: list[PiiSafetySignal] = []
        for text, source in ((input_text, "input"), (output_text, "output")):
            if not text:
                continue
            if self._fallback:
                chunk = _merge_span_signals(self._check_regex(text, call_id, fallback=True))
            else:
                chunk = self._merge_model_and_regex(text, call_id)
            for signal in chunk:
                signal.detail = f"{signal.detail} (in {source})"
                signals.append(signal)
        return signals

    def _merge_model_and_regex(
        self,
        text: str,
        call_id: str,
    ) -> Sequence[PiiSafetySignal]:
        combined = [
            *self._check_model(text, call_id),
            *self._check_regex(text, call_id, fallback=False),
        ]
        return _merge_span_signals(combined)

    def _check_model(
        self,
        text: str,
        call_id: str,
    ) -> Iterable[PiiSafetySignal]:
        if len(text) > 2048:
            log.warning("Text is too long (%d chars) for NER model, truncating", len(text))
            text = text[:2048]

        assert self._pipeline is not None
        results = self._pipeline(text)  # type: ignore[operator]
        for entity in results:
            ent_type = _normalize_entity_label(entity.get("entity_group", ""))
            if ent_type not in _PII_ENTITIES:
                continue
            start = entity.get("start")
            end = entity.get("end")
            if start is None or end is None:
                log.debug("Skipping NER entity without span: %s", entity)
                continue
            entity_score = entity.get("score", 1.0)
            span_start = int(start)
            span_end = int(end)
            yield PiiSafetySignal(
                entity_type=ent_type,
                span_start=span_start,
                span_end=span_end,
                signal_type="pii",
                severity="high",
                call_id=call_id,
                detail=(
                    f"PII detected: {ent_type} "
                    f"[{span_start}:{span_end}] (score={float(entity_score):.2f})"
                ),
                score=float(entity_score),
            )

    def _check_regex(
        self,
        text: str,
        call_id: str,
        *,
        fallback: bool,
    ) -> Iterable[PiiSafetySignal]:
        """Regex PII patterns. ``fallback=True`` when the NER model is unavailable."""
        tag = "regex fallback" if fallback else "pattern match"
        for pii_type, pattern in _REGEX_PATTERNS.items():
            for m in pattern.finditer(text):
                span_start, span_end = m.span()
                yield PiiSafetySignal(
                    entity_type=pii_type,
                    span_start=span_start,
                    span_end=span_end,
                    signal_type="pii",
                    severity="high",
                    call_id=call_id,
                    detail=(f"PII pattern detected: {pii_type} [{span_start}:{span_end}] ({tag})"),
                    score=1.0,
                )


__all__ = ["PiiDetector"]
