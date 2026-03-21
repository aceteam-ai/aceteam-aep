"""Content safety detector — uses a small local toxicity classifier."""

from __future__ import annotations

import logging

from .base import SafetySignal

log = logging.getLogger(__name__)


class ContentSafetyDetector:
    """Classifies text as safe/unsafe using a small local toxicity model.

    Model is lazy-loaded on first check() call. If transformers is not
    installed or model fails to load, detector silently disables itself.
    """

    name = "content_safety"

    def __init__(
        self,
        model_name: str = "s-nlp/roberta_toxicity_classifier",
        threshold: float = 0.7,
    ) -> None:
        self._model_name = model_name
        self._threshold = threshold
        self._pipeline: object | None = None
        self._available = True
        self._load_attempted = False

    def _load(self) -> None:
        self._load_attempted = True
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification",
                model=self._model_name,
                device=-1,
            )
        except ImportError:
            log.warning("transformers not installed, content safety detector disabled")
            self._available = False
        except Exception:
            log.warning("Content safety model unavailable", exc_info=True)
            self._available = False

    def check(
        self, *, input_text: str, output_text: str, call_id: str, **kwargs: object
    ) -> list[SafetySignal]:
        if not self._load_attempted:
            self._load()
        if not self._available:
            return []

        signals: list[SafetySignal] = []
        for text, source in [(output_text, "output"), (input_text, "input")]:
            if not text:
                continue
            try:
                result = self._pipeline(text[:512])  # type: ignore[operator]
                if result:
                    label = result[0].get("label", "").lower()
                    score = result[0].get("score", 0)
                    toxic_labels = ("toxic", "unsafe", "harmful", "label_1")
                    if label in toxic_labels and score >= self._threshold:
                        signals.append(
                            SafetySignal(
                                signal_type="content_safety",
                                severity="high" if score > 0.9 else "medium",
                                call_id=call_id,
                                detail=f"Unsafe content in {source} (score={score:.2f})",
                            )
                        )
            except Exception:
                log.warning("Content safety check failed for %s", source, exc_info=True)
        return signals


__all__ = ["ContentSafetyDetector"]
