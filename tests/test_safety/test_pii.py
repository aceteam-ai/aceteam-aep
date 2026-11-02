"""Tests for PII detector — uses real model if available, regex fallback otherwise."""

import pytest

from aceteam_aep.safety.pii import PiiDetector


@pytest.fixture(scope="module")
def detector() -> PiiDetector:
    return PiiDetector()


async def test_detects_ssn(detector: PiiDetector) -> None:
    signals = await detector.check(
        input_text="",
        output_text="His SSN is 123-45-6789",
        call_id="t1",
    )
    assert any(s.signal_type == "pii" for s in signals)


async def test_detects_email(detector: PiiDetector) -> None:
    signals = await detector.check(
        input_text="",
        output_text="Contact john.doe@company.com for help",
        call_id="t2",
    )
    assert any(s.signal_type == "pii" for s in signals)


async def test_clean_text_no_signal(detector: PiiDetector) -> None:
    signals = await detector.check(
        input_text="",
        output_text="The weather is nice today.",
        call_id="t3",
    )
    pii_signals = [s for s in signals if s.signal_type == "pii"]
    # Model may detect some false positives, regex should not
    # Allow either 0 signals or only low-confidence ones
    assert len(pii_signals) == 0 or all("regex" not in s.detail for s in pii_signals)


async def test_detects_phone_number(detector: PiiDetector) -> None:
    signals = await detector.check(
        input_text="",
        output_text="Call me at (555) 123-4567",
        call_id="t4",
    )
    assert any(s.signal_type == "pii" for s in signals)


def test_pipeline_shared_across_instances() -> None:
    """Two detectors for the same model share one cached pipeline (no rebuild)."""
    from aceteam_aep.safety import pii

    if "transformers" not in __import__("sys").modules:
        pytest.importorskip("transformers")

    d1 = PiiDetector()
    d1._load()
    if d1._fallback:
        pytest.skip("model unavailable, cache not exercised")
    d2 = PiiDetector()
    d2._load()
    assert d1._pipeline is d2._pipeline
    assert pii._PIPELINE_CACHE[d1._model_name] is d1._pipeline


async def test_regex_fallback_works() -> None:
    """Force regex fallback and verify it works."""
    det = PiiDetector(model_name="nonexistent/model-that-will-fail")
    signals = await det.check(
        input_text="",
        output_text="SSN: 123-45-6789",
        call_id="t5",
    )
    assert any(s.signal_type == "pii" for s in signals)
    assert any("regex fallback" in s.detail for s in signals)


def _regex_only_detector() -> PiiDetector:
    """Detector forced into regex fallback (no model load attempt)."""
    det = PiiDetector(model_name="nonexistent/model-that-will-fail")
    det._load_attempted = True
    det._fallback = True
    return det


async def test_email_regex_severity_low() -> None:
    det = _regex_only_detector()
    signals = await det.check(
        input_text="Contact noreply@anthropic.com for details",
        output_text="",
        call_id="t6",
    )
    email_signals = [s for s in signals if s.entity_type == "EMAIL"]
    assert email_signals
    assert all(s.severity == "low" for s in email_signals)


async def test_ssn_regex_severity_high() -> None:
    det = _regex_only_detector()
    signals = await det.check(
        input_text="",
        output_text="SSN: 123-45-6789",
        call_id="t7",
    )
    ssn_signals = [s for s in signals if s.entity_type == "SSN"]
    assert ssn_signals
    assert all(s.severity == "high" for s in ssn_signals)


async def test_phone_regex_severity_low() -> None:
    det = _regex_only_detector()
    signals = await det.check(
        input_text="",
        output_text="Call me at (555) 123-4567",
        call_id="t8",
    )
    phone_signals = [s for s in signals if s.entity_type == "PHONE"]
    assert phone_signals
    assert all(s.severity == "low" for s in phone_signals)


class _FakeNerPipeline:
    """Stand-in for the transformers token-classification pipeline."""

    def __init__(self, entities: list[dict]) -> None:
        self._entities = entities

    def __call__(self, text: str) -> list[dict]:
        return self._entities


def _model_detector(entities: list[dict]) -> PiiDetector:
    det = PiiDetector()
    det._load_attempted = True
    det._fallback = False
    det._pipeline = _FakeNerPipeline(entities)
    return det


async def test_ner_path_uses_severity_map() -> None:
    det = _model_detector(
        [
            {"entity_group": "PERSON", "start": 0, "end": 4, "score": 0.99},
            {"entity_group": "IP_ADDRESS", "start": 10, "end": 19, "score": 0.95},
            {"entity_group": "ID_NUM", "start": 25, "end": 34, "score": 0.90},
            {"entity_group": "CREDIT_CARD", "start": 40, "end": 56, "score": 0.97},
        ]
    )
    signals = await det.check(
        input_text="John lives at 10.0.0.1, id A12345678, card 4111111111111111",
        output_text="",
        call_id="t9",
    )
    by_type = {s.entity_type: s.severity for s in signals if "PII detected" in s.detail}
    assert by_type["PERSON"] == "low"
    assert by_type["IP_ADDRESS"] == "low"
    assert by_type["ID_NUM"] == "medium"
    assert by_type["CREDIT_CARD"] == "high"


async def test_ner_ssn_severity_high() -> None:
    det = _model_detector([{"entity_group": "B-SSN", "start": 8, "end": 19, "score": 0.98}])
    signals = await det.check(
        input_text="SSN is 987-65-4321",
        output_text="",
        call_id="t10",
    )
    ssn_model_signals = [
        s for s in signals if s.entity_type == "SSN" and "PII detected" in s.detail
    ]
    assert ssn_model_signals
    assert all(s.severity == "high" for s in ssn_model_signals)


def test_unknown_entity_type_defaults_to_medium() -> None:
    from aceteam_aep.safety.pii import _severity_for

    assert _severity_for("SOME_FUTURE_TYPE") == "medium"
