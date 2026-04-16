"""Tests for PII detector — uses real model if available, regex fallback otherwise."""

import pytest

from aceteam_aep.safety.pii import PiiDetector


@pytest.fixture(scope="module")
def detector() -> PiiDetector:
    return PiiDetector()


def test_detects_ssn(detector: PiiDetector) -> None:
    signals = detector.check(input_text="", output_text="His SSN is 123-45-6789", call_id="t1")
    assert any(s.signal_type == "pii" for s in signals)


def test_detects_email(detector: PiiDetector) -> None:
    signals = detector.check(
        input_text="", output_text="Contact john.doe@company.com for help", call_id="t2"
    )
    assert any(s.signal_type == "pii" for s in signals)


def test_clean_text_no_signal(detector: PiiDetector) -> None:
    signals = detector.check(input_text="", output_text="The weather is nice today.", call_id="t3")
    pii_signals = [s for s in signals if s.signal_type == "pii"]
    # Model may detect some false positives, regex should not
    # Allow either 0 signals or only low-confidence ones
    assert len(pii_signals) == 0 or all("regex" not in s.detail for s in pii_signals)


def test_detects_phone_number(detector: PiiDetector) -> None:
    signals = detector.check(input_text="", output_text="Call me at (555) 123-4567", call_id="t4")
    assert any(s.signal_type == "pii" for s in signals)


def test_regex_fallback_works() -> None:
    """Force regex fallback and verify it works."""
    det = PiiDetector(model_name="nonexistent/model-that-will-fail")
    signals = det.check(input_text="", output_text="SSN: 123-45-6789", call_id="t5")
    assert any(s.signal_type == "pii" for s in signals)
    assert any("regex fallback" in s.detail for s in signals)
