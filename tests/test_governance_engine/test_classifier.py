"""Tests for data classification."""

from aceteam_aep.governance_engine.classifier import DataClassifier


def test_clean_text_is_public() -> None:
    c = DataClassifier()
    assert c.classify_text("The weather is nice today.") == "public"


def test_pii_flag_makes_confidential() -> None:
    c = DataClassifier()
    assert c.classify_text("anything", pii_detected=True) == "confidential"


def test_financial_keywords_confidential() -> None:
    c = DataClassifier()
    assert c.classify_text("His SSN is 123-45-6789") == "confidential"
    assert c.classify_text("Bank account routing number 021000021") == "confidential"


def test_health_keywords_confidential() -> None:
    c = DataClassifier()
    assert c.classify_text("Patient diagnosis: stage 2 cancer") == "confidential"


def test_legal_keywords_confidential() -> None:
    c = DataClassifier()
    assert c.classify_text("This is attorney-client privileged") == "confidential"


def test_as_safety_detector() -> None:
    """DataClassifier implements SafetyDetector protocol."""
    c = DataClassifier()
    signals = c.check(
        input_text="", output_text="Patient diagnosis report",
        call_id="t1",
    )
    assert len(signals) == 1
    assert signals[0].signal_type == "data_classification"
    assert "confidential" in signals[0].detail


def test_public_text_no_signal() -> None:
    c = DataClassifier()
    signals = c.check(
        input_text="", output_text="The capital of France is Paris.",
        call_id="t1",
    )
    assert signals == []
