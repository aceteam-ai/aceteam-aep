"""Tests for the redactor."""

from aceteam_aep.governance_engine.redactor import Redactor


def test_redact_ssn() -> None:
    r = Redactor()
    text, redactions = r.redact("His SSN is 123-45-6789")
    assert "[REDACTED]" in text
    assert "123-45-6789" not in text
    assert any("SSN" in d for d in redactions)


def test_redact_email() -> None:
    r = Redactor()
    text, redactions = r.redact("Contact john@example.com for details")
    assert "john@example.com" not in text
    assert "[REDACTED]" in text


def test_redact_phone() -> None:
    r = Redactor()
    text, redactions = r.redact("Call (555) 123-4567")
    assert "(555) 123-4567" not in text


def test_clean_text_unchanged() -> None:
    r = Redactor()
    text, redactions = r.redact("The capital of France is Paris.")
    assert text == "The capital of France is Paris."
    assert redactions == []


def test_custom_marker() -> None:
    r = Redactor(marker="[***]")
    text, _ = r.redact("SSN: 123-45-6789")
    assert "[***]" in text


def test_redact_if_needed_public() -> None:
    r = Redactor()
    text, redactions = r.redact_if_needed("SSN: 123-45-6789", "public")
    assert "123-45-6789" in text  # not redacted for public
    assert redactions == []


def test_redact_if_needed_confidential() -> None:
    r = Redactor()
    text, redactions = r.redact_if_needed("SSN: 123-45-6789", "confidential")
    assert "123-45-6789" not in text  # redacted for confidential
    assert len(redactions) > 0


def test_multiple_patterns() -> None:
    r = Redactor()
    text, redactions = r.redact(
        "SSN: 123-45-6789, email: admin@test.com, phone: (555) 123-4567"
    )
    assert "123-45-6789" not in text
    assert "admin@test.com" not in text
    assert text.count("[REDACTED]") >= 3
