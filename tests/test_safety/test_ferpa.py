"""Tests for the FERPA education records detector."""

from __future__ import annotations

from aceteam_aep.safety.ferpa import FerpaDetector


def test_detects_student_id_with_context() -> None:
    det = FerpaDetector()
    signals = det.check(
        input_text="Look up student 123456789 enrollment status",
        output_text="",
        call_id="t1",
    )
    assert any(s.signal_type == "ferpa" for s in signals)


def test_detects_labeled_student_id() -> None:
    det = FerpaDetector()
    signals = det.check(
        input_text="Student ID: 87654321",
        output_text="",
        call_id="t1",
    )
    assert any("student id" in s.detail.lower() for s in signals)


def test_detects_gpa_with_student_context() -> None:
    det = FerpaDetector()
    signals = det.check(
        input_text="",
        output_text="The student has a GPA: 3.85 on their transcript record",
        call_id="t1",
    )
    assert any(s.signal_type == "ferpa" for s in signals)


def test_detects_grade_disclosure() -> None:
    det = FerpaDetector()
    signals = det.check(
        input_text="",
        output_text="She earned an A+ in Chemistry",
        call_id="t1",
    )
    assert any("grade" in s.detail.lower() for s in signals)


def test_detects_transcript_reference() -> None:
    det = FerpaDetector()
    signals = det.check(
        input_text="Pull the transcript for student Jane Doe",
        output_text="",
        call_id="t1",
    )
    assert any("transcript" in s.detail.lower() for s in signals)


def test_detects_financial_aid() -> None:
    det = FerpaDetector()
    signals = det.check(
        input_text="",
        output_text="Student was awarded a Pell Grant of $6,895",
        call_id="t1",
    )
    assert any("financial aid" in s.detail.lower() for s in signals)


def test_detects_disciplinary_record() -> None:
    det = FerpaDetector()
    signals = det.check(
        input_text="Show the disciplinary record for student John Smith",
        output_text="",
        call_id="t1",
    )
    assert any("disciplin" in s.detail.lower() for s in signals)


def test_detects_class_roster() -> None:
    det = FerpaDetector()
    signals = det.check(
        input_text="Print the class roster for CS 101",
        output_text="",
        call_id="t1",
    )
    assert any("roster" in s.detail.lower() or "bulk" in s.detail.lower() for s in signals)


def test_clean_text_no_signals() -> None:
    det = FerpaDetector()
    signals = det.check(
        input_text="What is the capital of France?",
        output_text="The capital of France is Paris.",
        call_id="t1",
    )
    assert len(signals) == 0


def test_severity_is_high_for_student_ids() -> None:
    det = FerpaDetector()
    signals = det.check(
        input_text="Student ID: 12345678",
        output_text="",
        call_id="t1",
    )
    high = [s for s in signals if s.severity == "high"]
    assert len(high) >= 1
