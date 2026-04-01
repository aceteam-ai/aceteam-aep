"""FERPA detector — identifies student education records and PII protected under FERPA.

The Family Educational Rights and Privacy Act (FERPA) protects:
- Student IDs, enrollment records, grades, transcripts
- Financial aid records, disciplinary records
- Student names + institution context (when identifiable)
- Directory information when restricted

This detector catches FERPA-regulated data patterns in both input and
output text, preventing accidental disclosure through AI agents.

Designed for CSU, UC, and K-12 systems adopting AI workflows.
"""

from __future__ import annotations

import re

from .base import SafetySignal

# FERPA-specific patterns (education records context)
_FERPA_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # Student IDs — common formats across CSU/UC/community college
    (
        re.compile(r"\b\d{9}\b(?=.*(?:student|enroll|gpa|grade|transcript))", re.IGNORECASE),
        "Student ID (9-digit with education context)",
        "high",
    ),
    (
        re.compile(r"\bstudent\s*(?:id|number|#)\s*[:\s]*\d{5,10}\b", re.IGNORECASE),
        "Labeled student ID number",
        "high",
    ),
    # GPA and grades with student context
    (
        re.compile(
            r"\b(?:gpa|grade\s*point)\s*[:\s]*\d\.\d{1,2}\b"
            r"(?=.*(?:student|name|record|transcript))",
            re.IGNORECASE,
        ),
        "GPA with student context (education record)",
        "high",
    ),
    (
        re.compile(
            r"\b(?:earned|received|got)\s+(?:an?\s+)?[A-F][+-]?\s+"
            r"(?:in|for)\s+\w+",
            re.IGNORECASE,
        ),
        "Grade disclosure (specific grade in course)",
        "medium",
    ),
    # Transcript and enrollment records
    (
        re.compile(r"\btranscript\b.*\b(?:student|name|record|ssn)\b", re.IGNORECASE),
        "Transcript reference with student identifiers",
        "high",
    ),
    (
        re.compile(
            r"\b(?:enrolled|enrollment|registration)\s+(?:in|for|status)\b"
            r".*\b(?:student|name)\b",
            re.IGNORECASE,
        ),
        "Enrollment record with student identifiers",
        "medium",
    ),
    # Financial aid records (bidirectional — "student awarded Pell Grant" or "Pell Grant for student")
    (
        re.compile(
            r"\b(?:financial\s+aid|fafsa|pell\s+grant|scholarship|loan)\b"
            r".*\b(?:student|name|amount|awarded)\b"
            r"|\b(?:student|name|amount|awarded)\b"
            r".*\b(?:financial\s+aid|fafsa|pell\s+grant|scholarship|loan)\b",
            re.IGNORECASE,
        ),
        "Financial aid record",
        "high",
    ),
    # Disciplinary records (bidirectional)
    (
        re.compile(
            r"\bdisciplin\w*\b.*\b(?:student|name)\b"
            r"|\b(?:student|name)\b.*\bdisciplin\w*\b"
            r"|\b(?:suspension|expulsion|probation|conduct\s+violation)\b"
            r".*\b(?:student|name)\b",
            re.IGNORECASE,
        ),
        "Disciplinary record with student identifiers",
        "high",
    ),
    # FERPA directory information opt-out
    (
        re.compile(r"\b(?:ferpa|directory\s+information)\s+(?:opt|restrict|block)", re.IGNORECASE),
        "FERPA directory information restriction",
        "medium",
    ),
    # Education record bulk patterns
    (
        re.compile(r"\b(?:class\s+roster|grade\s+book|student\s+list)\b", re.IGNORECASE),
        "Bulk education record reference (class roster/grade book)",
        "medium",
    ),
]


class FerpaDetector:
    """Detect FERPA-protected education records in text.

    Scans for student IDs, grades, transcripts, financial aid records,
    and disciplinary records that are protected under FERPA. Designed
    for education institutions (CSU, UC, K-12) using AI workflows.
    """

    name = "ferpa"

    def check(
        self, *, input_text: str, output_text: str, call_id: str, **kwargs: object
    ) -> list[SafetySignal]:
        signals: list[SafetySignal] = []
        combined = f"{input_text} {output_text}"

        for pattern, description, severity in _FERPA_PATTERNS:
            if pattern.search(combined):
                signals.append(
                    SafetySignal(
                        signal_type="ferpa",
                        severity=severity,
                        call_id=call_id,
                        detail=description,
                        detector="ferpa",
                        score=1.0,
                    )
                )

        return signals


__all__ = ["FerpaDetector"]
