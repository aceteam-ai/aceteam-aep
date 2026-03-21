"""Tests for CLI pretty-print output."""

from decimal import Decimal

from aceteam_aep.cli import format_session_summary
from aceteam_aep.safety.base import SafetySignal


def test_format_summary_clean() -> None:
    output = format_session_summary(cost=Decimal("0.0042"), signals=[], call_count=3)
    assert "$0.004200" in output
    assert "3" in output
    assert "PASS" in output


def test_format_summary_with_pii_signal() -> None:
    sig = SafetySignal(signal_type="pii", severity="high", call_id="x", detail="SSN found")
    output = format_session_summary(cost=Decimal("0.01"), signals=[sig], call_count=1)
    assert "pii" in output
    assert "BLOCK" in output
    assert "SSN found" in output


def test_format_summary_with_medium_signal() -> None:
    sig = SafetySignal(
        signal_type="cost_anomaly", severity="medium", call_id="x", detail="spike"
    )
    output = format_session_summary(cost=Decimal("0.05"), signals=[sig], call_count=5)
    assert "FLAG" in output


def test_format_summary_zero_cost() -> None:
    output = format_session_summary(cost=Decimal("0"), signals=[], call_count=0)
    assert "$0.000000" in output
    assert "0" in output
