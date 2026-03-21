"""Tests for the enforcement engine."""

from aceteam_aep.enforcement import EnforcementDecision, EnforcementPolicy, evaluate
from aceteam_aep.safety.base import SafetySignal


def _sig(signal_type: str, severity: str) -> SafetySignal:
    return SafetySignal(signal_type=signal_type, severity=severity, call_id="t", detail="test")


def test_no_signals_passes() -> None:
    result = evaluate([], EnforcementPolicy())
    assert result.action == "pass"


def test_high_severity_blocks() -> None:
    result = evaluate([_sig("content_safety", "high")], EnforcementPolicy())
    assert result.action == "block"


def test_medium_severity_flags() -> None:
    result = evaluate([_sig("cost_anomaly", "medium")], EnforcementPolicy())
    assert result.action == "flag"


def test_low_severity_passes() -> None:
    result = evaluate([_sig("info", "low")], EnforcementPolicy())
    assert result.action == "pass"


def test_custom_policy_overrides() -> None:
    policy = EnforcementPolicy(block_on=frozenset(), flag_on=frozenset())
    result = evaluate([_sig("content_safety", "high")], policy)
    assert result.action == "pass"


def test_allow_types_bypasses_signal() -> None:
    policy = EnforcementPolicy(allow_types=frozenset({"cost_anomaly"}))
    result = evaluate([_sig("cost_anomaly", "high")], policy)
    assert result.action == "pass"


def test_mixed_severities_picks_highest() -> None:
    signals = [_sig("pii", "high"), _sig("cost_anomaly", "medium")]
    result = evaluate(signals, EnforcementPolicy())
    assert result.action == "block"


def test_decision_includes_reason() -> None:
    result = evaluate([_sig("pii", "high")], EnforcementPolicy())
    assert result.action == "block"
    assert "pii" in result.reason
