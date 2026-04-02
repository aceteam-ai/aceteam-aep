"""Tests for the enforcement engine."""

import tempfile
from pathlib import Path

from aceteam_aep.enforcement import (
    DetectorPolicy,
    EnforcementPolicy,
    build_detectors_from_policy,
    evaluate,
)
from aceteam_aep.safety.base import SafetySignal


def _sig(
    signal_type: str, severity: str, score: float | None = None, detector: str = ""
) -> SafetySignal:
    return SafetySignal(
        signal_type=signal_type, severity=severity, call_id="t", detail="test",
        score=score, detector=detector,
    )


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


# --- Per-detector override tests ---


def test_override_blocks_regardless_of_severity() -> None:
    """A detector override with action=block should block even low severity."""
    policy = EnforcementPolicy(
        overrides={"pii": DetectorPolicy(action="block")},
    )
    result = evaluate([_sig("pii", "low", score=0.9)], policy)
    assert result.action == "block"


def test_override_passes_when_disabled() -> None:
    """Disabled detector overrides should pass."""
    policy = EnforcementPolicy(
        overrides={"pii": DetectorPolicy(action="block", enabled=False)},
    )
    result = evaluate([_sig("pii", "high", score=0.9)], policy)
    assert result.action == "pass"


def test_override_threshold_downgrades_below() -> None:
    """Score below threshold should downgrade the action."""
    policy = EnforcementPolicy(
        overrides={"pii": DetectorPolicy(action="block", threshold=0.8)},
    )
    # Score 0.5 < threshold 0.8 → downgraded
    result = evaluate([_sig("pii", "high", score=0.5)], policy)
    assert result.action in ("pass", "flag")
    assert result.action != "block"


def test_override_threshold_applies_above() -> None:
    """Score above threshold should use the override action."""
    policy = EnforcementPolicy(
        overrides={"pii": DetectorPolicy(action="block", threshold=0.8)},
    )
    result = evaluate([_sig("pii", "high", score=0.95)], policy)
    assert result.action == "block"


def test_override_no_score_always_applies() -> None:
    """Signal with score=None should always match (above any threshold)."""
    policy = EnforcementPolicy(
        overrides={"pii": DetectorPolicy(action="block", threshold=0.8)},
    )
    result = evaluate([_sig("pii", "high", score=None)], policy)
    assert result.action == "block"


def test_override_by_detector_name() -> None:
    """Overrides can match on detector field as well as signal_type."""
    policy = EnforcementPolicy(
        overrides={"my_detector": DetectorPolicy(action="flag")},
    )
    result = evaluate([_sig("pii", "high", score=1.0, detector="my_detector")], policy)
    assert result.action == "flag"


def test_mixed_overrides_and_severity() -> None:
    """Override applies to matching signal, severity fallback for others."""
    policy = EnforcementPolicy(
        overrides={"cost_anomaly": DetectorPolicy(action="pass")},
    )
    signals = [
        _sig("cost_anomaly", "medium", score=10.0),  # override → pass
        _sig("pii", "high", score=0.99),  # no override → severity → block
    ]
    result = evaluate(signals, policy)
    assert result.action == "block"


# --- from_dict / from_yaml ---


def test_from_dict_basic() -> None:
    policy = EnforcementPolicy.from_dict({
        "default_action": "flag",
        "detectors": {
            "pii": {"action": "block", "threshold": 0.8},
            "cost_anomaly": {"action": "pass", "multiplier": 10},
        },
    })
    assert policy.default_action == "flag"
    assert "pii" in policy.overrides
    assert policy.overrides["pii"].action == "block"
    assert policy.overrides["pii"].threshold == 0.8
    assert policy.overrides["cost_anomaly"].extra["multiplier"] == 10


def test_from_yaml() -> None:
    yaml_content = """\
default_action: flag
detectors:
  pii:
    action: block
    threshold: 0.8
  agent_threat:
    action: block
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        policy = EnforcementPolicy.from_yaml(f.name)

    assert policy.default_action == "flag"
    assert policy.overrides["pii"].threshold == 0.8
    assert policy.overrides["agent_threat"].action == "block"


def test_from_config_none() -> None:
    policy = EnforcementPolicy.from_config(None)
    assert policy == EnforcementPolicy()


def test_from_config_dict() -> None:
    policy = EnforcementPolicy.from_config({"default_action": "pass"})
    assert policy.default_action == "pass"


def test_from_config_passthrough() -> None:
    original = EnforcementPolicy(default_action="block")
    policy = EnforcementPolicy.from_config(original)
    assert policy is original


# --- build_detectors_from_policy ---


def test_build_detectors_respects_disabled() -> None:
    policy = EnforcementPolicy(
        overrides={
            "cost_anomaly": DetectorPolicy(enabled=False),
            "agent_threat": DetectorPolicy(action="block"),
        },
    )
    detectors = build_detectors_from_policy(policy)
    names = [d.name for d in detectors]
    assert "cost_anomaly" not in names
    assert "agent_threat" in names


def test_build_detectors_custom_multiplier() -> None:
    policy = EnforcementPolicy(
        overrides={
            "cost_anomaly": DetectorPolicy(extra={"multiplier": 20}),
        },
    )
    detectors = build_detectors_from_policy(policy)
    cost_det = next(d for d in detectors if d.name == "cost_anomaly")
    assert cost_det._multiplier == 20


def test_build_detectors_includes_trust_engine():
    """build_detectors_from_policy creates TrustEngineDetector when configured."""
    from aceteam_aep.enforcement import build_detectors_from_policy, EnforcementPolicy, DetectorPolicy
    policy = EnforcementPolicy(
        overrides={
            "trust_engine": DetectorPolicy(
                action="flag",
                enabled=True,
                extra={"dimensions": ["finance", "program", "web"]},
            ),
            "agent_threat": DetectorPolicy(enabled=False),
            "pii": DetectorPolicy(enabled=False),
            "cost_anomaly": DetectorPolicy(enabled=False),
            "content_safety": DetectorPolicy(enabled=False),
        }
    )
    detectors = build_detectors_from_policy(policy)
    names = [getattr(d, "name", type(d).__name__) for d in detectors]
    assert any("trust" in n.lower() for n in names), f"No trust engine detector in {names}"


def test_build_detectors_trust_engine_disabled():
    """Trust Engine detector not created when disabled."""
    from aceteam_aep.enforcement import build_detectors_from_policy, EnforcementPolicy, DetectorPolicy
    policy = EnforcementPolicy(
        overrides={
            "trust_engine": DetectorPolicy(enabled=False),
        }
    )
    detectors = build_detectors_from_policy(policy)
    names = [getattr(d, "name", type(d).__name__) for d in detectors]
    assert not any("trust" in n.lower() for n in names)
