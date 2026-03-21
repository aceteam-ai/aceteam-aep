"""Enforcement engine — PASS / FLAG / BLOCK decisions from safety signals."""

from __future__ import annotations

from dataclasses import dataclass, field

from .safety.base import SafetySignal


@dataclass(frozen=True)
class EnforcementPolicy:
    """Configurable policy for mapping signal severities to enforcement actions."""

    block_on: frozenset[str] = frozenset({"high"})
    flag_on: frozenset[str] = frozenset({"medium"})
    allow_types: frozenset[str] = frozenset()


@dataclass
class EnforcementDecision:
    """Result of evaluating safety signals against a policy."""

    action: str  # "pass", "flag", "block"
    signals: list[SafetySignal] = field(default_factory=list)
    reason: str = ""


def evaluate(signals: list[SafetySignal], policy: EnforcementPolicy) -> EnforcementDecision:
    """Evaluate safety signals against a policy and return an enforcement decision."""
    if not signals:
        return EnforcementDecision(action="pass")

    active = [s for s in signals if s.signal_type not in policy.allow_types]
    if not active:
        return EnforcementDecision(action="pass", signals=signals)

    if any(s.severity in policy.block_on for s in active):
        return EnforcementDecision(
            action="block",
            signals=active,
            reason="; ".join(
                f"{s.signal_type}: {s.detail}"
                for s in active
                if s.severity in policy.block_on
            ),
        )
    if any(s.severity in policy.flag_on for s in active):
        return EnforcementDecision(
            action="flag",
            signals=active,
            reason="; ".join(
                f"{s.signal_type}: {s.detail}"
                for s in active
                if s.severity in policy.flag_on
            ),
        )
    return EnforcementDecision(action="pass", signals=active)


__all__ = ["EnforcementDecision", "EnforcementPolicy", "evaluate"]
