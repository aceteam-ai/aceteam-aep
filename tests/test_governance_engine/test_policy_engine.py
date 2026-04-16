"""Tests for governance policy engine."""

from aceteam_aep.governance_engine.policy_engine import (
    GovernancePolicyEngine,
    OrgPolicy,
)


def test_no_policy_allows() -> None:
    engine = GovernancePolicyEngine()
    decision = engine.evaluate("org:unknown", "restricted")
    assert decision.action == "allow"


def test_clearance_allows() -> None:
    engine = GovernancePolicyEngine()
    engine.add_policy(OrgPolicy(entity="org:acme", max_classification="confidential"))
    decision = engine.evaluate("org:acme", "internal")
    assert decision.action == "allow"


def test_clearance_denies_above_level() -> None:
    engine = GovernancePolicyEngine()
    engine.add_policy(OrgPolicy(entity="org:acme", max_classification="internal"))
    decision = engine.evaluate("org:acme", "confidential")
    assert decision.action == "deny"
    assert "clearance" in decision.reason


def test_consent_check_passes() -> None:
    engine = GovernancePolicyEngine()
    engine.add_policy(
        OrgPolicy(
            entity="org:acme",
            max_classification="restricted",
            consent={"training": True, "sharing": True},
        )
    )
    decision = engine.evaluate(
        "org:acme",
        "public",
        consent_required={"training": True},
    )
    assert decision.action == "allow"


def test_consent_check_denies() -> None:
    engine = GovernancePolicyEngine()
    engine.add_policy(
        OrgPolicy(
            entity="org:acme",
            max_classification="restricted",
            consent={"training": False},
        )
    )
    decision = engine.evaluate(
        "org:acme",
        "public",
        consent_required={"training": True},
    )
    assert decision.action == "deny"
    assert "consent" in decision.reason.lower()


def test_multiple_violations() -> None:
    engine = GovernancePolicyEngine()
    engine.add_policy(
        OrgPolicy(
            entity="org:acme",
            max_classification="public",
            consent={"training": False},
        )
    )
    decision = engine.evaluate(
        "org:acme",
        "restricted",
        consent_required={"training": True},
    )
    assert decision.action == "deny"
    assert len(decision.violations) == 2
