from decimal import Decimal

from aceteam_aep.governance import (
    BudgetLimit,
    CitationClassification,
    CitationConstraints,
    GovernanceConfig,
    GovernancePolicy,
    PermissionScope,
    SecurityLevel,
)


def test_security_level_values():
    assert SecurityLevel.PUBLIC == "public"
    assert SecurityLevel.RESTRICTED == "restricted"


def test_governance_config_defaults():
    config = GovernanceConfig()
    assert config.default_security_level is None
    assert config.consent_required is False
    assert config.audit_all is False


def test_citation_classification():
    cc = CitationClassification(level=SecurityLevel.CONFIDENTIAL, regulations=["HIPAA"])
    assert cc.level == "confidential"


def test_budget_limit():
    bl = BudgetLimit(total=Decimal("10.00"))
    assert bl.currency == "USD"


def test_permission_scope():
    scope = PermissionScope(allowed_actions=["read", "write"])
    assert scope.is_action_allowed("read")
    assert not scope.is_action_allowed("delete")


def test_governance_policy_defaults():
    policy = GovernancePolicy()
    assert policy.security_level == SecurityLevel.INTERNAL


def test_citation_constraints():
    cc = CitationConstraints(audit_required=True, retention_days=365)
    assert cc.retention_days == 365
