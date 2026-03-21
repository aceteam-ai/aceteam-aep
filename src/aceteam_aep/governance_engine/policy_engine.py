"""Governance policy engine — enforce per-org data handling rules.

Evaluates whether a request/response is allowed given:
- Entity's maximum clearance level
- Consent directives (training, sharing, retention)
- Data classification of the content
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..proxy.headers import classification_rank


@dataclass(frozen=True)
class OrgPolicy:
    """Per-org governance policy."""

    entity: str
    max_classification: str = "public"  # highest level this entity can access
    consent: dict[str, bool] = field(default_factory=dict)
    # e.g., {"training": False, "sharing": True, "retention": True}


@dataclass
class GovernanceDecision:
    """Result of governance policy evaluation."""

    action: str  # "allow", "deny", "redact"
    reason: str = ""
    violations: list[str] = field(default_factory=list)


class GovernancePolicyEngine:
    """Evaluate governance policies against requests."""

    def __init__(self, policies: dict[str, OrgPolicy] | None = None) -> None:
        self._policies = policies or {}

    def add_policy(self, policy: OrgPolicy) -> None:
        self._policies[policy.entity] = policy

    def get_policy(self, entity: str) -> OrgPolicy | None:
        return self._policies.get(entity)

    def evaluate(
        self,
        entity: str,
        data_classification: str,
        consent_required: dict[str, bool] | None = None,
    ) -> GovernanceDecision:
        """Evaluate whether the entity can access data at the given classification."""
        policy = self._policies.get(entity)
        if not policy:
            # No policy = permissive (Layer 1 behavior)
            return GovernanceDecision(action="allow")

        violations: list[str] = []

        # Classification clearance check
        if classification_rank(data_classification) > classification_rank(
            policy.max_classification
        ):
            violations.append(
                f"Entity {entity} has clearance for {policy.max_classification} "
                f"but data is classified as {data_classification}"
            )

        # Consent check
        if consent_required:
            for key, required in consent_required.items():
                if required and not policy.consent.get(key, False):
                    violations.append(
                        f"Consent '{key}' required but entity {entity} "
                        f"has not granted it"
                    )

        if violations:
            return GovernanceDecision(
                action="deny",
                reason="; ".join(violations),
                violations=violations,
            )

        return GovernanceDecision(action="allow")


__all__ = ["GovernanceDecision", "GovernancePolicyEngine", "OrgPolicy"]
