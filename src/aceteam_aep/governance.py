"""AEP protocol governance types."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum


class SecurityLevel(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class PromptLayer:
    source: str
    role: str
    content: str
    priority: int = 0


@dataclass
class BudgetLimit:
    total: Decimal
    currency: str = "USD"


@dataclass
class PermissionScope:
    allowed_actions: list[str] = field(default_factory=list)
    allowed_services: list[str] = field(default_factory=list)
    budget: BudgetLimit | None = None
    max_api_calls: int | None = None

    def is_action_allowed(self, action: str) -> bool:
        return action in self.allowed_actions

    def is_service_allowed(self, service: str) -> bool:
        return service in self.allowed_services


@dataclass
class GovernanceConfig:
    default_security_level: SecurityLevel | None = None
    consent_required: bool = False
    audit_all: bool = False
    permission_scope: PermissionScope | None = None


@dataclass
class GovernancePolicy:
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    allowed_purposes: list[str] = field(default_factory=list)


@dataclass
class CitationConstraints:
    audit_required: bool = False
    retention_days: int | None = None


@dataclass
class CitationClassification:
    level: SecurityLevel
    regulations: list[str] = field(default_factory=list)
    constraints: CitationConstraints | None = None


__all__ = [
    "BudgetLimit",
    "CitationClassification",
    "CitationConstraints",
    "GovernanceConfig",
    "GovernancePolicy",
    "PermissionScope",
    "PromptLayer",
    "SecurityLevel",
]
