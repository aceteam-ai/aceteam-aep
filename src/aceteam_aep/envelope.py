"""AEP ExecutionEnvelope builder."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from .budget import BudgetState
from .costs import CostNode
from .spans import Span


@dataclass
class Citation:
    """Source attribution for content."""

    span_id: str
    source_type: str
    content: str
    confidence: float | None = None


@dataclass
class ExecutionError:
    """An error that occurred during execution."""

    code: str
    message: str
    node_id: str | None = None


@dataclass
class ExecutionEnvelope:
    """Execution result envelope - the standard return type for AEP-compliant executions."""

    execution_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: Literal["success", "partial", "failure"] = "success"
    outputs: dict[str, Any] | None = None
    cost_tree: list[CostNode] | None = None
    spans: list[Span] | None = None
    citations: list[Citation] | None = None
    errors: list[ExecutionError] | None = None
    warnings: list[str] | None = None
    budget_state: BudgetState | None = None

    def add_citation(self, citation: Citation) -> None:
        if self.citations is None:
            self.citations = []
        self.citations.append(citation)

    def add_cost(self, cost: CostNode) -> None:
        if self.cost_tree is None:
            self.cost_tree = []
        self.cost_tree.append(cost)

    def add_error(self, code: str, message: str, node_id: str | None = None) -> None:
        if self.errors is None:
            self.errors = []
        self.errors.append(ExecutionError(code=code, message=message, node_id=node_id))


__all__ = ["Citation", "ExecutionEnvelope", "ExecutionError"]
