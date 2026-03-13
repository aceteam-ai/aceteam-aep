"""AEP cost tracking - builds hierarchical cost trees from token usage."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

from .models import MODEL_REGISTRY
from .pricing import PricingProvider
from .types import Usage

# Fallback per-token costs for models not in the registry.
_FALLBACK_LLM_COSTS = (Decimal("0.000001"), Decimal("0.000002"))
_FALLBACK_EMBEDDING_COSTS = (Decimal("0.00000002"), Decimal("0"))


CostCategory = Literal["llm_tokens", "embedding", "compute", "storage", "api"]


@dataclass
class CostNode:
    """A node in the hierarchical cost tree."""

    id: str
    parent_id: str | None
    entity: str
    category: CostCategory
    compute_cost: Decimal
    value_added_fee: Decimal = Decimal("0")
    platform_fee: Decimal = Decimal("0")
    currency: str = "USD"
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def total_cost(self) -> Decimal:
        """Calculate total cost including fees."""
        return self.compute_cost + self.value_added_fee + self.platform_fee


class CostTracker:
    """Tracks costs during execution, building a hierarchical cost tree.

    Each LLM call, embedding call, or other billable operation produces
    a CostNode linked to its parent span.
    """

    def __init__(self, entity: str = "platform", pricing: PricingProvider | None = None) -> None:
        self._nodes: list[CostNode] = []
        self._entity = entity
        self._pricing = pricing

    def add_node(self, node: CostNode) -> None:
        """Add a pre-built CostNode to the tracker."""
        self._nodes.append(node)

    def record_llm_cost(
        self,
        span_id: str,
        model: str,
        usage: Usage,
        parent_cost_id: str | None = None,
    ) -> CostNode:
        """Record cost for an LLM call based on token usage."""
        if self._pricing:
            cost = self._pricing.get_cost(model, usage.prompt_tokens, usage.completion_tokens)
        else:
            info = MODEL_REGISTRY.get(model)
            input_rate = info.input_cost_per_token if info else _FALLBACK_LLM_COSTS[0]
            output_rate = info.output_cost_per_token if info else _FALLBACK_LLM_COSTS[1]
            cost = (input_rate * usage.prompt_tokens) + (output_rate * usage.completion_tokens)

        node = CostNode(
            id=uuid.uuid4().hex,
            parent_id=parent_cost_id,
            entity=self._entity,
            category="llm_tokens",
            compute_cost=cost,
            timestamp=datetime.now(UTC).isoformat(),
            metadata={
                "span_id": span_id,
                "model": model,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            },
        )
        self._nodes.append(node)
        return node

    def record_embedding_cost(
        self,
        span_id: str,
        model: str,
        token_count: int,
        parent_cost_id: str | None = None,
    ) -> CostNode:
        """Record cost for an embedding call."""
        info = MODEL_REGISTRY.get(model)
        input_rate = info.input_cost_per_token if info else _FALLBACK_EMBEDDING_COSTS[0]
        cost = input_rate * token_count

        node = CostNode(
            id=uuid.uuid4().hex,
            parent_id=parent_cost_id,
            entity=self._entity,
            category="embedding",
            compute_cost=cost,
            timestamp=datetime.now(UTC).isoformat(),
            metadata={
                "span_id": span_id,
                "model": model,
                "token_count": token_count,
            },
        )
        self._nodes.append(node)
        return node

    def get_cost_tree(self) -> list[CostNode]:
        """Get all cost nodes."""
        return list(self._nodes)

    def total_spent(self) -> Decimal:
        """Calculate total spent across all nodes."""
        return sum((node.total_cost() for node in self._nodes), Decimal("0"))


__all__ = ["CostCategory", "CostNode", "CostTracker"]
