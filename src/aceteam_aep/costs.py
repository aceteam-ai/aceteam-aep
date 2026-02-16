"""AEP cost tracking - builds hierarchical cost trees from token usage."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

from .types import Usage

# Approximate per-token costs (USD) for common models.
# These are rough estimates; actual costs vary by provider and plan.
MODEL_COSTS: dict[str, tuple[Decimal, Decimal]] = {
    # (input_cost_per_token, output_cost_per_token)
    # OpenAI
    "gpt-4o": (Decimal("0.0000025"), Decimal("0.000010")),
    "gpt-4o-mini": (Decimal("0.00000015"), Decimal("0.0000006")),
    "gpt-4-turbo": (Decimal("0.000010"), Decimal("0.000030")),
    "gpt-4.5-preview": (Decimal("0.000075"), Decimal("0.00015")),
    "o1": (Decimal("0.000015"), Decimal("0.000060")),
    "o3-mini": (Decimal("0.0000011"), Decimal("0.0000044")),
    # Anthropic
    "claude-opus-4-5-20250514": (Decimal("0.000015"), Decimal("0.000075")),
    "claude-sonnet-4-5-20250514": (Decimal("0.000003"), Decimal("0.000015")),
    "claude-haiku-4-5-20251001": (Decimal("0.0000008"), Decimal("0.000004")),
    # Google
    "gemini-2.5-flash": (Decimal("0.00000015"), Decimal("0.0000006")),
    "gemini-2.5-pro": (Decimal("0.00000125"), Decimal("0.000010")),
    # Embeddings
    "text-embedding-3-small": (Decimal("0.00000002"), Decimal("0")),
    "text-embedding-3-large": (Decimal("0.00000013"), Decimal("0")),
}


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

    def __init__(self, entity: str = "platform") -> None:
        self._nodes: list[CostNode] = []
        self._entity = entity

    def record_llm_cost(
        self,
        span_id: str,
        model: str,
        usage: Usage,
        parent_cost_id: str | None = None,
    ) -> CostNode:
        """Record cost for an LLM call based on token usage."""
        input_rate, output_rate = MODEL_COSTS.get(model, (Decimal("0.000001"), Decimal("0.000002")))
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
        input_rate, _ = MODEL_COSTS.get(model, (Decimal("0.00000002"), Decimal("0")))
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


__all__ = ["CostCategory", "CostNode", "CostTracker", "MODEL_COSTS"]
