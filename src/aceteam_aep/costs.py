"""AEP cost tracking - builds hierarchical cost trees from token usage."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from inspect import Parameter, signature
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


def _price_custom_usage(pricing: PricingProvider, model: str, usage: Usage) -> Decimal:
    """Adapt cached usage to providers written before the cache pricing arguments.

    Older providers have no cache rate contract. Their marginal input price is
    used for Anthropic's 0.1x reads and 1.25x writes, so cache usage cannot
    disappear from billing or raise an unexpected-keyword error.
    """
    if not (usage.cache_read_input_tokens or usage.cache_creation_input_tokens):
        return pricing.get_cost(model, usage.prompt_tokens, usage.completion_tokens)

    try:
        parameters = signature(pricing.get_cost).parameters
    except (TypeError, ValueError):
        parameters = {}
    # A variadic **kwargs can swallow a cache argument without billing it.
    # Only an explicit parameter establishes support for that token category.
    keyword_kinds = (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    read_parameter = parameters.get("cached_input_tokens")
    creation_parameter = parameters.get("cache_creation_input_tokens")
    accepts_read = read_parameter is not None and read_parameter.kind in keyword_kinds
    accepts_creation = creation_parameter is not None and creation_parameter.kind in keyword_kinds
    kwargs: dict[str, int] = {}
    if accepts_read:
        kwargs["cached_input_tokens"] = usage.cache_read_input_tokens
    if accepts_creation:
        kwargs["cache_creation_input_tokens"] = usage.cache_creation_input_tokens

    input_tokens = usage.prompt_tokens
    if not accepts_creation:
        input_tokens += usage.cache_creation_input_tokens
    cost = pricing.get_cost(model, input_tokens, usage.completion_tokens, **kwargs)
    if accepts_read and accepts_creation:
        return cost

    zero_kwargs = {"cached_input_tokens": 0} if accepts_read else {}
    zero_input_cost = pricing.get_cost(model, 0, 0, **zero_kwargs)
    is_anthropic = (info := MODEL_REGISTRY.get(model)) is not None and info.provider == "anthropic"
    if not accepts_read and usage.cache_read_input_tokens:
        read_cost = pricing.get_cost(model, usage.cache_read_input_tokens, 0, **zero_kwargs)
        cost += (read_cost - zero_input_cost) * (Decimal("0.1") if is_anthropic else 1)
    if not accepts_creation and is_anthropic and usage.cache_creation_input_tokens:
        write_cost = pricing.get_cost(model, usage.cache_creation_input_tokens, 0, **zero_kwargs)
        cost += (write_cost - zero_input_cost) * Decimal("0.25")
    return cost


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
            cost = _price_custom_usage(self._pricing, model, usage)
        else:
            info = MODEL_REGISTRY.get(model)
            input_rate = info.input_cost_per_token if info else _FALLBACK_LLM_COSTS[0]
            output_rate = info.output_cost_per_token if info else _FALLBACK_LLM_COSTS[1]
            if info and info.provider == "anthropic":
                cost = (
                    input_rate
                    * (
                        usage.prompt_tokens
                        + Decimal("0.1") * usage.cache_read_input_tokens
                        + Decimal("1.25") * usage.cache_creation_input_tokens
                    )
                    + output_rate * usage.completion_tokens
                )
            else:
                cost = (
                    input_rate
                    * (
                        usage.prompt_tokens
                        + usage.cache_read_input_tokens
                        + usage.cache_creation_input_tokens
                    )
                    + output_rate * usage.completion_tokens
                )

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
                "cache_read_input_tokens": usage.cache_read_input_tokens,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens,
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
