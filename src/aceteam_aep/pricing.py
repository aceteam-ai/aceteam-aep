"""Pricing provider protocol and default implementation."""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol, runtime_checkable

from aceteam_aep.models import get_model_info


@runtime_checkable
class PricingProvider(Protocol):
    """Protocol for computing LLM costs. Implement to provide custom pricing."""

    def get_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        cached_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        organization_id: str | None = None,
    ) -> Decimal: ...


class DefaultPricingProvider:
    """Uses the static ModelInfo registry from aceteam_aep.models."""

    def get_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        cached_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        organization_id: str | None = None,
    ) -> Decimal:
        info = get_model_info(model)
        if not info:
            return Decimal("0")
        input_rate = info.input_cost_per_token
        if info.provider == "anthropic":
            # This SDK emits only five-minute ephemeral cache markers.
            return (
                input_rate * input_tokens
                + input_rate * Decimal("0.1") * cached_input_tokens
                + input_rate * Decimal("1.25") * cache_creation_input_tokens
                + info.output_cost_per_token * output_tokens
            )
        return (
            input_rate * (input_tokens + cached_input_tokens + cache_creation_input_tokens)
            + info.output_cost_per_token * output_tokens
        )


__all__ = ["DefaultPricingProvider", "PricingProvider"]
