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
        organization_id: str | None = None,
    ) -> Decimal:
        info = get_model_info(model)
        if not info:
            return Decimal("0")
        return info.input_cost_per_token * input_tokens + info.output_cost_per_token * output_tokens


__all__ = ["DefaultPricingProvider", "PricingProvider"]
