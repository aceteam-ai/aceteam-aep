from decimal import Decimal

from aceteam_aep.pricing import DefaultPricingProvider, PricingProvider


def test_default_pricing_known_model():
    """DefaultPricingProvider returns non-zero cost for known models."""
    provider = DefaultPricingProvider()
    cost = provider.get_cost("gpt-4o", input_tokens=1000, output_tokens=500)
    assert cost > Decimal("0")


def test_default_pricing_unknown_model():
    """DefaultPricingProvider returns zero for unknown models."""
    provider = DefaultPricingProvider()
    cost = provider.get_cost("unknown-model-xyz", input_tokens=1000, output_tokens=500)
    assert cost == Decimal("0")


def test_default_pricing_zero_tokens():
    """Zero tokens = zero cost."""
    provider = DefaultPricingProvider()
    cost = provider.get_cost("gpt-4o", input_tokens=0, output_tokens=0)
    assert cost == Decimal("0")


def test_custom_pricing_provider():
    """A custom PricingProvider implementation works via protocol."""

    class FixedPricing:
        def get_cost(
            self,
            model: str,
            input_tokens: int,
            output_tokens: int,
            *,
            cached_input_tokens: int = 0,
            organization_id: str | None = None,
        ) -> Decimal:
            return Decimal("0.50")

    provider: PricingProvider = FixedPricing()
    assert provider.get_cost("any-model", 100, 100) == Decimal("0.50")
