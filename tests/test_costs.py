"""Tests for cost tracking."""

from decimal import Decimal

from aceteam_aep.costs import CostTracker
from aceteam_aep.models import MODEL_REGISTRY
from aceteam_aep.pricing import DefaultPricingProvider
from aceteam_aep.types import Usage


def test_record_llm_cost():
    tracker = CostTracker(entity="org:test")
    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    node = tracker.record_llm_cost(span_id="span-1", model="gpt-4o", usage=usage)

    assert node.category == "llm_tokens"
    assert node.entity == "org:test"
    assert node.compute_cost > 0
    assert node.metadata["model"] == "gpt-4o"
    assert node.metadata["prompt_tokens"] == 1000


def test_record_embedding_cost():
    tracker = CostTracker()
    node = tracker.record_embedding_cost(
        span_id="span-1", model="text-embedding-3-small", token_count=500
    )

    assert node.category == "embedding"
    assert node.compute_cost > 0


def test_total_spent():
    tracker = CostTracker()
    usage1 = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    usage2 = Usage(prompt_tokens=200, completion_tokens=100, total_tokens=300)

    tracker.record_llm_cost(span_id="s1", model="gpt-4o", usage=usage1)
    tracker.record_llm_cost(span_id="s2", model="gpt-4o", usage=usage2)

    total = tracker.total_spent()
    assert total > 0
    assert len(tracker.get_cost_tree()) == 2


def test_cost_node_total():
    tracker = CostTracker()
    usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    node = tracker.record_llm_cost(span_id="s1", model="gpt-4o", usage=usage)

    assert node.total_cost() == node.compute_cost + node.value_added_fee + node.platform_fee


def test_model_registry_has_common_models():
    assert "gpt-4o" in MODEL_REGISTRY
    assert "gpt-4o-mini" in MODEL_REGISTRY
    assert "text-embedding-3-small" in MODEL_REGISTRY


def test_unknown_model_gets_default_cost():
    tracker = CostTracker()
    usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    node = tracker.record_llm_cost(span_id="s1", model="unknown-model-xyz", usage=usage)
    assert node.compute_cost > 0  # Uses default rates


def test_parent_cost_id():
    tracker = CostTracker()
    usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    parent = tracker.record_llm_cost(span_id="s1", model="gpt-4o", usage=usage)
    child = tracker.record_llm_cost(
        span_id="s2", model="gpt-4o", usage=usage, parent_cost_id=parent.id
    )
    assert child.parent_id == parent.id


def test_anthropic_cache_read_and_write_are_priced_in_both_cost_paths():
    model = "claude-sonnet-4-5-20250514"
    input_rate = MODEL_REGISTRY[model].input_cost_per_token
    usage = Usage(
        prompt_tokens=1,
        completion_tokens=0,
        cache_read_input_tokens=1000,
        cache_creation_input_tokens=100,
    )
    expected = input_rate * (Decimal(1) + Decimal("0.1") * 1000 + Decimal("1.25") * 100)
    for pricing in (None, DefaultPricingProvider()):
        tracker = CostTracker(pricing=pricing)
        node = tracker.record_llm_cost("span", model, usage)
        assert node.compute_cost == expected
        assert tracker.total_spent() == expected
        assert node.metadata["cache_read_input_tokens"] == 1000
        assert node.metadata["cache_creation_input_tokens"] == 100


def test_custom_pricing_without_cache_arguments_still_handles_uncached_usage():
    class LegacyPricing:
        def get_cost(self, model: str, input_tokens: int, output_tokens: int) -> Decimal:
            return Decimal(input_tokens + output_tokens) / 1000

    tracker = CostTracker(pricing=LegacyPricing())  # type: ignore[arg-type]
    node = tracker.record_llm_cost("span", "custom", Usage(prompt_tokens=5, completion_tokens=3))
    assert node.compute_cost == Decimal("0.008")


def test_three_argument_pricing_accounts_for_anthropic_cache_usage():
    class LegacyPricing:
        def get_cost(self, model: str, input_tokens: int, output_tokens: int) -> Decimal:
            return Decimal(input_tokens + output_tokens) / 1000

    tracker = CostTracker(pricing=LegacyPricing())  # type: ignore[arg-type]
    usage = Usage(
        prompt_tokens=5,
        completion_tokens=3,
        cache_read_input_tokens=10,
        cache_creation_input_tokens=20,
    )
    node = tracker.record_llm_cost("span", "claude-sonnet-4-5-20250514", usage)
    assert node.compute_cost == Decimal("0.034")
    assert tracker.total_spent() == Decimal("0.034")


def test_prior_pricing_protocol_accounts_for_anthropic_cache_write():
    class PriorPricing:
        def get_cost(
            self,
            model: str,
            input_tokens: int,
            output_tokens: int,
            cached_input_tokens: int = 0,
        ) -> Decimal:
            return (
                Decimal(input_tokens + output_tokens) + Decimal("0.1") * cached_input_tokens
            ) / 1000

    tracker = CostTracker(pricing=PriorPricing())  # type: ignore[arg-type]
    usage = Usage(
        prompt_tokens=5,
        completion_tokens=3,
        cache_read_input_tokens=10,
        cache_creation_input_tokens=20,
    )
    node = tracker.record_llm_cost("span", "claude-sonnet-4-5-20250514", usage)
    assert node.compute_cost == Decimal("0.034")
