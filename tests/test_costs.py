"""Tests for cost tracking."""

from aceteam_aep.costs import MODEL_COSTS, CostTracker
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


def test_model_costs_has_common_models():
    assert "gpt-4o" in MODEL_COSTS
    assert "gpt-4o-mini" in MODEL_COSTS
    assert "text-embedding-3-small" in MODEL_COSTS


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
