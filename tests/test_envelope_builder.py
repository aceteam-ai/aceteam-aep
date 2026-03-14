from decimal import Decimal

import pytest

from aceteam_aep.costs import CostNode
from aceteam_aep.envelope import Citation, ExecutionError
from aceteam_aep.envelope_builder import EnvelopeBuilder, NodeRecord
from aceteam_aep.envelope_helpers import compute_duration, extract_primary_model, sum_cost_tree
from aceteam_aep.spans import Span
from aceteam_aep.types import Usage

# --- EnvelopeBuilder lifecycle ---


def test_single_node_lifecycle():
    """start -> begin_node -> end_node -> finish produces valid envelope with 2 spans."""
    builder = EnvelopeBuilder(execution_id="run-1", org_id="org-1")
    builder.start()
    span_id = builder.begin_node("node-1", "ExtractionNode")
    builder.end_node(span_id)
    envelope = builder.finish()
    assert envelope.execution_id == "run-1"
    assert envelope.status == "success"
    assert envelope.spans is not None
    assert len(envelope.spans) == 2  # root + child


def test_multi_node_with_citations_and_costs():
    """3 nodes with mixed citations and costs, all aggregated."""
    builder = EnvelopeBuilder(execution_id="run-2", org_id="org-2")
    builder.start()

    # Node 1: extraction with citations
    s1 = builder.begin_node("node-1", "Extraction")
    citations = [
        Citation(span_id="node-1", source_type="extraction", content="test", confidence=0.9),
    ]
    builder.end_node(s1, citations=citations)

    # Node 2: LLM with cost
    s2 = builder.begin_node("node-2", "Classification")
    usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    builder.record_llm_cost(s2, "gpt-4o", usage)
    builder.end_node(s2)

    # Node 3: no citations, no cost
    s3 = builder.begin_node("node-3", "Transform")
    builder.end_node(s3)

    envelope = builder.finish()
    assert len(envelope.spans) == 4  # root + 3 children
    assert len(envelope.citations) == 1
    assert envelope.cost_tree is not None
    assert len(envelope.cost_tree) > 0


def test_empty_workflow():
    """start -> finish with no nodes produces envelope with root span only."""
    builder = EnvelopeBuilder(execution_id="run-3", org_id="org-3")
    builder.start()
    envelope = builder.finish()
    assert len(envelope.spans) == 1
    assert envelope.citations is None
    assert envelope.cost_tree is None  # no costs recorded -> empty list -> None via `or None`


def test_node_with_zero_tokens():
    """Cost node with zero tokens has zero cost."""
    builder = EnvelopeBuilder(execution_id="run-4", org_id="org-4")
    builder.start()
    s1 = builder.begin_node("node-1", "LLM")
    usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    cost_node = builder.record_llm_cost(s1, "gpt-4o", usage)
    builder.end_node(s1)
    builder.finish()
    assert cost_node.compute_cost == Decimal("0")


def test_node_error():
    """end_node_error produces failed child span and error in envelope."""
    builder = EnvelopeBuilder(execution_id="run-5", org_id="org-5")
    builder.start()
    s1 = builder.begin_node("node-1", "Extraction")
    builder.end_node_error(s1, ExecutionError(code="NODE_FAIL", message="boom", node_id="node-1"))
    envelope = builder.finish(status="failure")
    assert envelope.status == "failure"
    assert len(envelope.errors) == 1
    assert envelope.errors[0].code == "NODE_FAIL"
    # Root span should be ERROR since there are errors
    root_span = [s for s in envelope.spans if s.parent_span_id is None][0]
    assert root_span.status == "ERROR"


def test_double_finish_raises():
    """Calling finish() twice raises RuntimeError."""
    builder = EnvelopeBuilder(execution_id="run-6", org_id="org-6")
    builder.start()
    builder.finish()
    with pytest.raises(RuntimeError):
        builder.finish()


def test_begin_node_after_finish_raises():
    """begin_node after finish raises RuntimeError."""
    builder = EnvelopeBuilder(execution_id="run-7", org_id="org-7")
    builder.start()
    builder.finish()
    with pytest.raises(RuntimeError):
        builder.begin_node("node-1", "Test")


# --- Reconstruction ---


def test_reconstruct_from_node_records():
    """Reconstruct produces envelope from NodeRecord list."""
    records = [
        NodeRecord(
            node_id="node-1",
            node_type="Extraction",
            status="SUCCESS",
            started_at="2026-03-13T10:00:00Z",
            finished_at="2026-03-13T10:00:05Z",
            output=None,
            error=None,
            model_name="gpt-4o",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            cost=Decimal("0.01"),
        ),
        NodeRecord(
            node_id="node-2",
            node_type="Classification",
            status="ERROR",
            started_at="2026-03-13T10:00:05Z",
            finished_at="2026-03-13T10:00:08Z",
            output=None,
            error="timeout",
            model_name=None,
            provider=None,
            input_tokens=0,
            output_tokens=0,
            cost=Decimal("0"),
        ),
    ]
    envelope = EnvelopeBuilder.reconstruct(
        execution_id="run-8",
        org_id="org-8",
        node_records=records,
    )
    assert envelope.status == "partial"
    assert len(envelope.spans) == 3  # root + 2 children
    assert len(envelope.errors) == 1


# --- Envelope helpers ---


def test_sum_cost_tree():
    nodes = [
        CostNode(
            id="1", parent_id=None, entity="org:1",
            category="llm_tokens", compute_cost=Decimal("0.10"),
        ),
        CostNode(
            id="2",
            parent_id=None,
            entity="org:1",
            category="llm_tokens",
            compute_cost=Decimal("0.05"),
            value_added_fee=Decimal("0.01"),
        ),
    ]
    assert sum_cost_tree(nodes) == Decimal("0.16")


def test_compute_duration():
    spans = [
        Span(
            span_id="root",
            executor_type="workflow",
            executor_id="run-1",
            started_at="2026-03-13T10:00:00+00:00",
            ended_at="2026-03-13T10:00:05+00:00",
        ),
        Span(
            span_id="child",
            executor_type="node",
            executor_id="n-1",
            parent_span_id="root",
            started_at="2026-03-13T10:00:01+00:00",
            ended_at="2026-03-13T10:00:03+00:00",
        ),
    ]
    assert compute_duration(spans) == 5000  # 5 seconds in ms


def test_extract_primary_model():
    nodes = [
        CostNode(
            id="1",
            parent_id=None,
            entity="org:1",
            category="embedding",
            compute_cost=Decimal("0.01"),
            metadata={"model": "text-embedding-3-small"},
        ),
        CostNode(
            id="2",
            parent_id=None,
            entity="org:1",
            category="llm_tokens",
            compute_cost=Decimal("0.10"),
            metadata={"model": "gpt-4o"},
        ),
    ]
    assert extract_primary_model(nodes) == "gpt-4o"


def test_extract_primary_model_empty():
    assert extract_primary_model(None) is None
    assert extract_primary_model([]) is None
