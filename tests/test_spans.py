"""Tests for span tracking."""

from aceteam_aep.spans import SpanTracker


def test_span_lifecycle():
    tracker = SpanTracker()
    span = tracker.start_span("agent_loop", "gpt-4o")
    assert span.status == "OK"
    assert span.started_at
    assert span.ended_at is None

    ended = tracker.end_span(span.span_id)
    assert ended.ended_at is not None
    assert ended.status == "OK"
    assert ended.duration_ms is not None
    assert ended.duration_ms >= 0


def test_child_spans():
    tracker = SpanTracker()
    root = tracker.start_span("agent_loop", "gpt-4o")
    child = tracker.start_span("llm_call", "gpt-4o", parent_span_id=root.span_id)

    assert child.parent_span_id == root.span_id
    assert len(tracker.get_spans()) == 2


def test_span_with_metadata():
    tracker = SpanTracker()
    span = tracker.start_span("llm_call", "gpt-4o", metadata={"tokens": 100})
    assert span.metadata["tokens"] == 100

    tracker.end_span(span.span_id, metadata={"completion_tokens": 50})
    assert span.metadata["completion_tokens"] == 50


def test_span_error_status():
    tracker = SpanTracker()
    span = tracker.start_span("tool_call", "search")
    tracker.end_span(span.span_id, status="ERROR")
    assert span.status == "ERROR"


def test_get_span():
    tracker = SpanTracker()
    span = tracker.start_span("agent_loop", "gpt-4o")
    found = tracker.get_span(span.span_id)
    assert found is span
    assert tracker.get_span("nonexistent") is None


def test_trace_id():
    tracker = SpanTracker(trace_id="my-trace")
    assert tracker.trace_id == "my-trace"

    auto_tracker = SpanTracker()
    assert len(auto_tracker.trace_id) > 0
