"""Tests for provenance tracker."""

from aceteam_aep.provenance.source_extractor import SourceRef
from aceteam_aep.provenance.tracker import ProvenanceTracker


def test_record_and_get_sources() -> None:
    tracker = ProvenanceTracker()
    src = SourceRef(source_type="tool_call", source_id="tool:call_1", content_preview="data")
    tracker.record_sources("call-a", [src])
    assert len(tracker.get_sources("call-a")) == 1
    assert tracker.get_sources("call-b") == []


def test_get_citations() -> None:
    tracker = ProvenanceTracker()
    src = SourceRef(
        source_type="tool_call",
        source_id="tool:call_1",
        content_preview="Revenue $4.2M",
        confidence=0.85,
    )
    tracker.record_sources("call-a", [src])
    citations = tracker.get_citations("call-a")
    assert len(citations) == 1
    assert citations[0].source_type == "tool_call"
    assert citations[0].confidence == 0.85


def test_get_all_citations() -> None:
    tracker = ProvenanceTracker()
    tracker.record_sources(
        "call-a",
        [
            SourceRef(source_type="tool_call", source_id="t1", content_preview="A"),
        ],
    )
    tracker.record_sources(
        "call-b",
        [
            SourceRef(source_type="url", source_id="u1", content_preview="B"),
        ],
    )
    all_citations = tracker.get_all_citations()
    assert len(all_citations) == 2


def test_source_count() -> None:
    tracker = ProvenanceTracker()
    assert tracker.source_count == 0
    tracker.record_sources(
        "c1",
        [
            SourceRef(source_type="a", source_id="1", content_preview=""),
            SourceRef(source_type="b", source_id="2", content_preview=""),
        ],
    )
    assert tracker.source_count == 2
