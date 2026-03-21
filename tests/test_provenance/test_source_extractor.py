"""Tests for provenance source extraction from messages."""

from aceteam_aep.provenance.source_extractor import SourceRef, extract_sources_from_messages


def test_extract_tool_call_results() -> None:
    messages = [
        {"role": "user", "content": "Search for revenue data"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_123"}]},
        {"role": "tool", "tool_call_id": "call_123", "content": "Revenue was $4.2M in Q3"},
    ]
    sources = extract_sources_from_messages(messages)
    assert any(s.source_type == "tool_call" for s in sources)
    assert any("4.2M" in s.content_preview for s in sources)


def test_extract_system_context() -> None:
    long_context = "Document content: " + "x" * 600
    messages = [
        {"role": "system", "content": long_context},
        {"role": "user", "content": "Summarize this"},
    ]
    sources = extract_sources_from_messages(messages)
    assert any(s.source_type == "system_context" for s in sources)


def test_short_system_not_treated_as_context() -> None:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi"},
    ]
    sources = extract_sources_from_messages(messages)
    assert not any(s.source_type == "system_context" for s in sources)


def test_extract_urls() -> None:
    messages = [
        {"role": "user", "content": "Check https://example.com/report.pdf please"},
    ]
    sources = extract_sources_from_messages(messages)
    assert any(s.source_type == "url" for s in sources)
    assert any("example.com" in s.source_id for s in sources)


def test_declared_sources_from_headers() -> None:
    sources = extract_sources_from_messages(
        [{"role": "user", "content": "Hi"}],
        declared_sources=["doc:contract-123", "db:customers"],
    )
    assert any(s.source_id == "doc:contract-123" for s in sources)
    assert any(s.source_id == "db:customers" for s in sources)


def test_deduplicates_sources() -> None:
    messages = [
        {"role": "tool", "tool_call_id": "call_1", "content": "Result A"},
        {"role": "tool", "tool_call_id": "call_1", "content": "Result A again"},
    ]
    sources = extract_sources_from_messages(messages)
    tool_sources = [s for s in sources if s.source_type == "tool_call"]
    assert len(tool_sources) == 1


def test_empty_messages() -> None:
    sources = extract_sources_from_messages([])
    assert sources == []
