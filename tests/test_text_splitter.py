"""Tests for text splitter."""

from aceteam_aep.text_splitter import split_text


def test_short_text():
    result = split_text("Hello world", chunk_size=100)
    assert result == ["Hello world"]


def test_empty_text():
    result = split_text("", chunk_size=100)
    assert result == []


def test_whitespace_only():
    result = split_text("   \n\n  ", chunk_size=100)
    assert result == []


def test_split_by_paragraph():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    result = split_text(text, chunk_size=30, chunk_overlap=0)
    assert len(result) >= 2
    assert "Paragraph one." in result[0]


def test_split_with_overlap():
    text = "A" * 50 + "\n\n" + "B" * 50 + "\n\n" + "C" * 50
    result = split_text(text, chunk_size=60, chunk_overlap=10)
    assert len(result) >= 2


def test_large_text():
    text = " ".join(f"word{i}" for i in range(1000))
    result = split_text(text, chunk_size=200, chunk_overlap=50)
    assert len(result) > 1
    # All chunks should be <= chunk_size (approximately)
    for chunk in result:
        assert len(chunk) <= 250  # Allow some slack due to word boundaries


def test_custom_separators():
    text = "a;b;c;d;e;f;g;h"
    result = split_text(text, chunk_size=5, chunk_overlap=0, separators=[";", ""])
    assert len(result) >= 2
