"""Tests for XML prompt helpers."""

from aceteam_aep.prompt import wrap_context, wrap_examples, wrap_file, wrap_xml


def test_wrap_xml_basic():
    result = wrap_xml("instructions", "Do the thing")
    assert result == "<instructions>\nDo the thing\n</instructions>"


def test_wrap_xml_with_attrs():
    result = wrap_xml("file", "content here", name="test.py", language="python")
    assert 'name="test.py"' in result
    assert 'language="python"' in result
    assert "<file" in result
    assert "</file>" in result


def test_wrap_file():
    result = wrap_file("print('hello')", "main.py", "python")
    assert '<file name="main.py" language="python">' in result
    assert "print('hello')" in result
    assert "</file>" in result


def test_wrap_file_no_language():
    result = wrap_file("some content", "data.txt")
    assert '<file name="data.txt">' in result
    assert "language=" not in result


def test_wrap_examples():
    examples = [
        {"input": "What is 2+2?", "output": "4"},
        {"input": "What is 3+3?", "output": "6"},
    ]
    result = wrap_examples(examples)
    assert "<examples>" in result
    assert "</examples>" in result
    assert '<example index="1">' in result
    assert '<example index="2">' in result
    assert "<input>" in result
    assert "<output>" in result
    assert "What is 2+2?" in result
    assert "4" in result


def test_wrap_context():
    result = wrap_context("Some relevant info", source="docs")
    assert '<context source="docs">' in result
    assert "Some relevant info" in result


def test_wrap_context_no_source():
    result = wrap_context("Just info")
    assert "<context>" in result
    assert "source=" not in result
