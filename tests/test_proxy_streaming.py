"""Tests for SSE streaming through the AEP proxy."""

from __future__ import annotations

from aceteam_aep.proxy.streaming import _accumulate_stream_chunks, _parse_sse_line


def test_parse_sse_data_line() -> None:
    result = _parse_sse_line('data: {"choices": [{"delta": {"content": "Hi"}}]}')
    assert result is not None
    assert result["choices"][0]["delta"]["content"] == "Hi"


def test_parse_sse_done() -> None:
    result = _parse_sse_line("data: [DONE]")
    assert result is None


def test_parse_sse_empty_line() -> None:
    assert _parse_sse_line("") is None
    assert _parse_sse_line("\n") is None


def test_parse_sse_non_data_line() -> None:
    assert _parse_sse_line("event: message") is None


def test_accumulate_chunks_text() -> None:
    chunks = [
        {"model": "gpt-4o", "choices": [{"delta": {"content": "Hello"}}]},
        {"model": "gpt-4o", "choices": [{"delta": {"content": " world"}}]},
        {
            "model": "gpt-4o",
            "choices": [{"delta": {}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    ]
    model, text, inp, out = _accumulate_stream_chunks(chunks)
    assert model == "gpt-4o"
    assert text == "Hello world"
    assert inp == 10
    assert out == 5


def test_accumulate_empty_chunks() -> None:
    model, text, inp, out = _accumulate_stream_chunks([])
    assert model == "unknown"
    assert text == ""
    assert inp == 0
    assert out == 0


def test_accumulate_no_usage() -> None:
    chunks = [
        {"model": "gpt-4o", "choices": [{"delta": {"content": "Hi"}}]},
    ]
    model, text, inp, out = _accumulate_stream_chunks(chunks)
    assert text == "Hi"
    assert inp == 0  # no usage chunk
    assert out == 0
