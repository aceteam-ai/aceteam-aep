"""Tests for the wrap() function — AEP interceptor for existing SDK clients."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aceteam_aep import AepSession, SafetySignal, wrap


# ---------------------------------------------------------------------------
# Helpers — fake OpenAI / Anthropic response objects
# ---------------------------------------------------------------------------


def _make_openai_response(
    model: str = "gpt-4o",
    input_tokens: int = 10,
    output_tokens: int = 20,
    content: str = "Hello!",
) -> MagicMock:
    resp = MagicMock()
    resp.model = model
    resp.usage.prompt_tokens = input_tokens
    resp.usage.completion_tokens = output_tokens
    choice = MagicMock()
    choice.message.content = content
    resp.choices = [choice]
    return resp


def _make_openai_client(response: Any | None = None) -> MagicMock:
    """Fake openai.OpenAI() — has .chat.completions.create."""
    client = MagicMock()
    client.chat.completions.create.return_value = response or _make_openai_response()
    # Make hasattr checks work
    del client.messages  # no .messages — so wrap() picks OpenAI path
    return client


def _make_anthropic_client(response: Any | None = None) -> MagicMock:
    """Fake anthropic.Anthropic() — has .messages but not .chat."""
    client = MagicMock(spec=["messages"])
    resp = response or MagicMock()
    resp.model = "claude-sonnet-4-5"
    resp.usage.input_tokens = 10
    resp.usage.output_tokens = 20
    block = MagicMock()
    block.text = "Hello!"
    resp.content = [block]
    client.messages.create.return_value = resp
    return client


# ---------------------------------------------------------------------------
# wrap() detection
# ---------------------------------------------------------------------------


def test_wrap_returns_same_object() -> None:
    client = _make_openai_client()
    wrapped = wrap(client, entity="test")
    assert wrapped is client


def test_wrap_attaches_aep_session() -> None:
    client = _make_openai_client()
    wrap(client, entity="org:test")
    assert hasattr(client, "aep")
    assert isinstance(client.aep, AepSession)
    assert client.aep.entity == "org:test"


def test_wrap_raises_on_unknown_client() -> None:
    with pytest.raises(TypeError, match="Cannot wrap client"):
        wrap(object())


# ---------------------------------------------------------------------------
# OpenAI interception
# ---------------------------------------------------------------------------


def test_openai_cost_recorded() -> None:
    client = _make_openai_client(_make_openai_response(input_tokens=100, output_tokens=200))
    wrap(client)
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert client.aep.cost_usd > Decimal("0")


def test_openai_spans_recorded() -> None:
    client = _make_openai_client()
    wrap(client)
    client.chat.completions.create(model="gpt-4o", messages=[])
    spans = client.aep.get_spans()
    assert len(spans) == 1
    assert spans[0].executor_type == "llm"


def test_openai_multiple_calls_accumulate() -> None:
    client = _make_openai_client(_make_openai_response(input_tokens=50, output_tokens=50))
    wrap(client)
    client.chat.completions.create(model="gpt-4o", messages=[])
    client.chat.completions.create(model="gpt-4o", messages=[])
    assert len(client.aep.get_spans()) == 2
    assert len(client.aep.get_cost_tree()) == 2


# ---------------------------------------------------------------------------
# Anthropic interception
# ---------------------------------------------------------------------------


def test_anthropic_cost_recorded() -> None:
    client = _make_anthropic_client()
    wrap(client)
    client.messages.create(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=100,
    )
    assert client.aep.cost_usd > Decimal("0")


def test_anthropic_spans_recorded() -> None:
    client = _make_anthropic_client()
    wrap(client)
    client.messages.create(model="claude-sonnet-4-5", messages=[], max_tokens=100)
    spans = client.aep.get_spans()
    assert len(spans) == 1


# ---------------------------------------------------------------------------
# T&S signal detection
# ---------------------------------------------------------------------------


def test_pii_detection_ssn() -> None:
    client = _make_openai_client(
        _make_openai_response(content="Your SSN is 123-45-6789.")
    )
    wrap(client)
    client.chat.completions.create(model="gpt-4o", messages=[])
    signals = client.aep.safety_signals
    assert any(s.signal_type == "pii_in_output" for s in signals)
    assert any(s.severity == "high" for s in signals)


def test_pii_detection_email() -> None:
    client = _make_openai_client(
        _make_openai_response(content="Contact admin@example.com for help.")
    )
    wrap(client)
    client.chat.completions.create(model="gpt-4o", messages=[])
    assert any(s.signal_type == "pii_in_output" for s in client.aep.safety_signals)


def test_no_false_positive_on_clean_output() -> None:
    client = _make_openai_client(_make_openai_response(content="The capital of France is Paris."))
    wrap(client)
    client.chat.completions.create(model="gpt-4o", messages=[])
    signals = client.aep.safety_signals
    assert not any(s.signal_type == "pii_in_output" for s in signals)


def test_cost_anomaly_detection() -> None:
    """After a baseline, a 10x more expensive call should raise a signal."""
    cheap = _make_openai_response(input_tokens=5, output_tokens=5)
    expensive = _make_openai_response(input_tokens=5000, output_tokens=5000)

    # Save the original mock before wrap() replaces .create
    client = _make_openai_client(cheap)
    original_mock = client.chat.completions.create
    wrap(client)

    # Establish baseline (3 cheap calls)
    for _ in range(3):
        client.chat.completions.create(model="gpt-4o", messages=[])

    # Switch the underlying mock to return the expensive response
    original_mock.return_value = expensive
    client.chat.completions.create(model="gpt-4o", messages=[])

    assert any(s.signal_type == "cost_anomaly" for s in client.aep.safety_signals)


def test_no_cost_anomaly_with_fewer_than_3_calls() -> None:
    """Cost anomaly detection requires at least 3 prior calls to establish baseline."""
    expensive = _make_openai_response(input_tokens=10000, output_tokens=10000)
    client = _make_openai_client(expensive)
    wrap(client)
    client.chat.completions.create(model="gpt-4o", messages=[])
    assert not any(s.signal_type == "cost_anomaly" for s in client.aep.safety_signals)


# ---------------------------------------------------------------------------
# Resilience — AEP instrumentation never breaks user code
# ---------------------------------------------------------------------------


def test_instrumentation_error_does_not_break_call() -> None:
    """If usage extraction fails, the original response is still returned."""
    bad_resp = MagicMock()
    bad_resp.usage = None  # will cause AttributeError in extraction
    bad_resp.model = "gpt-4o"
    bad_resp.choices = []

    client = _make_openai_client(bad_resp)
    wrap(client)
    result = client.chat.completions.create(model="gpt-4o", messages=[])
    assert result is bad_resp  # original returned even on AEP error
