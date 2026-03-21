"""Tests for async wrap() — AsyncOpenAI + AsyncAnthropic support."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aceteam_aep import AepSession, wrap
from aceteam_aep.safety.cost_anomaly import CostAnomalyDetector

_FAST_DETECTORS = [CostAnomalyDetector()]


def _make_async_openai_response(
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


def _make_async_openai_client(response: Any | None = None) -> MagicMock:
    """Fake AsyncOpenAI — .chat.completions.create is an AsyncMock."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=response or _make_async_openai_response()
    )
    del client.messages
    return client


def _make_async_anthropic_client(response: Any | None = None) -> MagicMock:
    """Fake AsyncAnthropic — .messages.create is an AsyncMock."""
    client = MagicMock(spec=["messages"])
    resp = response or MagicMock()
    resp.model = "claude-sonnet-4-5"
    resp.usage.input_tokens = 10
    resp.usage.output_tokens = 20
    block = MagicMock()
    block.text = "Hello!"
    resp.content = [block]
    client.messages.create = AsyncMock(return_value=resp)
    return client


@pytest.mark.asyncio
async def test_async_openai_cost_recorded() -> None:
    client = _make_async_openai_client(
        _make_async_openai_response(input_tokens=100, output_tokens=200)
    )
    wrap(client, detectors=_FAST_DETECTORS)
    await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert client.aep.cost_usd > Decimal("0")
    assert client.aep.call_count == 1


@pytest.mark.asyncio
async def test_async_openai_spans_recorded() -> None:
    client = _make_async_openai_client()
    wrap(client, detectors=_FAST_DETECTORS)
    await client.chat.completions.create(model="gpt-4o", messages=[])
    assert len(client.aep.get_spans()) == 1


@pytest.mark.asyncio
async def test_async_anthropic_cost_recorded() -> None:
    client = _make_async_anthropic_client()
    wrap(client, detectors=_FAST_DETECTORS)
    await client.messages.create(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=100,
    )
    assert client.aep.cost_usd > Decimal("0")
    assert client.aep.call_count == 1


@pytest.mark.asyncio
async def test_async_anthropic_spans_recorded() -> None:
    client = _make_async_anthropic_client()
    wrap(client, detectors=_FAST_DETECTORS)
    await client.messages.create(model="claude-sonnet-4-5", messages=[], max_tokens=100)
    assert len(client.aep.get_spans()) == 1


@pytest.mark.asyncio
async def test_async_pii_detection() -> None:
    from aceteam_aep.safety.pii import PiiDetector

    client = _make_async_openai_client(
        _make_async_openai_response(content="SSN: 123-45-6789")
    )
    wrap(client, detectors=[PiiDetector(model_name="nonexistent/force-regex")])
    await client.chat.completions.create(model="gpt-4o", messages=[])
    assert any(s.signal_type == "pii" for s in client.aep.safety_signals)
    assert client.aep.enforcement.action == "block"


@pytest.mark.asyncio
async def test_async_instrumentation_error_does_not_break() -> None:
    bad_resp = MagicMock()
    bad_resp.usage = None
    bad_resp.model = "gpt-4o"
    bad_resp.choices = []

    client = _make_async_openai_client(bad_resp)
    wrap(client, detectors=_FAST_DETECTORS)
    result = await client.chat.completions.create(model="gpt-4o", messages=[])
    assert result is bad_resp
