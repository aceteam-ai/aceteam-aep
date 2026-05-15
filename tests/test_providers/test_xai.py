"""Tests for the xAI/Grok provider — wire protocol is OpenAI-compatible,
so we only verify that ``StreamFailedError`` carries the xAI slug
when the inherited empty-stream guard fires.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from aceteam_aep.providers import StreamFailedError
from aceteam_aep.providers.xai import XAIClient


class _AsyncChunkIterator:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> AsyncIterator[Any]:
        async def _gen() -> AsyncIterator[Any]:
            for chunk in self._chunks:
                yield chunk

        return _gen()


def _make_stream_client(chunks: list[Any]) -> XAIClient:
    client = XAIClient(api_key="test", model="grok-test")
    client._client = MagicMock()

    async def _fake_create(**_kwargs: Any) -> _AsyncChunkIterator:
        return _AsyncChunkIterator(chunks)

    client._client.chat.completions.create = _fake_create
    return client


def test_xai_client_uses_xai_provider_slug() -> None:
    """The class-level slug is what gets stamped on the error — confirm
    the subclass override is wired up correctly without needing to drive
    a stream.
    """
    assert XAIClient._provider_slug == "xai"


@pytest.mark.asyncio
async def test_chat_stream_raises_with_xai_slug_on_empty_stream() -> None:
    client = _make_stream_client(chunks=[])

    with pytest.raises(StreamFailedError) as exc_info:
        async for _ in client.chat_stream(messages=[]):
            pass

    assert exc_info.value.provider == "xai"
    assert "grok-test" in str(exc_info.value)
