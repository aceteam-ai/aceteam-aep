"""Tests for the Ollama provider — wire protocol is OpenAI-compatible,
so we only verify that ``ProviderResponseError`` carries the Ollama slug
when the inherited empty-stream guard fires.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from aceteam_aep.providers import ProviderResponseError
from aceteam_aep.providers.ollama import OllamaClient


class _AsyncChunkIterator:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> AsyncIterator[Any]:
        async def _gen() -> AsyncIterator[Any]:
            for chunk in self._chunks:
                yield chunk

        return _gen()


def _make_stream_client(chunks: list[Any]) -> OllamaClient:
    client = OllamaClient(model="llama-test")
    client._client = MagicMock()

    async def _fake_create(**_kwargs: Any) -> _AsyncChunkIterator:
        return _AsyncChunkIterator(chunks)

    client._client.chat.completions.create = _fake_create
    return client


def test_ollama_client_uses_ollama_provider_slug() -> None:
    assert OllamaClient._provider_slug == "ollama"


@pytest.mark.asyncio
async def test_chat_stream_raises_with_ollama_slug_on_empty_stream() -> None:
    """An empty stream from a local Ollama server (model not found,
    container not serving, etc.) raises with the ollama slug.
    """
    client = _make_stream_client(chunks=[])

    with pytest.raises(ProviderResponseError) as exc_info:
        async for _ in client.chat_stream(messages=[]):
            pass

    assert exc_info.value.provider == "ollama"
    assert "llama-test" in str(exc_info.value)
    assert exc_info.value.user_message
