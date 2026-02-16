"""Embeddings protocol and OpenAI implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import openai


@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query text."""
        ...


class OpenAIEmbeddings:
    """OpenAI embeddings implementation."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        base_url: str | None = None,
    ) -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        kwargs: dict = {"input": texts, "model": self._model}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions

        response = await self._client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query text."""
        result = await self.embed([text])
        return result[0]


__all__ = ["EmbeddingClient", "OpenAIEmbeddings"]
