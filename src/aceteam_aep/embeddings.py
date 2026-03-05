"""Embeddings protocol and implementations.

Provides OpenAI, Ollama, and Cohere embedding clients that all
satisfy the ``EmbeddingClient`` protocol.
"""

from __future__ import annotations

import os
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
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        base_url: str | None = None,
    ) -> None:
        # When api_key is None, openai SDK falls back to OPENAI_API_KEY env var
        self._client = openai.AsyncOpenAI(
            api_key=api_key or None, base_url=base_url
        )
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


class OllamaEmbeddings:
    """Ollama embeddings using the ``ollama`` Python SDK."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str | None = None,
    ) -> None:
        try:
            import ollama as _ollama  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'ollama' package is required for OllamaEmbeddings. "
                "Install with: pip install ollama"
            ) from exc

        self._client = _ollama.AsyncClient(host=base_url or "http://localhost:11434")
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        results: list[list[float]] = []
        for text in texts:
            resp = await self._client.embed(model=self._model, input=text)
            results.append(resp["embeddings"][0])
        return results

    async def embed_query(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0]


class CohereEmbeddings:
    """Cohere embeddings using ``httpx`` (no extra SDK required)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "embed-english-v3.0",
        input_type: str = "search_document",
    ) -> None:
        self._api_key = api_key or os.environ.get("COHERE_API_KEY", "")
        self._model = model
        self._input_type = input_type

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        import httpx  # noqa: PLC0415

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.cohere.com/v1/embed",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "texts": texts,
                    "model": self._model,
                    "input_type": self._input_type,
                    "embedding_types": ["float"],
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["embeddings"]["float"]

    async def embed_query(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0]


__all__ = [
    "CohereEmbeddings",
    "EmbeddingClient",
    "OllamaEmbeddings",
    "OpenAIEmbeddings",
]
