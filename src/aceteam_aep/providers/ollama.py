"""Ollama (local) provider - uses OpenAI-compatible API."""

from __future__ import annotations

from .openai import OpenAIClient

OLLAMA_BASE_URL = "http://localhost:11434/v1"


class OllamaClient(OpenAIClient):
    """Ollama client using OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(
            api_key="ollama",  # Ollama doesn't need a real key
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )


__all__ = ["OllamaClient"]
