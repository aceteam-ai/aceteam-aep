"""xAI/Grok provider - uses OpenAI-compatible API."""

from __future__ import annotations

from .openai import OpenAIClient

XAI_BASE_URL = "https://api.x.ai/v1"


class XAIClient(OpenAIClient):
    """xAI/Grok client using OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str = "grok-3",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=XAI_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
        )


__all__ = ["XAIClient"]
