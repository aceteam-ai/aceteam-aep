"""create_client() - provider detection from model name."""

from __future__ import annotations

from typing import Any

from .client import ChatClient
from .models import PROVIDER_BASE_URLS, detect_provider
from .providers.anthropic import AnthropicClient
from .providers.google import GoogleClient
from .providers.openai import OpenAIClient


def create_client(
    model: str,
    api_key: str,
    *,
    provider: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs: Any,
) -> ChatClient:
    """Create a ChatClient for the given model.

    Auto-detects the provider from the model name, or uses the explicit
    provider parameter.

    Args:
        model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-5-20250514").
        api_key: API key for the provider.
        provider: Explicit provider override.
        base_url: Custom API base URL.
        temperature: Default temperature.
        max_tokens: Default max output tokens.

    Returns:
        A ChatClient instance for the detected/specified provider.
    """
    detected = provider or detect_provider(model)

    if detected == "anthropic":
        return AnthropicClient(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if detected == "google":
        return GoogleClient(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if detected == "xai":
        from .providers.xai import XAIClient

        return XAIClient(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if detected == "ollama":
        from .providers.ollama import OllamaClient

        actual_model = model.replace("ollama-", "") if model.startswith("ollama-") else model
        return OllamaClient(
            model=actual_model,
            base_url=base_url or "http://localhost:11434/v1",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # OpenAI or OpenAI-compatible
    url = base_url or PROVIDER_BASE_URLS.get(detected)
    return OpenAIClient(
        api_key=api_key,
        model=model,
        base_url=url,
        temperature=temperature,
        max_tokens=max_tokens,
    )


__all__ = ["create_client"]
