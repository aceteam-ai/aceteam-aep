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
    supports_temperature: bool = True,
    uses_max_completion_tokens: bool | None = None,
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
        supports_temperature: When False, the ``temperature`` parameter is
            never sent to the provider. Callers should drive this from their
            model catalog (e.g. ``model_pricing.no_temperature``) so that new
            no-temperature models — claude-opus-4-8, claude-sonnet-5 and
            successors that 400 when ``temperature`` is present — are handled
            without an aceteam-aep release. Defaults to True (unchanged
            behavior). Currently honored by the Anthropic and OpenAI-family
            clients.
        uses_max_completion_tokens: When not None, overrides the OpenAI-family
            client's ``max_completion_tokens`` vs. ``max_tokens`` heuristic.
            Defaults to None, in which case the registry/prefix-driven
            ``_uses_max_completion_tokens(model)`` check in
            ``providers/openai.py`` decides (unchanged behavior). Only
            honored by the OpenAI-family client (OpenAI, xAI, Ollama, and
            other OpenAI-compatible endpoints go through ``OpenAIClient``).

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
            supports_temperature=supports_temperature,
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
        supports_temperature=supports_temperature,
        uses_max_completion_tokens=uses_max_completion_tokens,
    )


__all__ = ["create_client"]
