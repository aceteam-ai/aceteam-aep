"""create_client() - provider detection from model name."""

from __future__ import annotations

from typing import Any

from .client import ChatClient
from .providers.anthropic import AnthropicClient
from .providers.google import GoogleClient
from .providers.openai import OpenAIClient

# Provider detection rules: (prefix/contains, provider, base_url_override)
_OPENAI_COMPATIBLE: dict[str, str] = {
    "sambanova": "https://api.sambanova.ai/v1",
    "theagentic": "https://api.theagentic.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
}


def _detect_provider(model: str) -> str:
    """Detect provider from model name."""
    model_lower = model.lower()

    if model_lower.startswith("claude") or "anthropic" in model_lower:
        return "anthropic"
    if model_lower.startswith("gemini") or "google" in model_lower:
        return "google"
    if model_lower.startswith("grok") or "xai" in model_lower:
        return "xai"
    if model_lower.startswith("ollama"):
        return "ollama"

    for prefix, _ in _OPENAI_COMPATIBLE.items():
        if prefix in model_lower:
            return prefix

    # Default to OpenAI (covers gpt-*, o1, o3, etc.)
    return "openai"


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
    detected = provider or _detect_provider(model)

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
    url = base_url or _OPENAI_COMPATIBLE.get(detected)
    return OpenAIClient(
        api_key=api_key,
        model=model,
        base_url=url,
        temperature=temperature,
        max_tokens=max_tokens,
    )


__all__ = ["create_client"]
