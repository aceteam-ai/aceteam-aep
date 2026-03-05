"""Model registry - authoritative source for model metadata and capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal


@dataclass(frozen=True)
class ModelInfo:
    """Metadata and capability descriptor for a single model."""

    provider: str
    """Provider identifier: 'openai', 'anthropic', 'google', 'xai', 'ollama',
    'sambanova', 'theagentic', 'deepseek'."""

    # Pricing (USD per token; 0 if not applicable)
    input_cost_per_token: Decimal = Decimal("0")
    output_cost_per_token: Decimal = Decimal("0")

    # Context limits
    context_window: int | None = None
    """Maximum input tokens the model accepts."""
    max_output_tokens: int | None = None
    """Maximum tokens the model can generate."""

    # Parameter support
    supports_temperature: bool = True
    """Whether the model accepts a temperature parameter."""
    supports_tools: bool = True
    """Whether the model supports function/tool calling."""
    supports_vision: bool = False
    """Whether the model accepts image inputs."""
    supports_system_prompt: bool = True
    """Whether the model accepts a system prompt."""
    uses_max_completion_tokens: bool = False
    """Whether to use max_completion_tokens instead of max_tokens (OpenAI o1/o3/gpt-5)."""

    # Embedding-only models
    is_embedding: bool = False
    """True for embedding-only models that don't support chat."""

    # Freeform extra metadata
    tags: tuple[str, ...] = field(default_factory=tuple)
    """Arbitrary tags, e.g. ('reasoning', 'vision', 'fast')."""


def _o(
    *,
    input: str,
    output: str,
    context_window: int | None = None,
    max_output_tokens: int | None = None,
    supports_temperature: bool = True,
    supports_tools: bool = True,
    supports_vision: bool = False,
    supports_system_prompt: bool = True,
    uses_max_completion_tokens: bool = False,
    is_embedding: bool = False,
    tags: tuple[str, ...] = (),
) -> ModelInfo:
    """Shorthand constructor for OpenAI models."""
    return ModelInfo(
        provider="openai",
        input_cost_per_token=Decimal(input),
        output_cost_per_token=Decimal(output),
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        supports_temperature=supports_temperature,
        supports_tools=supports_tools,
        supports_vision=supports_vision,
        supports_system_prompt=supports_system_prompt,
        uses_max_completion_tokens=uses_max_completion_tokens,
        is_embedding=is_embedding,
        tags=tags,
    )


def _a(
    *,
    input: str,
    output: str,
    context_window: int | None = None,
    max_output_tokens: int | None = None,
    supports_vision: bool = False,
    tags: tuple[str, ...] = (),
) -> ModelInfo:
    """Shorthand constructor for Anthropic models."""
    return ModelInfo(
        provider="anthropic",
        input_cost_per_token=Decimal(input),
        output_cost_per_token=Decimal(output),
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        supports_vision=supports_vision,
        tags=tags,
    )


def _g(
    *,
    input: str,
    output: str,
    context_window: int | None = None,
    max_output_tokens: int | None = None,
    supports_vision: bool = False,
    tags: tuple[str, ...] = (),
) -> ModelInfo:
    """Shorthand constructor for Google models."""
    return ModelInfo(
        provider="google",
        input_cost_per_token=Decimal(input),
        output_cost_per_token=Decimal(output),
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        supports_vision=supports_vision,
        tags=tags,
    )


def _x(
    *,
    input: str,
    output: str,
    context_window: int | None = None,
    max_output_tokens: int | None = None,
    supports_vision: bool = False,
    tags: tuple[str, ...] = (),
) -> ModelInfo:
    """Shorthand constructor for xAI models."""
    return ModelInfo(
        provider="xai",
        input_cost_per_token=Decimal(input),
        output_cost_per_token=Decimal(output),
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        supports_vision=supports_vision,
        tags=tags,
    )


MODEL_REGISTRY: dict[str, ModelInfo] = {
    # ── OpenAI ───────────────────────────────────────────────────────────────
    "gpt-4o": _o(
        input="0.0000025",
        output="0.000010",
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        tags=("vision",),
    ),
    "gpt-4o-mini": _o(
        input="0.00000015",
        output="0.0000006",
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        tags=("vision", "fast"),
    ),
    "gpt-4-turbo": _o(
        input="0.000010",
        output="0.000030",
        context_window=128_000,
        max_output_tokens=4_096,
        supports_vision=True,
        tags=("vision",),
    ),
    "gpt-4.5-preview": _o(
        input="0.000075",
        output="0.00015",
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        tags=("vision",),
    ),
    # o1 / o3 family — reasoning models with restricted parameters
    "o1": _o(
        input="0.000015",
        output="0.000060",
        context_window=200_000,
        max_output_tokens=100_000,
        supports_temperature=False,
        supports_vision=True,
        uses_max_completion_tokens=True,
        tags=("reasoning", "vision"),
    ),
    "o1-mini": _o(
        input="0.000003",
        output="0.000012",
        context_window=128_000,
        max_output_tokens=65_536,
        supports_temperature=False,
        uses_max_completion_tokens=True,
        tags=("reasoning", "fast"),
    ),
    "o1-preview": _o(
        input="0.000015",
        output="0.000060",
        context_window=128_000,
        max_output_tokens=32_768,
        supports_temperature=False,
        uses_max_completion_tokens=True,
        tags=("reasoning",),
    ),
    "o3": _o(
        input="0.000010",
        output="0.000040",
        context_window=200_000,
        max_output_tokens=100_000,
        supports_temperature=False,
        supports_vision=True,
        uses_max_completion_tokens=True,
        tags=("reasoning", "vision"),
    ),
    "o3-mini": _o(
        input="0.0000011",
        output="0.0000044",
        context_window=200_000,
        max_output_tokens=100_000,
        supports_temperature=False,
        uses_max_completion_tokens=True,
        tags=("reasoning", "fast"),
    ),
    # gpt-5 family
    "gpt-5": _o(
        input="0.000015",
        output="0.000060",
        context_window=1_000_000,
        max_output_tokens=32_768,
        supports_vision=True,
        uses_max_completion_tokens=True,
        tags=("vision",),
    ),
    "gpt-5-mini": _o(
        input="0.000003",
        output="0.000012",
        context_window=1_000_000,
        max_output_tokens=32_768,
        supports_vision=True,
        uses_max_completion_tokens=True,
        tags=("vision", "fast"),
    ),
    "gpt-5-nano": _o(
        input="0.00000015",
        output="0.0000006",
        context_window=1_000_000,
        max_output_tokens=32_768,
        supports_vision=True,
        uses_max_completion_tokens=True,
        tags=("vision", "fast"),
    ),
    # Embeddings
    "text-embedding-3-small": ModelInfo(
        provider="openai",
        input_cost_per_token=Decimal("0.00000002"),
        output_cost_per_token=Decimal("0"),
        supports_tools=False,
        supports_system_prompt=False,
        is_embedding=True,
        tags=("embedding",),
    ),
    "text-embedding-3-large": ModelInfo(
        provider="openai",
        input_cost_per_token=Decimal("0.00000013"),
        output_cost_per_token=Decimal("0"),
        supports_tools=False,
        supports_system_prompt=False,
        is_embedding=True,
        tags=("embedding",),
    ),
    # ── Anthropic ────────────────────────────────────────────────────────────
    "claude-opus-4-5-20250514": _a(
        input="0.000015",
        output="0.000075",
        context_window=200_000,
        max_output_tokens=32_000,
        supports_vision=True,
        tags=("vision",),
    ),
    "claude-sonnet-4-5-20250514": _a(
        input="0.000003",
        output="0.000015",
        context_window=200_000,
        max_output_tokens=8_096,
        supports_vision=True,
        tags=("vision", "fast"),
    ),
    "claude-haiku-4-5-20251001": _a(
        input="0.0000008",
        output="0.000004",
        context_window=200_000,
        max_output_tokens=8_096,
        supports_vision=True,
        tags=("vision", "fast"),
    ),
    # ── Google ───────────────────────────────────────────────────────────────
    "gemini-2.5-flash": _g(
        input="0.00000015",
        output="0.0000006",
        context_window=1_000_000,
        max_output_tokens=8_192,
        supports_vision=True,
        tags=("vision", "fast"),
    ),
    "gemini-2.5-pro": _g(
        input="0.00000125",
        output="0.000010",
        context_window=1_000_000,
        max_output_tokens=8_192,
        supports_vision=True,
        tags=("vision",),
    ),
    # ── xAI ──────────────────────────────────────────────────────────────────
    "grok-3": _x(
        input="0.000003",
        output="0.000015",
        context_window=131_072,
        max_output_tokens=8_192,
        tags=(),
    ),
    "grok-3-mini": _x(
        input="0.0000003",
        output="0.0000005",
        context_window=131_072,
        max_output_tokens=8_192,
        tags=("fast",),
    ),
    "grok-2-vision": _x(
        input="0.000002",
        output="0.000010",
        context_window=32_768,
        max_output_tokens=4_096,
        supports_vision=True,
        tags=("vision",),
    ),
}


# Provider base URLs for OpenAI-compatible third-party providers.
PROVIDER_BASE_URLS: dict[str, str] = {
    "sambanova": "https://api.sambanova.ai/v1",
    "theagentic": "https://api.theagentic.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
}

# Prefix rules for provider detection when a model isn't in the registry.
# Checked in order; first match wins.
_PROVIDER_PREFIXES: list[tuple[str, str]] = [
    ("claude", "anthropic"),
    ("gemini", "google"),
    ("grok", "xai"),
    ("ollama", "ollama"),
]

_PROVIDER_SUBSTRINGS: list[tuple[str, str]] = [
    ("anthropic", "anthropic"),
    ("google", "google"),
    ("xai", "xai"),
    *((p, p) for p in PROVIDER_BASE_URLS),
]


def get_model_info(model: str) -> ModelInfo | None:
    """Return the ModelInfo for a known model, or None if not in the registry."""
    return MODEL_REGISTRY.get(model)


def detect_provider(model: str) -> str:
    """Infer the provider for a model name.

    Checks the registry first, then falls back to prefix/substring matching.
    """
    info = MODEL_REGISTRY.get(model)
    if info is not None:
        return info.provider

    model_lower = model.lower()
    for prefix, provider in _PROVIDER_PREFIXES:
        if model_lower.startswith(prefix):
            return provider
    for substring, provider in _PROVIDER_SUBSTRINGS:
        if substring in model_lower:
            return provider

    return "openai"


__all__ = [
    "MODEL_REGISTRY",
    "PROVIDER_BASE_URLS",
    "ModelInfo",
    "detect_provider",
    "get_model_info",
]
