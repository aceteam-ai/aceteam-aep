"""Tests for the model registry and provider detection."""

from aceteam_aep.models import detect_provider


def test_detect_provider_registry_models():
    assert detect_provider("gpt-4o") == "openai"
    assert detect_provider("claude-sonnet-4-5-20250514") == "anthropic"
    assert detect_provider("gemini-2.5-flash") == "google"
    assert detect_provider("grok-3") == "xai"


def test_detect_provider_prefix_fallback():
    # Models not in registry but detectable by prefix
    assert detect_provider("claude-future-model") == "anthropic"
    assert detect_provider("gemini-future-model") == "google"
    assert detect_provider("grok-future-model") == "xai"


def test_detect_provider_ollama():
    assert detect_provider("ollama-llama3") == "ollama"
    assert detect_provider("ollama") == "ollama"


def test_detect_provider_openai_compatible_substrings():
    assert detect_provider("deepseek-chat") == "deepseek"
    assert detect_provider("sambanova-llama") == "sambanova"
    assert detect_provider("theagentic-model") == "theagentic"


def test_detect_provider_openai_default():
    assert detect_provider("gpt-5-future") == "openai"
    assert detect_provider("unknown-model-xyz") == "openai"
