"""Tests for OpenAI provider - format conversion only (no real API calls)."""

from aceteam_aep.models import MODEL_REGISTRY, get_model_info
from aceteam_aep.providers.openai import OpenAIClient, _format_messages, _uses_max_completion_tokens
from aceteam_aep.types import ChatMessage, ContentBlock, ToolCallRequest


def test_format_simple_messages():
    messages = [
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="Hello"),
    ]
    result = _format_messages(messages)
    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are helpful."
    assert result[1]["role"] == "user"


def test_format_multimodal_message():
    messages = [
        ChatMessage(
            role="user",
            content=[
                ContentBlock(type="text", text="What's in this image?"),
                ContentBlock(type="image_url", image_url="https://example.com/img.png"),
            ],
        )
    ]
    result = _format_messages(messages)
    assert len(result) == 1
    assert isinstance(result[0]["content"], list)
    assert result[0]["content"][0]["type"] == "text"
    assert result[0]["content"][1]["type"] == "image_url"


def test_format_tool_calls():
    messages = [
        ChatMessage(
            role="assistant",
            content="Let me search.",
            tool_calls=[ToolCallRequest(id="call_1", name="search", arguments={"query": "test"})],
        )
    ]
    result = _format_messages(messages)
    assert "tool_calls" in result[0]
    assert result[0]["tool_calls"][0]["function"]["name"] == "search"


def test_format_tool_result():
    messages = [
        ChatMessage(
            role="tool",
            content='{"result": "found"}',
            tool_call_id="call_1",
            name="search",
        )
    ]
    result = _format_messages(messages)
    assert result[0]["role"] == "tool"
    assert result[0]["tool_call_id"] == "call_1"


def test_openai_client_model_name():
    client = OpenAIClient(api_key="test", model="gpt-4o")
    assert client.model_name == "gpt-4o"


def test_uses_max_completion_tokens():
    # Models that require max_completion_tokens (registry-driven)
    for model in ("o1", "o1-mini", "o1-preview", "o3", "o3-mini", "gpt-5", "gpt-5-mini", "gpt-5-nano"):
        assert _uses_max_completion_tokens(model), f"{model} should use max_completion_tokens"

    # Models that use max_tokens
    for model in ("gpt-4o", "gpt-4o-mini", "gpt-4-turbo"):
        assert not _uses_max_completion_tokens(model), f"{model} should use max_tokens"


def test_registry_drives_max_completion_tokens():
    """_uses_max_completion_tokens must agree with the registry for all known models."""
    for model, info in MODEL_REGISTRY.items():
        if info.provider == "openai" and not info.is_embedding:
            assert _uses_max_completion_tokens(model) == info.uses_max_completion_tokens


def test_registry_model_info():
    info = get_model_info("gpt-4o")
    assert info is not None
    assert info.provider == "openai"
    assert info.supports_vision
    assert not info.uses_max_completion_tokens

    o1 = get_model_info("o1")
    assert o1 is not None
    assert o1.uses_max_completion_tokens
    assert not o1.supports_temperature

    assert get_model_info("not-a-real-model") is None
