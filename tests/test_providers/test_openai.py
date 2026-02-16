"""Tests for OpenAI provider - format conversion only (no real API calls)."""

from aceteam_aep.providers.openai import OpenAIClient, _format_messages
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
