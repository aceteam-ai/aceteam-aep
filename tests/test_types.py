"""Tests for core types."""

from aceteam_aep.types import ChatMessage, ContentBlock, ToolCallRequest, Usage


def test_chat_message_text_content():
    msg = ChatMessage(role="user", content="Hello")
    assert msg.text == "Hello"
    assert msg.role == "user"


def test_chat_message_multimodal_content():
    msg = ChatMessage(
        role="user",
        content=[
            ContentBlock(type="text", text="Describe this image"),
            ContentBlock(type="image_url", image_url="https://example.com/img.png"),
        ],
    )
    assert msg.text == "Describe this image"


def test_chat_message_with_tool_calls():
    msg = ChatMessage(
        role="assistant",
        content="Let me search for that.",
        tool_calls=[ToolCallRequest(id="call_1", name="search", arguments={"query": "test"})],
    )
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "search"


def test_tool_message():
    msg = ChatMessage(
        role="tool",
        content='{"result": "found it"}',
        tool_call_id="call_1",
        name="search",
    )
    assert msg.role == "tool"
    assert msg.tool_call_id == "call_1"


def test_usage_addition():
    u1 = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    u2 = Usage(prompt_tokens=200, completion_tokens=100, total_tokens=300)
    combined = u1 + u2
    assert combined.prompt_tokens == 300
    assert combined.completion_tokens == 150
    assert combined.total_tokens == 450


def test_usage_defaults():
    u = Usage()
    assert u.prompt_tokens == 0
    assert u.completion_tokens == 0
    assert u.total_tokens == 0
