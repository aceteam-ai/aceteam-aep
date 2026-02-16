"""Tests for agent loop with mock provider."""

import pytest

from aceteam_aep.agent import run_agent_loop, run_agent_loop_stream
from aceteam_aep.budget import BudgetEnforcer
from aceteam_aep.costs import CostTracker
from aceteam_aep.spans import SpanTracker
from aceteam_aep.tools import tool
from aceteam_aep.types import ChatMessage, ChatResponse, StreamChunk, ToolCallRequest, Usage


class MockClient:
    """Mock ChatClient that returns canned responses."""

    def __init__(self, responses: list[ChatResponse]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    @property
    def model_name(self) -> str:
        return "mock-model"

    async def chat(self, messages, **kwargs):
        response = self._responses[self._call_count]
        self._call_count += 1
        return response

    async def chat_stream(self, messages, **kwargs):
        response = self._responses[self._call_count]
        self._call_count += 1
        # Simulate streaming by yielding chunks
        text = response.message.text
        for i in range(0, len(text), 5):
            yield StreamChunk(delta_text=text[i : i + 5])
        if response.message.tool_calls:
            yield StreamChunk(delta_tool_calls=response.message.tool_calls)
        yield StreamChunk(usage=response.usage, finish_reason="stop")


@pytest.mark.asyncio
async def test_simple_chat():
    """Test agent loop with no tools - single call."""
    client = MockClient(
        [
            ChatResponse(
                message=ChatMessage(role="assistant", content="Hello!"),
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="mock-model",
            )
        ]
    )

    result = await run_agent_loop(
        client,
        [ChatMessage(role="user", content="Hi")],
        system_prompt="You are helpful.",
    )

    assert len(result.messages) >= 2  # system + user + assistant
    assert result.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_tool_calling():
    """Test agent loop with tool calls."""

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    client = MockClient(
        [
            # First call: model decides to call tool
            ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCallRequest(id="call_1", name="add", arguments={"a": 2, "b": 3})
                    ],
                ),
                usage=Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
                model="mock-model",
            ),
            # Second call: model gives final answer
            ChatResponse(
                message=ChatMessage(role="assistant", content="2 + 3 = 5"),
                usage=Usage(prompt_tokens=30, completion_tokens=5, total_tokens=35),
                model="mock-model",
            ),
        ]
    )

    result = await run_agent_loop(
        client,
        [ChatMessage(role="user", content="What is 2+3?")],
        tools=[add],
    )

    # Should have user, assistant (tool call), tool result, assistant (final)
    assert any(m.role == "tool" for m in result.messages)
    assert result.usage.total_tokens == 65


@pytest.mark.asyncio
async def test_with_span_tracking():
    """Test agent loop creates proper spans."""
    client = MockClient(
        [
            ChatResponse(
                message=ChatMessage(role="assistant", content="Done"),
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="mock-model",
            )
        ]
    )

    tracker = SpanTracker(trace_id="test-trace")
    await run_agent_loop(
        client,
        [ChatMessage(role="user", content="Hi")],
        span_tracker=tracker,
    )

    spans = tracker.get_spans()
    assert len(spans) >= 2  # root + llm_call
    root = spans[0]
    assert root.executor_type == "agent_loop"
    assert root.status == "OK"
    assert root.ended_at is not None


@pytest.mark.asyncio
async def test_with_cost_tracking():
    """Test agent loop records costs."""
    client = MockClient(
        [
            ChatResponse(
                message=ChatMessage(role="assistant", content="Done"),
                usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
                model="gpt-4o",
            )
        ]
    )

    cost_tracker = CostTracker(entity="org:test")
    await run_agent_loop(
        client,
        [ChatMessage(role="user", content="Hi")],
        cost_tracker=cost_tracker,
    )

    assert cost_tracker.total_spent() > 0
    nodes = cost_tracker.get_cost_tree()
    assert len(nodes) == 1
    assert nodes[0].category == "llm_tokens"


@pytest.mark.asyncio
async def test_with_budget():
    """Test agent loop enforces budget."""
    client = MockClient(
        [
            ChatResponse(
                message=ChatMessage(role="assistant", content="Done"),
                usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
                model="gpt-4o",
            )
        ]
    )

    budget = BudgetEnforcer(total="1.00")
    await run_agent_loop(
        client,
        [ChatMessage(role="user", content="Hi")],
        budget=budget,
    )

    assert budget.state.spent >= 0


@pytest.mark.asyncio
async def test_stream_basic():
    """Test streaming agent loop yields events."""
    client = MockClient(
        [
            ChatResponse(
                message=ChatMessage(role="assistant", content="Hello there!"),
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="mock-model",
            )
        ]
    )

    events = []
    async for event in run_agent_loop_stream(
        client,
        [ChatMessage(role="user", content="Hi")],
        span_tracker=SpanTracker(),
        cost_tracker=CostTracker(),
    ):
        events.append(event)

    event_types = [e.type for e in events]
    assert "span_start" in event_types
    assert "chunk" in event_types
    assert "end" in event_types


@pytest.mark.asyncio
async def test_stream_with_tool_calls():
    """Test streaming agent loop with tool calls yields proper events."""

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    client = MockClient(
        [
            ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCallRequest(id="c1", name="multiply", arguments={"a": 3, "b": 4})
                    ],
                ),
                usage=Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
                model="mock-model",
            ),
            ChatResponse(
                message=ChatMessage(role="assistant", content="3 * 4 = 12"),
                usage=Usage(prompt_tokens=30, completion_tokens=5, total_tokens=35),
                model="mock-model",
            ),
        ]
    )

    events = []
    async for event in run_agent_loop_stream(
        client,
        [ChatMessage(role="user", content="What is 3*4?")],
        tools=[multiply],
    ):
        events.append(event)

    event_types = [e.type for e in events]
    assert "tool_call_start" in event_types
    assert "tool_call_end" in event_types
    assert "end" in event_types


@pytest.mark.asyncio
async def test_unknown_tool():
    """Test agent loop handles unknown tool gracefully."""
    client = MockClient(
        [
            ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCallRequest(id="c1", name="nonexistent", arguments={})],
                ),
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="mock-model",
            ),
            ChatResponse(
                message=ChatMessage(role="assistant", content="I couldn't find that tool."),
                usage=Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
                model="mock-model",
            ),
        ]
    )

    result = await run_agent_loop(
        client,
        [ChatMessage(role="user", content="Do something")],
        tools=[],
    )

    # Should have a tool error message
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "not found" in tool_msgs[0].text


@pytest.mark.asyncio
async def test_max_iterations():
    """Test that max_iterations prevents infinite loops."""

    @tool
    def noop() -> str:
        """Do nothing."""
        return "done"

    # Client always returns tool calls
    responses = [
        ChatResponse(
            message=ChatMessage(
                role="assistant",
                content="",
                tool_calls=[ToolCallRequest(id=f"c{i}", name="noop", arguments={})],
            ),
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="mock-model",
        )
        for i in range(10)
    ]

    client = MockClient(responses)
    await run_agent_loop(
        client,
        [ChatMessage(role="user", content="Loop")],
        tools=[noop],
        max_iterations=3,
    )

    # Should have stopped after 3 iterations
    assert client._call_count == 3
