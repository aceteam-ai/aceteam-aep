"""Tests for tool protocol and decorator."""

import pytest

from aceteam_aep.tools import Tool, tool


def test_tool_decorator():
    @tool
    def calculator(expression: str) -> str:
        """Evaluate a math expression."""
        return str(eval(expression))

    assert isinstance(calculator, Tool)
    assert calculator.name == "calculator"
    assert calculator.description == "Evaluate a math expression."
    assert "expression" in calculator.parameters["properties"]
    assert calculator.parameters["properties"]["expression"]["type"] == "string"


def test_tool_decorator_with_args():
    @tool(name="calc", description="A calculator")
    def my_calculator(expression: str) -> str:
        return str(eval(expression))

    assert my_calculator.name == "calc"
    assert my_calculator.description == "A calculator"


def test_tool_openai_format():
    @tool
    def search(query: str, limit: int) -> str:
        """Search for something."""
        return f"Results for {query}"

    schema = search.to_openai_tool()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "search"
    assert "query" in schema["function"]["parameters"]["properties"]
    assert "limit" in schema["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_tool_invoke_sync():
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    result = await add.invoke({"a": 1, "b": 2})
    assert result == 3


@pytest.mark.asyncio
async def test_tool_invoke_async():
    @tool
    async def async_add(a: int, b: int) -> int:
        """Add two numbers asynchronously."""
        return a + b

    result = await async_add.invoke({"a": 3, "b": 4})
    assert result == 7


def test_tool_required_params():
    @tool
    def greet(name: str, greeting: str = "Hello") -> str:
        """Greet someone."""
        return f"{greeting}, {name}!"

    schema = greet.parameters
    assert "name" in schema.get("required", [])
    # greeting has default, so not required
    assert "greeting" not in schema.get("required", [])


def test_tool_manual_creation():
    t = Tool(
        name="test_tool",
        description="A test tool",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        func=lambda x: x.upper(),
    )
    assert t.name == "test_tool"
