"""Tool protocol and @tool decorator for AEP agent tools."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_type_hints


@dataclass
class Tool:
    """A tool that can be called by an agent."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    func: Callable[..., Any]

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def invoke(self, arguments: dict[str, Any]) -> Any:
        """Invoke the tool with the given arguments."""
        result = self.func(**arguments)
        if inspect.isawaitable(result):
            result = await result
        return result


def _python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation to JSON Schema."""
    if annotation is str or annotation == "str":
        return {"type": "string"}
    elif annotation is int or annotation == "int":
        return {"type": "integer"}
    elif annotation is float or annotation == "float":
        return {"type": "number"}
    elif annotation is bool or annotation == "bool":
        return {"type": "boolean"}
    elif annotation is list or (
        hasattr(annotation, "__origin__") and annotation.__origin__ is list
    ):
        return {"type": "array"}
    elif annotation is dict or (
        hasattr(annotation, "__origin__") and annotation.__origin__ is dict
    ):
        return {"type": "object"}
    else:
        return {"type": "string"}


def _build_parameters_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Build JSON Schema parameters from function signature."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        annotation = hints.get(name, str)
        prop = _python_type_to_json_schema(annotation)

        # Use docstring or parameter name as description
        prop["description"] = name

        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Decorator to create a Tool from a function.

    Usage:
        @tool
        def calculator(expression: str) -> str:
            '''Evaluate a math expression.'''
            return str(eval(expression))

        @tool(name="search", description="Search the web")
        def google_search(query: str) -> str:
            ...
    """

    def decorator(f: Callable[..., Any]) -> Tool:
        tool_name = name or f.__name__
        tool_desc = description or inspect.getdoc(f) or f.__name__
        parameters = _build_parameters_schema(f)
        return Tool(
            name=tool_name,
            description=tool_desc,
            parameters=parameters,
            func=f,
        )

    if func is not None:
        return decorator(func)
    return decorator


__all__ = ["Tool", "tool"]
