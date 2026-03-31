"""OpenAI + OpenAI-compatible provider (SambaNova, TheAgentic, DeepSeek, xAI)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import openai

from ..models import get_model_info
from ..types import ChatMessage, ChatResponse, StreamChunk, ToolCallRequest, Usage


def _format_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert ChatMessages to OpenAI API format."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        m: dict[str, Any] = {"role": msg.role}

        if isinstance(msg.content, str):
            m["content"] = msg.content
        else:
            parts: list[dict[str, Any]] = []
            for block in msg.content:
                if block.type == "text":
                    parts.append({"type": "text", "text": block.text or ""})
                elif block.type == "image_url":
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": block.image_url or "",
                                "detail": block.detail or "auto",
                            },
                        }
                    )
            m["content"] = parts

        if msg.tool_calls:
            m["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": _serialize_args(tc.arguments)},
                }
                for tc in msg.tool_calls
            ]

        if msg.tool_call_id:
            m["tool_call_id"] = msg.tool_call_id

        if msg.name:
            m["name"] = msg.name

        result.append(m)
    return result


def _serialize_args(args: dict[str, Any]) -> str:
    import json

    return json.dumps(args)


def _parse_tool_calls(
    tool_calls: list[Any] | None,
) -> list[ToolCallRequest] | None:
    if not tool_calls:
        return None
    import json

    result: list[ToolCallRequest] = []
    for tc in tool_calls:
        args_str = tc.function.arguments
        try:
            if isinstance(args_str, str) and args_str.strip():
                args = json.loads(args_str)
            else:
                args = args_str or {}
        except (json.JSONDecodeError, TypeError):
            args = {"raw": args_str}
        result.append(ToolCallRequest(id=tc.id, name=tc.function.name, arguments=args))
    return result


def _uses_max_completion_tokens(model: str) -> bool:
    """Return True if the model requires max_completion_tokens instead of max_tokens."""
    info = get_model_info(model)
    if info is not None:
        return info.uses_max_completion_tokens
    # Fallback prefix check for models not yet in the registry.
    prefixes = ("o1", "o3", "gpt-5")
    return any(model == p or model.startswith(p + "-") for p in prefixes)


def _supports_temperature(model: str) -> bool:
    """Return True if the model accepts temperature parameter."""
    info = get_model_info(model)
    if info is not None:
        return info.supports_temperature
    # Fallback: o1, o3, gpt-5 don't support temperature
    prefixes = ("o1", "o3", "gpt-5")
    return not any(model == p or model.startswith(p + "-") for p in prefixes)


def _extract_usage(usage: Any) -> Usage:
    if usage is None:
        return Usage()
    return Usage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
    )


class OpenAIClient:
    """OpenAI and OpenAI-compatible API client.

    Works with OpenAI, SambaNova, TheAgentic, DeepSeek, and any
    OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        token_param = (
            "max_completion_tokens" if _uses_max_completion_tokens(self._model) else "max_tokens"
        )
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": _format_messages(messages),
            token_param: max_tokens if max_tokens is not None else self._max_tokens,
        }

        if _supports_temperature(self._model):
            kwargs["temperature"] = temperature if temperature is not None else self._temperature

        if tools:
            kwargs["tools"] = tools

        if response_format:
            kwargs["response_format"] = response_format

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        tool_calls = _parse_tool_calls(msg.tool_calls)

        return ChatResponse(
            message=ChatMessage(
                role="assistant",
                content=msg.content or "",
                tool_calls=tool_calls,
            ),
            usage=_extract_usage(response.usage),
            model=response.model,
            finish_reason=choice.finish_reason,
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        token_param = (
            "max_completion_tokens" if _uses_max_completion_tokens(self._model) else "max_tokens"
        )
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": _format_messages(messages),
            token_param: max_tokens if max_tokens is not None else self._max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if _supports_temperature(self._model):
            kwargs["temperature"] = temperature if temperature is not None else self._temperature

        if tools:
            kwargs["tools"] = tools

        stream = await self._client.chat.completions.create(**kwargs)

        # Accumulate partial tool calls across chunks
        import json

        partial_tool_calls: dict[int, dict[str, Any]] = {}

        async for chunk in stream:
            if not chunk.choices:
                # Final chunk with usage
                if chunk.usage:
                    yield StreamChunk(usage=_extract_usage(chunk.usage))
                continue

            delta = chunk.choices[0].delta

            # Handle text
            text = delta.content or ""

            # Handle tool calls
            completed_tool_calls: list[ToolCallRequest] | None = None
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in partial_tool_calls:
                        partial_tool_calls[idx] = {
                            "id": tc_delta.id or "",
                            "name": getattr(tc_delta.function, "name", None) or "",
                            "arguments": "",
                        }
                    else:
                        if tc_delta.id:
                            partial_tool_calls[idx]["id"] = tc_delta.id
                        if getattr(tc_delta.function, "name", None):
                            partial_tool_calls[idx]["name"] = tc_delta.function.name

                    if getattr(tc_delta.function, "arguments", None):
                        partial_tool_calls[idx]["arguments"] += tc_delta.function.arguments

            # Check for finished tool calls
            finish = chunk.choices[0].finish_reason
            if finish == "tool_calls" and partial_tool_calls:
                completed_tool_calls = []
                for tc_data in partial_tool_calls.values():
                    try:
                        raw_args = tc_data["arguments"]
                        args = json.loads(raw_args) if raw_args.strip() else {}
                    except (json.JSONDecodeError, TypeError):
                        args = {"raw": tc_data["arguments"]}
                    completed_tool_calls.append(
                        ToolCallRequest(
                            id=tc_data["id"],
                            name=tc_data["name"],
                            arguments=args,
                        )
                    )
                partial_tool_calls.clear()

            yield StreamChunk(
                delta_text=text,
                delta_tool_calls=completed_tool_calls,
                finish_reason=finish,
                model=chunk.model,
            )


__all__ = ["OpenAIClient"]
