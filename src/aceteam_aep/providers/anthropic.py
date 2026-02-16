"""Anthropic Claude provider."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from ..types import ChatMessage, ChatResponse, StreamChunk, ToolCallRequest, Usage


def _format_messages(
    messages: list[ChatMessage],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert ChatMessages to Anthropic API format.

    Returns (system_prompt, messages) since Anthropic uses a separate system param.
    """
    system_prompt: str | None = None
    result: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "system":
            # Anthropic uses a separate system parameter
            text = msg.text
            system_prompt = (system_prompt + "\n\n" + text) if system_prompt else text
            continue

        m: dict[str, Any] = {"role": msg.role}

        if msg.role == "tool":
            # Anthropic expects tool results as content blocks
            m["role"] = "user"
            m["content"] = [
                {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id or "",
                    "content": msg.text,
                }
            ]
        elif isinstance(msg.content, str):
            content_blocks: list[dict[str, Any]] = []
            if msg.content:
                content_blocks.append({"type": "text", "text": msg.content})
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
            m["content"] = content_blocks if content_blocks else msg.content
        else:
            parts: list[dict[str, Any]] = []
            for block in msg.content:
                if block.type == "text":
                    parts.append({"type": "text", "text": block.text or ""})
                elif block.type == "image_url" and block.image_url:
                    # Anthropic uses base64 image format
                    if block.image_url.startswith("data:"):
                        media_type, _, data = block.image_url.partition(";base64,")
                        media_type = media_type.replace("data:", "")
                        parts.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data,
                                },
                            }
                        )
                    else:
                        parts.append(
                            {
                                "type": "image",
                                "source": {"type": "url", "url": block.image_url},
                            }
                        )
            m["content"] = parts

        result.append(m)

    return system_prompt, result


def _tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-format tools to Anthropic format."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool["function"]
            result.append(
                {
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                }
            )
    return result


class AnthropicClient:
    """Anthropic Claude client."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
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
        system_prompt, formatted = _format_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": formatted,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        if tools:
            kwargs["tools"] = _tools_to_anthropic(tools)

        response = await self._client.messages.create(**kwargs)

        # Extract text and tool calls
        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCallRequest(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        usage = Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        return ChatResponse(
            message=ChatMessage(
                role="assistant",
                content="\n".join(text_parts),
                tool_calls=tool_calls if tool_calls else None,
            ),
            usage=usage,
            model=response.model,
            finish_reason=response.stop_reason,
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        system_prompt, formatted = _format_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": formatted,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        if tools:
            kwargs["tools"] = _tools_to_anthropic(tools)

        async with self._client.messages.stream(**kwargs) as stream:
            current_tool: dict[str, Any] | None = None
            input_tokens = 0
            output_tokens = 0

            async for event in stream:
                if event.type == "message_start":
                    if hasattr(event.message, "usage"):
                        input_tokens = event.message.usage.input_tokens

                elif event.type == "content_block_start":
                    if (
                        hasattr(event.content_block, "type")
                        and event.content_block.type == "tool_use"
                    ):
                        current_tool = {
                            "id": event.content_block.id,
                            "name": event.content_block.name,
                            "arguments": "",
                        }

                elif event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        yield StreamChunk(delta_text=event.delta.text)
                    elif hasattr(event.delta, "partial_json") and current_tool:
                        current_tool["arguments"] += event.delta.partial_json

                elif event.type == "content_block_stop":
                    if current_tool:
                        try:
                            args = json.loads(current_tool["arguments"])
                        except (json.JSONDecodeError, TypeError):
                            args = {"raw": current_tool["arguments"]}
                        yield StreamChunk(
                            delta_tool_calls=[
                                ToolCallRequest(
                                    id=current_tool["id"],
                                    name=current_tool["name"],
                                    arguments=args,
                                )
                            ]
                        )
                        current_tool = None

                elif event.type == "message_delta":
                    if hasattr(event, "usage") and event.usage:
                        output_tokens = event.usage.output_tokens
                    finish = getattr(event.delta, "stop_reason", None)
                    if finish:
                        yield StreamChunk(
                            finish_reason=finish,
                            usage=Usage(
                                prompt_tokens=input_tokens,
                                completion_tokens=output_tokens,
                                total_tokens=input_tokens + output_tokens,
                            ),
                        )


__all__ = ["AnthropicClient"]
