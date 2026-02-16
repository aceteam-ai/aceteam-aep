"""Google GenAI (Gemini) provider."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import types as genai_types

from ..types import ChatMessage, ChatResponse, StreamChunk, ToolCallRequest, Usage


def _format_contents(
    messages: list[ChatMessage],
) -> tuple[str | None, list[genai_types.Content]]:
    """Convert ChatMessages to Google GenAI format.

    Returns (system_instruction, contents).
    """
    system_parts: list[str] = []
    contents: list[genai_types.Content] = []

    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.text)
            continue

        role = "model" if msg.role == "assistant" else "user"
        if msg.role == "tool":
            role = "user"

        parts: list[genai_types.Part] = []

        if isinstance(msg.content, str):
            if msg.content:
                parts.append(genai_types.Part(text=msg.content))
        else:
            for block in msg.content:
                if block.type == "text" and block.text:
                    parts.append(genai_types.Part(text=block.text))

        if msg.tool_calls:
            for tc in msg.tool_calls:
                parts.append(
                    genai_types.Part(
                        function_call=genai_types.FunctionCall(name=tc.name, args=tc.arguments)
                    )
                )

        if msg.role == "tool" and msg.tool_call_id:
            try:
                result = json.loads(msg.text)
            except (json.JSONDecodeError, TypeError):
                result = {"result": msg.text}
            parts = [
                genai_types.Part(
                    function_response=genai_types.FunctionResponse(
                        name=msg.name or "tool", response=result
                    )
                )
            ]

        if parts:
            contents.append(genai_types.Content(role=role, parts=parts))

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


def _tools_to_google(tools: list[dict[str, Any]]) -> list[genai_types.Tool]:
    """Convert OpenAI-format tools to Google GenAI format."""
    declarations: list[genai_types.FunctionDeclaration] = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool["function"]
            declarations.append(
                genai_types.FunctionDeclaration(
                    name=func["name"],
                    description=func.get("description", ""),
                    parameters=func.get("parameters"),
                )
            )
    return [genai_types.Tool(function_declarations=declarations)] if declarations else []


class GoogleClient:
    """Google GenAI (Gemini) client."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
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
        system_instruction, contents = _format_contents(messages)

        config = genai_types.GenerateContentConfig(
            temperature=temperature if temperature is not None else self._temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self._max_tokens,
        )

        if system_instruction:
            config.system_instruction = system_instruction

        if tools:
            config.tools = _tools_to_google(tools)

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        # Extract text and tool calls
        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []

        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts or []:
                if part.text:
                    text_parts.append(part.text)
                if part.function_call:
                    tool_calls.append(
                        ToolCallRequest(
                            id=f"call_{part.function_call.name}",
                            name=part.function_call.name,
                            arguments=(
                                dict(part.function_call.args) if part.function_call.args else {}
                            ),
                        )
                    )

        # Extract usage
        usage = Usage()
        if response.usage_metadata:
            usage = Usage(
                prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                completion_tokens=response.usage_metadata.candidates_token_count or 0,
                total_tokens=response.usage_metadata.total_token_count or 0,
            )

        return ChatResponse(
            message=ChatMessage(
                role="assistant",
                content="\n".join(text_parts),
                tool_calls=tool_calls if tool_calls else None,
            ),
            usage=usage,
            model=self._model,
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        system_instruction, contents = _format_contents(messages)

        config = genai_types.GenerateContentConfig(
            temperature=temperature if temperature is not None else self._temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self._max_tokens,
        )

        if system_instruction:
            config.system_instruction = system_instruction

        if tools:
            config.tools = _tools_to_google(tools)

        async for chunk in await self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=config,
        ):
            text = ""
            tool_call_list: list[ToolCallRequest] = []

            if chunk.candidates and chunk.candidates[0].content:
                for part in chunk.candidates[0].content.parts or []:
                    if part.text:
                        text += part.text
                    if part.function_call:
                        tool_call_list.append(
                            ToolCallRequest(
                                id=f"call_{part.function_call.name}",
                                name=part.function_call.name,
                                arguments=(
                                    dict(part.function_call.args) if part.function_call.args else {}
                                ),
                            )
                        )

            usage_chunk: Usage | None = None
            if chunk.usage_metadata:
                usage_chunk = Usage(
                    prompt_tokens=chunk.usage_metadata.prompt_token_count or 0,
                    completion_tokens=chunk.usage_metadata.candidates_token_count or 0,
                    total_tokens=chunk.usage_metadata.total_token_count or 0,
                )

            yield StreamChunk(
                delta_text=text,
                delta_tool_calls=tool_call_list if tool_call_list else None,
                usage=usage_chunk,
            )


__all__ = ["GoogleClient"]
