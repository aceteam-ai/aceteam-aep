"""Anthropic Claude provider."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from typing import Any

import anthropic

from ..client import (
    GatewayResponseError,
    _context_headers,
    _gateway_metadata,
    _validate_gateway_url,
)
from ..types import (
    AepRequestContext,
    AepResponseMetadata,
    ChatMessage,
    ChatResponse,
    StreamChunk,
    ToolCallRequest,
    Usage,
)
from .errors import StreamFailedError


def _extract_json_schema(response_format: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the JSON schema dict from an OpenAI-style response_format."""
    fmt_type = response_format.get("type")
    if fmt_type == "json_schema":
        spec = response_format.get("json_schema", {})
        return spec.get("schema")
    return None


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
        supports_temperature: bool = True,
        trusted_gateway_url: str | None = None,
    ) -> None:
        if trusted_gateway_url is not None:
            trusted_gateway_url = _validate_gateway_url(trusted_gateway_url)
        self._client = anthropic.AsyncAnthropic(  # pyright: ignore[reportAttributeAccessIssue]
            api_key=api_key, base_url=trusted_gateway_url
        )
        self._trusted_gateway = trusted_gateway_url is not None
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        # When False, the ``temperature`` key is never sent to the API.
        # Newer Anthropic models (e.g. claude-opus-4-8, claude-sonnet-5)
        # reject the request outright if ``temperature`` is present. The
        # caller drives this from the model catalog rather than a hardcoded
        # registry so new no-temperature models don't require an AEP release.
        self._supports_temperature = supports_temperature

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
        return await self._chat(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            request_context=None,
        )

    async def chat_with_context(
        self,
        messages: list[ChatMessage],
        *,
        request_context: AepRequestContext,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        if not self._trusted_gateway:
            raise ValueError("Request context requires a trusted gateway")
        return await self._chat(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            request_context=request_context,
        )

    async def _chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
        request_context: AepRequestContext | None,
    ) -> ChatResponse:
        system_prompt, formatted = _format_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": formatted,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }

        if self._supports_temperature:
            kwargs["temperature"] = temperature if temperature is not None else self._temperature

        if system_prompt:
            kwargs["system"] = system_prompt

        if tools:
            kwargs["tools"] = _tools_to_anthropic(tools)

        if response_format:
            schema = _extract_json_schema(response_format)
            if schema:
                schema_prompt = (
                    "You must respond with a valid JSON object conforming to this schema:\n"
                    f"```json\n{json.dumps(schema, indent=2)}\n```\n"
                    "Output ONLY the JSON object, no other text."
                )
                if "system" in kwargs:
                    kwargs["system"] += f"\n\n{schema_prompt}"
                else:
                    kwargs["system"] = schema_prompt

        metadata = None
        if self._trusted_gateway:
            kwargs["extra_headers"] = _context_headers(request_context)
            raw = await self._client.messages.with_raw_response.create(**kwargs)
            metadata = _gateway_metadata(raw.headers)
            try:
                response = raw.parse()
            except Exception as exc:
                raise GatewayResponseError(exc, metadata) from exc
        else:
            response = await self._client.messages.create(**kwargs)

        try:
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
                            origin=metadata,
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
                response_metadata=metadata,
            )
        except Exception as exc:
            if metadata is not None:
                raise GatewayResponseError(exc, metadata) from exc
            raise

    async def _stream_events(
        self,
        kwargs: dict[str, Any],
        request_context: AepRequestContext | None,
    ) -> AsyncGenerator[tuple[AepResponseMetadata | None, Any | None], None]:
        if self._trusted_gateway:
            kwargs["extra_headers"] = _context_headers(request_context)
            kwargs["stream"] = True
            async with self._client.messages.with_streaming_response.create(**kwargs) as raw:
                metadata = _gateway_metadata(raw.headers)
                yield metadata, None
                try:
                    stream = await raw.parse()
                    async for event in stream:
                        yield metadata, event
                except Exception as exc:
                    raise GatewayResponseError(exc, metadata) from exc
        else:
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    yield None, event

    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        return self._chat_stream(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            request_context=None,
        )

    def chat_stream_with_context(
        self,
        messages: list[ChatMessage],
        *,
        request_context: AepRequestContext,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        if not self._trusted_gateway:
            raise ValueError("Request context requires a trusted gateway")
        return self._chat_stream(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            request_context=request_context,
        )

    async def _chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
        request_context: AepRequestContext | None,
    ) -> AsyncIterator[StreamChunk]:
        system_prompt, formatted = _format_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": formatted,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }

        if self._supports_temperature:
            kwargs["temperature"] = temperature if temperature is not None else self._temperature

        if system_prompt:
            kwargs["system"] = system_prompt

        if tools:
            kwargs["tools"] = _tools_to_anthropic(tools)

        metadata: AepResponseMetadata | None = None
        current_tool: dict[str, Any] | None = None
        input_tokens = 0
        output_tokens = 0
        # If the stream closes with this still false, Anthropic
        # accepted the request and returned an SSE stream that
        # closed without emitting a single text, tool-call, or
        # stop-reason event. Observed in production from revoked
        # BYOK keys where the upstream rejection arrives as a soft
        # close. Raise so callers see a real error rather than a
        # blank assistant reply.
        produced_anything = False

        source = self._stream_events(kwargs, request_context)
        async with aclosing(source):
            async for metadata, event in source:
                if event is None:
                    yield StreamChunk(response_metadata=metadata)
                    continue
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
                        produced_anything = True
                        yield StreamChunk(delta_text=event.delta.text, response_metadata=metadata)
                    elif hasattr(event.delta, "partial_json") and current_tool:
                        current_tool["arguments"] += event.delta.partial_json

                elif event.type == "content_block_stop":
                    if current_tool:
                        try:
                            raw_args = current_tool["arguments"]
                            args = json.loads(raw_args) if raw_args.strip() else {}
                        except (json.JSONDecodeError, TypeError):
                            args = {"raw": current_tool["arguments"]}
                        produced_anything = True
                        yield StreamChunk(
                            delta_tool_calls=[
                                ToolCallRequest(
                                    id=current_tool["id"],
                                    name=current_tool["name"],
                                    arguments=args,
                                    origin=metadata,
                                )
                            ],
                            response_metadata=metadata,
                        )
                        current_tool = None

                elif event.type == "message_delta":
                    if hasattr(event, "usage") and event.usage:
                        output_tokens = event.usage.output_tokens
                    finish = getattr(event.delta, "stop_reason", None)
                    if finish:
                        produced_anything = True
                        # Flush any in-progress tool call that was truncated
                        # (e.g. by max_tokens). Anthropic skips content_block_stop
                        # when the response is cut short, so the accumulated
                        # partial JSON would otherwise be silently dropped.
                        if current_tool:
                            try:
                                raw_args = current_tool["arguments"]
                                args = json.loads(raw_args) if raw_args.strip() else {}
                            except (json.JSONDecodeError, TypeError):
                                args = {"raw": current_tool["arguments"]}
                            yield StreamChunk(
                                delta_tool_calls=[
                                    ToolCallRequest(
                                        id=current_tool["id"],
                                        name=current_tool["name"],
                                        arguments=args,
                                        origin=metadata,
                                    )
                                ],
                                response_metadata=metadata,
                            )
                            current_tool = None
                        yield StreamChunk(
                            finish_reason=finish,
                            usage=Usage(
                                prompt_tokens=input_tokens,
                                completion_tokens=output_tokens,
                                total_tokens=input_tokens + output_tokens,
                            ),
                            response_metadata=metadata,
                        )

        if not produced_anything:
            error = StreamFailedError(
                f"Anthropic stream closed with no content for model {self._model!r}",
                provider="anthropic",
            )
            if metadata is not None:
                raise GatewayResponseError(error, metadata) from error
            raise error
        if self._trusted_gateway:
            yield StreamChunk(response_metadata=metadata, response_complete=True)


__all__ = ["AnthropicClient"]
