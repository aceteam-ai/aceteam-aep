"""Anthropic Claude provider."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from ..types import ChatMessage, ChatResponse, StreamChunk, ToolCallRequest, Usage
from .errors import StreamFailedError

_CACHE_CONTROL: dict[str, str] = {"type": "ephemeral"}

# Environment values that turn a boolean AEP flag off. Mirrors the truthy
# convention used elsewhere in the package (config.py), inverted: prompt
# caching defaults ON, so only an explicit off-value disables it.
_ENV_FALSE = frozenset({"0", "false", "no", "off"})


def _prompt_caching_default() -> bool:
    """Read the ``AEP_PROMPT_CACHING`` env toggle (default ON).

    Caching is behavior-neutral (identical outputs, lower cost), so it ships
    on by default. Set ``AEP_PROMPT_CACHING`` to ``0``/``false``/``no``/``off``
    to disable it if anything looks off. That path emits zero
    ``cache_control`` markers, producing a byte-identical request to the
    pre-caching behavior.
    """
    return os.environ.get("AEP_PROMPT_CACHING", "").strip().lower() not in _ENV_FALSE


def _cache_usage(usage: Any) -> tuple[int, int]:
    """Pull (cache_read, cache_creation) input tokens off an Anthropic usage.

    The SDK reports these as ``None`` when caching is inactive, so coerce to 0.
    """
    read = getattr(usage, "cache_read_input_tokens", 0) or 0
    creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return read, creation


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


def _tools_to_anthropic(
    tools: list[dict[str, Any]], *, cache: bool = False
) -> list[dict[str, Any]]:
    """Convert OpenAI-format tools to Anthropic format.

    When ``cache`` is True, a ``cache_control`` breakpoint is placed on the
    final tool definition. Tool schemas render first in the Anthropic prompt,
    so this caches the tool-definitions prefix (and covers the no-system-prompt
    case, where the tools are the last static block). Freshly built dicts are
    returned, so the caller's tool list is never mutated.
    """
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
    if cache and result:
        result[-1] = {**result[-1], "cache_control": dict(_CACHE_CONTROL)}
    return result


def _system_param(system_prompt: str, *, cache: bool) -> str | list[dict[str, Any]]:
    """Build the Anthropic ``system`` parameter.

    Without caching, the plain string is returned (unchanged behavior). With
    caching, the fully-assembled system text is wrapped in a single text block
    carrying a ``cache_control`` breakpoint. System renders after tools, so
    this block is the last static content and caches the tools+system prefix
    together. The text is byte-identical either way, keeping outputs neutral.
    """
    if not cache:
        return system_prompt
    return [{"type": "text", "text": system_prompt, "cache_control": dict(_CACHE_CONTROL)}]


class AnthropicClient:
    """Anthropic Claude client."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        supports_temperature: bool = True,
        prompt_caching: bool | None = None,
    ) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        # When False, the ``temperature`` key is never sent to the API.
        # Newer Anthropic models (e.g. claude-opus-4-8, claude-sonnet-5)
        # reject the request outright if ``temperature`` is present. The
        # caller drives this from the model catalog rather than a hardcoded
        # registry so new no-temperature models don't require an AEP release.
        self._supports_temperature = supports_temperature
        # When True, ``cache_control`` breakpoints are attached to the static
        # prefix (tool definitions + system prompt) so Anthropic can serve the
        # repeated ~26k-token prefix from cache at ~0.1x input price instead of
        # re-billing it every turn. Behavior-neutral: identical outputs, lower
        # cost. Defaults to the ``AEP_PROMPT_CACHING`` env toggle (ON) so it can
        # be disabled without an AEP release if anything looks off.
        self._prompt_caching = (
            _prompt_caching_default() if prompt_caching is None else prompt_caching
        )

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
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }

        if self._supports_temperature:
            kwargs["temperature"] = temperature if temperature is not None else self._temperature

        if tools:
            kwargs["tools"] = _tools_to_anthropic(tools, cache=self._prompt_caching)

        # Assemble the complete system text (base prompt + any schema
        # instruction) *before* attaching the cache_control breakpoint, so the
        # breakpoint sits on the fully-assembled static block.
        system_text = system_prompt
        if response_format:
            schema = _extract_json_schema(response_format)
            if schema:
                schema_prompt = (
                    "You must respond with a valid JSON object conforming to this schema:\n"
                    f"```json\n{json.dumps(schema, indent=2)}\n```\n"
                    "Output ONLY the JSON object, no other text."
                )
                system_text = f"{system_text}\n\n{schema_prompt}" if system_text else schema_prompt

        if system_text:
            kwargs["system"] = _system_param(system_text, cache=self._prompt_caching)

        response = await self._client.messages.create(**kwargs)

        cache_read, cache_creation = _cache_usage(response.usage)

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
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
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
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }

        if self._supports_temperature:
            kwargs["temperature"] = temperature if temperature is not None else self._temperature

        if system_prompt:
            kwargs["system"] = _system_param(system_prompt, cache=self._prompt_caching)

        if tools:
            kwargs["tools"] = _tools_to_anthropic(tools, cache=self._prompt_caching)

        async with self._client.messages.stream(**kwargs) as stream:
            current_tool: dict[str, Any] | None = None
            input_tokens = 0
            output_tokens = 0
            cache_read = 0
            cache_creation = 0
            # If the stream closes with this still false, Anthropic
            # accepted the request and returned an SSE stream that
            # closed without emitting a single text, tool-call, or
            # stop-reason event. Observed in production from revoked
            # BYOK keys where the upstream rejection arrives as a soft
            # close. Raise so callers see a real error rather than a
            # blank assistant reply.
            produced_anything = False

            async for event in stream:
                if event.type == "message_start":
                    if hasattr(event.message, "usage"):
                        input_tokens = event.message.usage.input_tokens
                        # Cache token counts are only reported on message_start
                        # (message_delta carries output tokens only).
                        cache_read, cache_creation = _cache_usage(event.message.usage)

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
                        yield StreamChunk(delta_text=event.delta.text)
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
                                )
                            ]
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
                                    )
                                ]
                            )
                            current_tool = None
                        yield StreamChunk(
                            finish_reason=finish,
                            usage=Usage(
                                prompt_tokens=input_tokens,
                                completion_tokens=output_tokens,
                                total_tokens=input_tokens + output_tokens,
                                cache_read_input_tokens=cache_read,
                                cache_creation_input_tokens=cache_creation,
                            ),
                        )

        if not produced_anything:
            raise StreamFailedError(
                f"Anthropic stream closed with no content for model {self._model!r}",
                provider="anthropic",
            )


__all__ = ["AnthropicClient"]
