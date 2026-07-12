"""OpenAI + OpenAI-compatible provider (SambaNova, TheAgentic, DeepSeek, xAI)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import openai

from ..models import get_model_info
from ..types import ChatMessage, ChatResponse, StreamChunk, ToolCallRequest, Usage
from .errors import StreamFailedError


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


def _matches_family(model: str, prefixes: tuple[str, ...]) -> bool:
    """Match a model against family prefixes, tolerating both dash- and
    dot-separated variants (``gpt-5-mini``, ``gpt-5.6-terra``) and the bare
    family name (``gpt-5``).

    OpenAI has started shipping dotted point-release names for the ``gpt-5``
    family (``gpt-5.1``, ``gpt-5.4-mini``, ``gpt-5.6-terra``, ...). A naive
    ``model.startswith(prefix + "-")`` check misses these because the
    separator after the family name is a dot, not a dash. This helper is the
    single place both fallback heuristics below consult so they can never
    drift apart again.
    """
    return any(model == p or model.startswith((p + "-", p + ".")) for p in prefixes)


# Family prefixes for OpenAI reasoning models that (a) require
# max_completion_tokens instead of max_tokens and (b) don't accept a
# temperature parameter. Used as a fallback for models not yet seeded into
# the registry (see MODEL_REGISTRY in models.py).
_REASONING_FAMILY_PREFIXES: tuple[str, ...] = ("o1", "o3", "gpt-5")


def _uses_max_completion_tokens(model: str) -> bool:
    """Return True if the model requires max_completion_tokens instead of max_tokens."""
    info = get_model_info(model)
    if info is not None:
        return info.uses_max_completion_tokens
    # Fallback prefix check for models not yet in the registry.
    return _matches_family(model, _REASONING_FAMILY_PREFIXES)


def _supports_temperature(model: str) -> bool:
    """Return True if the model accepts temperature parameter."""
    info = get_model_info(model)
    if info is not None:
        return info.supports_temperature
    # Fallback: o1, o3, gpt-5 (and dotted point releases) don't support temperature.
    return not _matches_family(model, _REASONING_FAMILY_PREFIXES)


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

    # Provider slug stamped on ``StreamFailedError`` when the upstream
    # silently closes a stream. Subclasses targeting a specific vendor
    # (xAI, Ollama) override this so downstream callers can distinguish
    # the actual provider from the wire protocol shared with OpenAI.
    _provider_slug: str = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        supports_temperature: bool = True,
        uses_max_completion_tokens: bool | None = None,
    ) -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        # Explicit caller-supplied override. When False, ``temperature`` is
        # never sent regardless of the registry guard below. Defaults to True
        # so the registry-driven ``_supports_temperature(model)`` check remains
        # the sole gate for existing callers (backward compatible).
        self._supports_temperature = supports_temperature
        # Explicit caller-supplied override for which token-limit param to
        # send. When None (the default), the registry/prefix-driven
        # ``_uses_max_completion_tokens(model)`` heuristic decides, so
        # existing callers are unaffected. Mirrors ``supports_temperature``
        # above: an inert lever for the aceteam side to flip per-model
        # without waiting on an aceteam-aep release.
        self._uses_max_completion_tokens = uses_max_completion_tokens

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
        if self._uses_max_completion_tokens is not None:
            uses_mct = self._uses_max_completion_tokens
        else:
            uses_mct = _uses_max_completion_tokens(self._model)
        token_param = "max_completion_tokens" if uses_mct else "max_tokens"
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": _format_messages(messages),
            token_param: max_tokens if max_tokens is not None else self._max_tokens,
        }

        if self._supports_temperature and _supports_temperature(self._model):
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
        if self._uses_max_completion_tokens is not None:
            uses_mct = self._uses_max_completion_tokens
        else:
            uses_mct = _uses_max_completion_tokens(self._model)
        token_param = "max_completion_tokens" if uses_mct else "max_tokens"
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": _format_messages(messages),
            token_param: max_tokens if max_tokens is not None else self._max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if self._supports_temperature and _supports_temperature(self._model):
            kwargs["temperature"] = temperature if temperature is not None else self._temperature

        if tools:
            kwargs["tools"] = tools

        stream = await self._client.chat.completions.create(**kwargs)

        # Accumulate partial tool calls across chunks
        import json

        partial_tool_calls: dict[int, dict[str, Any]] = {}

        # If the stream closes with this still false, the upstream
        # accepted the request and returned a 200 SSE stream that closed
        # without a single content delta, tool-call delta, or finish
        # reason. Observed on revoked BYOK keys / silent aggregator
        # rejections. Raise so callers don't render a blank reply.
        # Usage-only chunks (no choices) don't flip this — some
        # OpenAI-compatible aggregators emit a usage frame even when
        # the underlying choices stream was silently rejected.
        produced_anything = False

        async for chunk in stream:
            if not chunk.choices:
                if chunk.usage:
                    yield StreamChunk(usage=_extract_usage(chunk.usage))
                continue

            delta = chunk.choices[0].delta

            # Handle text
            text = delta.content or ""
            if text:
                produced_anything = True

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
            if finish:
                produced_anything = True
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

            if completed_tool_calls:
                produced_anything = True

            yield StreamChunk(
                delta_text=text,
                delta_tool_calls=completed_tool_calls,
                finish_reason=finish,
                model=chunk.model,
            )

        if not produced_anything:
            raise StreamFailedError(
                f"{self._provider_slug.capitalize()} stream closed with no "
                f"content for model {self._model!r}",
                provider=self._provider_slug,
            )


__all__ = ["OpenAIClient"]
