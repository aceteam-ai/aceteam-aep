"""Agent loop - replaces LangGraph StateGraph with a simple while loop.

run_agent_loop() for non-streaming, run_agent_loop_stream() for streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any

from .budget import BudgetEnforcer
from .client import ChatClient, ContextChatClient, _tool_origin
from .costs import CostTracker
from .spans import SpanTracker
from .stream import (
    StreamEvent,
    chunk_event,
    cost_event,
    end_event,
    error_event,
    response_metadata_event,
    span_end_event,
    span_start_event,
    tool_call_end_event,
    tool_call_start_event,
)
from .tools import Tool
from .types import (
    AepRequestContext,
    AepResponseMetadata,
    AgentResult,
    ChatMessage,
    ChatResponse,
    Usage,
)

_DEFAULT_TOOL_TIMEOUT = 300.0  # 5 minutes

logger = logging.getLogger(__name__)

# Default estimated cost per LLM call for budget reservation
_DEFAULT_RESERVATION = Decimal("0.01")


def _build_tool_schemas(tools: list[Tool]) -> list[dict[str, Any]]:
    """Convert Tool objects to OpenAI function calling format."""
    return [t.to_openai_tool() for t in tools]


async def _invoke_with_origin(
    tool: Tool,
    arguments: dict[str, Any],
    metadata: AepResponseMetadata | None,
) -> Any:
    token = _tool_origin.set(metadata)
    try:
        return await tool.invoke(arguments)
    finally:
        _tool_origin.reset(token)


async def run_agent_loop(
    client: ChatClient,
    messages: list[ChatMessage],
    *,
    tools: list[Tool] | None = None,
    system_prompt: str = "",
    budget: BudgetEnforcer | None = None,
    span_tracker: SpanTracker | None = None,
    cost_tracker: CostTracker | None = None,
    max_iterations: int = 25,
    temperature: float | None = None,
    max_tokens: int | None = None,
    request_context: AepRequestContext | None = None,
) -> AgentResult:
    """Run an agent loop: call model -> check tool_calls -> execute -> loop.

    Args:
        client: The LLM client to use.
        messages: Initial conversation messages.
        tools: Available tools for the agent.
        system_prompt: System prompt to prepend.
        budget: Optional budget enforcer.
        span_tracker: Optional span tracker for AEP compliance.
        cost_tracker: Optional cost tracker for AEP compliance.
        max_iterations: Maximum number of model calls.
        temperature: Override temperature.
        max_tokens: Override max tokens.

    Returns:
        AgentResult with final messages and aggregated usage.
    """
    tool_schemas = _build_tool_schemas(tools) if tools else None
    tools_by_name = {t.name: t for t in (tools or [])}

    # Build working messages list
    working: list[ChatMessage] = []
    if system_prompt:
        working.append(ChatMessage(role="system", content=system_prompt))
    working.extend(messages)

    total_usage = Usage()
    last_finish_reason: str | None = None
    response_metadata: list[AepResponseMetadata] = []
    _iteration = -1
    root_span = None

    if span_tracker:
        root_span = span_tracker.start_span("agent_loop", client.model_name)

    try:
        for _iteration in range(max_iterations):
            # Budget check
            reservation = None
            if budget:
                reservation = budget.reserve(_DEFAULT_RESERVATION)

            # Call LLM
            llm_span = None
            if span_tracker and root_span:
                llm_span = span_tracker.start_span(
                    "llm_call", client.model_name, parent_span_id=root_span.span_id
                )

            if request_context is not None:
                if not isinstance(client, ContextChatClient):
                    raise TypeError("Client does not support AEP request context")
                response: ChatResponse = await client.chat_with_context(
                    working,
                    request_context=request_context,
                    tools=tool_schemas,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                response = await client.chat(
                    working,
                    tools=tool_schemas,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            if response.response_metadata is not None:
                response_metadata.append(response.response_metadata)

            total_usage = total_usage + response.usage
            last_finish_reason = response.finish_reason

            # Record cost
            if cost_tracker:
                cost_tracker.record_llm_cost(
                    span_id=llm_span.span_id if llm_span else "unknown",
                    model=response.model,
                    usage=response.usage,
                )

            if llm_span and span_tracker:
                span_tracker.end_span(llm_span.span_id)

            # Settle budget reservation
            if budget and reservation:
                actual = cost_tracker.total_spent() if cost_tracker else Decimal("0")
                budget.settle(reservation, actual)

            # Add assistant message
            working.append(response.message)

            # Check for tool calls
            if not response.message.tool_calls:
                break

            # Execute tool calls
            for tc in response.message.tool_calls:
                tool = tools_by_name.get(tc.name)
                if not tool:
                    working.append(
                        ChatMessage(
                            role="tool",
                            content=json.dumps({"error": f"Tool '{tc.name}' not found"}),
                            tool_call_id=tc.id,
                            name=tc.name,
                        )
                    )
                    continue

                tool_span = None
                if span_tracker and root_span:
                    tool_span = span_tracker.start_span(
                        "tool_call", tc.name, parent_span_id=root_span.span_id
                    )

                try:
                    result = await _invoke_with_origin(
                        tool, tc.arguments, response.response_metadata
                    )
                    result_str = json.dumps(result) if not isinstance(result, str) else result
                    working.append(
                        ChatMessage(
                            role="tool",
                            content=result_str,
                            tool_call_id=tc.id,
                            name=tc.name,
                        )
                    )
                    if tool_span and span_tracker:
                        span_tracker.end_span(tool_span.span_id)
                except Exception as e:
                    working.append(
                        ChatMessage(
                            role="tool",
                            content=json.dumps({"error": str(e)}),
                            tool_call_id=tc.id,
                            name=tc.name,
                        )
                    )
                    if tool_span and span_tracker:
                        span_tracker.end_span(tool_span.span_id, status="ERROR")

        if root_span and span_tracker:
            span_tracker.end_span(root_span.span_id)

        return AgentResult(
            messages=working,
            usage=total_usage,
            iterations=min(_iteration + 1, max_iterations),
            finish_reason=last_finish_reason,
            response_metadata=response_metadata,
        )

    except Exception:
        if root_span and span_tracker:
            span_tracker.end_span(root_span.span_id, status="ERROR")
        raise


async def run_agent_loop_stream(
    client: ChatClient,
    messages: list[ChatMessage],
    *,
    tools: list[Tool] | None = None,
    system_prompt: str = "",
    budget: BudgetEnforcer | None = None,
    span_tracker: SpanTracker | None = None,
    cost_tracker: CostTracker | None = None,
    max_iterations: int = 25,
    temperature: float | None = None,
    max_tokens: int | None = None,
    request_context: AepRequestContext | None = None,
) -> AsyncIterator[StreamEvent]:
    """Stream agent loop, yielding AEP stream events.

    Yields: span_start, chunk, tool_call_start, tool_call_end, cost, span_end, end
    """
    tool_schemas = _build_tool_schemas(tools) if tools else None
    tools_by_name = {t.name: t for t in (tools or [])}

    working: list[ChatMessage] = []
    if system_prompt:
        working.append(ChatMessage(role="system", content=system_prompt))
    working.extend(messages)

    total_usage = Usage()
    last_finish_reason: str | None = None
    response_metadata: AepResponseMetadata | None = None
    root_span = None

    if span_tracker:
        root_span = span_tracker.start_span("agent_loop", client.model_name)
        yield span_start_event(root_span.span_id, "agent_loop", client.model_name)

    try:
        for _iteration in range(max_iterations):
            # Budget check
            reservation = None
            if budget:
                reservation = budget.reserve(_DEFAULT_RESERVATION)

            # LLM streaming call
            llm_span = None
            if span_tracker and root_span:
                llm_span = span_tracker.start_span(
                    "llm_call", client.model_name, parent_span_id=root_span.span_id
                )
                yield span_start_event(
                    llm_span.span_id, "llm_call", client.model_name, root_span.span_id
                )

            accumulated_text = ""
            accumulated_tool_calls = []
            call_usage = Usage()
            cost_node = None
            ok = False
            response_metadata = None

            # `finally` runs on any exit — exception, cancellation, or
            # forced close — so the llm_span ends and the reservation
            # settles even when chat_stream raises (e.g.
            # StreamFailedError on a silent upstream rejection). On
            # success we additionally yield the cost + span_end events
            # below; on failure observers can reconstruct the span
            # status from the tracker (we can't safely yield during
            # cancellation or forced close).
            try:
                if request_context is not None:
                    if not isinstance(client, ContextChatClient):
                        raise TypeError("Client does not support AEP request context")
                    chunks = client.chat_stream_with_context(
                        working,
                        request_context=request_context,
                        tools=tool_schemas,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                else:
                    chunks = client.chat_stream(
                        working,
                        tools=tool_schemas,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                async for stream_chunk in chunks:
                    if stream_chunk.response_metadata is not None and response_metadata is None:
                        response_metadata = stream_chunk.response_metadata
                        yield response_metadata_event(
                            stream_chunk.response_metadata, complete=False
                        )
                    if stream_chunk.delta_text:
                        accumulated_text += stream_chunk.delta_text
                        event = chunk_event(stream_chunk.delta_text)
                        event.response_metadata = response_metadata
                        yield event

                    if stream_chunk.delta_tool_calls:
                        accumulated_tool_calls.extend(stream_chunk.delta_tool_calls)

                    if stream_chunk.usage:
                        call_usage = stream_chunk.usage

                    if stream_chunk.finish_reason:
                        last_finish_reason = stream_chunk.finish_reason

                total_usage = total_usage + call_usage

                if cost_tracker and llm_span:
                    cost_node = cost_tracker.record_llm_cost(
                        span_id=llm_span.span_id,
                        model=client.model_name,
                        usage=call_usage,
                    )
                    yield cost_event(cost_node)

                ok = True
                if response_metadata is not None:
                    yield response_metadata_event(response_metadata, complete=True)
            finally:
                if llm_span and span_tracker:
                    span_tracker.end_span(llm_span.span_id, status="OK" if ok else "ERROR")
                if budget and reservation:
                    actual_cost = cost_node.total_cost() if ok and cost_node else Decimal("0")
                    budget.settle(reservation, actual_cost)

            if llm_span and span_tracker:
                yield span_end_event(llm_span.span_id)

            # Build assistant message
            assistant_msg = ChatMessage(
                role="assistant",
                content=accumulated_text,
                tool_calls=accumulated_tool_calls if accumulated_tool_calls else None,
            )
            working.append(assistant_msg)

            # No tool calls -> done
            if not accumulated_tool_calls:
                break

            # If the response was truncated by max_tokens, tool call arguments
            # are likely incomplete (truncated JSON).  Rather than executing
            # broken calls or giving up, remove the partial assistant message
            # and ask the model to retry with a more concise approach.
            if last_finish_reason == "max_tokens":
                working.pop()  # remove partial assistant message
                working.append(
                    ChatMessage(
                        role="user",
                        content=(
                            "Your previous response was cut off because it exceeded the "
                            "output token limit. Please try again with a more concise "
                            "version. If the content is too large for a single tool call, "
                            "split it into smaller parts across multiple calls."
                        ),
                    )
                )
                yield chunk_event("[Retrying -- previous response exceeded token limit]\n\n")
                last_finish_reason = None
                continue

            # Execute tool calls
            for tc in accumulated_tool_calls:
                event = tool_call_start_event(tc.id, tc.name, tc.arguments)
                event.response_metadata = response_metadata
                yield event

                tool = tools_by_name.get(tc.name)
                if not tool:
                    error_msg = f"Tool '{tc.name}' not found"
                    working.append(
                        ChatMessage(
                            role="tool",
                            content=json.dumps({"error": error_msg}),
                            tool_call_id=tc.id,
                            name=tc.name,
                        )
                    )
                    event = tool_call_end_event(tc.id, None, error=error_msg)
                    event.response_metadata = response_metadata
                    yield event
                    continue

                tool_span = None
                if span_tracker and root_span:
                    tool_span = span_tracker.start_span(
                        "tool_call", tc.name, parent_span_id=root_span.span_id
                    )

                try:
                    result = await asyncio.wait_for(
                        _invoke_with_origin(tool, tc.arguments, response_metadata),
                        timeout=_DEFAULT_TOOL_TIMEOUT,
                    )
                    result_str = json.dumps(result) if not isinstance(result, str) else result
                    working.append(
                        ChatMessage(
                            role="tool",
                            content=result_str,
                            tool_call_id=tc.id,
                            name=tc.name,
                        )
                    )
                    if tool_span and span_tracker:
                        span_tracker.end_span(tool_span.span_id)
                    event = tool_call_end_event(tc.id, result)
                    event.response_metadata = response_metadata
                    yield event
                except Exception as e:
                    working.append(
                        ChatMessage(
                            role="tool",
                            content=json.dumps({"error": str(e)}),
                            tool_call_id=tc.id,
                            name=tc.name,
                        )
                    )
                    if tool_span and span_tracker:
                        span_tracker.end_span(tool_span.span_id, status="ERROR")
                    event = tool_call_end_event(tc.id, None, error=str(e))
                    event.response_metadata = response_metadata
                    yield event

        if root_span and span_tracker:
            span_tracker.end_span(root_span.span_id)
            yield span_end_event(root_span.span_id)

        yield end_event(total_usage, finish_reason=last_finish_reason)

    except Exception as e:
        if root_span and span_tracker:
            span_tracker.end_span(root_span.span_id, status="ERROR")
            yield span_end_event(root_span.span_id, status="ERROR")
        yield error_event("agent_error", str(e))
        raise


__all__ = ["run_agent_loop", "run_agent_loop_stream"]
