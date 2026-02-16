"""Agent loop - replaces LangGraph StateGraph with a simple while loop.

run_agent_loop() for non-streaming, run_agent_loop_stream() for streaming.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any

from .budget import BudgetEnforcer
from .client import ChatClient
from .costs import CostTracker
from .spans import SpanTracker
from .stream import (
    StreamEvent,
    chunk_event,
    cost_event,
    end_event,
    error_event,
    span_end_event,
    span_start_event,
    tool_call_end_event,
    tool_call_start_event,
)
from .tools import Tool
from .types import AgentResult, ChatMessage, ChatResponse, Usage

logger = logging.getLogger(__name__)

# Default estimated cost per LLM call for budget reservation
_DEFAULT_RESERVATION = Decimal("0.01")


def _build_tool_schemas(tools: list[Tool]) -> list[dict[str, Any]]:
    """Convert Tool objects to OpenAI function calling format."""
    return [t.to_openai_tool() for t in tools]


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

            response: ChatResponse = await client.chat(
                working,
                tools=tool_schemas,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            total_usage = total_usage + response.usage

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
                    result = await tool.invoke(tc.arguments)
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

            async for stream_chunk in client.chat_stream(
                working,
                tools=tool_schemas,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                if stream_chunk.delta_text:
                    accumulated_text += stream_chunk.delta_text
                    yield chunk_event(stream_chunk.delta_text)

                if stream_chunk.delta_tool_calls:
                    accumulated_tool_calls.extend(stream_chunk.delta_tool_calls)

                if stream_chunk.usage:
                    call_usage = stream_chunk.usage

            total_usage = total_usage + call_usage

            # Record cost
            cost_node = None
            if cost_tracker and llm_span:
                cost_node = cost_tracker.record_llm_cost(
                    span_id=llm_span.span_id,
                    model=client.model_name,
                    usage=call_usage,
                )
                yield cost_event(cost_node)

            if llm_span and span_tracker:
                span_tracker.end_span(llm_span.span_id)
                yield span_end_event(llm_span.span_id)

            # Settle budget
            if budget and reservation and cost_node:
                budget.settle(reservation, cost_node.total_cost())
            elif budget and reservation:
                budget.settle(reservation, Decimal("0"))

            # Build assistant message
            assistant_msg = ChatMessage(
                role="assistant",
                content=accumulated_text,
                tool_calls=accumulated_tool_calls if accumulated_tool_calls else None,
            )
            working.append(assistant_msg)

            # No tool calls → done
            if not accumulated_tool_calls:
                break

            # Execute tool calls
            for tc in accumulated_tool_calls:
                yield tool_call_start_event(tc.id, tc.name, tc.arguments)

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
                    yield tool_call_end_event(tc.id, None, error=error_msg)
                    continue

                tool_span = None
                if span_tracker and root_span:
                    tool_span = span_tracker.start_span(
                        "tool_call", tc.name, parent_span_id=root_span.span_id
                    )

                try:
                    result = await tool.invoke(tc.arguments)
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
                    yield tool_call_end_event(tc.id, result)
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
                    yield tool_call_end_event(tc.id, None, error=str(e))

        if root_span and span_tracker:
            span_tracker.end_span(root_span.span_id)
            yield span_end_event(root_span.span_id)

        yield end_event(total_usage)

    except Exception as e:
        if root_span and span_tracker:
            span_tracker.end_span(root_span.span_id, status="ERROR")
            yield span_end_event(root_span.span_id, status="ERROR")
        yield error_event("agent_error", str(e))
        raise


__all__ = ["run_agent_loop", "run_agent_loop_stream"]
