"""SSE streaming handler for the AEP proxy.

When a request has `"stream": true`, the proxy:
1. Forwards to the target API with streaming
2. Passes each SSE chunk through to the client immediately (low latency)
3. Buffers chunks in parallel to accumulate the full response
4. After stream completes, runs safety checks on the accumulated output
5. If safety BLOCK: appends a final SSE event with the block signal
6. Records cost from the accumulated usage data
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from starlette.responses import StreamingResponse

from ..enforcement import EnforcementPolicy, evaluate
from ..safety.base import DetectorRegistry

log = logging.getLogger(__name__)


def _parse_sse_line(line: str) -> dict[str, Any] | None:
    """Parse a single SSE data line into a dict."""
    line = line.strip()
    if not line.startswith("data: "):
        return None
    data = line[6:]
    if data == "[DONE]":
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def _accumulate_stream_chunks(
    chunks: list[dict[str, Any]],
) -> tuple[str, str, int, int]:
    """Accumulate SSE chunks into (model, output_text, input_tokens, output_tokens)."""
    model = "unknown"
    text_parts: list[str] = []
    input_tokens = 0
    output_tokens = 0

    for chunk in chunks:
        if "model" in chunk:
            model = chunk["model"]

        # Accumulate text from delta
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                text_parts.append(content)

        # Usage is typically in the final chunk
        usage = chunk.get("usage")
        if usage:
            input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
            output_tokens = (
                usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0
            )

    return model, "".join(text_parts), input_tokens, output_tokens


async def handle_streaming_request(
    *,
    target_url: str,
    body_bytes: bytes,
    headers: dict[str, str],
    call_id: str,
    input_text: str,
    registry: DetectorRegistry,
    policy: EnforcementPolicy,
    on_complete: Any = None,
) -> StreamingResponse:
    """Handle a streaming request through the proxy.

    Args:
        target_url: Full URL to forward to (e.g., https://api.openai.com/v1/chat/completions)
        body_bytes: Raw request body
        headers: Headers to forward (auth, content-type)
        call_id: Unique call ID for this request
        input_text: Extracted input text for safety checking
        registry: Safety detector registry
        policy: Enforcement policy
        on_complete: Callback(model, input_tokens, output_tokens, output_text, signals, decision)
    """

    async def stream_generator() -> Any:
        accumulated_chunks: list[dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=120.0) as client, client.stream(
            "POST",
            target_url,
            content=body_bytes,
            headers=headers,
        ) as upstream:
            # Pass through each line immediately
            async for line in upstream.aiter_lines():
                # Buffer for post-stream safety
                parsed = _parse_sse_line(line)
                if parsed:
                    accumulated_chunks.append(parsed)

                # Pass through to client immediately
                yield f"{line}\n"

        # Stream complete — run safety checks on accumulated output
        model, output_text, input_tokens, output_tokens = _accumulate_stream_chunks(
            accumulated_chunks
        )

        # Run safety detectors
        signals = registry.run_all(
            input_text=input_text,
            output_text=output_text,
            call_id=call_id,
        )

        decision = evaluate(signals, policy)

        # If blocked, append a safety event to the stream
        if decision.action == "block":
            safety_event = {
                "aep_safety_block": True,
                "action": decision.action,
                "reason": decision.reason,
                "signals": [
                    {"type": s.signal_type, "severity": s.severity, "detail": s.detail}
                    for s in signals
                ],
            }
            yield f"data: {json.dumps(safety_event)}\n\n"

        # Notify caller of completion for cost tracking
        if on_complete:
            on_complete(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                output_text=output_text,
                signals=signals,
                decision=decision,
            )

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-AEP-Call-ID": call_id,
        },
    )


__all__ = ["handle_streaming_request"]
