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
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from starlette.responses import Response, StreamingResponse

from ..enforcement import EnforcementPolicy, evaluate, evaluate_pipeline
from ..safety.base import DetectorRegistry

log = logging.getLogger(__name__)


def _sse_field_value(line: str) -> tuple[str, str] | None:
    """Parse an SSE field, removing the one optional space after its colon."""
    if not line or line.startswith(":"):
        return None
    field, separator, value = line.partition(":")
    if not separator:
        return field, ""
    return field, value[1:] if value.startswith(" ") else value


def _parse_sse_line(line: str) -> dict[str, Any] | None:
    """Parse a single SSE data line into a dict."""
    field_value = _sse_field_value(line)
    if field_value is None or field_value[0] != "data":
        return None
    data = field_value[1]
    if data == "[DONE]":
        return None
    try:
        parsed = json.loads(data)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _is_terminal_sse_line(line: str) -> bool:
    field_value = _sse_field_value(line)
    return (
        field_value is not None and field_value[0] == "data" and field_value[1].strip() == "[DONE]"
    )


def _is_message_stop_sse_line(line: str) -> bool:
    field_value = _sse_field_value(line)
    return field_value is not None and field_value == ("event", "message_stop")


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
            output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0

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
    pipeline: Any = None,
    on_complete: Any = None,
    debug: bool = False,
) -> Response:
    """Handle a streaming request through the proxy.

    Opens the upstream connection and checks its status BEFORE committing to a
    200 SSE response. A non-2xx upstream is passed through as-is (status code +
    error body, JSON not SSE) so clients see the provider's real error instead
    of hanging on an empty stream. A mid-stream failure emits a final SSE
    ``event: error`` with an Anthropic-format payload before closing.

    Args:
        target_url: Full URL to forward to (e.g., https://api.openai.com/v1/chat/completions)
        body_bytes: Raw request body
        headers: Headers to forward (auth, content-type)
        call_id: Unique call ID for this request
        input_text: Extracted input text for safety checking
        registry: Safety detector registry
        policy: Enforcement policy
        on_complete: Callback(model, input_tokens, output_tokens, output_text, signals, decision)
        debug: Enable debug logging for the stream
    """
    if debug:
        log.debug("STREAM REQUEST %s: %s", call_id, target_url)

    client = httpx.AsyncClient(timeout=120.0)
    try:
        upstream = await client.send(
            client.build_request(
                "POST",
                target_url,
                content=body_bytes,
                headers=headers,
            ),
            stream=True,
        )
    except Exception:
        await client.aclose()
        raise

    # Upstream rejected the request before any bytes were streamed: pass the
    # provider's status + error body through verbatim (Anthropic/OpenAI error
    # shape preserved) instead of returning an empty 200 SSE stream.
    if upstream.status_code < 200 or upstream.status_code >= 300:
        try:
            error_body = await upstream.aread()
        finally:
            await upstream.aclose()
            await client.aclose()
        log.warning(
            "STREAM UPSTREAM ERROR %s: status=%d body=%s",
            call_id,
            upstream.status_code,
            error_body[:500].decode("utf-8", errors="replace"),
        )
        return Response(
            content=error_body,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
            headers={"X-AEP-Call-ID": call_id},
        )

    async def stream_generator() -> AsyncGenerator[str, None]:
        accumulated_chunks: list[dict[str, Any]] = []
        upstream_done = False
        terminal_event: str | None = None
        event_lines: list[str] = []

        try:
            # Forward content promptly, but withhold the terminal marker until
            # the output safety verdict is available. Clients stop at [DONE].
            async for line in upstream.aiter_lines():
                if upstream_done:
                    if line.strip():
                        log.warning("Ignoring upstream data after [DONE] for %s", call_id)
                    continue
                event_lines.append(line)
                if line:
                    continue

                # SSE fields form an event only at the blank-line boundary.
                # Classify the whole frame so terminal fields in either order
                # are withheld, without swallowing later non-terminal events.
                frame = "".join(f"{event_line}\n" for event_line in event_lines)
                if any(_is_terminal_sse_line(event_line) for event_line in event_lines):
                    upstream_done = True
                    event_lines.clear()
                    continue
                parsed_chunks = [
                    parsed
                    for event_line in event_lines
                    if (parsed := _parse_sse_line(event_line)) is not None
                ]
                is_message_stop = any(_is_message_stop_sse_line(item) for item in event_lines)
                is_message_stop |= any(
                    parsed.get("type") == "message_stop" for parsed in parsed_chunks
                )
                event_lines.clear()
                if is_message_stop:
                    if terminal_event is None:
                        terminal_event = frame
                    continue
                accumulated_chunks.extend(parsed_chunks)

                # Debug: log each chunk (truncated)
                if debug:
                    log.debug("STREAM CHUNK %s: %s", call_id, frame[:200])

                yield frame

            # Preserve a trailing OpenAI [DONE] even without its final blank
            # line, as the previous line-forwarding path did. Other incomplete
            # terminal frames are not valid SSE and must not be replayed.
            if event_lines:
                if any(_is_terminal_sse_line(item) for item in event_lines):
                    upstream_done = True
                else:
                    partial_chunks = [
                        parsed
                        for event_line in event_lines
                        if (parsed := _parse_sse_line(event_line)) is not None
                    ]
                    is_message_stop = any(
                        _is_message_stop_sse_line(item) for item in event_lines
                    ) or any(parsed.get("type") == "message_stop" for parsed in partial_chunks)
                    if not is_message_stop:
                        accumulated_chunks.extend(partial_chunks)
                        yield "".join(f"{event_line}\n" for event_line in event_lines)
        except Exception as exc:
            # Bytes were already sent under a 200; the status can't change.
            # Emit a final Anthropic-format SSE error event so clients
            # terminate instead of waiting forever, then close the stream.
            log.warning("STREAM INTERRUPTED %s: %s: %s", call_id, type(exc).__name__, exc)
            error_event = {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"AEP proxy: upstream stream interrupted: {exc}",
                },
            }
            yield "event: error\n"
            yield f"data: {json.dumps(error_event)}\n\n"
            return
        finally:
            await upstream.aclose()
            await client.aclose()

        # Stream complete — run safety checks on accumulated output
        model, output_text, input_tokens, output_tokens = _accumulate_stream_chunks(
            accumulated_chunks
        )

        if debug:
            log.debug(
                "STREAM COMPLETE %s: model=%s, input_tokens=%d, output_tokens=%d, output_text=%s",
                call_id,
                model,
                input_tokens,
                output_tokens,
                output_text[:500] if output_text else "",
            )

        # Run safety detectors
        if pipeline:
            pipeline_result = await pipeline.evaluate(
                input_text=input_text,
                output_text=output_text,
                call_id=call_id,
            )
            signals = pipeline_result.signals
            decision = evaluate_pipeline(pipeline_result, policy)
        else:
            signals = await registry.run_all(
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

        # Forward exactly one OpenAI terminal marker, after any safety block.
        # Other SSE protocols (for example Anthropic message_stop) do not use
        # [DONE], so do not synthesize one for them.
        if upstream_done:
            yield "data: [DONE]\n\n"
        elif terminal_event is not None:
            yield terminal_event

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
