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

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from starlette.responses import Response, StreamingResponse

from ..enforcement import (
    EnforcementDecision,
    EnforcementPolicy,
    evaluate,
    evaluate_pipeline_with_residual,
)
from ..safety.base import DetectorRegistry
from ..safety.secrets import SecretDetector

log = logging.getLogger(__name__)

_MAX_FRAME_CHARS = 64 * 1024
_MAX_STREAM_CHARS = 1024 * 1024
_STREAM_SECONDS = 120
_TOKEN_END = re.compile(r"[A-Za-z0-9_+=/.-]$")
_LABEL_TAIL = re.compile(
    r"(?i)(?:api[_-]?key|apikey|secret[_-]?key|access[_-]?token|"
    r"aws_secret_access_key|aws_secret)\s*(?:[:=]\s*['\"]?)?$"
)


def _stream_fields(chunk: dict[str, Any]) -> dict[str, str]:
    """Return every semantic string by stable stream channel.

    Protocol metadata is checked in the complete raw frame separately. Its
    values are not delta fragments and would otherwise hold every frame until
    the stream limit (for example a repeated ``id`` or ``model``).
    """
    fields: dict[str, str] = {}

    def collect(value: Any, path: str) -> None:
        if isinstance(value, str):
            fields[path] = value
        elif isinstance(value, dict):
            for key, item in value.items():
                if key not in {
                    "model",
                    "id",
                    "type",
                    "object",
                    "role",
                    "index",
                    "finish_reason",
                    "created",
                    "created_at",
                }:
                    collect(item, f"{path}/{key}")
        elif isinstance(value, list):
            for position, item in enumerate(value):
                channel_index = item.get("index", position) if isinstance(item, dict) else position
                collect(item, f"{path}/{channel_index}")

    collect(chunk, "stream")
    return fields


async def contains_blocked_secret(
    text: str, registry: DetectorRegistry, policy: EnforcementPolicy, call_id: str
) -> bool:
    """Use the configured detector and policy, including disable and allow rules."""
    if "secret_leak" in policy.allow_types:
        return False
    for detector in registry._detectors:
        if not isinstance(detector, SecretDetector):
            continue
        signals = await detector.check(input_text="", output_text=text, call_id=call_id)
        for signal in signals:
            signal.detector = detector.name
        return evaluate(signals, policy).action == "block"
    return False


def _needs_holdback(text: str) -> bool:
    """Keep an unfinished credential token or label until its boundary is known."""
    return bool(_TOKEN_END.search(text) or _LABEL_TAIL.search(text) or "-----BEGIN" in text)


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

        text_parts.extend(_stream_fields(chunk).values())

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
    residual_registry: DetectorRegistry | None = None,
    on_complete: Any = None,
    debug: bool = False,
    safety_enabled: bool = True,
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
        residual_registry: Detectors not covered by pipeline layers, if enabled
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

    secret_guard = safety_enabled and await contains_blocked_secret(
        "AKIA" + "A" * 16, registry, policy, call_id
    )

    # Upstream rejected the request before any bytes were streamed: pass the
    # provider's status + error body through verbatim (Anthropic/OpenAI error
    # shape preserved) instead of returning an empty 200 SSE stream.
    if upstream.status_code < 200 or upstream.status_code >= 300:
        try:
            error_parts: list[bytes] = []
            error_size = 0
            error_iterator = upstream.aiter_bytes().__aiter__()
            error_deadline = time.monotonic() + _STREAM_SECONDS
            while True:
                try:
                    part = await asyncio.wait_for(
                        error_iterator.__anext__(),
                        timeout=max(0, error_deadline - time.monotonic()),
                    )
                except StopAsyncIteration:
                    break
                error_size += len(part)
                if error_size > _MAX_FRAME_CHARS:
                    raise ValueError("upstream error body too large")
                error_parts.append(part)
            error_body = b"".join(error_parts)
        except (TimeoutError, ValueError):
            error_body = b'{"error":{"message":"AEP proxy: upstream response unavailable"}}'
        finally:
            await upstream.aclose()
            await client.aclose()
        log.warning("STREAM UPSTREAM ERROR %s: status=%d", call_id, upstream.status_code)
        if secret_guard and await contains_blocked_secret(
            error_body.decode("utf-8", errors="replace"), registry, policy, call_id
        ):
            error_body = json.dumps(
                {
                    "error": {
                        "message": "AEP safety: upstream response blocked",
                        "code": "safety_block",
                    }
                }
            ).encode()
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
        pending_frames: list[str] = []
        pending_fields: dict[str, str] = {}
        pending_size = 0
        stream_size = 0
        blocked_early = False
        deadline = time.monotonic() + _STREAM_SECONDS

        async def inspect_frame(frame: str, chunks: list[dict[str, Any]]) -> tuple[bool, bool]:
            """Return (blocked, safe to flush) for a complete pending frame."""
            nonlocal pending_size
            pending_frames.append(frame)
            pending_size += len(frame)
            if pending_size > _MAX_FRAME_CHARS:
                raise ValueError("stream holdback limit exceeded")
            for chunk in chunks:
                raw_text = json.dumps(chunk, ensure_ascii=False)
                if await contains_blocked_secret(raw_text, registry, policy, call_id):
                    pending_fields["raw/frame"] = raw_text
                    return True, False
                for channel, fragment in _stream_fields(chunk).items():
                    pending_fields[channel] = pending_fields.get(channel, "") + fragment
                    if await contains_blocked_secret(
                        pending_fields[channel], registry, policy, call_id
                    ):
                        return True, False
            return False, not any(_needs_holdback(value) for value in pending_fields.values())

        try:
            # Forward content promptly, but withhold the terminal marker until
            # the output safety verdict is available. Clients stop at [DONE].
            line_iterator = upstream.aiter_lines().__aiter__()
            while True:
                try:
                    line = await asyncio.wait_for(
                        line_iterator.__anext__(), timeout=max(0, deadline - time.monotonic())
                    )
                except StopAsyncIteration:
                    break
                stream_size += len(line) + 1
                if (
                    stream_size > _MAX_STREAM_CHARS
                    or sum(map(len, event_lines)) + len(line) > _MAX_FRAME_CHARS
                ):
                    raise ValueError("stream size limit exceeded")
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
                frame = "".join(
                    f"{event_line}\n"
                    for event_line in event_lines
                    if not (secret_guard and event_line.startswith(":"))
                )
                if any(_is_terminal_sse_line(event_line) for event_line in event_lines):
                    upstream_done = True
                    event_lines.clear()
                    continue
                parsed_chunks = [
                    parsed
                    for event_line in event_lines
                    if (parsed := _parse_sse_line(event_line)) is not None
                ]
                if secret_guard:
                    data_lines = [
                        item
                        for item in event_lines
                        if (_sse_field_value(item) or (None,))[0] == "data"
                    ]
                    if len(data_lines) != len(parsed_chunks):
                        raise ValueError("unsupported stream data frame")
                    for item in event_lines:
                        field_value = _sse_field_value(item)
                        if field_value is None:
                            continue
                        if field_value[0] not in {"data", "event"}:
                            raise ValueError("unsupported stream field")
                        if field_value[0] == "event" and field_value[1] not in {
                            "message",
                            "message_start",
                            "message_stop",
                            "message_delta",
                            "content_block_start",
                            "content_block_delta",
                            "content_block_stop",
                            "ping",
                            "error",
                        }:
                            raise ValueError("unsupported stream event")
                    if not parsed_chunks and not any(
                        _is_message_stop_sse_line(item) for item in event_lines
                    ):
                        event_lines.clear()
                        continue
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
                    log.debug("STREAM CHUNK %s: %d chars", call_id, len(frame))

                if secret_guard:
                    blocked_early, safe = await inspect_frame(frame, parsed_chunks)
                    if blocked_early:
                        break
                    if safe:
                        yield "".join(pending_frames)
                        pending_frames.clear()
                        pending_fields.clear()
                        pending_size = 0
                else:
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
                        if secret_guard and not partial_chunks:
                            raise ValueError("unsupported trailing stream frame")
                        accumulated_chunks.extend(partial_chunks)
                        frame = "".join(f"{event_line}\n" for event_line in event_lines)
                        if secret_guard:
                            blocked_early, safe = await inspect_frame(frame, partial_chunks)
                            if safe:
                                yield "".join(pending_frames)
                                pending_frames.clear()
                                pending_fields.clear()
                        else:
                            yield frame
            if secret_guard and pending_frames and not blocked_early:
                for channel_text in pending_fields.values():
                    if await contains_blocked_secret(channel_text, registry, policy, call_id):
                        blocked_early = True
                        break
                if not blocked_early:
                    yield "".join(pending_frames)
        except Exception as exc:
            # Bytes were already sent under a 200; the status can't change.
            # Emit a final Anthropic-format SSE error event so clients
            # terminate instead of waiting forever, then close the stream.
            log.warning("STREAM INTERRUPTED %s: %s", call_id, type(exc).__name__)
            error_event = {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "AEP proxy: upstream stream interrupted",
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
                "STREAM COMPLETE %s: model=%s, input_tokens=%d, output_tokens=%d",
                call_id,
                model,
                input_tokens,
                output_tokens,
            )

        # A blocked secret is never released, even when a pipeline handles
        # other signal types or the upstream terminates without a final frame.
        if blocked_early:
            signals = []
            for detector in registry._detectors:
                if isinstance(detector, SecretDetector):
                    for channel_text in pending_fields.values():
                        detected = await detector.check(
                            input_text="", output_text=channel_text, call_id=call_id
                        )
                        for signal in detected:
                            signal.detector = detector.name
                        signals.extend(detected)
            decision = evaluate(signals, policy)
        elif not safety_enabled:
            signals = []
            decision = EnforcementDecision(action="pass")
        elif pipeline:
            pipeline_result = await pipeline.evaluate(
                input_text=input_text,
                output_text=output_text,
                call_id=call_id,
            )
            residual_signals = (
                await residual_registry.run_all(
                    input_text=input_text,
                    output_text=output_text,
                    call_id=call_id,
                )
                if residual_registry is not None
                else ()
            )
            signals = [*pipeline_result.signals, *residual_signals]
            decision = evaluate_pipeline_with_residual(pipeline_result, residual_signals, policy)
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
                "signals": [{"type": s.signal_type, "severity": s.severity} for s in signals],
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
        if blocked_early:
            if "/v1/messages" in target_url:
                yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'
            else:
                yield "data: [DONE]\n\n"
        elif upstream_done:
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
