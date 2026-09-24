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
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx
from starlette.responses import Response, StreamingResponse

from ..enforcement import EnforcementDecision, EnforcementPolicy, evaluate, evaluate_pipeline
from ..safety.base import DetectorRegistry, SafetySignal

log = logging.getLogger(__name__)

TransportOutcome = Literal[
    "completed", "http_error", "network_error", "stream_error", "interrupted"
]
EvaluationState = Literal["completed", "not_evaluated", "unavailable", "interrupted"]
EnforcementAction = Literal["pass", "flag", "block"]


@dataclass(frozen=True)
class TerminalSignal:
    """Signal identifiers only; detector details may contain sensitive text."""

    signal_type: str
    severity: str
    detector: str


@dataclass(frozen=True)
class TerminalOutcome:
    """Final state of one admitted relay call, without request or response text.

    Evaluation observes output already sent to the consumer. A ``block`` action
    therefore does not mean that the output was withheld.
    """

    call_id: str
    transport: TransportOutcome
    evaluation: EvaluationState
    action: EnforcementAction | None = None
    signals: tuple[TerminalSignal, ...] = ()
    scope: Literal["observed_output", "not_applicable"] = "observed_output"
    http_status: int | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """Result returned by an embedding application's policy-aware runner.

    Runners must report ``unavailable`` if any detector failed, including when
    that failure was caught internally and no signals were returned.
    """

    signals: Sequence[SafetySignal]
    decision: EnforcementDecision | None
    state: Literal["completed", "not_evaluated", "unavailable"]


class EvaluationRunner(Protocol):
    """Optional existing evaluation runner supplied by the embedding app."""

    def __call__(
        self, *, input_text: str, output_text: str, call_id: str
    ) -> Awaitable[EvaluationResult]: ...


def _terminal_signals(signals: Sequence[SafetySignal]) -> tuple[TerminalSignal, ...]:
    return tuple(TerminalSignal(s.signal_type, s.severity, s.detector) for s in signals)


async def _run_registry_with_status(
    registry: DetectorRegistry, *, input_text: str, output_text: str, call_id: str
) -> tuple[list[SafetySignal], bool]:
    """Mirror registry execution while retaining failures hidden by run_all."""

    async def run_one(detector: Any) -> tuple[list[SafetySignal], bool]:
        try:
            signals = list(
                await detector.check(
                    input_text=input_text, output_text=output_text, call_id=call_id
                )
            )
            for signal in signals:
                signal.detector = detector.name
            return signals, False
        except Exception:
            log.warning(
                "Detector %s failed, skipping",
                getattr(detector, "name", "unknown"),
                exc_info=True,
            )
            return [], True

    results = await asyncio.gather(*(run_one(d) for d in registry._detectors))
    return [signal for signals, _ in results for signal in signals], any(
        failed for _, failed in results
    )


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
    on_terminal: Callable[[TerminalOutcome], None] | None = None,
    evaluation_runner: EvaluationRunner | None = None,
    evaluation_enabled: bool = True,
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
        on_terminal: Optional exactly-once callback with text-free terminal metadata
        evaluation_runner: Optional policy-aware evaluation runner
        evaluation_enabled: Whether output safety evaluation is enabled
        debug: Enable debug logging for the stream
    """
    terminal_sent = False

    def notify(
        transport: TransportOutcome,
        evaluation: EvaluationState,
        *,
        decision: EnforcementDecision | None = None,
        signals: Sequence[SafetySignal] = (),
        http_status: int | None = None,
    ) -> None:
        nonlocal terminal_sent
        if terminal_sent:
            return
        terminal_sent = True
        if on_terminal is None:
            return
        action: EnforcementAction | None = None
        if (
            evaluation == "completed"
            and decision is not None
            and decision.action in ("pass", "flag", "block")
        ):
            action = decision.action
        try:
            on_terminal(
                TerminalOutcome(
                    call_id=call_id,
                    transport=transport,
                    evaluation=evaluation,
                    action=action,
                    signals=_terminal_signals(signals),
                    scope=(
                        "not_applicable"
                        if http_status is None or transport in ("http_error", "network_error")
                        else "observed_output"
                    ),
                    http_status=http_status,
                )
            )
        except BaseException:
            log.warning("Terminal callback failed for %s", call_id, exc_info=True)

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
    except BaseException as exc:
        interrupted = isinstance(exc, (asyncio.CancelledError, GeneratorExit))
        try:
            await client.aclose()
        finally:
            notify(
                "interrupted" if interrupted else "network_error",
                "interrupted" if interrupted else "not_evaluated",
            )
        raise

    # Upstream rejected the request before any bytes were streamed: pass the
    # provider's status + error body through verbatim (Anthropic/OpenAI error
    # shape preserved) instead of returning an empty 200 SSE stream.
    if upstream.status_code < 200 or upstream.status_code >= 300:
        try:
            error_body = await upstream.aread()
        except BaseException as exc:
            interrupted = isinstance(exc, (asyncio.CancelledError, GeneratorExit))
            notify(
                "interrupted" if interrupted else "network_error",
                "interrupted" if interrupted else "not_evaluated",
                http_status=upstream.status_code,
            )
            raise
        finally:
            try:
                await upstream.aclose()
            finally:
                await client.aclose()
        log.warning(
            "STREAM UPSTREAM ERROR %s: status=%d body=%s",
            call_id,
            upstream.status_code,
            error_body[:500].decode("utf-8", errors="replace"),
        )
        notify("http_error", "not_evaluated", http_status=upstream.status_code)
        return Response(
            content=error_body,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
            headers={"X-AEP-Call-ID": call_id},
        )

    async def stream_generator() -> AsyncGenerator[str, None]:
        accumulated_chunks: list[dict[str, Any]] = []
        saw_terminal_marker = False
        transport: TransportOutcome = "interrupted"
        evaluation: EvaluationState = "interrupted"
        terminal_decision: EnforcementDecision | None = None
        terminal_signals: Sequence[SafetySignal] = ()
        finished = False
        try:
            try:
                # Pass through each line immediately
                async for line in upstream.aiter_lines():
                    if line == "data: [DONE]" or line == "event: message_stop":
                        saw_terminal_marker = True
                    parsed = _parse_sse_line(line)
                    if parsed:
                        accumulated_chunks.append(parsed)
                    if debug:
                        log.debug("STREAM CHUNK %s: %s", call_id, line[:200] if line else "<empty>")
                    yield f"{line}\n"
            except Exception as exc:
                # Bytes were already sent under a 200; preserve the legacy error event.
                transport = "stream_error"
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
                try:
                    await upstream.aclose()
                finally:
                    await client.aclose()

            transport = "completed" if saw_terminal_marker else "interrupted"
            # Keep the legacy cost and wire behavior at EOF, including unmarked EOF.
            model, output_text, input_tokens, output_tokens = _accumulate_stream_chunks(
                accumulated_chunks
            )
            if debug:
                log.debug(
                    "STREAM COMPLETE %s: model=%s, input_tokens=%d, "
                    "output_tokens=%d, output_text=%s",
                    call_id,
                    model,
                    input_tokens,
                    output_tokens,
                    output_text[:500] if output_text else "",
                )

            signals: Sequence[SafetySignal] = ()
            decision: EnforcementDecision
            if not evaluation_enabled:
                evaluation = "not_evaluated"
                decision = evaluate(signals, policy)
            elif evaluation_runner is not None:
                try:
                    result = await evaluation_runner(
                        input_text=input_text, output_text=output_text, call_id=call_id
                    )
                    signals = result.signals
                    evaluation = result.state
                    decision = result.decision or evaluate(signals, policy)
                    if evaluation == "completed" and result.decision is None:
                        evaluation = "unavailable"
                except Exception:
                    log.warning("Stream evaluation failed for %s", call_id, exc_info=True)
                    evaluation = "unavailable"
                    terminal_signals = signals
                    finished = True
                    raise
            elif pipeline:
                try:
                    pipeline_result = await pipeline.evaluate(
                        input_text=input_text, output_text=output_text, call_id=call_id
                    )
                    signals = pipeline_result.signals
                    decision = evaluate_pipeline(pipeline_result, policy)
                    evaluation = "completed"
                except Exception:
                    log.warning("Stream evaluation failed for %s", call_id, exc_info=True)
                    evaluation = "unavailable"
                    terminal_signals = signals
                    finished = True
                    raise
                else:
                    if on_terminal is not None:
                        layers = getattr(pipeline, "_layers", None)
                        if layers is None:
                            evaluation = "unavailable"
                        else:
                            expected = [layer.name for layer in layers]
                            actual = [result.layer_name for result in pipeline_result.layer_results]
                            if not expected:
                                evaluation = "not_evaluated"
                            elif actual != expected[: len(actual)] or (
                                pipeline_result.short_circuited_at is None
                                and len(actual) != len(expected)
                            ):
                                evaluation = "unavailable"
            else:
                if on_terminal is not None:
                    if registry._detectors:
                        signals, failed = await _run_registry_with_status(
                            registry,
                            input_text=input_text,
                            output_text=output_text,
                            call_id=call_id,
                        )
                        evaluation = "unavailable" if failed else "completed"
                    else:
                        evaluation = "not_evaluated"
                else:
                    signals = await registry.run_all(
                        input_text=input_text,
                        output_text=output_text,
                        call_id=call_id,
                    )
                decision = evaluate(signals, policy)

            terminal_signals = signals
            terminal_decision = decision
            if transport != "completed":
                evaluation = "interrupted"

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

            finished = True
            if on_complete:
                on_complete(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    output_text=output_text,
                    signals=signals,
                    decision=decision,
                )
        finally:
            if not finished and transport == "completed":
                transport = "interrupted"
                evaluation = "interrupted"
            notify(
                transport,
                evaluation,
                decision=terminal_decision,
                signals=terminal_signals,
                http_status=upstream.status_code,
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


__all__ = [
    "EvaluationResult",
    "EvaluationRunner",
    "TerminalOutcome",
    "TerminalSignal",
    "handle_streaming_request",
]
