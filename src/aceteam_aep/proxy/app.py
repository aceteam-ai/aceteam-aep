"""AEP reverse proxy — sits between agent and LLM API.

Intercepts OpenAI-compatible requests, runs safety checks on input and output,
tracks cost, and enforces PASS/FLAG/BLOCK decisions.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from ..costs import CostTracker
from ..enforcement import EnforcementDecision, EnforcementPolicy, evaluate
from ..safety.base import DetectorRegistry, SafetySignal
from ..safety.cost_anomaly import CostAnomalyDetector
from ..spans import SpanTracker
from ..types import Usage
from .headers import build_response_headers, parse_aep_headers, strip_aep_headers

log = logging.getLogger(__name__)

# Models where usage field uses different names
_ANTHROPIC_STYLE = frozenset({"claude", "anthropic"})


def _extract_text_from_messages(messages: list[dict[str, Any]]) -> str:
    """Extract text content from OpenAI-format messages."""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return " ".join(parts)


def _extract_output_text(data: dict[str, Any]) -> str:
    """Extract assistant text from an OpenAI-format response."""
    parts: list[str] = []
    for choice in data.get("choices", []):
        msg = choice.get("message", {})
        content = msg.get("content")
        if content:
            parts.append(content)
    return " ".join(parts)


def _extract_usage(data: dict[str, Any]) -> tuple[str, int, int]:
    """Extract (model, input_tokens, output_tokens) from response."""
    model = data.get("model", "unknown")
    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0
    return model, input_tokens, output_tokens


class ProxyState:
    """Shared state for the proxy — cost tracking, safety signals, spans."""

    def __init__(
        self,
        target_base_url: str = "https://api.openai.com",
        detectors: list[Any] | None = None,
        policy: EnforcementPolicy | dict[str, Any] | str | None = None,
    ) -> None:
        self.target_base_url = target_base_url.rstrip("/")
        self.cost_tracker = CostTracker()
        self.span_tracker = SpanTracker()
        self.registry = DetectorRegistry()
        self.policy = EnforcementPolicy.from_config(policy)
        self.signals: list[SafetySignal] = []
        self.call_count = 0
        self.decisions: list[EnforcementDecision] = []
        self._call_costs: list[Decimal] = []
        self.governance_contexts: list[dict[str, Any]] = []
        self._started_at = datetime.now(UTC)

        # Register detectors
        for det in detectors or _default_proxy_detectors():
            self.registry.add(det)

    @property
    def cost_usd(self) -> Decimal:
        return self.cost_tracker.total_spent()

    @property
    def latest_enforcement(self) -> EnforcementDecision:
        if self.decisions:
            return self.decisions[-1]
        return EnforcementDecision(action="pass")

    def _cost_by_span_id(self) -> dict[str, float]:
        """Build a lookup of span_id → cost in USD."""
        lookup: dict[str, float] = {}
        for node in self.cost_tracker.get_cost_tree():
            sid = node.metadata.get("span_id")
            if sid:
                lookup[sid] = lookup.get(sid, 0.0) + float(node.compute_cost)
        return lookup

    def to_dict(self) -> dict[str, Any]:
        """State as JSON-serializable dict for the dashboard."""
        decision = self.latest_enforcement
        cost_lookup = self._cost_by_span_id()
        return {
            "cost": float(self.cost_usd),
            "calls": self.call_count,
            "action": decision.action,
            "reason": decision.reason,
            "session_started": self._started_at.isoformat(),
            "signals": [
                {
                    "type": s.signal_type,
                    "severity": s.severity,
                    "detail": s.detail,
                    "call_id": s.call_id,
                    "detector": s.detector,
                    "timestamp": s.timestamp,
                    "score": s.score,
                }
                for s in self.signals
            ],
            "spans": [
                {
                    "id": s.span_id,
                    "call_id": s.metadata.get("call_id"),
                    "executor": s.executor_id,
                    "status": s.status,
                    "duration_ms": s.duration_ms,
                    "started_at": s.started_at,
                    "cost": cost_lookup.get(s.span_id, 0.0),
                }
                for s in self.span_tracker.get_spans()
            ],
            "governance": self.governance_contexts,
        }


def _default_proxy_detectors() -> list[Any]:
    """Default detectors for the proxy."""
    from ..safety.agent_threat import AgentThreatDetector

    detectors: list[Any] = [CostAnomalyDetector(), AgentThreatDetector()]
    try:
        from ..safety.pii import PiiDetector

        detectors.append(PiiDetector())
    except Exception:
        pass
    try:
        from ..safety.content import ContentSafetyDetector

        detectors.append(ContentSafetyDetector())
    except Exception:
        pass
    return detectors


def create_proxy_app(
    target_base_url: str = "https://api.openai.com",
    detectors: list[Any] | None = None,
    policy: EnforcementPolicy | dict[str, Any] | str | None = None,
    dashboard: bool = True,
) -> Starlette:
    """Create the AEP proxy ASGI app."""

    state = ProxyState(
        target_base_url=target_base_url,
        detectors=detectors,
        policy=policy,
    )

    async def proxy_handler(request: Request) -> Response:
        """Forward request to target API with safety interception."""
        call_id = uuid.uuid4().hex[:8]
        path = request.url.path
        body_bytes = await request.body()

        # Parse request body
        try:
            body = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            body = {}

        # --- PARSE AEP GOVERNANCE HEADERS ---
        aep_ctx = parse_aep_headers(request.headers)
        gov_context: dict[str, Any] | None = None
        if aep_ctx.entity != "default" or aep_ctx.classification != "public" or aep_ctx.consent:
            gov_context = {
                "call_id": call_id,
                "entity": aep_ctx.entity,
                "classification": aep_ctx.classification,
                "consent": aep_ctx.consent,
                "budget": str(aep_ctx.budget) if aep_ctx.budget is not None else None,
                "sources": aep_ctx.sources,
                "trace_id": aep_ctx.trace_id,
            }
            state.governance_contexts.append(gov_context)

        # --- INPUT SAFETY CHECK ---
        input_text = ""
        if "messages" in body:
            input_text = _extract_text_from_messages(body["messages"])

        input_signals = state.registry.run_all(
            input_text=input_text,
            output_text="",
            call_id=call_id,
        )

        if input_signals:
            input_decision = evaluate(input_signals, state.policy)
            if input_decision.action == "block":
                state.signals.extend(input_signals)
                state.decisions.append(input_decision)
                state.call_count += 1
                log.warning("BLOCKED request %s: %s", call_id, input_decision.reason)
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": (f"AEP safety: request blocked — {input_decision.reason}"),
                            "type": "aep_safety_block",
                            "code": "safety_block",
                        }
                    },
                )

        # --- FORWARD TO TARGET API ---
        target_url = f"{state.target_base_url}{path}"

        # Forward headers (especially Authorization), stripping AEP governance headers
        forward_headers: dict[str, str] = {}
        if request.headers.get("authorization"):
            forward_headers["Authorization"] = request.headers["authorization"]
        if request.headers.get("content-type"):
            forward_headers["Content-Type"] = request.headers["content-type"]
        # Forward anthropic-specific headers
        for key in ("x-api-key", "anthropic-version"):
            if request.headers.get(key):
                forward_headers[key] = request.headers[key]
        # Strip X-AEP-* headers so they don't leak to the LLM provider
        forward_headers = strip_aep_headers(forward_headers)

        # --- STREAMING BRANCH ---
        if body.get("stream"):
            from .streaming import handle_streaming_request

            # Start span BEFORE the stream begins to measure actual latency
            stream_span = state.span_tracker.start_span(
                executor_type="llm",
                executor_id="pending",
                metadata={"call_id": call_id, "streaming": True},
            )

            def on_stream_complete(**kwargs: Any) -> None:
                model = kwargs.get("model", "unknown")
                inp = kwargs.get("input_tokens", 0)
                out = kwargs.get("output_tokens", 0)
                signals = kwargs.get("signals", [])
                decision = kwargs.get("decision")

                # Update span with actual model info
                stream_span.executor_id = model
                usage = Usage(
                    prompt_tokens=inp,
                    completion_tokens=out,
                    total_tokens=inp + out,
                )
                state.cost_tracker.record_llm_cost(
                    span_id=stream_span.span_id,
                    model=model,
                    usage=usage,
                )
                state.span_tracker.end_span(stream_span.span_id)
                state.call_count += 1
                state.signals.extend(signals)
                if decision:
                    state.decisions.append(decision)

            return await handle_streaming_request(
                target_url=target_url,
                body_bytes=body_bytes,
                headers=forward_headers,
                call_id=call_id,
                input_text=input_text,
                registry=state.registry,
                policy=state.policy,
                on_complete=on_stream_complete,
            )

        # --- NON-STREAMING BRANCH ---
        # Start span BEFORE the upstream call to measure actual latency
        span = state.span_tracker.start_span(
            executor_type="llm",
            executor_id="pending",
            metadata={"call_id": call_id},
        )

        async with httpx.AsyncClient(timeout=120.0) as client:
            upstream_resp = await client.request(
                method=request.method,
                url=target_url,
                content=body_bytes,
                headers=forward_headers,
            )

        # If upstream errored, pass through
        if upstream_resp.status_code >= 400:
            state.span_tracker.end_span(span.span_id, status="ERROR")
            return Response(
                content=upstream_resp.content,
                status_code=upstream_resp.status_code,
                headers=dict(upstream_resp.headers),
            )

        # Parse response
        try:
            resp_data = upstream_resp.json()
        except json.JSONDecodeError:
            state.span_tracker.end_span(span.span_id, status="ERROR")
            return Response(
                content=upstream_resp.content,
                status_code=upstream_resp.status_code,
            )

        # --- RECORD COST + SPANS ---
        model, input_tokens, output_tokens = _extract_usage(resp_data)
        output_text = _extract_output_text(resp_data)

        # Update span with actual model info now that we have the response
        span.executor_id = model
        usage = Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        cost_node = state.cost_tracker.record_llm_cost(
            span_id=span.span_id,
            model=model,
            usage=usage,
        )
        state.span_tracker.end_span(span.span_id)
        state.call_count += 1

        # --- OUTPUT SAFETY CHECK ---
        output_signals = state.registry.run_all(
            input_text=input_text,
            output_text=output_text,
            call_id=call_id,
            call_cost=cost_node.compute_cost,
        )

        all_signals = input_signals + output_signals
        state.signals.extend(all_signals)

        decision = evaluate(all_signals, state.policy)
        state.decisions.append(decision)

        if decision.action == "block":
            log.warning("BLOCKED response %s: %s", call_id, decision.reason)
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": (f"AEP safety: response blocked — {decision.reason}"),
                        "type": "aep_safety_block",
                        "code": "safety_block",
                    }
                },
            )

        # --- PASS THROUGH (with AEP metadata header) ---
        resp_headers = build_response_headers(
            cost=cost_node.compute_cost,
            enforcement=decision.action,
            call_id=call_id,
            classification=aep_ctx.classification,
            flag_reason=decision.reason if decision.action == "flag" else "",
            trace_id=aep_ctx.trace_id,
        )

        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=resp_headers,
        )

    # Dashboard routes
    routes: list[Route | Mount] = [
        # Catch-all proxy route for OpenAI-compatible paths
        Route("/v1/{path:path}", proxy_handler, methods=["POST", "GET"]),
        # Also handle Anthropic paths
        Route("/v1/messages", proxy_handler, methods=["POST"]),
    ]

    if dashboard:
        from ..dashboard.app import create_app as create_dashboard

        dashboard_app = create_dashboard(get_state=state.to_dict)
        routes.extend(
            [
                Route("/aep/", dashboard_app.routes[0].endpoint),  # type: ignore[union-attr]
                Route("/aep/api/state", dashboard_app.routes[1].endpoint),  # type: ignore[union-attr]
            ]
        )

    return Starlette(routes=routes)


__all__ = ["ProxyState", "create_proxy_app"]
