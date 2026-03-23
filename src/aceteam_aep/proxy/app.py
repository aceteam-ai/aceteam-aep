"""AEP reverse proxy — sits between agent and LLM API.

Intercepts OpenAI-compatible requests, runs safety checks on input and output,
tracks cost, and enforces PASS/FLAG/BLOCK decisions.
"""

from __future__ import annotations

import json
import logging
import uuid
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
        policy: EnforcementPolicy | None = None,
    ) -> None:
        self.target_base_url = target_base_url.rstrip("/")
        self.cost_tracker = CostTracker()
        self.span_tracker = SpanTracker()
        self.registry = DetectorRegistry()
        self.policy = policy or EnforcementPolicy()
        self.signals: list[SafetySignal] = []
        self.call_count = 0
        self.decisions: list[EnforcementDecision] = []
        self._call_costs: list[Decimal] = []

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

    def to_dict(self) -> dict[str, Any]:
        """State as JSON-serializable dict for the dashboard."""
        decision = self.latest_enforcement
        return {
            "cost": float(self.cost_usd),
            "calls": self.call_count,
            "action": decision.action,
            "reason": decision.reason,
            "signals": [
                {
                    "type": s.signal_type,
                    "severity": s.severity,
                    "detail": s.detail,
                    "call_id": s.call_id,
                    "detector": s.detector,
                    "timestamp": s.timestamp,
                }
                for s in self.signals
            ],
            "spans": [
                {
                    "id": s.span_id,
                    "executor": s.executor_id,
                    "status": s.status,
                    "duration_ms": s.duration_ms,
                    "started_at": s.started_at,
                }
                for s in self.span_tracker.get_spans()
            ],
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
    policy: EnforcementPolicy | None = None,
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

        # Forward headers (especially Authorization)
        forward_headers: dict[str, str] = {}
        if request.headers.get("authorization"):
            forward_headers["Authorization"] = request.headers["authorization"]
        if request.headers.get("content-type"):
            forward_headers["Content-Type"] = request.headers["content-type"]
        # Forward anthropic-specific headers
        for key in ("x-api-key", "anthropic-version"):
            if request.headers.get(key):
                forward_headers[key] = request.headers[key]

        # --- STREAMING BRANCH ---
        if body.get("stream"):
            from .streaming import handle_streaming_request

            def on_stream_complete(**kwargs: Any) -> None:
                model = kwargs.get("model", "unknown")
                inp = kwargs.get("input_tokens", 0)
                out = kwargs.get("output_tokens", 0)
                signals = kwargs.get("signals", [])
                decision = kwargs.get("decision")

                span = state.span_tracker.start_span(
                    executor_type="llm",
                    executor_id=model,
                    metadata={"call_id": call_id, "streaming": True},
                )
                usage = Usage(
                    prompt_tokens=inp,
                    completion_tokens=out,
                    total_tokens=inp + out,
                )
                state.cost_tracker.record_llm_cost(
                    span_id=span.span_id,
                    model=model,
                    usage=usage,
                )
                state.span_tracker.end_span(span.span_id)
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
        async with httpx.AsyncClient(timeout=120.0) as client:
            upstream_resp = await client.request(
                method=request.method,
                url=target_url,
                content=body_bytes,
                headers=forward_headers,
            )

        # If upstream errored, pass through
        if upstream_resp.status_code >= 400:
            return Response(
                content=upstream_resp.content,
                status_code=upstream_resp.status_code,
                headers=dict(upstream_resp.headers),
            )

        # Parse response
        try:
            resp_data = upstream_resp.json()
        except json.JSONDecodeError:
            return Response(
                content=upstream_resp.content,
                status_code=upstream_resp.status_code,
            )

        # --- RECORD COST + SPANS ---
        model, input_tokens, output_tokens = _extract_usage(resp_data)
        output_text = _extract_output_text(resp_data)

        span = state.span_tracker.start_span(
            executor_type="llm",
            executor_id=model,
            metadata={"call_id": call_id},
        )
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
        resp_headers = {
            "X-AEP-Cost": str(cost_node.compute_cost),
            "X-AEP-Enforcement": decision.action,
            "X-AEP-Call-ID": call_id,
        }
        if decision.action == "flag":
            resp_headers["X-AEP-Flag-Reason"] = decision.reason

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
