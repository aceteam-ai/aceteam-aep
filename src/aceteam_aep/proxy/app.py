"""AEP reverse proxy — sits between agent and LLM API.

Intercepts OpenAI-compatible requests, runs safety checks on input and output,
tracks cost, and enforces PASS/FLAG/BLOCK decisions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from ..costs import CostTracker
from ..enforcement import EnforcementDecision, EnforcementPolicy, evaluate
from ..safety.agent_threat import AgentThreatDetector
from ..safety.base import DetectorRegistry, SafetyDetector, SafetySignal
from ..safety.content import ContentSafetyDetector
from ..safety.cost_anomaly import CostAnomalyDetector
from ..safety.custom import (
    CustomPolicy,
    CustomPolicyStore,
    CustomSafetyDetector,
    default_custom_policies,
)
from ..safety.pii import PiiDetector
from ..spans import SpanTracker
from ..types import Usage
from .headers import build_response_headers, parse_aep_headers, strip_aep_headers
from .logutil import configure_proxy_debug_logging
from .redis_publisher import build_event, publish_event

log = logging.getLogger(__name__)

# Models where usage field uses different names
_ANTHROPIC_STYLE = frozenset({"claude", "anthropic"})


async def _read_json_body(request: Request) -> tuple[Any | None, Response | None]:
    try:
        return await request.json(), None
    except Exception:
        return None, Response(
            '{"error": "invalid JSON"}', status_code=400, media_type="application/json"
        )


def _parse_custom_policy_write_fields(body: Any) -> dict[str, Any] | JSONResponse:
    """Require a JSON object with exactly ``name``, ``rule``, and ``enabled``."""
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
    fields = {k: body[k] for k in ("name", "rule", "enabled") if k in body}
    if set(fields.keys()) != {"name", "rule", "enabled"}:
        return JSONResponse(
            {"error": "name, rule, and enabled are required"},
            status_code=400,
        )
    return fields


def _custom_policy_from_write_fields(
    fields: dict[str, Any], *, policy_id: str | None = None
) -> CustomPolicy | JSONResponse:
    """Build a ``CustomPolicy`` from parsed fields; POST omits ``policy_id`` (new uuid)."""
    payload = dict(fields)
    if policy_id is not None:
        payload["id"] = policy_id
    try:
        return CustomPolicy.model_validate(payload).eager()
    except ValidationError as exc:
        return JSONResponse(
            {"error": "validation failed", "detail": exc.errors()},
            status_code=422,
        )


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
        detectors: Sequence[SafetyDetector] | None = None,
        policy: EnforcementPolicy | dict[str, Any] | str | None = None,
        budget: float | None = None,
        budget_per_session: float | None = None,
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
        self.blocked_count = 0
        self.safety_enabled = True
        self.budget = Decimal(str(budget)) if budget is not None else None
        self.budget_per_session = (
            Decimal(str(budget_per_session)) if budget_per_session is not None else None
        )

        to_register: list[SafetyDetector] = list(
            detectors if detectors is not None else _default_proxy_detectors()
        )
        custom_dets = [d for d in to_register if isinstance(d, CustomSafetyDetector)]
        if len(custom_dets) > 1:
            raise ValueError(
                "At most one CustomSafetyDetector may be supplied; "
                f"found {len(custom_dets)} instances"
            )
        if custom_dets:
            self.custom_policy_store = custom_dets[0].store
        else:
            self.custom_policy_store = CustomPolicyStore(initial=default_custom_policies())
            to_register.append(CustomSafetyDetector(self.custom_policy_store))
        for det in to_register:
            self.registry.add(det)

    @property
    def cost_usd(self) -> Decimal:
        return self.cost_tracker.total_spent()

    def check_budget(self) -> str | None:
        """Check if budget is exceeded. Returns error message or None."""
        cost = self.cost_usd
        if self.budget is not None and cost >= self.budget:
            return f"Total budget exceeded: ${cost:.4f} >= ${self.budget:.4f}"
        if self.budget_per_session is not None and cost >= self.budget_per_session:
            return f"Session budget exceeded: ${cost:.4f} >= ${self.budget_per_session:.4f}"
        return None

    @property
    def budget_remaining(self) -> float | None:
        """Remaining budget in USD, or None if no budget set."""
        if self.budget is not None:
            return float(self.budget - self.cost_usd)
        if self.budget_per_session is not None:
            return float(self.budget_per_session - self.cost_usd)
        return None

    @property
    def latest_enforcement(self) -> EnforcementDecision:
        if self.decisions:
            return self.decisions[-1]
        return EnforcementDecision(action="pass")

    @property
    def estimated_savings(self) -> Decimal:
        """Estimated cost saved by blocking calls before they reach the LLM."""
        if not self._call_costs:
            return Decimal("0")
        avg = Decimal(sum(self._call_costs) / len(self._call_costs))
        return avg * self.blocked_count

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
            "safety_enabled": self.safety_enabled,
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
            "budget": {
                "total": float(self.budget) if self.budget else None,
                "per_session": float(self.budget_per_session) if self.budget_per_session else None,
                "spent": float(self.cost_usd),
                "remaining": self.budget_remaining,
            }
            if (self.budget or self.budget_per_session)
            else None,
            "savings": {
                "blocked_calls": self.blocked_count,
                "estimated_savings_usd": float(self.estimated_savings),
                "avg_call_cost_usd": float(sum(self._call_costs) / len(self._call_costs))
                if self._call_costs
                else 0.0,
            },
            "attestation": None,  # populated by proxy when signing enabled
        }


def _default_proxy_detectors() -> Sequence[SafetyDetector]:
    """Default detectors for the proxy."""
    return (
        CostAnomalyDetector(),
        AgentThreatDetector(),
        PiiDetector(),
        ContentSafetyDetector(),
    )


def create_proxy_app(
    target_base_url: str = "https://api.openai.com",
    detectors: Sequence[SafetyDetector] | None = None,
    policy: EnforcementPolicy | dict[str, Any] | str | None = None,
    dashboard: bool = True,
    sign_key: Any | None = None,
    signer_id: str = "proxy:default",
    budget: float | None = None,
    budget_per_session: float | None = None,
    debug: bool = False,
) -> Starlette:
    """Create the AEP proxy ASGI app."""

    state = ProxyState(
        target_base_url=target_base_url,
        detectors=detectors,
        policy=policy,
        budget=budget,
        budget_per_session=budget_per_session,
    )

    # Enable debug logging if requested (see logutil; uvicorn/defaults drop DEBUG)
    if debug:
        configure_proxy_debug_logging()

    # Attestation engine (optional — enabled when sign_key is provided)
    attestation_engine = None
    if sign_key is not None:
        from ..attestation import AttestationEngine

        attestation_engine = AttestationEngine(_private_key=sign_key, signer_id=signer_id)

    # Override state.to_dict to include attestation data
    _orig_to_dict = state.to_dict

    def _state_with_attestation() -> dict[str, Any]:
        d = _orig_to_dict()
        if attestation_engine is not None:
            d["attestation"] = {
                "enabled": True,
                "signer_id": attestation_engine.signer_id,
                "chain_height": attestation_engine.chain_height,
                "latest_hash": (
                    attestation_engine.chain[-1]["chain_hash"] if attestation_engine.chain else None
                ),
            }
        return d

    # Replace the get_state callable used by dashboard
    _get_state = _state_with_attestation

    async def proxy_handler(request: Request) -> Response:
        """Forward request to target API with safety interception."""
        call_id = uuid.uuid4().hex[:8]
        path = request.url.path
        body_bytes = await request.body()

        # Debug logging for all requests (both input and output)
        if debug:
            try:
                body = json.loads(body_bytes) if body_bytes else {}
            except json.JSONDecodeError:
                body = {}
            log.debug(
                "REQUEST %s %s %s: headers=%s, body=%s",
                call_id,
                request.method,
                path,
                dict(request.headers),
                body,
            )

        # Parse request body
        try:
            body = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            body = {}

        # --- BUDGET CHECK ---
        budget_error = state.check_budget()
        if budget_error:
            _instance_id = os.environ.get("AEP_INSTANCE_ID", "")
            if _instance_id:
                await publish_event(
                    build_event(
                        instance_id=_instance_id,
                        event_type="budget_warning",
                        action="block",
                        message=budget_error,
                    )
                )
            return Response(
                json.dumps(
                    {
                        "error": {
                            "message": f"AEP budget: {budget_error}",
                            "type": "aep_budget_exceeded",
                            "code": "budget_exceeded",
                        }
                    }
                ),
                status_code=429,
                media_type="application/json",
            )

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

        if state.safety_enabled:
            input_signals = await state.registry.run_all(
                input_text=input_text,
                output_text="",
                call_id=call_id,
            )

            if len(input_signals) > 0:
                input_decision = evaluate(input_signals, state.policy)
                if input_decision.action == "block":
                    state.signals.extend(input_signals)
                    state.decisions.append(input_decision)
                    state.call_count += 1
                    state.blocked_count += 1
                    log.warning("BLOCKED request %s: %s", call_id, input_decision.reason)
                    _instance_id = os.environ.get("AEP_INSTANCE_ID", "")
                    if _instance_id:
                        _detector = input_signals[0].detector if input_signals else None
                        _severity = input_signals[0].severity if input_signals else None
                        await publish_event(
                            build_event(
                                instance_id=_instance_id,
                                event_type="safety_block",
                                action="block",
                                message=input_decision.reason or "Input blocked by safety filter",
                                detector=_detector,
                                severity=_severity,
                            )
                        )
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": {
                                "message": (
                                    f"AEP safety: request blocked — {input_decision.reason}"
                                ),
                                "type": "aep_safety_block",
                                "code": "safety_block",
                            }
                        },
                    )
        else:
            input_signals = ()

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
                cost_node = state.cost_tracker.record_llm_cost(
                    span_id=stream_span.span_id,
                    model=model,
                    usage=usage,
                )
                state.span_tracker.end_span(stream_span.span_id)
                state.call_count += 1
                state.signals.extend(signals)
                if decision:
                    state.decisions.append(decision)

                _instance_id = os.environ.get("AEP_INSTANCE_ID", "")
                if _instance_id and decision:
                    _event_type = "safety_block" if decision.action == "block" else "llm_call"
                    _detector = decision.signals[0].detector if decision.signals else None
                    _severity = decision.signals[0].severity if decision.signals else None
                    asyncio.ensure_future(
                        publish_event(
                            build_event(
                                instance_id=_instance_id,
                                event_type=_event_type,
                                action=decision.action,
                                message=decision.reason
                                or f"{model} streaming call: {inp} in, {out} out",
                                cost_usd=float(cost_node.compute_cost) if cost_node else 0,
                                model=model,
                                tokens_in=inp,
                                tokens_out=out,
                                detector=_detector,
                                severity=_severity,
                            )
                        )
                    )

            return await handle_streaming_request(
                target_url=target_url,
                body_bytes=body_bytes,
                headers=forward_headers,
                call_id=call_id,
                input_text=input_text,
                registry=state.registry,
                policy=state.policy,
                on_complete=on_stream_complete,
                debug=debug,
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
        state._call_costs.append(cost_node.compute_cost)
        state.span_tracker.end_span(span.span_id)
        state.call_count += 1

        # --- OUTPUT SAFETY CHECK ---
        if state.safety_enabled:
            output_signals = await state.registry.run_all(
                input_text=input_text,
                output_text=output_text,
                call_id=call_id,
                call_cost=cost_node.compute_cost,
            )

            all_signals = (*input_signals, *output_signals)
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
        else:
            all_signals = []
            decision = EnforcementDecision(action="pass")
            state.decisions.append(decision)

        # --- PUBLISH EVENT TO REDIS ---
        _instance_id = os.environ.get("AEP_INSTANCE_ID", "")
        if _instance_id:
            _event_type = "safety_block" if decision.action == "block" else "llm_call"
            _detector = decision.signals[0].detector if decision.signals else None
            _severity = decision.signals[0].severity if decision.signals else None
            await publish_event(
                build_event(
                    instance_id=_instance_id,
                    event_type=_event_type,
                    action=decision.action,
                    message=decision.reason
                    or f"{model} call: {input_tokens} in, {output_tokens} out",
                    cost_usd=float(cost_node.compute_cost) if cost_node else 0,
                    model=model,
                    tokens_in=input_tokens if usage else None,
                    tokens_out=output_tokens if usage else None,
                    detector=_detector,
                    severity=_severity,
                )
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

        # Sign verdict and add attestation headers (if signing enabled)
        if attestation_engine is not None:
            signal_dicts = [
                {"signal_type": s.signal_type, "severity": s.severity, "detail": s.detail}
                for s in all_signals
            ]
            attest_headers = attestation_engine.sign_verdict(
                call_id=call_id,
                action=decision.action,
                signals=signal_dicts,
                confidence=next(
                    (
                        s.score
                        for s in all_signals
                        if s.signal_type == "trust_engine" and s.score is not None
                    ),
                    None,
                ),
            )
            resp_headers.update(attest_headers)

        # Debug: log the response
        if debug:
            log.debug(
                (
                    "RESPONSE %s %s %s: model=%s, input_tokens=%d, output_tokens=%d, "
                    "status=%d, body=%s"
                ),
                call_id,
                request.method,
                path,
                model,
                input_tokens,
                output_tokens,
                upstream_resp.status_code,
                resp_data,
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

        dashboard_app = create_dashboard(get_state=_get_state)
        # Look up endpoints by path to avoid fragile positional indices
        dash_endpoints = {r.path: r.endpoint for r in dashboard_app.routes}  # type: ignore[union-attr]
        routes.extend(
            [
                Route("/aep/", dash_endpoints["/"]),
                Route("/aep/ciso", dash_endpoints["/ciso"]),
                Route("/aep/api/state", dash_endpoints["/api/state"]),
            ]
        )

    # Server-side feedback store — path is NOT user-configurable via API
    # to prevent path traversal attacks.
    from ..feedback import FeedbackStore

    _feedback_store = FeedbackStore("aep-feedback.jsonl")

    # Feedback API — POST /aep/api/feedback
    async def feedback_handler(request: Request) -> Response:
        """Record a signal verdict (confirmed/dismissed) from an operator."""
        try:
            body = await request.json()
        except Exception:
            return Response('{"error": "invalid JSON"}', status_code=400)

        signal_type = body.get("signal_type")
        verdict = body.get("verdict")
        if not signal_type or verdict not in ("confirmed", "dismissed"):
            return Response(
                '{"error": "signal_type and verdict (confirmed|dismissed) required"}',
                status_code=400,
            )

        v = _feedback_store.record(
            signal_type,
            score=body.get("score"),
            verdict=verdict,
            detail=body.get("detail", ""),
        )
        return Response(
            json.dumps({"ok": True, "verdict": v.to_dict()}),
            media_type="application/json",
        )

    # Feedback summary API — GET /aep/api/feedback/summary
    async def feedback_summary_handler(request: Request) -> Response:
        """Return feedback analysis and threshold recommendations."""
        from ..feedback import recommend_thresholds

        store = _feedback_store

        # Extract current thresholds from policy
        current_thresholds: dict[str, float | None] = {}
        for name, override in state.policy.overrides.items():
            current_thresholds[name] = override.threshold

        summary = recommend_thresholds(
            store,
            current_thresholds=current_thresholds,
            min_verdicts=int(request.query_params.get("min_verdicts", "5")),
        )

        recs = {}
        for sig_type, rec in summary.recommendations.items():
            recs[sig_type] = {
                "current_threshold": rec.current_threshold,
                "suggested_threshold": rec.suggested_threshold,
                "total_verdicts": rec.total_verdicts,
                "confirmed": rec.confirmed,
                "dismissed": rec.dismissed,
                "false_positive_rate": round(rec.false_positive_rate, 3),
                "reason": rec.reason,
            }

        return Response(
            json.dumps({"total_verdicts": summary.total_verdicts, "recommendations": recs}),
            media_type="application/json",
        )

    # Safety toggle API — GET/POST /aep/api/safety
    async def safety_toggle_handler(request: Request) -> Response:
        """Toggle safety on/off or swap the enforcement policy at runtime.

        GET returns current state. POST accepts {"enabled": bool} and/or {"policy": dict}.
        """
        if request.method == "GET":
            return Response(
                json.dumps(
                    {
                        "safety_enabled": state.safety_enabled,
                        "policy": {
                            "default_action": state.policy.default_action,
                            "block_on": sorted(state.policy.block_on),
                            "flag_on": sorted(state.policy.flag_on),
                            "detectors": {
                                k: {
                                    "action": v.action,
                                    "threshold": v.threshold,
                                    "enabled": v.enabled,
                                    **({"extra": dict(v.extra)} if v.extra else {}),
                                }
                                for k, v in state.policy.overrides.items()
                            },
                        },
                    }
                ),
                media_type="application/json",
            )

        try:
            body = await request.json()
        except Exception:
            return Response('{"error": "invalid JSON"}', status_code=400)

        # Toggle safety enabled/disabled
        if "enabled" in body:
            state.safety_enabled = bool(body["enabled"])
            log.info("Safety %s", "enabled" if state.safety_enabled else "disabled")

        # Hot-swap policy from inline dict (not file paths — prevents path traversal)
        if "policy" in body:
            policy_data = body["policy"]
            if isinstance(policy_data, dict):
                state.policy = EnforcementPolicy.from_dict(policy_data)
                log.info("Policy swapped at runtime")
            else:
                return Response(
                    '{"error": "policy must be a JSON object, not a file path"}',
                    status_code=400,
                )

        return Response(
            json.dumps(
                {
                    "safety_enabled": state.safety_enabled,
                    "policy": {
                        "default_action": state.policy.default_action,
                        "block_on": sorted(state.policy.block_on),
                        "flag_on": sorted(state.policy.flag_on),
                        "detectors": {
                            k: {
                                "action": v.action,
                                "threshold": v.threshold,
                                "enabled": v.enabled,
                                **({"extra": dict(v.extra)} if v.extra else {}),
                            }
                            for k, v in state.policy.overrides.items()
                        },
                    },
                }
            ),
            media_type="application/json",
        )

    def _normalize_policy_id(policy_id: str) -> str | None:
        try:
            return str(uuid.UUID(policy_id))
        except ValueError:
            return None

    # Custom policies CRUD (collection + item routes below).
    async def custom_policies_collection_handler(request: Request) -> Response:
        if request.method == "GET":
            policies = [p.model_dump() for p in state.custom_policy_store.all()]
            return Response(
                json.dumps({"policies": policies}),
                media_type="application/json",
            )

        body, json_err = await _read_json_body(request)
        if json_err is not None:
            return json_err
        fields = _parse_custom_policy_write_fields(body)
        if isinstance(fields, JSONResponse):
            return fields
        policy = _custom_policy_from_write_fields(fields, policy_id=None)
        if isinstance(policy, JSONResponse):
            return policy
        state.custom_policy_store.upsert(policy)
        return Response(
            json.dumps(policy.model_dump()),
            status_code=201,
            media_type="application/json",
        )

    async def custom_policy_item_handler(request: Request) -> Response:
        raw_id = request.path_params["policy_id"]
        policy_id = _normalize_policy_id(raw_id)
        if policy_id is None:
            return JSONResponse({"error": "invalid policy id"}, status_code=400)

        if request.method == "GET":
            policy = state.custom_policy_store.get(policy_id)
            if policy is None:
                return JSONResponse({"error": "custom policy not found"}, status_code=404)
            return Response(json.dumps(policy.model_dump()), media_type="application/json")

        if request.method == "DELETE":
            if state.custom_policy_store.get(policy_id) is None:
                return JSONResponse({"error": "custom policy not found"}, status_code=404)
            state.custom_policy_store.delete(policy_id)
            return Response(status_code=204)

        # PUT — replace name, rule, enabled (id fixed from URL)
        body, json_err = await _read_json_body(request)
        if json_err is not None:
            return json_err
        fields = _parse_custom_policy_write_fields(body)
        if isinstance(fields, JSONResponse):
            return fields
        if state.custom_policy_store.get(policy_id) is None:
            return JSONResponse({"error": "custom policy not found"}, status_code=404)
        updated = _custom_policy_from_write_fields(fields, policy_id=policy_id)
        if isinstance(updated, JSONResponse):
            return updated
        state.custom_policy_store.upsert(updated)
        return Response(json.dumps(updated.model_dump()), media_type="application/json")

    routes.extend(
        [
            Route("/aep/api/feedback", feedback_handler, methods=["POST"]),
            Route("/aep/api/feedback/summary", feedback_summary_handler, methods=["GET"]),
            Route("/aep/api/safety", safety_toggle_handler, methods=["POST", "GET"]),
            Route(
                "/aep/api/custom-policies/{policy_id}",
                custom_policy_item_handler,
                methods=["GET", "PUT", "DELETE"],
            ),
            Route(
                "/aep/api/custom-policies",
                custom_policies_collection_handler,
                methods=["GET", "POST"],
            ),
        ]
    )

    # Mount MCP gateway if fastmcp is installed
    mcp_http_app = None
    try:
        from ..mcp_gateway import create_mcp_app

        mcp_http_app = create_mcp_app(state)
        if mcp_http_app is not None:
            from starlette.routing import Mount

            routes.append(Mount("/mcp", app=mcp_http_app))
            log.info("MCP gateway mounted at /mcp/")
    except Exception as exc:
        log.debug("MCP gateway not available: %s", exc)

    # Wire MCP lifespan into parent app (required by FastMCP for task group init)
    if mcp_http_app is not None and hasattr(mcp_http_app, "lifespan"):
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def lifespan(app):
            async with mcp_http_app.lifespan(app):
                yield

        return Starlette(routes=routes, lifespan=lifespan)

    return Starlette(routes=routes)


__all__ = ["ProxyState", "create_proxy_app"]
