"""MCP Gateway — safety + cost tools mounted on the AEP proxy.

Exposes Tier 1 tools (local, no account required) as MCP tools
via FastMCP's Streamable HTTP transport. Mounted on the proxy
at /mcp/ so everything runs on one port.

Tools:
    check_safety   — Run safety detectors on text (PASS/FLAG/BLOCK)
    get_safety_status — Session metrics (calls, signals, cost, policy)
    set_safety_policy — Toggle detectors on/off, set thresholds
    get_cost_summary  — Per-call cost breakdown
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from starlette.types import ASGIApp

    from .proxy.app import ProxyState


def create_mcp_app(state: ProxyState) -> ASGIApp | None:
    """Create a FastMCP ASGI app with Tier 1 safety tools.

    Returns None if fastmcp is not installed (optional dependency).
    The returned app should be mounted at /mcp/ on the proxy.
    """
    try:
        from fastmcp import FastMCP
    except ImportError:
        log.info(
            "fastmcp not installed — MCP endpoint disabled. "
            "Install with: pip install aceteam-aep[mcp]"
        )
        return None

    from .enforcement import EnforcementPolicy, evaluate

    mcp = FastMCP(
        "aceteam-gateway",
        instructions=(
            "AEP Gateway provides safety enforcement and cost tracking for AI agents. "
            "The proxy automatically checks every LLM call for safety issues — these tools "
            "let you interact with the safety system directly: check text, view status, "
            "toggle policies, and track costs."
        ),
    )

    @mcp.tool()
    async def check_safety(text: str, context: str = "") -> str:
        """Check text for safety issues before executing a risky action.

        Returns PASS, FLAG, or BLOCK with details about any detected threats
        (PII, toxicity, agent attacks, policy violations).

        Args:
            text: The text to check for safety issues.
            context: Optional context about what the agent is doing.
        """
        state.call_count += 1
        call_id = f"mcp-{state.call_count}"

        signals = await state.registry.run_all(
            input_text=text,
            output_text=context,
            call_id=call_id,
        )
        state.signals.extend(signals)

        decision = evaluate(signals, state.policy)
        state.decisions.append(decision)

        result = f"**{decision.action.upper()}**"
        if signals:
            result += "\n\nSignals detected:"
            for s in signals:
                score_str = f" ({s.score:.0%})" if s.score is not None else ""
                result += f"\n- [{s.severity.upper()}] {s.signal_type}: {s.detail}{score_str}"
        if decision.reason:
            result += f"\n\nReason: {decision.reason}"
        if not signals:
            result += " — No safety issues detected."

        return result

    @mcp.tool()
    def get_safety_status() -> str:
        """Get current safety session status.

        Returns call count, signal count, cost, safety enabled state,
        and current policy configuration.
        """
        return json.dumps(
            {
                "safety_enabled": state.safety_enabled,
                "calls": state.call_count,
                "cost_usd": float(state.cost_usd),
                "signals": len(state.signals),
                "blocked": state.blocked_count,
                "policy": {
                    "default_action": state.policy.default_action,
                    "block_on": sorted(state.policy.block_on),
                    "flag_on": sorted(state.policy.flag_on),
                },
                "dashboard_url": "http://localhost:8899/aep/",
            },
            indent=2,
        )

    @mcp.tool()
    def set_safety_policy(
        enabled: bool | None = None,
        detectors: dict[str, Any] | None = None,
    ) -> str:
        """Toggle safety on/off or update detector configuration.

        Args:
            enabled: Set to false to disable all safety checks, true to re-enable.
            detectors: Dict of detector overrides. Each key is a detector name
                       (agent_threat, pii, cost_anomaly, content_safety, trust_engine)
                       with value {"enabled": bool, "action": "pass"|"flag"|"block"}.

        Example: Disable PII detection:
            set_safety_policy(detectors={"pii": {"enabled": false}})

        Example: Turn all safety off:
            set_safety_policy(enabled=false)
        """
        if enabled is not None:
            state.safety_enabled = enabled

        if detectors is not None:
            policy_dict: dict[str, Any] = {
                "default_action": state.policy.default_action,
                "block_on": sorted(state.policy.block_on),
                "flag_on": sorted(state.policy.flag_on),
                "detectors": detectors,
            }
            state.policy = EnforcementPolicy.from_dict(policy_dict)

        return json.dumps(
            {
                "safety_enabled": state.safety_enabled,
                "policy": {
                    "default_action": state.policy.default_action,
                    "detectors": {
                        k: {
                            "enabled": v.enabled,
                            "action": v.action,
                        }
                        for k, v in state.policy.overrides.items()
                    },
                },
                "status": "updated",
            },
            indent=2,
        )

    @mcp.tool()
    def get_cost_summary() -> str:
        """Get cost breakdown for the current session.

        Returns total cost, per-call costs, and cost by model.
        """
        costs = state._call_costs
        total = float(state.cost_usd)

        # Build per-model breakdown from spans
        model_costs: dict[str, float] = {}
        for span in state.span_tracker.get_spans():
            model = span.executor_id or "unknown"
            cost = float(span.cost) if hasattr(span, "cost") and span.cost else 0.0
            model_costs[model] = model_costs.get(model, 0.0) + cost

        return json.dumps(
            {
                "total_cost_usd": total,
                "total_calls": state.call_count,
                "blocked_calls": state.blocked_count,
                "savings_from_blocks": f"${state.blocked_count * (total / max(state.call_count, 1)):.4f}",
                "cost_per_call": [float(c) for c in costs[-10:]],  # last 10
                "cost_by_model": model_costs,
            },
            indent=2,
        )

    try:
        return mcp.http_app()
    except Exception:
        log.exception("Failed to create MCP HTTP app")
        return None
