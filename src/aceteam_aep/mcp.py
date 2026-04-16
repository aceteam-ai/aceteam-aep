"""AEP MCP Server — safety enforcement as an MCP tool.

Registers as an MCP server that provides safety checking as a tool.
Any MCP-compatible agent (Claude Code, OpenClaw, Codex) gets safety
enforcement by adding one config line.

Usage in claude_desktop_config.json or OpenClaw settings::

    {
      "mcpServers": {
        "aep-safety": {
          "command": "aceteam-aep",
          "args": ["mcp-server"]
        }
      }
    }

Or with a policy::

    {
      "mcpServers": {
        "aep-safety": {
          "command": "aceteam-aep",
          "args": ["mcp-server", "--policy", "strict.yaml"]
        }
      }
    }

The server exposes one tool: ``check_safety`` which evaluates text
for safety signals and returns PASS/FLAG/BLOCK verdicts.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aceteam_aep.enforcement import EnforcementPolicy
    from aceteam_aep.safety.base import DetectorRegistry

log = logging.getLogger(__name__)


def _build_detectors(policy_path: str | None = None) -> tuple[DetectorRegistry, EnforcementPolicy]:
    """Build detectors and policy from optional policy path."""
    from .enforcement import EnforcementPolicy, build_detectors_from_policy
    from .safety.base import DetectorRegistry

    if policy_path:
        policy = EnforcementPolicy.from_yaml(policy_path)
        detectors = build_detectors_from_policy(policy)
    else:
        from .safety.agent_threat import AgentThreatDetector
        from .safety.cost_anomaly import CostAnomalyDetector
        from .safety.pii import PiiDetector

        policy = EnforcementPolicy()
        detectors = [CostAnomalyDetector(), AgentThreatDetector(), PiiDetector()]

    registry = DetectorRegistry()
    for det in detectors:
        registry.add(det)

    return registry, policy


async def run_mcp_server(policy_path: str | None = None) -> None:
    """Run the AEP safety MCP server using stdin/stdout JSON-RPC."""
    registry, policy = _build_detectors(policy_path)

    from .enforcement import evaluate

    # MCP uses JSON-RPC 2.0 over stdin/stdout
    tools = [
        {
            "name": "check_safety",
            "description": (
                "Check text for safety issues. Returns PASS, FLAG, or BLOCK "
                "with details about any detected threats (PII, toxicity, "
                "agent attacks, policy violations). Use before executing "
                "any potentially risky action."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to check for safety issues",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context (e.g. 'agent wants to execute this code')",
                    },
                },
                "required": ["text"],
            },
        },
        {
            "name": "get_safety_status",
            "description": (
                "Get current safety session status: total calls checked, "
                "signals raised, current enforcement action."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
    ]

    call_count = 0
    all_signals: list[Any] = []

    async def handle_request(req: dict[str, Any]) -> dict[str, Any]:
        nonlocal call_count

        method = req.get("method", "")
        req_id = req.get("id")
        params = req.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "aep-safety",
                        "version": "0.5.6",
                    },
                },
            }

        if method == "notifications/initialized":
            return {}  # no response needed

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": tools},
            }

        if method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})

            if tool_name == "check_safety":
                text = arguments.get("text", "")
                context = arguments.get("context", "")
                call_id = f"mcp-{call_count}"
                call_count += 1

                signals = await registry.run_all(
                    input_text=text,
                    output_text=context,
                    call_id=call_id,
                )
                all_signals.extend(signals)

                decision = evaluate(signals, policy)

                result_text = f"**{decision.action.upper()}**"
                if signals:
                    result_text += "\n\nSignals detected:"
                    for s in signals:
                        result_text += f"\n- [{s.severity.upper()}] {s.signal_type}: {s.detail}"
                if decision.reason:
                    result_text += f"\n\nReason: {decision.reason}"
                if not signals:
                    result_text += " — No safety issues detected."

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": result_text}],
                    },
                }

            if tool_name == "get_safety_status":
                status = (
                    f"AEP Safety Status:\n"
                    f"- Calls checked: {call_count}\n"
                    f"- Total signals: {len(all_signals)}\n"
                    f"- Policy: {'custom' if policy_path else 'default'}"
                )
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": status}],
                    },
                }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }

        # Unknown method
        if req_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            }
        return {}

    # Main loop: read JSON-RPC from stdin, write to stdout
    log.info("AEP MCP server starting (stdin/stdout)")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = await handle_request(req)
            if resp:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
        except Exception as e:
            log.error("MCP error: %s", e)
            error_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": str(e)},
            }
            sys.stdout.write(json.dumps(error_resp) + "\n")
            sys.stdout.flush()
