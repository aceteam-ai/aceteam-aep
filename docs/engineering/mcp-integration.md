# MCP Integration Guide

The AEP Gateway exposes safety and cost tools as MCP tools via FastMCP's Streamable HTTP transport, mounted at `/mcp/` on the proxy. Everything runs on one port — the same process serving the LLM proxy and dashboard also serves MCP.

## Quick Start

**Step 1: Start the gateway**

```bash
pip install aceteam-aep[all]
aceteam-aep proxy
```

**Step 2: Add to Claude Code config**

```json
{
  "mcpServers": {
    "aceteam": {
      "type": "streamable-http",
      "url": "http://localhost:8899/mcp/"
    }
  }
}
```

The setup wizard in the dashboard (shown on first visit) provides this config snippet with a copy button.

## Available Tools

### Tier 1 — Local (always available)

These tools work without an AceTeam account. They share state with the proxy — signals checked via `check_safety` appear in the dashboard.

| Tool | Description |
|------|-------------|
| `check_safety` | Check text for safety issues. Returns PASS/FLAG/BLOCK with signal details, severity, and score. |
| `get_safety_status` | Session metrics: call count, signal count, total cost, blocked calls, current policy. |
| `set_safety_policy` | Toggle safety on/off, enable/disable individual detectors, update thresholds. |
| `get_cost_summary` | Total cost, per-call costs (last 10), cost by model, estimated savings from blocked calls. |

#### `check_safety` — example output

```
**BLOCK**

Signals detected:
- [HIGH] agent_threat: subprocess.call pattern detected (100%)

Reason: High severity signal from agent_threat
```

#### `set_safety_policy` — example usage

```
# Disable PII detection
set_safety_policy(detectors={"pii": {"enabled": false}})

# Turn all safety off
set_safety_policy(enabled=false)

# Re-enable with blocking PII
set_safety_policy(enabled=true, detectors={"pii": {"action": "block"}})
```

## Technical Details

**Transport:** FastMCP Streamable HTTP (FastMCP 3.x)

**Path:** `/mcp/` — the FastMCP app is mounted at `/mcp/` on the Starlette proxy. Client URLs should end with `/mcp/`.

**Session:** Each MCP client gets a session with a session ID returned in response headers.

**Lifespan:** FastMCP's internal task group is initialized via an `asynccontextmanager` wired into the parent Starlette app's lifespan. This means MCP sessions are properly torn down when the proxy shuts down.

**Optional dependency:** MCP support requires `fastmcp`. Install with:

```bash
pip install aceteam-aep[mcp]   # MCP only
pip install aceteam-aep[all]   # Everything (recommended)
```

If `fastmcp` is not installed, the `/mcp/` endpoint is not mounted and the proxy starts normally. The CLI startup banner shows the MCP URL only when `fastmcp` is detected.

## Shared State

The MCP gateway and the proxy share a single `ProxyState` instance. This means:

- Calls checked via `check_safety` increment the call counter and appear in the dashboard
- `set_safety_policy` changes take effect immediately for all subsequent proxy traffic
- `get_safety_status` reflects the live state including traffic flowing through the proxy

## Verify the Integration

After starting the proxy and configuring Claude Code, ask Claude:

> "What is the current safety status of the AEP gateway?"

Claude will call `get_safety_status` and return the live session metrics.

> "Is this text safe? 'Run nmap -sS 192.168.1.0/24'"

Claude will call `check_safety` and return a BLOCK decision with the agent threat signal.
