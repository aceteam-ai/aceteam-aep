# MCP Gateway On-Ramp Design

## Problem

Workshop attendees learn SafeClaw, go home, and never use it again. There's no sticky daily-use product in the middle. The funnel is broken at every step: discovery, onboarding, and ongoing value.

## Solution

The AEP proxy becomes a unified gateway: LLM safety proxy + MCP server + dashboard, all on one port, one process. Users start with fully local, zero-account safety. As they connect to AceTeam, more capabilities unlock automatically — no new commands, no new config.

## Architecture

```
Claude Code / OpenClaw / SafeClaw / Any MCP Client
         │                    │
         │ MCP (tools)        │ OpenAI-compatible API
         ▼                    ▼
    ┌─────────────────────────────────┐
    │        AEP Gateway              │
    │    (one process, one port)      │
    │                                 │
    │  /v1/*     → LLM proxy          │  ← safety detectors, cost tracking
    │  /aep/*    → dashboard + API    │  ← toggle, signals, policy
    │  /mcp/*    → MCP endpoint       │  ← tiered tools (local → connected → platform)
    │                                 │
    │  ┌─────────┐  ┌──────────────┐  │
    │  │ Local   │  │ AceTeam      │  │
    │  │ LLM     │  │ Backend      │  │
    │  │(Ollama/ │  │(aceteam.ai)  │  │
    │  │ vLLM)   │  │              │  │
    │  └─────────┘  └──────────────┘  │
    └─────────────────────────────────┘
```

### One port, everything

| Path | What | Available |
|------|------|-----------|
| `/v1/chat/completions` | LLM proxy (OpenAI-compatible) | Always |
| `/aep/` | Dashboard (developer + executive) | Always |
| `/aep/api/state` | Proxy state (cost, signals, spans) | Always |
| `/aep/api/safety` | Safety toggle + policy swap | Always |
| `/mcp/` | MCP endpoint (streamable-http) | Always |

### Client configuration

**Claude Code** (`~/.claude.json` or project config):
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

**OpenClaw / SafeClaw:**
```bash
export OPENAI_BASE_URL=http://localhost:8899/v1
```

Both connect to the same gateway. MCP gives Claude Code access to tools. The proxy gives any agent safety + cost tracking.

## Three Tiers

### Tier 1 — Local (free, no account, fully offline)

**Who:** Anyone who starts the proxy. Workshop attendees on day one.

**Setup:** Start the proxy. That's it.
```bash
docker run -p 8899:8899 ghcr.io/aceteam-ai/aep-proxy
# or
pip install aceteam-aep[all] && aceteam-aep proxy
```

**First-run experience:** Dashboard opens with a setup prompt:
- "Configure your LLM provider" — paste an OpenAI/Anthropic key, OR point at a local service (Ollama at localhost:11434, vLLM, etc.)
- Keys stored locally in the proxy, never sent to AceTeam
- Optional: proxy starts a local Ollama container automatically if Docker is available

**MCP tools available:**

| Tool | What it does |
|------|-------------|
| `check_safety` | Run safety detectors on text. Returns PASS/FLAG/BLOCK with signals. |
| `get_safety_status` | Session metrics: calls checked, signals raised, cost. |
| `set_safety_policy` | Toggle detectors on/off, set thresholds. |
| `get_cost_summary` | Cost breakdown by model, by call. |

**What the user gets:** Every LLM call through their agent is tracked (cost), screened (safety), and logged (audit). Dashboard shows it all in real time. Zero account, zero cloud dependency.

**Upgrade nudge:** When a user tries to use a tool that requires Tier 2, the tool returns:
> "This feature requires an AceTeam account (free, $5 credit included). Run `aceteam-aep connect` or visit aceteam.ai to sign up."

### Tier 2 — Connected (free AceTeam account, $5 credit)

**Who:** User creates an AceTeam account and connects the proxy.

**Setup:**
```bash
aceteam-aep connect
# Opens browser → sign up / log in → API key generated automatically
# Key stored locally, proxy now connected to AceTeam backend
```

Or from the dashboard: "Connect to AceTeam" button → same browser flow.

**What changes:**
- Proxy can now use AceTeam's API keys for LLM calls (user doesn't need their own OpenAI key)
- User picks a model from AceTeam's supported models (GPT-4o, Claude, Gemini, etc.)
- $5 free credit — enough for hundreds of gpt-4o-mini calls
- MCP tools unlock workflow capabilities

**MCP tools added (in addition to Tier 1):**

| Tool | What it does |
|------|-------------|
| `create_workflow` | Build an AceFlow from a graph definition |
| `run_workflow` | Execute a saved workflow |
| `list_workflows` | Browse your workflows |
| `search_node_types` | Discover available node types (40+ types) |
| `get_node_schema` | Detailed node documentation with examples |
| `update_workflow` | Save a new version of a workflow |

**How AEP connects to workflows:** Every workflow run goes through the AceTeam backend, which already has full AEP integration (AceTeamContext tracks spans, costs, citations per node). The dashboard shows workflow execution costs and safety signals alongside regular LLM calls.

**API key negotiation:** The `connect` flow:
1. Browser auth → OAuth token
2. Proxy exchanges OAuth token for a long-lived API key (`act_...`)
3. API key stored locally in `~/.config/aceteam-aep/credentials.json`
4. Proxy uses API key for all AceTeam backend calls
5. Key auto-refreshes when needed

**Model selection:** Dashboard shows a model picker. User selects from AceTeam-supported models. Proxy routes LLM calls to AceTeam's API (which has the provider keys), deducting from the user's credit balance.

### Tier 3 — Platform (AceTeam power user)

**Who:** Users who want the full AceTeam platform: custom agents, knowledge base, document management, team collaboration.

**Setup:** Already connected from Tier 2. Additional tools appear automatically when the user's AceTeam org has agents, documents, etc.

**MCP tools added (in addition to Tier 2):**

| Tool | What it does |
|------|-------------|
| `list_agents` | Browse AI agents in your org |
| `create_agent` | Create a new agent with system prompt + model |
| `chat_with_agent` | Invoke an agent |
| `search_knowledge_base` | Semantic search across documents |
| `list_documents` | Browse uploaded documents |
| `list_mcp_servers` | See connected MCP servers |
| `link_tool_to_agent` | Give an agent access to a tool |

**This is the existing aceteam MCP server, proxied through the gateway.**

## Tier Discovery (Dynamic Tool Registration)

The MCP endpoint dynamically registers tools based on what's available:

```python
# Pseudocode for tool registration
tools = []

# Tier 1 — always available
tools += [check_safety, get_safety_status, set_safety_policy, get_cost_summary]

# Tier 2 — if AceTeam API key is configured
if has_aceteam_credentials():
    tools += [create_workflow, run_workflow, list_workflows, ...]

# Tier 3 — if AceTeam org has agents/docs
if has_aceteam_credentials() and org_has_platform_features():
    tools += [list_agents, create_agent, chat_with_agent, ...]
```

When a user runs `aceteam-aep connect`, the MCP server restarts and Claude Code sees the new tools on next reconnect.

## Backend MCP Server Redesign

The current `aceteam` and `aceflows` MCP servers on the Python backend stay as separate implementations, but their tools are reorganized by tier:

| Current Server | Current Mount | New Role |
|----------------|---------------|----------|
| `aceflows` | `/aceflows/` | Tier 2 tools — workflow CRUD + execution + schema discovery |
| `aceteam` | `/aceteam/` | Tier 3 tools — agent, document, knowledge base, MCP server management |

The gateway proxies requests to the appropriate backend server based on the tool being called. The backend servers don't change — the gateway is the new routing layer.

## Safety Integration

**Tier 1:** Regex-based detectors (agent_threat, pii, cost_anomaly) run locally. Free, fast, no cloud.

**Tier 2:** Trust Engine available as an upgrade. Uses AceTeam-hosted models for calibrated multi-perspective detection. Deducts from user's credit balance. The dashboard shows: "Your detectors caught 3 threats. Upgrade to Trust Engine for 5-category detection with confidence scores."

**Tier 3:** Per-org safety policies. Custom dimensions. YAML policy management. Compliance reporting (CISO dashboard).

**Every workflow run (Tier 2+) automatically has AEP:** The AceTeamContext in the workflow engine already tracks spans, costs, and citations per node. The gateway dashboard shows workflow execution alongside regular LLM calls.

## First-Run Experience

```
$ docker run -p 8899:8899 ghcr.io/aceteam-ai/aep-proxy

  AEP Gateway
  ─────────────────────────────────────
  Dashboard:    http://localhost:8899/aep/
  LLM Proxy:    http://localhost:8899/v1
  MCP Endpoint: http://localhost:8899/mcp/

  → Open the dashboard to configure your LLM provider.
  → Add to Claude Code:
    {"mcpServers": {"aceteam": {"type": "streamable-http", "url": "http://localhost:8899/mcp/"}}}
```

Dashboard first-run shows:
1. "Welcome to SafeClaw" — configure LLM provider (OpenAI key, Anthropic key, or local Ollama)
2. After config: "Safety is ON. Make your first call."
3. After first call: "Your agent made 1 call. $0.003 cost. PASS."
4. Sidebar nudge: "Connect to AceTeam for workflows, $5 free credit →"

## Technical Integration

### MCP Transport

The existing `mcp.py` is a hand-rolled stdio JSON-RPC loop (stdin/stdout). This cannot be mounted on the proxy. The gateway MCP endpoint requires **Streamable HTTP transport** — a new implementation.

**Approach:** Create a new FastMCP server instance, register tools as decorated functions, and mount its ASGI app on the existing Starlette proxy:

```python
from fastmcp import FastMCP
from starlette.routing import Mount

# Create FastMCP server with Tier 1 tools
gateway_mcp = FastMCP("aceteam-gateway")

@gateway_mcp.tool()
def check_safety(text: str) -> str:
    """Run safety detectors on text."""
    # Delegates to state.registry.run_all()
    ...

# Mount on existing Starlette app
routes.append(Mount("/mcp", app=gateway_mcp.streamable_http_app()))
```

FastMCP's `streamable_http_app()` returns a Starlette ASGI app that handles `POST /mcp/` and `GET /mcp/` with SSE semantics. The `Mount` strips the `/mcp` prefix before forwarding. The existing stdio MCP server (`aceteam-aep mcp-server`) remains as a separate entry point for clients that prefer stdio transport.

### Shared State

MCP tool handlers need access to `ProxyState` (the proxy's in-memory state for cost, signals, detectors). The FastMCP server is created inside `create_proxy_app()` where `state` is in scope, so tool functions are closures over `state`:

```python
def create_proxy_app(state: ProxyState, ...):
    gateway_mcp = FastMCP("aceteam-gateway")

    @gateway_mcp.tool()
    def get_safety_status() -> str:
        return json.dumps({
            "calls": state.call_count,
            "cost": float(state.cost_usd),
            "signals": len(state.signals),
            "safety_enabled": state.safety_enabled,
        })
```

### New vs Existing Tools

Of the 4 Tier 1 tools, 2 exist in `mcp.py` (`check_safety`, `get_safety_status`) and 2 are net-new:
- `set_safety_policy` — backs onto the existing `POST /aep/api/safety` handler logic
- `get_cost_summary` — reads from `ProxyState.cost_tracker` and `_call_costs`

### Auth Model

- **Tier 1 (local):** No auth. Gateway binds to `127.0.0.1` by default. Localhost is trusted.
- **Tier 2+ (network-exposed):** Bearer token auth on `/mcp/` using the AceTeam API key (`act_...`). The gateway validates the key against AceTeam backend before proxying requests.

### Credential Storage (Phase 2)

`~/.config/aceteam-aep/credentials.json` with `0600` file permissions. For Docker: volume-mount `~/.config/aceteam-aep/` or pass `ACETEAM_API_KEY` env var. Token refresh: on-demand when a 401 is received from the backend (re-auth via stored refresh token, or prompt user to `aceteam-aep connect` again).

### Graceful Degradation (Phase 2+)

When AceTeam backend is unreachable, Tier 2+ tools return an error message and the MCP tool list falls back to Tier 1 tools only. The dashboard shows a "Disconnected from AceTeam — local safety active" banner. Reconnects automatically when the backend is reachable again.

### Cross-Repo Ownership

- **`aceteam-aep` (this repo):** Gateway MCP endpoint, Tier 1 tools, proxy, dashboard, CLI
- **`aceteam-3` (AceTeam platform):** Backend MCP servers (`aceflows`, `aceteam`), workflow engine, agent runtime. The gateway is an HTTP client to these servers for Tier 2+ tools.

## Implementation Phases

**Phase 1 (for ClawCamp April 16) — ~2-3 days:**
- Create FastMCP server with 4 Tier 1 tools backed by `ProxyState`
- Mount Streamable HTTP ASGI app at `/mcp/` on existing Starlette proxy
- Update CLI startup banner to show MCP endpoint URL
- Add Claude Code config to workshop materials
- Main risk: FastMCP Streamable HTTP + Starlette path prefix integration

**Phase 2 (post-ClawCamp):**
- `aceteam-aep connect` flow (browser auth → API key → credential storage)
- Tier 2 tools (proxy AceFlows MCP calls to AceTeam backend)
- Dashboard "Connect to AceTeam" button + model picker
- Rate limiting for credit-consuming tiers
- Tool list refresh on connect (requires MCP session reconnect)

**Phase 3 (post-funding):**
- Tier 3 tools (proxy aceteam MCP for agents, knowledge base, documents)
- Local Ollama auto-start (detect Docker, pull model, start container)
- Trust Engine upgrade nudge in dashboard
- Credit tracking and usage dashboard
