# MCP Gateway On-Ramp Design

## Problem

Workshop attendees learn SafeClaw, go home, and never use it again. There's no sticky daily-use product in the middle. The funnel is broken at every step: discovery, onboarding, and ongoing value.

## Solution

The AEP proxy becomes a unified gateway: LLM safety proxy + MCP server + dashboard, all on one port, one process, in a container. Users start with fully local, zero-account safety. As they connect to AceTeam, more capabilities unlock automatically — no new commands, no new config.

**Key principle: the proxy is the product, MCP tools are supplementary.** Setting `OPENAI_BASE_URL=http://localhost:8899/v1` gives every agent safety + cost tracking automatically. MCP tools let power users interact with safety from within their agent (toggle policies, check status, build workflows). But the proxy delivers value even if the user never touches MCP.

## Architecture

```
Claude Code / OpenClaw / SafeClaw / Any Agent
         │                    │
         │ MCP (tools)        │ OpenAI-compatible API
         ▼                    ▼
    ┌─────────────────────────────────┐
    │      AEP Gateway (container)    │
    │      one process, one port      │
    │                                 │
    │  /v1/*     → LLM proxy          │  ← safety detectors, cost tracking
    │  /dashboard/*    → dashboard + API    │  ← toggle, signals, policy, setup
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
| `/dashboard/` | Dashboard (developer + executive + setup wizard) | Always |
| `/dashboard/api/state` | Proxy state (cost, signals, spans) | Always |
| `/dashboard/api/safety` | Safety toggle + policy swap | Always |
| `/mcp/` | MCP endpoint (streamable-http) | Always |

## Getting Started

### Opinionated default: container

```bash
podman run -p 8899:8899 ghcr.io/aceteam-ai/aep-proxy
# or
docker run -p 8899:8899 ghcr.io/aceteam-ai/aep-proxy
```

Container is the default because:
- **Safer:** No host filesystem access. Mailbox pattern for file exchange.
- **Simpler:** One command, everything bundled. No Python version issues.
- **Consistent:** Same environment on Mac, Windows, Linux.

### Escape hatch: pip install

```bash
pip install aceteam-aep[all]
aceteam-aep setup
```

The `setup` command:
1. Detects Podman or Docker → pulls and starts container with env vars forwarded
2. If no container runtime → starts proxy natively as background process
3. Detects Claude Code → writes MCP config automatically
4. Adds `export OPENAI_BASE_URL=http://localhost:8899/v1` to shell profile
5. Opens dashboard in browser

`--print-config` shows all config without writing. `--no-shell` skips shell profile modification.

## First-Run Experience

```
$ podman run -p 8899:8899 ghcr.io/aceteam-ai/aep-proxy

  SafeClaw Gateway
  ─────────────────────────────────────
  Dashboard:    http://localhost:8899/dashboard/
  LLM Proxy:    http://localhost:8899/v1
  MCP Endpoint: http://localhost:8899/mcp/

  → Open the dashboard to get started.
```

### Dashboard setup wizard

The dashboard first-run shows a setup wizard (not a wall of config):

**Step 1: "How do you want to use LLMs?"**

Two equal buttons:
- **"I have API keys"** → paste OpenAI/Anthropic key. Auto-detects keys from env vars (`-e OPENAI_API_KEY=...` passed to container) and shows "Using OpenAI key from environment ✓"
- **"Use AceTeam (free, $5 credit)"** → "Connect to AceTeam" button opens browser OAuth. Works from dashboard (no CLI needed, no Docker escape required). After auth: proxy has AceTeam API key, user picks a model, $5 credit ready.

Both paths end at the same place: proxy configured, safety on, ready for first call.

**Step 2: "Point your agent at the proxy"**

Shows the one line to add:
```bash
export OPENAI_BASE_URL=http://localhost:8899/v1
```

With copy button. And: "Using Claude Code? Add this to your config:" with the MCP JSON and a copy button. Links to exact file path for each platform.

**Step 3: "Make your first call"**

Dashboard waits for the first call, then celebrates:
> "Your agent made 1 call. $0.003 cost. PASS. ✓ Safety is on."

### After setup

Dashboard sidebar shows:
- Live cost counter
- Safety status (PASS/FLAG/BLOCK badges)
- Policy controls (toggle checkboxes — already shipped)
- "Connect to AceTeam" nudge (if not connected): "Build workflows, $5 free credit →"

## Three Tiers

### Tier 1 — Local (free, no account, fully offline)

**Who:** Anyone who starts the container. Workshop attendees on day one.

**What they get:**
- Every LLM call tracked (cost, tokens, model)
- Dangerous requests blocked before reaching the LLM ($0 cost)
- Dashboard with real-time monitoring
- Policy controls (toggle detectors on/off from dashboard)
- Signed audit trail (optional, via `--sign-key`)

**MCP tools (supplementary — proxy does the real work):**

| Tool | What it does |
|------|-------------|
| `check_safety` | Run safety detectors on text. Returns PASS/FLAG/BLOCK with signals. |
| `get_safety_status` | Session metrics: calls checked, signals raised, cost, safety enabled. |
| `set_safety_policy` | Toggle detectors on/off, set thresholds. |
| `get_cost_summary` | Cost breakdown by model, by call. |

**The proxy is what delivers value.** A user who never touches MCP still gets safety and cost tracking on every LLM call. MCP tools let Claude Code interact with safety (e.g., "check if this text is safe before I send it" or "how much have I spent?").

**Upgrade nudge:** When user hits a Tier 2 feature (from MCP or dashboard), the response includes:
> "Build workflows and get $5 free LLM credit — connect to AceTeam →"

### Tier 2 — Connected (free AceTeam account, $5 credit)

**Who:** User clicks "Connect to AceTeam" in the dashboard (or runs `aceteam-aep connect` from CLI).

**Setup:** Dashboard button → browser opens → sign up / log in → API key generated automatically → dashboard shows "Connected ✓". Works from inside Docker/Podman (browser opens on host).

**What changes:**
- User can use AceTeam's LLM APIs (no need for own OpenAI key)
- $5 free credit — enough for hundreds of gpt-4o-mini calls
- Model picker in dashboard (GPT-4o, Claude, Gemini, etc.)
- Workflow tools unlock (MCP + dashboard)

**MCP tools added:**

| Tool | What it does |
|------|-------------|
| `create_workflow` | Build an AceFlow from a graph definition |
| `run_workflow` | Execute a saved workflow |
| `list_workflows` | Browse your workflows |
| `search_node_types` | Discover available node types (40+ types) |
| `get_node_schema` | Detailed node documentation with examples |
| `update_workflow` | Save a new version of a workflow |

**Non-MCP path to workflows:** Dashboard gets a "Create Workflow" link to the AceTeam web UI. Users on OpenClaw (no MCP) can still access workflows through the browser. The proxy tracks workflow runs regardless of how they're triggered.

**AEP in every workflow:** Every workflow run goes through the AceTeam backend, which has full AEP integration (AceTeamContext tracks spans, costs, citations per node). The dashboard shows workflow costs and safety signals alongside regular LLM calls.

### Tier 3 — Platform (AceTeam power user)

**Who:** Teams who want custom agents, knowledge base, document management, collaboration.

**Setup:** Already connected from Tier 2. Additional tools appear automatically.

**MCP tools added:**

| Tool | What it does |
|------|-------------|
| `list_agents` | Browse AI agents in your org |
| `create_agent` | Create a new agent with system prompt + model |
| `chat_with_agent` | Invoke an agent |
| `search_knowledge_base` | Semantic search across documents |
| `list_documents` | Browse uploaded documents |
| `list_mcp_servers` | See connected MCP servers |
| `link_tool_to_agent` | Give an agent access to a tool |

## Tier Discovery (Dynamic Tool Registration)

The MCP endpoint dynamically registers tools based on what's available:

```python
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

When credentials change (connect or disconnect), the MCP server signals a tool list refresh. Claude Code picks up new tools on next session or reconnect.

## Backend MCP Server Redesign

The current `aceteam` and `aceflows` MCP servers on the AceTeam Python backend stay as separate implementations:

| Current Server | Mount | New Role |
|----------------|-------|----------|
| `aceflows` | `/aceflows/` | Tier 2 — workflow CRUD + execution + schema discovery |
| `aceteam` | `/aceteam/` | Tier 3 — agent, document, knowledge base management |

The gateway proxies Tier 2+ MCP calls to these backend servers over HTTP. Backend servers don't change — the gateway is the routing layer.

## Safety Integration

**Tier 1:** Regex-based detectors (agent_threat, pii, cost_anomaly) run locally in the container. Free, fast, no cloud.

**Tier 2:** Trust Engine available as an upgrade. Uses AceTeam-hosted models for calibrated multi-perspective detection. Deducts from user's credit balance. Dashboard shows: "Your detectors caught 3 threats. Upgrade to Trust Engine for 5-category detection with confidence scores."

**Tier 3:** Per-org safety policies. Custom dimensions. YAML policy management. Compliance reporting (CISO dashboard).

**Every workflow run (Tier 2+) automatically has AEP.** The AceTeamContext in the workflow engine already tracks spans, costs, and citations per node.

## Technical Integration

### MCP Transport

The existing `mcp.py` is a hand-rolled stdio JSON-RPC loop (stdin/stdout). This cannot be mounted on the proxy. The gateway MCP endpoint requires **Streamable HTTP transport** — a new implementation.

**Approach:** Create a new FastMCP server instance, register tools as decorated functions, and mount its ASGI app on the existing Starlette proxy:

```python
from fastmcp import FastMCP
from starlette.routing import Mount

gateway_mcp = FastMCP("aceteam-gateway")

# Mount on existing Starlette app
routes.append(Mount("/mcp", app=gateway_mcp.streamable_http_app()))
```

FastMCP's `streamable_http_app()` returns a Starlette ASGI app. The `Mount` strips the `/mcp` prefix before forwarding. The existing stdio MCP server (`aceteam-aep mcp-server`) remains as a separate entry point for clients that prefer stdio transport.

### Shared State

MCP tool handlers access `ProxyState` via closures — the FastMCP server is created inside `create_proxy_app()` where `state` is in scope:

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
- `set_safety_policy` — backs onto existing `POST /dashboard/api/safety` handler logic
- `get_cost_summary` — reads from `ProxyState.cost_tracker` and `_call_costs`

### Auth Model

- **Tier 1 (local):** No auth. Gateway binds to `127.0.0.1` by default.
- **Tier 2+ (network-exposed):** Bearer token auth on `/mcp/` using AceTeam API key (`act_...`).

### Credential Storage (Phase 2)

For container: env var `ACETEAM_API_KEY` or volume-mount `~/.config/aceteam-aep/`. For pip install: `~/.config/aceteam-aep/credentials.json` with `0600` permissions. Token refresh: on-demand when 401 received.

### Graceful Degradation (Phase 2+)

When AceTeam backend is unreachable, Tier 2+ tools return errors and the dashboard shows "Disconnected from AceTeam — local safety active." Tier 1 tools always work.

### Cross-Repo Ownership

- **`aceteam-aep`:** Gateway MCP endpoint, Tier 1 tools, proxy, dashboard, CLI, container image
- **`aceteam-3`:** Backend MCP servers (`aceflows`, `aceteam`), workflow engine, agent runtime

## Implementation Phases

**Phase 1 (for ClawCamp April 16) — ~2-3 days:**
- Create FastMCP server with 4 Tier 1 tools backed by `ProxyState`
- Mount Streamable HTTP ASGI app at `/mcp/` on existing Starlette proxy
- Update CLI startup banner to show MCP endpoint URL
- Add Claude Code MCP config to workshop materials
- Main risk: FastMCP Streamable HTTP + Starlette path prefix integration

**Phase 2 (post-ClawCamp):**
- Dashboard setup wizard (LLM config + "Connect to AceTeam" button with browser OAuth)
- `aceteam-aep setup` CLI command (detect container runtime, write Claude Code config, configure shell)
- Tier 2 tools (proxy AceFlows MCP calls to AceTeam backend)
- Dashboard model picker + credit display
- Rate limiting for credit-consuming tiers

**Phase 3 (post-funding):**
- Tier 3 tools (proxy aceteam MCP for agents, knowledge base, documents)
- Local Ollama auto-start in container
- Trust Engine upgrade nudge
- Credit tracking dashboard
- Non-MCP workflow creation via dashboard UI
