# AEP Gateway Architecture

## Overview

The AEP Gateway (SafeClaw) is a unified process serving three endpoints on a single port:

| Path | Protocol | Purpose |
|------|----------|---------|
| `/v1/*` | OpenAI-compatible HTTP | LLM reverse proxy with safety enforcement |
| `/aep/*` | HTTP + HTML | Dashboard, safety API, policy controls |
| `/mcp/*` | MCP Streamable HTTP | Tool access for Claude Code and MCP clients |

All three surfaces share one `ProxyState` — cost tracking, safety signals, decisions, and policy are live across every interface simultaneously.

## Components

### Proxy (`proxy/app.py`)

- Intercepts all LLM traffic at the network level (OpenAI-compatible `/v1/*`)
- Runs safety detectors on input and output
- Enforces PASS/FLAG/BLOCK decisions
- Tracks cost per call via `CostTracker`
- Optionally signs verdicts (Ed25519 + Merkle chain via `attestation.py`)
- Serves the dashboard at `/aep/` and state API at `/aep/api/*`

### Dashboard (`dashboard/templates/index.html`)

A dark-themed local web UI that auto-refreshes every 2 seconds.

**Developer view:**
- Individual call log with safety signals and cost per call
- Call timeline, governance context, enforcement decisions

**Executive/CISO view:**
- Enforcement coverage %, threats blocked, compliance status
- Safety breakdown (PII / threats / toxicity / anomalies)
- Cost attribution by entity

**Policy controls:**
- Per-detector checkboxes (enable/disable each detector individually)
- Trust Engine category toggles (finance, iot, software, web, program R-Judge dimensions)
- Safety on/off master toggle in the header

**Setup wizard:**
- First-run overlay shown when `localStorage` has no `safeclaw-setup-done` flag and call count is 0
- Two-option flow: BYOK (paste API key) or AceTeam hosted ($5 credit, coming soon)
- Step 2 shows the `OPENAI_BASE_URL` export command and Claude Code MCP JSON config to copy

### MCP Gateway (`mcp_gateway.py`)

- FastMCP Streamable HTTP server mounted at `/mcp/` on the proxy
- Shares `ProxyState` with the proxy via closure
- Lifespan task group wired into the parent Starlette app's lifespan context
- Optional dependency: only active when `fastmcp` is installed (`pip install aceteam-aep[mcp]`)

**Tier 1 tools (always available locally):**

| Tool | Description |
|------|-------------|
| `check_safety` | Check text for safety issues. Returns PASS/FLAG/BLOCK with signal details. |
| `get_safety_status` | Session metrics: calls, signals, cost, safety enabled state, policy. |
| `set_safety_policy` | Toggle detectors on/off or update policy thresholds. |
| `get_cost_summary` | Cost breakdown: total, per-call (last 10), by model, savings from blocks. |

### Safety Detectors (`safety/`)

All detectors implement `check(*, input_text, output_text, call_id, **kwargs) -> list[SafetySignal]`.

| Detector | Model | What it catches |
|----------|-------|----------------|
| `AgentThreatDetector` | Regex (11 patterns) | Subprocess, socket, port scan, reverse shell, credential access |
| `PiiDetector` | iiiorg/piiranha (~110MB) + regex fallback | SSN, credit card, email, phone, IP address |
| `CostAnomalyDetector` | Statistical (no ML) | Cost spikes >Nx session average |
| `ContentSafetyDetector` | s-nlp/roberta_toxicity (~125MB) | Toxic, harmful, unsafe content |
| `FerpaDetector` | Regex patterns | Student IDs, grades, transcripts, financial aid |
| `TrustEngineDetector` | Multi-perspective LLM | Calibrated confidence across configurable dimensions |

### Enforcement (`enforcement.py`)

- Policy-driven: YAML config → `DetectorPolicy` per detector
- Priority: per-detector action override → severity fallback → default action
- `build_detectors_from_policy()` creates detector instances from a policy YAML
- `evaluate(signals, policy)` → `EnforcementDecision` with action (pass/flag/block) and reason

### CLI (`proxy/cli.py`)

| Command | Description |
|---------|-------------|
| `proxy` | Start the gateway. Prints LLM proxy URL, dashboard URL, and MCP URL on startup. |
| `wrap` | Wrap any command with AEP proxy — sets `OPENAI_BASE_URL` and `ANTHROPIC_BASE_URL` in the child process env. |
| `keygen` | Generate Ed25519 keypair (`aep.key` / `aep.pub`) for verdict signing. |
| `verify` | Verify a Merkle audit chain JSONL file against a public key. |
| `mcp-server` | Stdio MCP server (legacy — for non-HTTP MCP clients). |

## Module Map

```
src/aceteam_aep/
├── wrap.py              # SDK wrapper (wrap mode) — AepSession, monkey-patching
├── proxy/
│   ├── app.py           # Starlette ASGI app — ProxyState, request handling, route mounting
│   ├── cli.py           # CLI: proxy, wrap, keygen, verify, mcp-server
│   ├── headers.py       # X-AEP-* header parsing, building, stripping
│   └── streaming.py     # SSE stream interception for streaming responses
├── mcp_gateway.py       # FastMCP Streamable HTTP gateway — 4 Tier 1 tools
├── safety/
│   ├── base.py          # SafetySignal, DetectorRegistry — detector interface
│   ├── pii.py           # PII detector (HuggingFace piiranha + regex fallback)
│   ├── content.py       # Content safety (HuggingFace toxicity classifier)
│   ├── cost_anomaly.py  # Cost anomaly detection (statistical)
│   ├── agent_threat.py  # Agent threat patterns (11 regex patterns)
│   ├── ferpa.py         # FERPA education records detector
│   └── trust_engine.py  # Trust Engine — multi-perspective + ensemble + external judge
├── enforcement.py       # EnforcementPolicy, evaluate(), build_detectors_from_policy()
├── attestation.py       # Ed25519 signed verdicts + Merkle audit chains
├── config.py            # Unified YAML config: proxy, enforcement, budget
├── feedback.py          # Signal feedback loop: verdicts → threshold recommendations
├── instrument.py        # Global SDK patching: instrument() patches OpenAI/Anthropic
├── provenance/          # Source attribution: extractor + tracker
├── costs.py             # CostTracker, CostNode — per-call cost accounting
├── spans.py             # SpanTracker — execution trace recording
├── types.py             # Usage, shared types
├── dashboard/
│   └── templates/       # HTML dashboard (developer + executive views, policy controls, setup wizard)
└── __init__.py          # Public API
```

## Trust Engine

Multi-perspective safety evaluation with calibrated confidence scoring. Three modes:

| Mode | How it works | When to use |
|------|-------------|-------------|
| `multi-perspective` | One LLM call, N dimensions in prompt | Default — cheap, fast |
| `ensemble` | N separate LLM calls | Diverse model families |
| `judge_service_url` | Calls external Flask service | R-Judge specialist judges |

Dimensions are safety perspectives. Two sets:

- **Default**: `pii`, `toxicity`, `agent_threat`, `policy_compliance`, `irreversibility`
- **R-Judge domains**: `finance`, `iot`, `software`, `web`, `program` (SJTU R-Judge benchmark, EMNLP 2024)

Dimensions are toggleable per policy YAML. The external judge service (AdaExtract2 `judge_service.py`) wraps specialist prompts as a Flask API.

## Runtime Safety Toggle

`POST /aep/api/safety` enables/disables safety at runtime without restart:

```bash
# Safety off
curl -X POST localhost:8899/aep/api/safety -d '{"enabled": false}'

# Hot-swap policy
curl -X POST localhost:8899/aep/api/safety -d '{"policy": {"default_action": "block"}}'

# Check state
curl localhost:8899/aep/api/safety
```

The dashboard has a master toggle switch in the header. Per-detector and per-category checkboxes allow fine-grained control without restart.

## Signal Feedback Loop

Operators mark flagged signals as confirmed (true positive) or dismissed (false positive) via the dashboard review buttons or `POST /aep/api/feedback`:

```
POST /aep/api/feedback → JSONL store → analyze FP rate → recommend threshold → apply to YAML
```

The system uses the 90th percentile of dismissed scores (capped below the lowest confirmed score) to suggest thresholds. Requires 5+ verdicts per detector.

## Attestation

Optional Ed25519 signed verdicts with Merkle chaining for tamper-evident audit trails:

```bash
# Generate keypair
aceteam-aep keygen --output ./keys

# Start proxy with signing
aceteam-aep proxy --sign-key ./keys/aep.key

# Verify chain
aceteam-aep verify --pub-key ./keys/aep.pub --chain audit.jsonl
```

Each verdict in the chain contains the previous chain hash, so any tampering is detectable.

## Vertical Policy Templates

Pre-built policies in `policies/`:

| Policy | PII Threshold | Cost Multiplier | Default Action |
|--------|:------------:|:---------------:|:--------------:|
| `finance.yaml` | 0.5 | 3x | block |
| `healthcare.yaml` | 0.3 | 4x | block |
| `legal.yaml` | 0.6 | 8x | flag |
| `education.yaml` | 0.5 | 4x | flag |
| `startup.yaml` | 0.8 | 10x | flag |
| `clawcamp.yaml` | 0.6 | 5x | flag (5 R-Judge categories) |
| `safety-off.yaml` | — | — | pass (all detectors disabled) |

## Docker Deployment

Pre-built image: `ghcr.io/aceteam-ai/aep-proxy:latest`

- Python 3.12 slim base
- Safety models pre-downloaded (~235MB)
- Binds to `0.0.0.0:8899`
- Healthcheck on `/aep/`
- Entrypoint: `aceteam-aep proxy`

Published on tag push via GitHub Actions.

## Integration Points

| Client | How to integrate |
|--------|-----------------|
| Claude Code | MCP config → `http://localhost:8899/mcp/` (see `docs/engineering/mcp-integration.md`) |
| OpenClaw / SafeClaw | `export OPENAI_BASE_URL=http://localhost:8899/v1` |
| Any Python agent | `aceteam-aep wrap -- python my_agent.py` |
| Docker sidecar | `OPENAI_BASE_URL=http://aep-proxy:8899/v1` in compose |
| AceTeam Platform | Gateway proxies Tier 2+ MCP calls to backend (planned) |
