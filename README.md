# aceteam-aep

AceTeam™ trust & safety infrastructure for AI agents. The Agentic Execution Protocol™ (AEP™) adds cost tracking, safety detection, and enforcement to any LLM-powered tool — **zero code changes required.**

## Installation

```bash
pip install aceteam-aep[all]               # Everything (recommended)
pip install aceteam-aep[safety,proxy]      # Safety detectors + proxy
pip install aceteam-aep                    # Core only (cost tracking + regex safety)
```

## Quick Start — Make OpenClaw (or any agent) Safe

No code changes. Just run the proxy and point your agent at it:

```bash
# Terminal 1: Start the AEP safety proxy
aceteam-aep proxy --port 8080

# Terminal 2: Run OpenClaw through the proxy
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=sk-your-key
openclaw run "analyze these financial statements"
```

Open **http://localhost:8080/aep/** — the dashboard shows every LLM call flowing through in real-time: cost, safety signals, and enforcement decisions.

The proxy intercepts **both directions**:
- **Incoming requests** — blocks dangerous prompts before they reach the API
- **Outgoing responses** — blocks PII, toxic content, and cost anomalies before the agent sees them

Works with OpenClaw, LangChain, CrewAI, curl, or any tool that calls the OpenAI API.

## What the Proxy Sees

The proxy is a reverse proxy (man-in-the-middle by design). It reads the full request AND full response. It can block in either direction.

```
Your Agent
    │
    ├─── REQUEST ──────────────────────────────┐
    │    messages: [user prompt, tool results]  │
    │                                           ▼
    │                                    ┌─────────────┐
    │                                    │  AEP Proxy   │
    │                                    │              │
    │                                    │  ✓ Input     │──── if dangerous ──→ BLOCK (never reaches API)
    │                                    │    text      │
    │                                    │              │──── if safe ──→ forward to OpenAI
    │                                    │              │
    │                                    │  ✓ Output    │──── if PII/toxic ──→ BLOCK (agent never sees it)
    │                                    │    text      │
    │                                    │              │──── if safe ──→ return to agent
    │                                    │  ✓ Cost      │
    │                                    │  ✓ Tool calls│
    │                                    └─────────────┘
    │                                           │
    ◄─── RESPONSE ─────────────────────────────┘
         assistant message, token usage
```

| Data | Proxy Sees It? | Details |
|------|:--------------:|---------|
| User messages (input text) | **Yes** | Full message array from request body |
| LLM response (output text) | **Yes** | Full response including all choices |
| Tool call requests | **Yes** | What functions the LLM asks to call |
| Tool call results | **Yes** | Included in next request's messages |
| Token usage + cost | **Yes** | From response usage field |
| **Agent actions between calls** | **No** | File writes, code execution, browser actions happen inside the agent, not via the LLM API |
| **Application context** | **No** | Who is calling, data classification — unless sent via `X-AEP-*` headers |

**The proxy sees every word going to and from the LLM.** It cannot see what the agent does *between* LLM calls. For that, use the SDK (Layer 2).

## Two Layers: Proxy + SDK

Think **WireGuard + Tailscale**. WireGuard is a minimal wire protocol. Tailscale adds identity and management on top. Same here:

**Layer 1 — AEP Proxy (free, zero code changes)**
- Sees all LLM traffic (input, output, tool calls, cost)
- Runs safety detectors, enforces PASS/FLAG/BLOCK
- Dashboard at `/aep/`
- Works with any language, any framework

**Layer 2 — AEP SDK (application-level context)**
- Adds identity: `X-AEP-Entity: org:acme`
- Adds governance: `X-AEP-Classification: confidential`
- Adds provenance: citation chains, source tracking
- Via HTTP headers through the proxy, or via Python `wrap()`

Layer 1 gets developers in the door. Layer 2 is what enterprises need for compliance.

## Python SDK — Wrap Your Existing Client

```python
import openai
from aceteam_aep import wrap

client = wrap(openai.OpenAI())

# Use exactly as before — AEP intercepts transparently
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)

# AEP tracks everything
print(client.aep.cost_usd)            # $0.000150
print(client.aep.enforcement.action)   # "pass"
print(client.aep.safety_signals)       # []
client.aep.print_summary()             # Colored CLI output
```

Works with **OpenAI**, **Anthropic**, and any OpenAI-compatible client. Sync and async.

```python
import anthropic
from aceteam_aep import wrap

client = wrap(anthropic.Anthropic())
# Same API — client.aep.cost_usd, client.aep.safety_signals, etc.
```

## Safety Signals

Every LLM call is evaluated by pluggable safety detectors:

| Detector | What It Catches | Model |
|----------|----------------|-------|
| **PII** | SSN, email, phone, credit cards in output | `iiiorg/piiranha-v1-detect-personal-information` (~110M) |
| **Content Safety** | Toxic, harmful, or unsafe content | `s-nlp/roberta_toxicity_classifier` (~125M) |
| **Cost Anomaly** | Spend spikes >5x session average | Statistical (no model) |

Models lazy-load on first use, run on CPU. Falls back to regex if `transformers` not installed.

### Enforcement: PASS / FLAG / BLOCK

Every call produces an enforcement decision based on signal severity:

- **PASS** — No signals or low severity. Safe to proceed.
- **FLAG** — Medium severity. Route to human review.
- **BLOCK** — High severity (PII, toxic content). Prevent delivery.

```python
client = wrap(openai.OpenAI())
response = client.chat.completions.create(...)

match client.aep.enforcement.action:
    case "pass":
        return response
    case "flag":
        queue_for_review(response)
    case "block":
        return reject(client.aep.enforcement.reason)
```

### Custom Detectors

```python
from aceteam_aep import wrap
from aceteam_aep.safety.base import SafetySignal

class MyDetector:
    name = "my_detector"

    def check(self, *, input_text, output_text, call_id, **kwargs):
        if "secret" in output_text.lower():
            return [SafetySignal(
                signal_type="data_leak",
                severity="high",
                call_id=call_id,
                detail="Potential secret in output",
            )]
        return []

client = wrap(openai.OpenAI(), detectors=[MyDetector()])
```

## Dashboard

```python
client.aep.serve_dashboard()  # http://localhost:8899
```

Dark-themed local web UI showing cost, safety status, signal timeline, and call history. Auto-refreshes every 2 seconds.

## CLI Output

```python
client.aep.print_summary()
```

```
──────────────────────────────────────────────────
  AEP Session Summary
──────────────────────────────────────────────────
  Calls:  5
  Cost:   $0.004200
  Safety: PASS
──────────────────────────────────────────────────
```

## Agent Loop (Advanced)

For building agents from scratch with full AEP compliance:

```python
from aceteam_aep import create_client, run_agent_loop, ChatMessage, tool

client = create_client("gpt-4o", api_key="sk-...")

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

result = await run_agent_loop(
    client,
    [ChatMessage(role="user", content="Search for AEP protocol")],
    tools=[search],
    system_prompt="You are a helpful assistant.",
)
```

## Workshop Guide

Step-by-step setup in 5 minutes — from install to safety signals firing:

**[docs/workshop-guide.md](docs/workshop-guide.md)**

Covers: proxy setup, routing agents (Python/OpenClaw/curl), triggering safety signals, governance headers, custom detectors. Works for workshops, onboarding, or self-guided evaluation.

## Providers

- **OpenAI** (GPT-4o, GPT-5, o1, o3)
- **Anthropic** (Claude Opus, Sonnet, Haiku)
- **Google** (Gemini 2.5, 3.0)
- **xAI** (Grok)
- **Ollama** (local models)
- **OpenAI-compatible** (SambaNova, TheAgentic, DeepSeek)

## Trademarks

"Agentic Execution Protocol," "AEP," and "AceTeam" are trademarks of AceTeam. The software is licensed under Apache 2.0. The trademark is not included in the license grant — you may not use these names to endorse or promote derivative works without written permission.
