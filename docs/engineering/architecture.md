# AEP Architecture

## Overview

AEP (Agentic Execution Protocol) provides safety and accountability for AI agents through two deployment modes:

1. **Wrap mode** — Python SDK that monkey-patches LLM client libraries (OpenAI, Anthropic) to intercept calls in-process
2. **Proxy mode** — HTTP reverse proxy that intercepts all LLM traffic at the network level

Both modes provide the same four pillars: cost tracking, safety enforcement, provenance, and governance.

## Four Pillars

```
                    ┌─────────────────────┐
                    │    GOVERNANCE        │  X-AEP-Entity, Classification, Consent
                    │    (who & what)      │
                    ├─────────────────────┤
                    │    PROVENANCE        │  Sources, citations, trace IDs
                    │    (where from)      │
                    ├─────────────────────┤
                    │    SAFETY            │  PII, toxicity, agent threats → PASS/FLAG/BLOCK
                    │    (is it safe)      │
                    ├─────────────────────┤
                    │    COST              │  Tokens, model pricing, cumulative spend
                    │    (what it costs)   │
                    └─────────────────────┘
```

## Module Map

```
src/aceteam_aep/
├── wrap.py              # SDK wrapper (wrap mode) — AepSession, AepSource, monkey-patching
├── proxy/
│   ├── app.py           # Starlette ASGI app (proxy mode) — ProxyState, request handling
│   ├── cli.py           # CLI: aceteam-aep proxy, aceteam-aep wrap
│   ├── headers.py       # X-AEP-* header parsing, building, stripping
│   └── streaming.py     # SSE stream interception for streaming responses
├── safety/
│   ├── base.py          # SafetySignal, DetectorRegistry — detector interface
│   ├── pii.py           # PII detector (HuggingFace piiranha model + regex fallback)
│   ├── content.py       # Content safety (HuggingFace toxicity classifier)
│   ├── cost_anomaly.py  # Cost anomaly detection (statistical, no ML)
│   └── agent_threat.py  # Agent threat patterns (regex: port scans, subprocess, credential access)
├── enforcement.py       # EnforcementPolicy, evaluate(), discover_policy()
├── costs.py             # CostTracker, CostNode — per-call cost accounting
├── spans.py             # SpanTracker — execution trace recording
├── types.py             # Usage, shared types
├── dashboard/
│   └── templates/       # Jinja2 HTML dashboard
└── __init__.py          # Public API: wrap, AepSession, AepSource
```

## Wrap Mode

```python
from aceteam_aep import wrap
client = wrap(openai.OpenAI())
```

The `wrap()` function:
1. Creates an `AepSession` attached as `client.aep`
2. Monkey-patches `client.chat.completions.create()` (and async variant)
3. For each call:
   a. **Pre-flight check** — run detectors on input text. If BLOCK, raise `AepPreflightBlock` (request never sent)
   b. **Forward call** to the real OpenAI/Anthropic API
   c. **Post-flight** — extract usage, run detectors on output text, record cost + span + signals
   d. **Enforce** — evaluate signals against policy → PASS/FLAG/BLOCK decision

### Pre-flight Blocking

```
Input text → Detectors → Signals → Policy evaluation
                                         │
                                    ┌────┴────┐
                                    │  BLOCK  │ → AepPreflightBlock raised, $0 cost
                                    └─────────┘
                                    │  PASS   │ → Continue to API call
                                    └─────────┘
```

## Proxy Mode

```bash
aceteam-aep proxy --port 8899 --host 0.0.0.0
```

The proxy:
1. Starts a Starlette ASGI server
2. Intercepts all `/v1/*` requests (OpenAI-compatible API)
3. For each request:
   a. Parse body (extract messages/input text)
   b. Parse X-AEP-* governance headers
   c. **Pre-flight** — run detectors on input. If BLOCK, return HTTP 400 (request never forwarded)
   d. Strip X-AEP-* headers before forwarding upstream
   e. Forward to target API (OpenAI, Anthropic, etc.)
   f. **Post-flight** — run detectors on response, record cost + signals
   g. Add X-AEP-* response headers (cost, enforcement, call_id, classification, trace_id)
4. Serves dashboard at `/aep/` and state API at `/aep/api/state`

## Safety Detectors

All detectors implement the same interface:

```python
class MyDetector:
    name = "my_detector"

    def check(self, *, input_text: str, output_text: str, call_id: str, **kwargs) -> list[SafetySignal]:
        # Return list of SafetySignal if issues detected, empty list if clean
        return []
```

Built-in detectors:

| Detector | Model | What it catches |
|----------|-------|----------------|
| `PiiDetector` | iiiorg/piiranha (110MB) + regex fallback | SSN, email, phone, credit card, IP address |
| `ContentSafetyDetector` | s-nlp/roberta_toxicity (125MB) | Toxic, harmful, unsafe content |
| `CostAnomalyDetector` | Statistical (no ML) | Cost spikes >Nx session average |
| `AgentThreatDetector` | Regex patterns | Port scans, subprocess, socket, credential access |

## Enforcement Policy

Signals are evaluated against a configurable policy:

```yaml
# aep-policy.yaml
default_action: flag
block_on: [high]
flag_on: [medium]
detectors:
  pii:
    action: block
    threshold: 0.8
  agent_threat:
    action: block
```

Policy discovery chain: explicit argument → `AEP_POLICY` env var → file walk (aep-policy.yaml) → built-in default.

## Provenance

The SDK tracks data sources via `client.aep.add_source()`:

```python
client.aep.add_source("file:///data/report.pdf", kind="file", label="Q1 Report")
client.aep.add_source("https://api.example.com/v1/data", kind="api", label="Customer API")

# After calls
print(client.aep.sources)            # all sources
print(client.aep.get_citations("call-123"))  # sources for a specific call
print(client.aep.provenance_summary) # summary with counts by kind
```

The proxy tracks provenance via `X-AEP-Sources` and `X-AEP-Trace-Id` headers.

## Governance Headers

Both modes support governance context:

| Header | Purpose | Example |
|--------|---------|---------|
| `X-AEP-Entity` | Organization/agent identity | `org:acme-corp` |
| `X-AEP-Classification` | Data sensitivity level | `confidential` |
| `X-AEP-Consent` | Data processing consent | `analytics=true,training=false` |
| `X-AEP-Budget` | Cost budget cap (USD) | `10.00` |
| `X-AEP-Sources` | Data sources used | `file:///data/report.pdf` |
| `X-AEP-Trace-Id` | Distributed trace correlation | `workflow-1234` |

In wrap mode, set via env vars: `AEP_ENTITY`, `AEP_CLASSIFICATION`, etc.
In proxy mode, send as HTTP headers on each request.

## Docker Deployment

Pre-built image: `ghcr.io/aceteam-ai/aep-proxy:latest`

- Python 3.12 slim base
- Safety models pre-downloaded (~235MB)
- Binds to `0.0.0.0:8899`
- Healthcheck on `/aep/`
- Entrypoint: `aceteam-aep proxy`

Published on tag push via GitHub Actions.
