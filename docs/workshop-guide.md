# AEP Trust Engine — Workshop Guide

> Add safety, cost tracking, and accountability to any AI agent in 5 minutes.

This guide works for anyone — workshop attendees, solo developers, or enterprise teams evaluating agent safety infrastructure.

---

## What You'll Build

By the end of this guide, you'll have:
- A safety proxy intercepting every LLM call your agent makes
- Real-time dashboard showing cost, safety signals, and enforcement decisions
- PII detection, toxicity classification, and cost anomaly alerts — running locally
- PASS / FLAG / BLOCK enforcement on every agent output

No code changes to your agent. One command to start.

---

## Prerequisites

- Python 3.12+
- An OpenAI API key (or Anthropic)
- 5 minutes

---

## Part 1: Install + Start the Proxy (2 minutes)

```bash
# Install AEP with all safety detectors + dashboard
pip install aceteam-aep[all]

# Start the safety proxy
aceteam-aep proxy --port 8080
```

You'll see:
```
  AEP Proxy
  ─────────────────────────────────
  Listening:  http://localhost:8080
  Target:     https://api.openai.com
  Dashboard:  http://localhost:8080/aep/

  Usage:
    export OPENAI_BASE_URL=http://localhost:8080/v1
    openclaw run "your task here"
```

Open **http://localhost:8080/aep/** in your browser. You'll see an empty dashboard — it'll light up as soon as calls flow through.

---

## Part 2: Route Your Agent Through the Proxy (1 minute)

### Option A: Python (OpenAI SDK)

Open a new terminal:

```bash
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=sk-your-key-here
python
```

```python
import openai

client = openai.OpenAI()

# This call now flows through the AEP proxy
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print(response.choices[0].message.content)
```

Check the dashboard — you'll see the call appear with cost tracked.

### Option B: Python (Anthropic SDK)

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080/v1
export ANTHROPIC_API_KEY=sk-ant-your-key
python
```

```python
import anthropic

client = anthropic.Anthropic(base_url="http://localhost:8080/v1")
message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(message.content[0].text)
```

### Option C: OpenClaw

Edit your OpenClaw model config to point at the proxy:

```json
{
  "api": "openai-completions",
  "id": "gpt-4o",
  "name": "GPT-4o (via AEP)",
  "baseUrl": "http://localhost:8080/v1"
}
```

Then run OpenClaw normally — all calls route through AEP.

### Option D: curl (any language)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Check the response headers:
```
X-AEP-Cost: 0.000042
X-AEP-Enforcement: pass
X-AEP-Call-ID: a1b2c3d4
```

---

## Part 3: Trigger Safety Signals (1 minute)

Now let's see the safety detectors in action.

### Trigger PII Detection

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": "Generate a fake person profile with name, SSN, email, and phone number."
    }],
)
print(response.choices[0].message.content)
```

Check the dashboard — you should see:
- 🔴 **BLOCK** badge (PII detected in output)
- Signal: `pii: PII detected: SSN (score=0.95)`

The proxy blocked this response because it contained a Social Security Number. Your agent never saw the PII.

### Trigger Content Safety

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": "Write an extremely hostile, threatening message."
    }],
)
```

Dashboard shows:
- 🔴 **BLOCK** or 🟡 **FLAG** (depending on toxicity score)
- Signal: `content_safety: Unsafe content in output (score=0.92)`

### Trigger Cost Anomaly

Make 3 cheap calls, then one expensive one:

```python
# Baseline: 3 cheap calls
for _ in range(3):
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hi"}],
        max_tokens=5,
    )

# Anomaly: expensive call
client.chat.completions.create(
    model="gpt-4o",  # more expensive model
    messages=[{"role": "user", "content": "Write a 2000-word essay about AI safety."}],
    max_tokens=4000,
)
```

Dashboard shows:
- 🟡 **FLAG** — cost anomaly detected
- Signal: `cost_anomaly: Cost $0.012000 is >5x session avg $0.000150`

This is how you catch the $135K surprise bill before it happens.

---

## Part 4: The Python SDK (Alternative to Proxy)

If you prefer in-process wrapping over a proxy:

```python
import openai
from aceteam_aep import wrap

# One line to add safety
client = wrap(openai.OpenAI())

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Access AEP data programmatically
print(f"Cost: ${client.aep.cost_usd}")
print(f"Safety: {client.aep.enforcement.action}")
print(f"Signals: {client.aep.safety_signals}")

# CLI summary
client.aep.print_summary()

# Or launch the dashboard
# client.aep.serve_dashboard()  # http://localhost:8899
```

---

## Part 5: Add Governance Context (Advanced)

Add identity and data classification to your calls via HTTP headers:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-AEP-Entity: org:acme" \
  -H "X-AEP-Classification: confidential" \
  -H "X-AEP-Consent: training=no" \
  -H "X-AEP-Budget: 1.00" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Summarize this contract."}]
  }'
```

The proxy:
1. Reads `X-AEP-*` headers (identity, classification, consent, budget)
2. Enforces governance rules (block if over budget or classification violation)
3. Strips headers before forwarding to OpenAI (upstream never sees them)
4. Adds `X-AEP-*` response headers with cost + enforcement decision

This works from any language — Python, Node.js, Go, curl. No SDK required.

---

## Part 6: Custom Detectors (Build Your Own)

Create a detector for your specific domain:

```python
from aceteam_aep import wrap
from aceteam_aep.safety.base import SafetySignal

class ComplianceDetector:
    name = "compliance"

    def check(self, *, input_text, output_text, call_id, **kwargs):
        # Flag if the agent mentions competitors
        competitors = ["acme corp", "globex", "initech"]
        for comp in competitors:
            if comp in output_text.lower():
                return [SafetySignal(
                    signal_type="compliance",
                    severity="medium",
                    call_id=call_id,
                    detail=f"Competitor mentioned: {comp}",
                )]
        return []

client = wrap(openai.OpenAI(), detectors=[ComplianceDetector()])
```

Detectors are just Python classes with a `check()` method. Write them for:
- Domain-specific compliance rules
- Custom PII patterns (employee IDs, internal codes)
- Business logic validation
- Output format enforcement

---

## Architecture: How It Works

```
Your Agent (OpenClaw, LangChain, Python, curl)
        │
        │  OPENAI_BASE_URL=http://localhost:8080/v1
        ▼
┌─────────────────────────────────────────┐
│         AEP Proxy (Layer 1)             │
│  ─────────────────────────────────      │
│  → Intercept request                     │
│  → Run input safety checks               │
│  → BLOCK if dangerous (before OpenAI)    │
│  → Forward to real API                   │
│  → Intercept response                    │
│  → Run output safety checks              │
│  → BLOCK if PII/toxic (before agent)     │
│  → Track cost + spans                    │
│  → Serve dashboard at /aep/              │
└─────────────────┬───────────────────────┘
                  │
                  ▼
         OpenAI / Anthropic API
```

**Layer 1 (Proxy):** Zero code changes. Cost + safety. Free.
**Layer 2 (SDK/Headers):** Application context. Provenance + governance. Enterprise.

Think WireGuard (simple, free, wire-level) vs Tailscale ($1.6B, identity + management on top).

---

## What's Next

- **Star the repo:** https://github.com/aceteam-ai/aceteam-aep
- **Read the docs:** Full API reference in the README
- **Join the community:** [AceTeam Discord / GitHub Discussions]
- **Enterprise?** Contact jason@aceteam.ai for governance, compliance, and sovereign deployment

---

## Troubleshooting

**"Connection refused" on port 8080:**
Make sure `aceteam-aep proxy --port 8080` is running in another terminal.

**"Model not found" errors:**
The proxy forwards to `https://api.openai.com` by default. For Anthropic, use `--target https://api.anthropic.com`.

**Safety models downloading slowly:**
First run downloads ~300MB of HuggingFace models. They're cached after that. Use `--no-safety` to skip model-based detectors and use regex only.

**OpenClaw not routing through proxy:**
OpenClaw hardcodes its API base URL. You need to edit the model config — see Option C above.

**Dashboard not updating:**
The dashboard polls `/aep/api/state` every 2 seconds. If calls aren't appearing, verify your agent is actually pointing at `localhost:8080`.
