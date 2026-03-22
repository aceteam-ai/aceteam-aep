# AEP™ Business Model: Free Proxy → Paid Safety

## The Two-Layer Model

The Agentic Execution Protocol™ (AEP™) ships as one package with two tiers of integration. The proxy (Layer 1) is the distribution wedge. The SDK + active safety (Layer 2) is the revenue engine.

```
┌─────────────────────────────────────────────────────┐
│  PAID: Layer 2 — Active Safety + Governance          │
│  ─────────────────────────────────────               │
│  • Model-based PII detection (NER transformer)       │
│  • Content safety classification (toxicity model)    │
│  • Prompt injection detection (DeBERTa model)        │
│  • Data governance enforcement (classification +     │
│    consent + redaction)                              │
│  • Provenance chains (source attribution)            │
│  • Trust Engine (ensemble of judges, calibrated      │
│    confidence — from research collaboration)         │
│                                                      │
│  COST: Runs ML models per call → compute/tokens      │
│  VALUE: Enterprise compliance, audit trails, T&S     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  FREE: Layer 1 — Passive Observation                 │
│  ─────────────────────────────────────               │
│  • Cost tracking (token counting + pricing)          │
│  • Budget enforcement (hard cap per session/call)    │
│  • Basic regex safety (pattern matching, no models)  │
│  • Execution spans (call timeline)                   │
│  • Dashboard (read-only monitoring)                  │
│                                                      │
│  COST: Near-zero — no ML inference, just counting    │
│  VALUE: Visibility, runaway cost prevention           │
└─────────────────────────────────────────────────────┘
```

## Why Safety Is the Upsell

### The Compute Cost Argument

Layer 1 (free) is essentially free to operate — it counts tokens and applies regex patterns. The proxy adds <1ms latency per call.

Layer 2 (paid) runs ML models on every call:

| Detector | Model Size | Inference Time (CPU) | Cost per 1K calls |
|----------|-----------|---------------------|-------------------|
| PII (NER) | ~110M params | ~50ms | ~$0.00 (local) |
| Content Safety | ~125M params | ~30ms | ~$0.00 (local) |
| Prompt Injection | ~86M params | ~40ms | ~$0.00 (local) |
| Trust Engine (future) | 3-5 judge models × ~7B each | ~2-5s | ~$0.02-0.10 |

Local models (PII, content, injection) are cheap — the cost is in the GPU/CPU time, not API calls. But at scale (millions of agent calls per day), the compute adds up.

The Trust Engine (ensemble of judges) is where the real cost is. Running 3-5 judge models per call to get calibrated confidence scores is expensive — this is the premium tier.

### The Value Argument

**Free tier value:** "I can see what my agents are spending."
→ Solves the $135K surprise bill problem. Immediate, tangible, no commitment.

**Paid tier value:** "I can trust what my agents are doing."
→ Solves enterprise compliance. PII protection, audit trails, data governance. Required for regulated industries (healthcare, finance, legal).

The gap between "see" and "trust" is the business.

## Conversion Funnel

```
1. Developer discovers AEP (GitHub, ClawCamp, blog post)
   ↓
2. `pip install aceteam-aep` — wraps their OpenAI client
   FREE: cost tracking + regex safety
   ↓
3. Sees dashboard, realizes agent is spending $X/day
   Sees safety signals (regex catches obvious PII)
   ↓
4. Wants better detection → enables model-based detectors
   `pip install aceteam-aep[safety]`
   PAID: runs local ML models, better detection
   ↓
5. Enterprise needs compliance → governance + audit
   `pip install aceteam-aep[all]` + config file
   PAID: data classification, consent, redaction, audit trails
   ↓
6. At scale → managed AEP service (hosted proxy + dashboard)
   PAID: we run the infrastructure, they get the API
```

## Pricing Model Options

### Option A: Usage-Based (Sentry/Datadog playbook)

| Tier | Calls/month | Features | Price |
|------|-------------|----------|-------|
| Free | Unlimited | Cost tracking, regex safety, dashboard | $0 |
| Pro | Up to 100K | + Model-based safety detectors | $49/mo |
| Enterprise | Unlimited | + Governance, provenance, Trust Engine, SLA | Custom |

### Option B: Compute-Based

Charge for the ML inference compute:
- Free: regex safety (no compute cost to us)
- Paid: $X per 1K safety evaluations (model-based)
- Premium: $X per 1K Trust Engine evaluations (multi-judge ensemble)

### Option C: Seat-Based + Compute

- Free: open source, self-hosted, unlimited
- Paid: managed dashboard + hosted safety evaluation
- Per-seat pricing for the dashboard/compliance portal

## The Network Effect Moat

Every wrapped agent generates anonymized telemetry:
- Cost distributions by model/provider
- Safety signal frequencies by type
- Enforcement action rates

Across thousands of wrapped agents:
→ Better detection baselines (what's "normal" for a coding agent vs a customer service agent?)
→ Domain-specific calibration (legal agents have different failure modes than medical agents)
→ Anomaly detection from fleet-wide patterns (new attack vector detected across multiple orgs)

**This data doesn't exist anywhere else.** It's the moat.

## Integration with AEP Headers (Layer 1 → Layer 2 Bridge)

Any client can send governance context without using the Python SDK:

```bash
# curl, Node.js, Go, Ruby — any language
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-..." \
  -H "X-AEP-Entity: org:acme" \
  -H "X-AEP-Classification: confidential" \
  -H "X-AEP-Consent: training=no" \
  -H "X-AEP-Budget: 5.00" \
  -H "X-AEP-Sources: doc:contract-123" \
  -d '{"model": "gpt-4o", "messages": [...]}'

# Response includes:
# X-AEP-Cost: 0.0042
# X-AEP-Enforcement: pass
# X-AEP-Classification: confidential
# X-AEP-Call-ID: abc123
```

The proxy:
1. Reads X-AEP-* headers
2. Strips them before forwarding to OpenAI (upstream never sees them)
3. Runs governance enforcement based on entity + classification + consent
4. Adds X-AEP-* response headers with cost + enforcement decision

This means **any language, any framework** can get Layer 2 governance by just adding headers. No SDK required.

## ClawCamp Demo Script

```bash
# Terminal 1: Start AEP proxy
pip install aceteam-aep[all]
aceteam-aep proxy --port 8080

# Terminal 2: Run OpenClaw through AEP
export OPENAI_BASE_URL=http://localhost:8080/v1
openclaw run "analyze these financial statements"

# Terminal 3: Watch the dashboard
open http://localhost:8080/aep/
```

**What attendees see:**
1. Every OpenClaw LLM call flowing through the dashboard
2. Cost accumulating in real-time
3. Safety signals firing (PII in financial docs, cost anomalies)
4. PASS/FLAG/BLOCK decisions on each call

**What they take home:**
- `pip install aceteam-aep` on their laptop
- A GitHub star
- The understanding that their agents now have a safety layer

**The pitch in one sentence:**
> "OpenClaw gives your agents superpowers. AEP gives them a safety net."
