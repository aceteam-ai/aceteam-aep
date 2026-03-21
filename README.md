# aceteam-aep

Trust & safety infrastructure for AI agents. Wrap any LLM client with one line — get cost tracking, safety signals, and enforcement for free.

## Installation

```bash
pip install aceteam-aep                    # Core (cost tracking + regex safety)
pip install aceteam-aep[safety]            # + HuggingFace model-based detectors
pip install aceteam-aep[safety,dashboard]  # + Local web dashboard
pip install aceteam-aep[all]               # Everything
```

## Quick Start — Wrap Your Existing Client

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

## Providers

- **OpenAI** (GPT-4o, GPT-5, o1, o3)
- **Anthropic** (Claude Opus, Sonnet, Haiku)
- **Google** (Gemini 2.5, 3.0)
- **xAI** (Grok)
- **Ollama** (local models)
- **OpenAI-compatible** (SambaNova, TheAgentic, DeepSeek)
