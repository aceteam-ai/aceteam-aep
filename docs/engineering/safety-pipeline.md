# Cascading Confidence Pipeline

## Overview

The safety pipeline replaces the flat "run all detectors in parallel" model with a sequential cascade that short-circuits when confident enough. Each layer produces a probability (not a boolean), and the system stops running expensive layers once it has enough signal to decide.

```
Agent Action
    |
    v
Layer 0: Regex          free, <1ms     deterministic patterns
    | no match
    v
Layer 1: PAW             free, ~50ms   local LoRA classifier + logprobs
    | uncertain
    v
Layer 2: Content Model   free, ~100ms  local toxicity transformer
    | still uncertain
    v
Layer 3: TrustEngine     ~$0.001, ~500ms  LLM API multi-perspective
    |
    v
Calibration: weighted average of all layer scores
    |
    v
Escalation: PASS / FLAG / BLOCK based on thresholds
```

Most actions are obviously safe — regex and PAW handle them locally for free. Only ambiguous cases hit the API. Only truly uncertain cases reach a human.

## How It Works

### Layer Protocol

Every layer implements `CascadeLayer` and returns a `LayerResult`:

```python
@dataclass
class LayerResult:
    layer_name: str
    p_unsafe: float      # 0.0 = certainly safe, 1.0 = certainly unsafe
    confidence: float    # 0.0 = no opinion, 1.0 = fully certain
    signals: list[SafetySignal]
    latency_ms: float
```

### Short-Circuit Logic

After each layer, the pipeline computes a weighted average of all results so far. If combined confidence exceeds `confidence_threshold` and p_unsafe is decisively high or low, it stops:

- `p_unsafe >= block_above` with sufficient confidence → BLOCK (skip remaining layers)
- `p_unsafe < pass_below` with sufficient confidence → PASS (skip remaining layers)
- Otherwise → continue to next layer

### Layer Adapters

Each adapter wraps an existing detector:

| Adapter | Wraps | Confidence Source |
|---------|-------|-------------------|
| `RegexLayer` | PiiDetector, AgentThreatDetector | Hit = 1.0, miss = 0.1 (regex absence doesn't mean safe) |
| `PawLayer` | CustomSafetyDetector | logprob softmax P(Y) vs P(N) from llama.cpp |
| `ContentModelLayer` | ContentSafetyDetector | Toxicity classifier score |
| `TrustEngineLayer` | TrustEngineDetector | Multi-perspective P(safe) with reasoning |

### PAW Logprob Extraction

The key innovation: PAW (ProgramAsWeights) compiles natural language policy rules into LoRA adapters on Qwen 0.6B. Previously it returned binary Y/N with no confidence. Now:

1. After `llm.eval(input_tokens)`, we read logits via `llama_get_logits_ith(ctx, -1)`
2. Extract logits for Y and N token IDs
3. Compute softmax: `p_compliant = exp(logit_Y) / (exp(logit_Y) + exp(logit_N))`
4. Return as `score` on the SafetySignal

Real-world results on SSN detection rule:
- Input with SSN "123-45-6789": `p_compliant = 0.0495` (95% sure it's unsafe)
- Clean text: `p_compliant = 0.9999` (near-certain safe)

Falls back to boolean-only if logprob extraction fails (different llama.cpp version, etc).

See `safety/custom.py`: `_extract_yn_probability()`, `_paw_call_with_logprobs()`.

### Calibration

Currently uses confidence-weighted average of layer scores. The research track (IDEaS grant with UWaterloo) will replace this with:

- Temperature scaling (post-hoc, one scalar per detector)
- Linear probes on LLM hidden-state activations (Bayesian, ECE target <= 0.05)
- GP-based active learning for escalation threshold tuning

## Enabling the Pipeline

Add a `pipeline:` section to your policy YAML:

```yaml
pipeline:
  enabled: true
  pass_below: 0.3      # p_unsafe < 0.3 -> auto-PASS
  block_above: 0.7     # p_unsafe >= 0.7 -> auto-BLOCK
  confidence_threshold: 0.7
  layers:
    - name: regex
      weight: 1.0
    - name: paw
      weight: 1.5
    - name: trust_engine
      weight: 2.0
```

Without this section, behavior is identical to before (parallel detectors, severity-based enforcement). The pipeline is opt-in and backward compatible.

## Module Map

```
safety/
├── pipeline.py          # NEW: CascadeLayer, LayerResult, SafetyPipeline, adapters
├── custom.py            # UPDATED: PAW logprobs (PawResult, _extract_yn_probability)
├── base.py              # SafetySignal (score field), DetectorRegistry (unchanged)
├── trust_engine.py      # TrustEngineDetector (unchanged, wrapped by TrustEngineLayer)
├── pii.py               # PiiDetector (unchanged, wrapped by RegexLayer)
├── agent_threat.py      # AgentThreatDetector (unchanged, wrapped by RegexLayer)
├── content.py           # ContentSafetyDetector (unchanged, wrapped by ContentModelLayer)
└── ...

enforcement.py           # UPDATED: PipelinePolicy, evaluate_pipeline(), build_pipeline_from_policy()
proxy/app.py             # UPDATED: pipeline construction and evaluation in request flow
proxy/streaming.py       # UPDATED: pipeline support for streaming responses
```

## Observability

`PipelineResult` tracks everything needed to debug the cascade:

```python
@dataclass
class PipelineResult:
    p_unsafe: float              # combined score
    confidence: float            # combined confidence
    verdict: str                 # "pass", "flag", "block"
    signals: list[SafetySignal]  # all signals from all layers
    layer_results: list[LayerResult]  # per-layer detail
    layers_executed: int         # how many layers ran
    short_circuited_at: str | None    # which layer caused early exit
    total_latency_ms: float
```

## Thresholds Guide

| Use case | pass_below | block_above | confidence_threshold |
|----------|-----------|-------------|---------------------|
| Strict (bank, healthcare) | 0.1 | 0.5 | 0.8 |
| Balanced (enterprise) | 0.3 | 0.7 | 0.7 |
| Permissive (startup, dev) | 0.5 | 0.9 | 0.6 |

Lower `pass_below` = more actions flagged for human review. Lower `block_above` = more auto-blocks. Higher `confidence_threshold` = more layers run before deciding.
