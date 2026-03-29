# AEP-Confidence — Calibrated Safety Scoring via Ensemble Judges

**Status:** Draft
**Date:** 2026-03-29
**Authors:** AceTeam Engineering
**License:** Apache 2.0

---

## 1. Overview

AEP-Confidence defines how safety verdicts include calibrated confidence scores from an ensemble of diverse judge models. It replaces binary severity (HIGH/MEDIUM/LOW) with a continuous probability: "73% confident this output is safe."

This enables:
- Threshold-based enforcement (block above 0.6, flag above 0.3)
- Progressive autonomy (high-confidence agents get more latitude)
- Meaningful audit records (numeric scores, not just labels)
- Cross-org comparability (calibrated scores have consistent meaning)

## 2. Trust Engine Interface

A Trust Engine is any system that evaluates agent outputs using multiple independent judges and produces a calibrated confidence score.

### 2.1 Judge Protocol

Each judge evaluates a single (input, output) pair and returns a safety assessment:

```
JudgeRequest = {
    input_text:   string    // the prompt sent to the LLM
    output_text:  string    // the response from the LLM
    policy:       string    // the safety policy to evaluate against (natural language)
    judge_id:     string    // unique identifier for this judge instance
}

JudgeResponse = {
    safe:        boolean    // binary safety judgment
    score:       float      // confidence in the judgment (0.0 = uncertain, 1.0 = certain)
    reasoning:   string     // explanation of the judgment (optional)
    judge_id:    string     // matches request
    latency_ms:  integer    // time taken for this judge
}
```

Judges MUST be stateless: same input always produces the same output (modulo model temperature, which should be 0 for safety evaluation).

### 2.2 Judge Diversity

Calibration improves with judge diversity. Diversity is achieved across:

| Dimension | Examples |
|-----------|---------|
| **Model family** | Gemma, Llama, Mistral, Qwen |
| **Model size** | 7B, 9B, 14B |
| **Prompt strategy** | Direct ask, chain-of-thought, few-shot, role-play |
| **Few-shot examples** | Different example sets per judge |

Research backing (UWaterloo/Vector Institute):
- 4 models × 3 prompts × 4 few-shot strategies = 48 diverse judges
- 51 judges → AUC 0.75 discriminative ability
- Adding 2 additional judge models improved ECE from 0.70 to 0.45

### 2.3 Model Serving

Judges are served via any OpenAI-compatible chat completions endpoint:

```
POST /v1/chat/completions
{
    "model": "gemma-2-9b-it",
    "messages": [
        {"role": "system", "content": "You are a safety evaluator..."},
        {"role": "user", "content": "Evaluate this agent output for safety..."}
    ],
    "temperature": 0
}
```

Supported backends:
- Ollama (local, consumer GPU)
- vLLM / SGLang (self-hosted, production)
- Any OpenAI-compatible API

The Trust Engine implementation is responsible for model serving. The protocol only defines the judge request/response format.

## 3. Confidence Aggregation

### 3.1 Basic Aggregation

Given N judge responses, the aggregate confidence is:

```
P(safe) = (1/N) * sum(judge_i.score for each judge where judge_i.safe == true)
         + (1/N) * sum((1 - judge_i.score) for each judge where judge_i.safe == false)
```

Simplified: if a judge says "safe with 0.9 confidence", that contributes 0.9 to P(safe). If a judge says "unsafe with 0.8 confidence", that contributes 0.2 to P(safe).

### 3.2 Weighted Aggregation

Judges may have different reliability. Weights can be assigned based on:
- Historical accuracy on labeled data
- Model capability (larger models may get higher weight)
- Prompt strategy effectiveness

```
P(safe) = sum(w_i * adjusted_score_i) / sum(w_i)
```

Weights are implementation-specific and not part of the protocol.

### 3.3 Confidence in the AEP Signal

The confidence score appears in the `SafetySignal.score` field:

```json
{
    "signal_type": "trust_engine",
    "severity": "medium",
    "detail": "Trust Engine: 3 judges, P(safe)=0.73",
    "score": 0.73,
    "detector": "trust_engine:v1"
}
```

And in the response header:

```http
X-AEP-Confidence: 0.73
```

## 4. Enforcement Integration

Confidence scores integrate with `EnforcementPolicy` thresholds:

```yaml
detectors:
  trust_engine:
    enabled: true
    action: block
    threshold: 0.6    # block if P(safe) < 0.6
```

Enforcement logic:
- `P(safe) >= threshold` → PASS (or defer to other detectors)
- `P(safe) < threshold` → action specified in policy (block/flag)

This replaces binary severity mapping with continuous threshold-based decisions.

## 5. Caching

### 5.1 Verdict Cache

Trust Engine evaluation is expensive (N model inference calls per LLM call). Caching avoids recomputation for repeated inputs.

Cache key:

```
cache_key = SHA-256(input_text || output_text || policy_hash)
```

Cache entry:

```
{
    confidence: float,
    signals: Signal[],
    judges: JudgeResponse[],
    cached_at: ISO 8601,
    ttl_seconds: integer
}
```

### 5.2 Prefix Caching (Conversations)

Multi-turn conversations repeat the full message history. Only the latest turn needs evaluation.

The cache uses prefix-based lookup:

```
Conversation: [msg1, msg2, msg3, msg4]

Key: SHA-256(msg1 || msg2 || msg3)  → cached verdict for prefix
Only evaluate: msg4 in context of cached prefix verdict
```

Implementation may use radix trees for efficient prefix matching. This is an implementation detail, not part of the protocol.

### 5.3 Cache Invalidation

- **TTL-based:** cache entries expire after a configurable duration (default: 5 minutes)
- **Policy change:** cache is invalidated when the enforcement policy changes
- **Explicit:** `aceteam-aep cache clear` or API call

## 6. Wire Format

### 6.1 Request Headers

When a client wants to request Trust Engine evaluation:

```http
X-AEP-Require-Confidence: true
X-AEP-Min-Judges: 3
```

These are hints. The proxy may ignore them if Trust Engine is not available.

### 6.2 Response Headers

```http
X-AEP-Confidence: 0.73
X-AEP-Judges: 3
X-AEP-Cache-Hit: true
```

### 6.3 State API

The proxy state endpoint includes confidence data:

```json
{
    "confidence": {
        "last_score": 0.73,
        "judges_used": 3,
        "cache_hit_rate": 0.62,
        "avg_latency_ms": 1240
    }
}
```

## 7. Conformance

An AEP-Confidence conformant implementation MUST:

1. Use the JudgeRequest/JudgeResponse format (Section 2.1)
2. Aggregate confidence using at least the basic method (Section 3.1)
3. Report confidence in `X-AEP-Confidence` response header
4. Report judge count in `X-AEP-Judges` response header
5. Integrate with EnforcementPolicy thresholds (Section 4)

An implementation MAY:
- Use weighted aggregation (Section 3.2)
- Implement caching (Section 5)
- Support prefix caching for conversations (Section 5.2)

## 8. Security Considerations

- **Judge manipulation:** if an attacker controls a judge model, they can skew confidence scores. Mitigate with diverse judges from independent providers.
- **Cache poisoning:** if an attacker can poison the cache, subsequent lookups return wrong scores. Mitigate with cache key integrity checks.
- **Model drift:** judge models may change behavior over time. Mitigate with periodic recalibration against labeled data.
- **Cost attacks:** requesting Trust Engine evaluation on high volumes creates compute costs. Mitigate with rate limiting and caching.
