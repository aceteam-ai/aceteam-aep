# Layer 2 Protocol — Provenance + Governance Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the remaining two AEP pillars (Provenance + Governance) that require application-level integration. This is the paid tier — the SDK that makes agents enterprise-ready.

**Architecture:** Layer 2 sits above Layer 1 (proxy). It requires the caller to use the AEP SDK or send AEP headers. The proxy reads these headers and enforces governance rules. The SDK constructs ExecutionEnvelopes with citation chains and governance records.

**Tech Stack:** Python 3.12+, pydantic, existing AEP types (ExecutionEnvelope, Citation, GovernanceConfig)

**Repo:** `aceteam-ai/aceteam-aep`

**Issues:** #12 (provenance), #13 (governance)

---

## File Structure

```
src/aceteam_aep/
├── provenance/
│   ├── __init__.py
│   ├── tracker.py               # ProvenanceTracker — builds citation chains
│   ├── source_extractor.py      # Extract source refs from tool_call results
│   └── attribution.py           # Map output claims to input sources
├── governance/
│   ├── __init__.py
│   ├── classifier.py            # Classify data sensitivity per message
│   ├── consent.py               # Consent tracking + enforcement
│   ├── redactor.py              # Redact sensitive fields based on policy
│   └── policy_engine.py         # Evaluate GovernancePolicy against request
├── proxy/
│   ├── headers.py               # Parse/emit X-AEP-* headers
│   └── app.py                   # MODIFY — enforce governance from headers
├── wrap.py                      # MODIFY — track provenance in wrap() calls
├── envelope_builder.py          # MODIFY — attach provenance + governance
```

---

### Task 1: AEP Headers in Proxy

The bridge between Layer 1 and Layer 2. The proxy reads `X-AEP-*` headers from the caller, enforces governance rules, and strips them before forwarding.

**Files:**
- Create: `src/aceteam_aep/proxy/headers.py`
- Modify: `src/aceteam_aep/proxy/app.py`
- Create: `tests/test_proxy_headers.py`

**Headers spec:**
```
X-AEP-Entity: org:acme                    # Who is calling
X-AEP-Classification: confidential        # Data sensitivity level
X-AEP-Consent: training=no,sharing=org    # Consent directives
X-AEP-Budget: 5.00                        # Max budget for this call
X-AEP-Trace-ID: abc123                    # Link to parent execution
```

**Behavior:**
- Proxy reads headers, creates `ExecutionContext` for this call
- If `X-AEP-Classification: restricted` and no `X-AEP-Entity`, BLOCK (can't have restricted data without knowing who's asking)
- If `X-AEP-Budget` and cumulative cost would exceed, BLOCK
- Strip all `X-AEP-*` headers before forwarding to upstream
- Add `X-AEP-*` response headers with cost, enforcement, trace info

- [ ] Write tests (header parsing, budget enforcement, stripped before forward)
- [ ] Implement header parsing + enforcement
- [ ] Run tests, verify pass
- [ ] Commit

---

### Task 2: Provenance Tracker

Track what sources contributed to each LLM call.

**Files:**
- Create: `src/aceteam_aep/provenance/__init__.py`
- Create: `src/aceteam_aep/provenance/tracker.py`
- Create: `src/aceteam_aep/provenance/source_extractor.py`
- Create: `tests/test_provenance.py`

**ProvenanceTracker:**
```python
class ProvenanceTracker:
    def record_source(self, call_id: str, source: SourceRef) -> None:
        """Record that a source was used in this call's context."""

    def record_tool_result(self, call_id: str, tool_name: str, result: str) -> None:
        """Record a tool call result as a source."""

    def get_citations(self, call_id: str) -> list[Citation]:
        """Get all citations for a call."""

@dataclass
class SourceRef:
    source_type: str          # "document", "tool_call", "database", "url"
    source_id: str            # Document ID, URL, tool name
    content_preview: str      # First 200 chars
    confidence: float = 0.0   # From Trust Engine (future)
```

**Source extraction from messages:**
The `source_extractor` parses the messages array to find:
- `role: tool` messages → tool call results as sources
- `role: system` messages with document context → RAG sources
- URLs in message content → web sources

This is heuristic — it catches the obvious patterns. Deep source attribution (which sentence came from which source) is the Trust Engine research problem and would be a future `AttributionDetector`.

- [ ] Write tests (extract tool results, extract system doc context, build citations)
- [ ] Implement ProvenanceTracker + source_extractor
- [ ] Run tests, verify pass
- [ ] Commit

---

### Task 3: Provenance in wrap()

Wire provenance tracking into the `wrap()` function so Python SDK users get citations automatically.

**Files:**
- Modify: `src/aceteam_aep/wrap.py`
- Modify: `tests/test_wrap.py`

Add to `AepSession`:
```python
@property
def citations(self) -> list[Citation]:
    """All source citations recorded this session."""
    return self._provenance.get_all_citations()
```

The `_record_call` method now also runs the source extractor on the messages array.

- [ ] Write tests (citations populated from tool calls in messages)
- [ ] Wire provenance into AepSession._record_call
- [ ] Run tests, verify pass
- [ ] Commit

---

### Task 4: Governance Classifier

Classify data sensitivity of inputs and outputs.

**Files:**
- Create: `src/aceteam_aep/governance/__init__.py`
- Create: `src/aceteam_aep/governance/classifier.py`
- Create: `tests/test_governance/__init__.py`
- Create: `tests/test_governance/test_classifier.py`

**DataClassifier** — a SafetyDetector that classifies text sensitivity:
```python
class DataClassifier:
    name = "data_classification"

    def check(self, *, input_text, output_text, call_id, **kwargs):
        # Classify using: PII presence → confidential, keywords → internal, etc.
        # Returns SafetySignal with signal_type="data_classification"
        # and detail containing the classification level
```

Uses the PII detector as a sub-signal: if PII is detected → at least "confidential". Additional rules for specific patterns (financial data, health data, legal data).

This is a safety detector that plugs into the existing DetectorRegistry.

- [ ] Write tests (classify PII as confidential, clean text as public)
- [ ] Implement DataClassifier
- [ ] Run tests, verify pass
- [ ] Commit

---

### Task 5: Governance Policy Engine

Enforce governance policies at the proxy level.

**Files:**
- Create: `src/aceteam_aep/governance/policy_engine.py`
- Create: `tests/test_governance/test_policy_engine.py`

```python
class GovernancePolicyEngine:
    def evaluate(
        self,
        entity: str,
        classification: SecurityLevel,
        consent: dict[str, bool],
        policy: GovernancePolicy,
    ) -> GovernanceDecision:
        # Check: does this entity have clearance for this classification?
        # Check: does consent allow this action?
        # Returns: allow / deny / redact
```

Policies are per-org, loaded from config:
```yaml
governance:
  policies:
    org:acme:
      max_classification: confidential  # can access up to confidential
      consent: { training: false, sharing: same_org }
    org:beta:
      max_classification: internal
```

- [ ] Write tests (clearance check, consent check, deny on violation)
- [ ] Implement GovernancePolicyEngine
- [ ] Run tests, verify pass
- [ ] Commit

---

### Task 6: Redactor

Redact sensitive fields from responses based on governance policy.

**Files:**
- Create: `src/aceteam_aep/governance/redactor.py`
- Create: `tests/test_governance/test_redactor.py`

```python
class Redactor:
    def redact(self, text: str, signals: list[SafetySignal]) -> str:
        """Replace detected PII with [REDACTED] markers."""
        # Uses PII detector spans to know WHAT to redact and WHERE
```

This is the enforcement action for governance: instead of BLOCK, we REDACT. The response still goes through, but sensitive fields are replaced.

- [ ] Write tests (SSN redacted, email redacted, clean text unchanged)
- [ ] Implement Redactor
- [ ] Run tests, verify pass
- [ ] Commit

---

### Task 7: Wire Governance into Proxy

Connect everything: proxy reads headers, runs governance classifier, enforces policy, redacts if needed.

**Files:**
- Modify: `src/aceteam_aep/proxy/app.py`
- Create: `tests/test_proxy_governance.py`

New proxy flow with governance:
1. Read `X-AEP-*` headers → extract entity, classification, consent
2. Run input safety + governance classifier on input
3. If governance DENY → BLOCK request
4. Forward to upstream
5. Run output safety + governance classifier on output
6. If governance DENY → BLOCK or REDACT response
7. Return with `X-AEP-*` response headers including classification

- [ ] Write tests (governance block, redact, passthrough, missing headers)
- [ ] Wire governance into proxy_handler
- [ ] Run tests, verify pass
- [ ] Commit

---

### Task 8: ExecutionEnvelope Builder Integration

Wire everything into the envelope builder so each call produces a complete AEP envelope.

**Files:**
- Modify: `src/aceteam_aep/envelope_builder.py`
- Modify: `src/aceteam_aep/wrap.py`

The envelope now includes all 4 pillars:
```python
envelope = EnvelopeBuilder() \
    .set_cost_tree(cost_nodes) \
    .set_citations(provenance.get_citations(call_id)) \
    .set_governance(governance_record) \
    .set_safety(safety_signals, enforcement_decision) \
    .build()
```

Add `get_envelope()` to AepSession:
```python
client.aep.get_envelope()  # Returns full ExecutionEnvelope
```

- [ ] Write tests (envelope contains all 4 pillars)
- [ ] Wire into AepSession and proxy
- [ ] Run tests, verify pass
- [ ] Commit

---

## Milestone Checkpoints

| Milestone | Tasks | What It Unlocks |
|-----------|-------|-----------------|
| **AEP Headers** | 1 | Proxy can receive governance metadata from any client |
| **Provenance** | 2-3 | Citation chains in wrap() and proxy |
| **Governance** | 4-6 | Data classification, policy enforcement, redaction |
| **Integration** | 7-8 | Full 4-pillar AEP envelopes from both proxy and SDK |

## Dependencies

- Layer 1 production plan (storage, config) should be completed first
- Trust Engine research (confidence scores) feeds into provenance citation confidence — but is not blocking
- GovernanceConfig types already exist in `governance.py` — reuse, don't redefine
