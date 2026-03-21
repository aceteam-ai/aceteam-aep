# AEP Trust Engine Wrapper — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `pip install aceteam-aep` with a `wrap()` function that intercepts OpenAI/Anthropic clients, adds cost tracking + safety signals (PII, content safety, cost anomaly), and outputs to both CLI and a local web dashboard. Demo-ready for ClawCamp Oakland April 16.

**Architecture:** The `wrap()` function (already on `feat/wrap-function`) monkey-patches SDK clients to intercept every LLM call. Each call gets: (1) cost tracked via CostTracker, (2) safety evaluated by pluggable detectors (small local HuggingFace models, not regex), (3) enforcement decision (PASS/FLAG/BLOCK). Results surface via `client.aep` and optionally to a local web dashboard served by the package itself.

**Tech Stack:** Python 3.12+, pydantic, transformers (for local safety classifiers), httpx, uvicorn (for dashboard), jinja2 (dashboard templates), openai SDK, anthropic SDK

**Repo:** `aceteam-ai/aceteam-aep` on branch `feat/wrap-function` (already has wrap.py with regex-based T&S)

**Current state:** `wrap()` exists with regex PII + keyword content detection + cost anomaly. Tests pass. Need to upgrade detectors to real models, add enforcement, add async support, add dashboard, add CLI output.

---

## File Structure

```
src/aceteam_aep/
├── wrap.py                      # MODIFY — wrap() public API, AepSession, enforcement logic
├── safety/                      # CREATE — pluggable safety detection engine
│   ├── __init__.py              # SafetyDetector protocol, DetectorRegistry
│   ├── base.py                  # SafetyDetector protocol + SafetySignal dataclass (move from wrap.py)
│   ├── pii.py                   # PII detector using small transformer model
│   ├── content.py               # Content safety detector (Llama Guard / DeBERTa)
│   ├── cost_anomaly.py          # Cost spike detector (move from wrap.py)
│   └── prompt_injection.py      # Prompt injection detector
├── enforcement.py               # CREATE — PASS/FLAG/BLOCK decision engine
├── dashboard/                   # CREATE — local web dashboard
│   ├── __init__.py              # serve_dashboard() entry point
│   ├── app.py                   # uvicorn ASGI app
│   └── templates/
│       └── index.html           # Single-page dashboard
├── cli.py                       # CREATE — CLI pretty-print for safety signals
├── __init__.py                  # MODIFY — export new modules
tests/
├── test_wrap.py                 # MODIFY — update for new detector architecture
├── test_safety/                 # CREATE
│   ├── test_pii.py
│   ├── test_content.py
│   ├── test_cost_anomaly.py
│   └── test_prompt_injection.py
├── test_enforcement.py          # CREATE
├── test_dashboard.py            # CREATE
└── test_cli.py                  # CREATE
```

---

### Task 1: Extract Safety Framework — Base Protocol + Registry

**Files:**
- Create: `src/aceteam_aep/safety/__init__.py`
- Create: `src/aceteam_aep/safety/base.py`
- Modify: `src/aceteam_aep/wrap.py`
- Create: `tests/test_safety/__init__.py`
- Create: `tests/test_safety/test_base.py`

- [ ] **Step 1: Write tests for SafetyDetector protocol and DetectorRegistry**

```python
# tests/test_safety/test_base.py
from aceteam_aep.safety.base import SafetyDetector, SafetySignal, DetectorRegistry

class FakeDetector:
    name = "fake"
    def check(self, *, input_text: str, output_text: str, call_id: str, **kwargs) -> list[SafetySignal]:
        if "bad" in output_text:
            return [SafetySignal(signal_type="test", severity="high", call_id=call_id, detail="found bad")]
        return []

def test_detector_registry_runs_all_detectors():
    reg = DetectorRegistry()
    reg.add(FakeDetector())
    signals = reg.run_all(input_text="", output_text="something bad", call_id="abc")
    assert len(signals) == 1
    assert signals[0].signal_type == "test"

def test_detector_registry_empty_on_clean_input():
    reg = DetectorRegistry()
    reg.add(FakeDetector())
    signals = reg.run_all(input_text="", output_text="all good", call_id="abc")
    assert signals == []

def test_detector_registry_catches_detector_errors():
    class BrokenDetector:
        name = "broken"
        def check(self, **kwargs) -> list[SafetySignal]:
            raise RuntimeError("boom")
    reg = DetectorRegistry()
    reg.add(BrokenDetector())
    signals = reg.run_all(input_text="", output_text="test", call_id="abc")
    assert signals == []  # never crashes, just skips
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/jason/workspace/aceteam-ai/aceteam-aep && uv run pytest tests/test_safety/test_base.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement SafetySignal, SafetyDetector protocol, DetectorRegistry**

```python
# src/aceteam_aep/safety/base.py
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)

@dataclass
class SafetySignal:
    signal_type: str          # "pii", "content_safety", "cost_anomaly", "prompt_injection"
    severity: str             # "low", "medium", "high"
    call_id: str
    detail: str
    detector: str = ""        # which detector raised this
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

@runtime_checkable
class SafetyDetector(Protocol):
    name: str
    def check(self, *, input_text: str, output_text: str, call_id: str, **kwargs) -> list[SafetySignal]: ...

class DetectorRegistry:
    def __init__(self) -> None:
        self._detectors: list[SafetyDetector] = []

    def add(self, detector: SafetyDetector) -> None:
        self._detectors.append(detector)

    def run_all(self, *, input_text: str, output_text: str, call_id: str, **kwargs) -> list[SafetySignal]:
        signals: list[SafetySignal] = []
        for det in self._detectors:
            try:
                results = det.check(input_text=input_text, output_text=output_text, call_id=call_id, **kwargs)
                for s in results:
                    s.detector = det.name
                signals.extend(results)
            except Exception:
                log.warning("Detector %s failed, skipping", getattr(det, "name", "unknown"), exc_info=True)
        return signals
```

```python
# src/aceteam_aep/safety/__init__.py
from .base import DetectorRegistry, SafetyDetector, SafetySignal

__all__ = ["DetectorRegistry", "SafetyDetector", "SafetySignal"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/jason/workspace/aceteam-ai/aceteam-aep && uv run pytest tests/test_safety/test_base.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/jason/workspace/aceteam-ai/aceteam-aep
git add src/aceteam_aep/safety/ tests/test_safety/
git commit -m "feat(safety): extract SafetyDetector protocol and DetectorRegistry"
```

---

### Task 2: PII Detector — HuggingFace Model-Based

Replace regex PII detection with a small local transformer model.

**Model candidate:** `iiiorg/piiranha-v1-detect-personal-information` (BERT-based NER, ~110M params, runs on CPU) or `bigcode/starpii` or `lakshyakh93/deberta_finetuned_pii`. Evaluate at implementation time — pick the smallest one that detects SSN, email, phone, credit card.

**Files:**
- Create: `src/aceteam_aep/safety/pii.py`
- Create: `tests/test_safety/test_pii.py`

- [ ] **Step 1: Write tests for PII detector**

```python
# tests/test_safety/test_pii.py
import pytest
from aceteam_aep.safety.pii import PiiDetector

@pytest.fixture(scope="module")
def detector():
    return PiiDetector()

def test_detects_ssn(detector):
    signals = detector.check(input_text="", output_text="His SSN is 123-45-6789", call_id="t1")
    assert any(s.signal_type == "pii" for s in signals)

def test_detects_email(detector):
    signals = detector.check(input_text="", output_text="Contact john.doe@company.com", call_id="t2")
    assert any(s.signal_type == "pii" for s in signals)

def test_clean_text_no_signal(detector):
    signals = detector.check(input_text="", output_text="The weather is nice today.", call_id="t3")
    assert not any(s.signal_type == "pii" for s in signals)

def test_detects_phone_number(detector):
    signals = detector.check(input_text="", output_text="Call me at (555) 123-4567", call_id="t4")
    assert any(s.signal_type == "pii" for s in signals)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/jason/workspace/aceteam-ai/aceteam-aep && uv run pytest tests/test_safety/test_pii.py -v`
Expected: FAIL

- [ ] **Step 3: Implement PiiDetector with lazy-loaded HuggingFace model**

```python
# src/aceteam_aep/safety/pii.py
from __future__ import annotations
import logging
from .base import SafetyDetector, SafetySignal

log = logging.getLogger(__name__)

# PII entity types we care about
_PII_ENTITIES = {"SSN", "EMAIL", "PHONE", "CREDIT_CARD", "IP_ADDRESS", "PERSON", "LOCATION"}

class PiiDetector:
    """Detects PII in text using a small local transformer NER model.

    Model is lazy-loaded on first check() call to avoid import-time overhead.
    Falls back to regex if transformers is not installed.
    """

    name = "pii"

    def __init__(self, model_name: str = "iiiorg/piiranha-v1-detect-personal-information") -> None:
        self._model_name = model_name
        self._pipeline = None
        self._fallback = False

    def _load(self) -> None:
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "token-classification",
                model=self._model_name,
                aggregation_strategy="simple",
                device=-1,  # CPU
            )
        except ImportError:
            log.warning("transformers not installed, PII detector falling back to regex")
            self._fallback = True
        except Exception:
            log.warning("Failed to load PII model %s, falling back to regex", self._model_name, exc_info=True)
            self._fallback = True

    def check(self, *, input_text: str, output_text: str, call_id: str, **kwargs) -> list[SafetySignal]:
        if self._pipeline is None and not self._fallback:
            self._load()

        if self._fallback:
            return self._check_regex(output_text, call_id)
        return self._check_model(output_text, call_id)

    def _check_model(self, text: str, call_id: str) -> list[SafetySignal]:
        assert self._pipeline is not None
        # Truncate to avoid OOM on large outputs
        results = self._pipeline(text[:2048])
        signals: list[SafetySignal] = []
        seen_types: set[str] = set()
        for entity in results:
            ent_type = entity.get("entity_group", "").upper()
            if ent_type in _PII_ENTITIES and ent_type not in seen_types:
                seen_types.add(ent_type)
                signals.append(SafetySignal(
                    signal_type="pii",
                    severity="high",
                    call_id=call_id,
                    detail=f"PII detected: {ent_type} (score={entity.get('score', 0):.2f})",
                ))
        return signals

    def _check_regex(self, text: str, call_id: str) -> list[SafetySignal]:
        """Fallback regex detection when transformers unavailable."""
        import re
        patterns = {
            "SSN": re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),
            "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
            "EMAIL": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
            "PHONE": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        }
        signals: list[SafetySignal] = []
        for pii_type, pattern in patterns.items():
            if pattern.search(text):
                signals.append(SafetySignal(
                    signal_type="pii",
                    severity="high",
                    call_id=call_id,
                    detail=f"PII pattern detected: {pii_type} (regex fallback)",
                ))
        return signals
```

- [ ] **Step 4: Add `transformers` + `torch` as optional dependency in pyproject.toml**

```toml
# Add to [project.optional-dependencies]
safety = ["transformers>=4.40", "torch>=2.0"]
all = ["aceteam-aep[xai,ollama,safety]"]
dev = ["aceteam-aep[all]", ...]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/jason/workspace/aceteam-ai/aceteam-aep && uv sync --extra dev && uv run pytest tests/test_safety/test_pii.py -v`
Expected: PASS (4 tests). Note: first run downloads model (~440MB).

- [ ] **Step 6: Commit**

```bash
git add src/aceteam_aep/safety/pii.py tests/test_safety/test_pii.py pyproject.toml
git commit -m "feat(safety): add PII detector with HuggingFace model + regex fallback"
```

---

### Task 3: Content Safety Detector — Local Classifier

**Model candidate:** `valurank/distilroberta-toxicity` (~82M params, binary toxic/non-toxic, CPU-friendly) or `unitary/toxic-bert`. For v1, a toxicity classifier is sufficient. Llama Guard (8B) is too large for inline per-call evaluation — save for the multi-agent judge debate architecture (Trust Engine v2).

**Files:**
- Create: `src/aceteam_aep/safety/content.py`
- Create: `tests/test_safety/test_content.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_safety/test_content.py
import pytest
from aceteam_aep.safety.content import ContentSafetyDetector

@pytest.fixture(scope="module")
def detector():
    return ContentSafetyDetector()

def test_flags_toxic_content(detector):
    signals = detector.check(
        input_text="", output_text="I will kill you and your entire family",
        call_id="t1"
    )
    assert any(s.signal_type == "content_safety" for s in signals)

def test_clean_content_passes(detector):
    signals = detector.check(
        input_text="", output_text="The quarterly earnings report shows 12% growth.",
        call_id="t2"
    )
    assert not any(s.signal_type == "content_safety" for s in signals)

def test_flags_harmful_instructions(detector):
    signals = detector.check(
        input_text="", output_text="Here is how to synthesize methamphetamine step by step",
        call_id="t3"
    )
    assert any(s.signal_type == "content_safety" for s in signals)
```

- [ ] **Step 2: Run tests, verify fail**

- [ ] **Step 3: Implement ContentSafetyDetector**

```python
# src/aceteam_aep/safety/content.py
from __future__ import annotations
import logging
from .base import SafetyDetector, SafetySignal

log = logging.getLogger(__name__)

class ContentSafetyDetector:
    """Classifies text as safe/unsafe using a small local toxicity model."""

    name = "content_safety"

    def __init__(
        self,
        model_name: str = "valurank/distilroberta-toxicity",
        threshold: float = 0.7,
    ) -> None:
        self._model_name = model_name
        self._threshold = threshold
        self._pipeline = None
        self._available = True

    def _load(self) -> None:
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-classification",
                model=self._model_name,
                device=-1,
            )
        except Exception:
            log.warning("Content safety model unavailable", exc_info=True)
            self._available = False

    def check(self, *, input_text: str, output_text: str, call_id: str, **kwargs) -> list[SafetySignal]:
        if self._pipeline is None and self._available:
            self._load()
        if not self._available:
            return []

        signals: list[SafetySignal] = []
        # Check output (primary) and input (for prompt injection / jailbreak detection)
        for text, source in [(output_text, "output"), (input_text, "input")]:
            if not text:
                continue
            try:
                result = self._pipeline(text[:512])
                if result and result[0].get("label", "").lower() in ("toxic", "unsafe", "harmful"):
                    score = result[0].get("score", 0)
                    if score >= self._threshold:
                        signals.append(SafetySignal(
                            signal_type="content_safety",
                            severity="high" if score > 0.9 else "medium",
                            call_id=call_id,
                            detail=f"Unsafe content in {source} (score={score:.2f}, model={self._model_name})",
                        ))
            except Exception:
                log.warning("Content safety check failed for %s", source, exc_info=True)
        return signals
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git add src/aceteam_aep/safety/content.py tests/test_safety/test_content.py
git commit -m "feat(safety): add content safety detector with local toxicity model"
```

---

### Task 4: Cost Anomaly Detector — Extract from wrap.py

Move cost anomaly detection from inline in wrap.py to its own detector class.

**Files:**
- Create: `src/aceteam_aep/safety/cost_anomaly.py`
- Create: `tests/test_safety/test_cost_anomaly.py`
- Modify: `src/aceteam_aep/wrap.py` — remove inline cost anomaly logic

- [ ] **Step 1: Write tests**

```python
# tests/test_safety/test_cost_anomaly.py
from decimal import Decimal
from aceteam_aep.safety.cost_anomaly import CostAnomalyDetector

def test_no_anomaly_with_few_calls():
    det = CostAnomalyDetector()
    signals = det.check(input_text="", output_text="", call_id="1", call_cost=Decimal("1.00"))
    assert signals == []

def test_detects_spike_after_baseline():
    det = CostAnomalyDetector(min_calls=3, multiplier=5)
    for i in range(3):
        det.check(input_text="", output_text="", call_id=str(i), call_cost=Decimal("0.01"))
    signals = det.check(input_text="", output_text="", call_id="spike", call_cost=Decimal("1.00"))
    assert any(s.signal_type == "cost_anomaly" for s in signals)

def test_no_false_positive_on_normal_variation():
    det = CostAnomalyDetector(min_calls=3, multiplier=5)
    for i in range(3):
        det.check(input_text="", output_text="", call_id=str(i), call_cost=Decimal("0.01"))
    signals = det.check(input_text="", output_text="", call_id="normal", call_cost=Decimal("0.03"))
    assert not any(s.signal_type == "cost_anomaly" for s in signals)
```

- [ ] **Step 2: Run tests, verify fail**
- [ ] **Step 3: Implement CostAnomalyDetector**

```python
# src/aceteam_aep/safety/cost_anomaly.py
from __future__ import annotations
from decimal import Decimal
from .base import SafetySignal

class CostAnomalyDetector:
    name = "cost_anomaly"

    def __init__(self, min_calls: int = 3, multiplier: int = 5) -> None:
        self._min_calls = min_calls
        self._multiplier = multiplier
        self._history: list[Decimal] = []

    def check(self, *, input_text: str, output_text: str, call_id: str, call_cost: Decimal = Decimal("0"), **kwargs) -> list[SafetySignal]:
        signals: list[SafetySignal] = []
        self._history.append(call_cost)
        if len(self._history) <= self._min_calls:
            return signals
        prior = self._history[:-1]
        avg = sum(prior, Decimal("0")) / len(prior)
        if avg > 0 and call_cost > avg * self._multiplier:
            signals.append(SafetySignal(
                signal_type="cost_anomaly",
                severity="medium",
                call_id=call_id,
                detail=f"Cost ${call_cost:.6f} is >{self._multiplier}x session avg ${avg:.6f}",
            ))
        return signals
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Update safety/__init__.py exports**

```python
from .cost_anomaly import CostAnomalyDetector
from .content import ContentSafetyDetector
from .pii import PiiDetector
```

- [ ] **Step 6: Commit**

```bash
git add src/aceteam_aep/safety/cost_anomaly.py tests/test_safety/test_cost_anomaly.py src/aceteam_aep/safety/__init__.py
git commit -m "feat(safety): extract cost anomaly detector as pluggable module"
```

---

### Task 5: Enforcement Engine — PASS / FLAG / BLOCK

**Files:**
- Create: `src/aceteam_aep/enforcement.py`
- Create: `tests/test_enforcement.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_enforcement.py
from aceteam_aep.enforcement import EnforcementPolicy, EnforcementDecision, evaluate
from aceteam_aep.safety.base import SafetySignal

def _sig(signal_type: str, severity: str) -> SafetySignal:
    return SafetySignal(signal_type=signal_type, severity=severity, call_id="t", detail="test")

def test_no_signals_passes():
    result = evaluate([], EnforcementPolicy())
    assert result.action == "pass"

def test_high_severity_blocks():
    result = evaluate([_sig("content_safety", "high")], EnforcementPolicy())
    assert result.action == "block"

def test_medium_severity_flags():
    result = evaluate([_sig("cost_anomaly", "medium")], EnforcementPolicy())
    assert result.action == "flag"

def test_custom_policy_overrides():
    policy = EnforcementPolicy(block_on=frozenset(), flag_on=frozenset())  # nothing triggers
    result = evaluate([_sig("content_safety", "high")], policy)
    assert result.action == "pass"
```

- [ ] **Step 2: Run tests, verify fail**
- [ ] **Step 3: Implement enforcement**

```python
# src/aceteam_aep/enforcement.py
from __future__ import annotations
from dataclasses import dataclass, field
from .safety.base import SafetySignal

@dataclass(frozen=True)
class EnforcementPolicy:
    block_on: frozenset[str] = frozenset({"high"})
    flag_on: frozenset[str] = frozenset({"medium"})
    # Per-signal-type overrides (e.g., always allow cost_anomaly)
    allow_types: frozenset[str] = frozenset()

@dataclass
class EnforcementDecision:
    action: str  # "pass", "flag", "block"
    signals: list[SafetySignal] = field(default_factory=list)
    reason: str = ""

def evaluate(signals: list[SafetySignal], policy: EnforcementPolicy) -> EnforcementDecision:
    if not signals:
        return EnforcementDecision(action="pass")

    active = [s for s in signals if s.signal_type not in policy.allow_types]
    if not active:
        return EnforcementDecision(action="pass", signals=signals)

    if any(s.severity in policy.block_on for s in active):
        return EnforcementDecision(
            action="block", signals=active,
            reason="; ".join(f"{s.signal_type}: {s.detail}" for s in active if s.severity in policy.block_on),
        )
    if any(s.severity in policy.flag_on for s in active):
        return EnforcementDecision(
            action="flag", signals=active,
            reason="; ".join(f"{s.signal_type}: {s.detail}" for s in active if s.severity in policy.flag_on),
        )
    return EnforcementDecision(action="pass", signals=active)
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git add src/aceteam_aep/enforcement.py tests/test_enforcement.py
git commit -m "feat: add PASS/FLAG/BLOCK enforcement engine"
```

---

### Task 6: Rewire wrap.py — Use DetectorRegistry + Enforcement

Replace the inline regex/keyword checks in wrap.py with the pluggable detector registry.

**Files:**
- Modify: `src/aceteam_aep/wrap.py`
- Modify: `tests/test_wrap.py`

- [ ] **Step 1: Update wrap.py**

Key changes:
- `AepSession` gets a `DetectorRegistry` and `EnforcementPolicy`
- `_record_call()` runs `registry.run_all()` then `evaluate()`
- Remove `_PII_PATTERNS`, `_HARMFUL_KEYWORDS`, `_SENSITIVE_TOPICS` and all inline detection methods
- Add `enforcement` property to AepSession
- `wrap()` accepts optional `detectors` list and `policy` parameter
- Default detectors: `[CostAnomalyDetector(), PiiDetector(), ContentSafetyDetector()]`
- Default detectors that require `transformers` are only added if available

- [ ] **Step 2: Update tests to use new detector architecture**

Existing tests should still pass — the behavior is the same, just the internal implementation changed. Update the import of `SafetySignal` to come from `safety.base` instead of `wrap`.

- [ ] **Step 3: Run full test suite**

Run: `cd /home/jason/workspace/aceteam-ai/aceteam-aep && uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add src/aceteam_aep/wrap.py tests/test_wrap.py
git commit -m "refactor: rewire wrap() to use pluggable DetectorRegistry + enforcement"
```

---

### Task 7: CLI Pretty-Print Output

**Files:**
- Create: `src/aceteam_aep/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_cli.py
from decimal import Decimal
from aceteam_aep.cli import format_session_summary
from aceteam_aep.safety.base import SafetySignal

def test_format_summary_clean():
    output = format_session_summary(cost=Decimal("0.0042"), signals=[], call_count=3)
    assert "$0.0042" in output
    assert "3 calls" in output
    assert "PASS" in output

def test_format_summary_with_signals():
    sig = SafetySignal(signal_type="pii", severity="high", call_id="x", detail="SSN found")
    output = format_session_summary(cost=Decimal("0.01"), signals=[sig], call_count=1)
    assert "pii" in output.lower()
    assert "BLOCK" in output or "FLAG" in output
```

- [ ] **Step 2: Implement CLI formatter**

```python
# src/aceteam_aep/cli.py
from __future__ import annotations
from decimal import Decimal
from .safety.base import SafetySignal
from .enforcement import EnforcementPolicy, evaluate

def format_session_summary(
    *, cost: Decimal, signals: list[SafetySignal], call_count: int,
    policy: EnforcementPolicy | None = None,
) -> str:
    policy = policy or EnforcementPolicy()
    decision = evaluate(signals, policy)
    action_color = {"pass": "\033[32m", "flag": "\033[33m", "block": "\033[31m"}
    reset = "\033[0m"
    color = action_color.get(decision.action, "")

    lines = [
        f"\n{'─' * 50}",
        f"  AEP Session Summary",
        f"{'─' * 50}",
        f"  Calls: {call_count}",
        f"  Cost:  ${cost:.6f}",
        f"  Safety: {color}{decision.action.upper()}{reset}",
    ]
    if signals:
        lines.append(f"  Signals ({len(signals)}):")
        for s in signals:
            sev_color = {"high": "\033[31m", "medium": "\033[33m", "low": "\033[36m"}.get(s.severity, "")
            lines.append(f"    [{sev_color}{s.severity.upper()}{reset}] {s.signal_type}: {s.detail}")
    lines.append(f"{'─' * 50}\n")
    return "\n".join(lines)
```

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Add `print_summary()` method to AepSession in wrap.py**
- [ ] **Step 5: Commit**

```bash
git add src/aceteam_aep/cli.py tests/test_cli.py src/aceteam_aep/wrap.py
git commit -m "feat: add CLI pretty-print for AEP session summary"
```

---

### Task 8: Local Web Dashboard

A minimal local dashboard served by the package. `client.aep.serve_dashboard()` starts a local uvicorn server showing cost + signals in real-time.

**Files:**
- Create: `src/aceteam_aep/dashboard/__init__.py`
- Create: `src/aceteam_aep/dashboard/app.py`
- Create: `src/aceteam_aep/dashboard/templates/index.html`
- Create: `tests/test_dashboard.py`

- [ ] **Step 1: Add dashboard dependencies to pyproject.toml**

```toml
dashboard = ["uvicorn>=0.30", "jinja2>=3.1", "starlette>=0.38"]
all = ["aceteam-aep[xai,ollama,safety,dashboard]"]
```

- [ ] **Step 2: Write test for dashboard ASGI app**

```python
# tests/test_dashboard.py
import pytest
from starlette.testclient import TestClient
from decimal import Decimal
from aceteam_aep.dashboard.app import create_app

def test_dashboard_returns_200():
    app = create_app(get_state=lambda: {"cost": Decimal("0.01"), "signals": [], "calls": 2, "spans": []})
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "$0.01" in resp.text

def test_dashboard_api_json():
    app = create_app(get_state=lambda: {"cost": Decimal("0.01"), "signals": [], "calls": 2, "spans": []})
    client = TestClient(app)
    resp = client.get("/api/state")
    assert resp.status_code == 200
    data = resp.json()
    assert data["calls"] == 2
```

- [ ] **Step 3: Implement dashboard app**

Single-page Starlette app with:
- `GET /` — HTML page with auto-refresh (polls `/api/state` every 2s)
- `GET /api/state` — JSON of current session state
- Dashboard shows: total cost, call count, safety signals list, enforcement status, per-call timeline

- [ ] **Step 4: Implement `index.html` template**

Clean, dark-themed single page with:
- Cost counter (large number, top)
- Safety status badge (green PASS / yellow FLAG / red BLOCK)
- Signal list with severity colors
- Call timeline (spans)
- Auto-refreshes via `fetch('/api/state')` every 2s

- [ ] **Step 5: Add `serve_dashboard()` to AepSession**

```python
# In wrap.py AepSession:
def serve_dashboard(self, port: int = 8899) -> None:
    from .dashboard import serve
    serve(self, port=port)
```

- [ ] **Step 6: Run tests, verify pass**
- [ ] **Step 7: Commit**

```bash
git add src/aceteam_aep/dashboard/ tests/test_dashboard.py pyproject.toml src/aceteam_aep/wrap.py
git commit -m "feat: add local web dashboard for AEP session monitoring"
```

---

### Task 9: Async Support for wrap()

The current `wrap()` only patches sync `.create()`. Need to also patch async `.create()` for `AsyncOpenAI` and `AsyncAnthropic`.

**Files:**
- Modify: `src/aceteam_aep/wrap.py`
- Modify: `tests/test_wrap.py`

- [ ] **Step 1: Write async tests**

```python
# In tests/test_wrap.py
import asyncio

@pytest.mark.asyncio
async def test_async_openai_cost_recorded():
    client = _make_async_openai_client()
    wrap(client)
    await client.chat.completions.create(model="gpt-4o", messages=[])
    assert client.aep.cost_usd > Decimal("0")
```

- [ ] **Step 2: Add async wrappers in wrap.py**

For each `_wrap_openai` / `_wrap_anthropic`, check if the client is async (has `__aenter__` or `isinstance(AsyncOpenAI)`) and patch accordingly with `async def patched_create`.

- [ ] **Step 3: Run tests, verify pass**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat: add async support for wrap() — AsyncOpenAI + AsyncAnthropic"
```

---

### Task 10: Integration Test + README Update

**Files:**
- Create: `examples/quickstart.py`
- Modify: `README.md`

- [ ] **Step 1: Create quickstart example**

```python
# examples/quickstart.py
"""AEP wrapper quickstart — run with: python examples/quickstart.py"""
import openai
from aceteam_aep import wrap

client = wrap(openai.OpenAI(), entity="demo")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)

print(response.choices[0].message.content)
client.aep.print_summary()

# To launch the dashboard:
# client.aep.serve_dashboard()  # opens http://localhost:8899
```

- [ ] **Step 2: Update README.md with wrap() usage, safety signals, dashboard**

- [ ] **Step 3: Run full test suite**

Run: `cd /home/jason/workspace/aceteam-ai/aceteam-aep && uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 4: Commit and push**

```bash
git add examples/ README.md
git commit -m "docs: add quickstart example and update README for wrap() + safety"
git push origin feat/wrap-function
```

---

### Task 11: Prompt Injection Detector (Stretch)

**Model candidate:** `protectai/deberta-v3-base-prompt-injection-v2` (~86M params, binary classification). Only if time permits before ClawCamp.

**Files:**
- Create: `src/aceteam_aep/safety/prompt_injection.py`
- Create: `tests/test_safety/test_prompt_injection.py`

Same pattern as PII and Content detectors — lazy-load model, classify input text, return SafetySignal if injection detected.

---

## Milestone Checkpoints

| Milestone | Tasks | What's Demo-able |
|-----------|-------|-------------------|
| **M1: Pluggable Safety** | 1-4 | `wrap()` with real model-based PII + content + cost detection |
| **M2: Enforcement** | 5-6 | PASS/FLAG/BLOCK decisions on every call |
| **M3: Visibility** | 7-8 | CLI summary + local web dashboard |
| **M4: Production-Ready** | 9-10 | Async support, examples, docs |
| **M5: Stretch** | 11 | Prompt injection detection |

## ClawCamp Demo Script (April 16)

```bash
pip install aceteam-aep[safety,dashboard]
```

```python
from aceteam_aep import wrap
import openai

client = wrap(openai.OpenAI(), entity="workshop-attendee")

# Normal call — PASS
client.chat.completions.create(model="gpt-4o-mini", messages=[...])

# Trigger PII detection — FLAG/BLOCK
client.chat.completions.create(model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is John's SSN?"}])

# Show dashboard
client.aep.serve_dashboard()  # http://localhost:8899
client.aep.print_summary()    # CLI output
```

Attendee leaves with: a GitHub star, a working safety wrapper, and telemetry flowing.
