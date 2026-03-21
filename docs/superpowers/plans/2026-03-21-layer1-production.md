# Layer 1 Production Readiness — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the AEP proxy production-ready: persistent storage, config system, multi-tenant isolation, streaming support, prompt injection detector.

**Architecture:** Layer 1 (proxy) handles Cost + Safety from the wire. This plan fills the gaps to go from demo to production: data survives restarts, multiple orgs can share one proxy, streaming works, and the proxy is configurable without code changes.

**Tech Stack:** Python 3.12+, SQLite (aiosqlite), Starlette, pydantic, YAML (pyyaml), SSE

**Repo:** `aceteam-ai/aceteam-aep` on `feat/wrap-function`

**Issues:** #10 (storage), #11 (multi-tenant), #14 (config), #15 (streaming)

---

## File Structure

```
src/aceteam_aep/
├── storage/
│   ├── __init__.py              # StorageBackend protocol
│   ├── memory.py                # InMemoryBackend (current, default)
│   ├── sqlite.py                # SqliteBackend (local persistence)
│   └── supabase.py              # SupabaseBackend (platform integration, future)
├── config.py                    # Config loading: YAML + env + CLI
├── safety/
│   └── prompt_injection.py      # Prompt injection detector
├── proxy/
│   ├── app.py                   # MODIFY — use storage, config, multi-tenant
│   ├── cli.py                   # MODIFY — load config file
│   └── streaming.py             # SSE streaming handler
```

---

### Task 1: Storage Protocol + InMemoryBackend

**Files:**
- Create: `src/aceteam_aep/storage/__init__.py`
- Create: `src/aceteam_aep/storage/memory.py`
- Create: `tests/test_storage/__init__.py`
- Create: `tests/test_storage/test_memory.py`

**StorageBackend protocol:**
```python
class StorageBackend(Protocol):
    async def save_call(self, entity: str, call: CallRecord) -> None: ...
    async def save_signals(self, entity: str, signals: list[SafetySignal]) -> None: ...
    async def get_session(self, entity: str) -> SessionState: ...
    async def get_calls(self, entity: str, *, limit: int = 100) -> list[CallRecord]: ...
    async def get_signals(self, entity: str, *, limit: int = 100) -> list[SafetySignal]: ...
```

**CallRecord dataclass:**
```python
@dataclass
class CallRecord:
    call_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    enforcement_action: str  # pass/flag/block
    timestamp: str
    input_text_preview: str = ""   # first 200 chars
    output_text_preview: str = ""  # first 200 chars
```

- [ ] Write tests for InMemoryBackend
- [ ] Implement StorageBackend protocol + InMemoryBackend
- [ ] Run tests, verify pass
- [ ] Commit

---

### Task 2: SQLite Storage Backend

**Files:**
- Create: `src/aceteam_aep/storage/sqlite.py`
- Create: `tests/test_storage/test_sqlite.py`

Uses `aiosqlite` for async SQLite access. Schema:

```sql
CREATE TABLE calls (
    id TEXT PRIMARY KEY,
    entity TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    enforcement TEXT,
    input_preview TEXT,
    output_preview TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX idx_calls_entity ON calls(entity, created_at);

CREATE TABLE signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity TEXT NOT NULL,
    call_id TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    detail TEXT,
    detector TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX idx_signals_entity ON signals(entity, created_at);
```

Add `aiosqlite>=0.20` to `[project.optional-dependencies] proxy`.

- [ ] Write tests (save_call, save_signals, get_session, persistence across instances)
- [ ] Implement SqliteBackend with auto-migration on first connect
- [ ] Run tests, verify pass
- [ ] Commit

---

### Task 3: Config System

**Files:**
- Create: `src/aceteam_aep/config.py`
- Create: `tests/test_config.py`

Pydantic model for config with YAML loading:

```python
class AepConfig(BaseModel):
    proxy: ProxyConfig = ProxyConfig()
    safety: SafetyConfig = SafetyConfig()
    enforcement: EnforcementConfig = EnforcementConfig()
    budget: BudgetConfig | None = None
    storage: StorageConfig = StorageConfig()
    entity: str = "default"

class ProxyConfig(BaseModel):
    port: int = 8080
    target: str = "https://api.openai.com"

class SafetyConfig(BaseModel):
    pii: DetectorConfig = DetectorConfig(enabled=True)
    content_safety: DetectorConfig = DetectorConfig(enabled=True)
    cost_anomaly: CostAnomalyConfig = CostAnomalyConfig()
    prompt_injection: DetectorConfig = DetectorConfig(enabled=False)

class StorageConfig(BaseModel):
    type: Literal["memory", "sqlite"] = "memory"
    path: str = "./aep_data.db"
```

Resolution: `aep.yaml` in cwd → `~/.config/aceteam-aep/config.yaml` → env vars (`AEP_PROXY_PORT`) → CLI args.

Add `pyyaml>=6.0` to proxy deps.

- [ ] Write tests (load from YAML, env override, defaults)
- [ ] Implement config loading
- [ ] Wire into proxy CLI
- [ ] Run tests, verify pass
- [ ] Commit

---

### Task 4: Multi-Tenant Proxy

**Files:**
- Modify: `src/aceteam_aep/proxy/app.py`
- Create: `tests/test_proxy_multitenant.py`

Replace single global `ProxyState` with per-entity states:

```python
class ProxyRouter:
    def __init__(self, storage: StorageBackend, ...):
        self._sessions: dict[str, ProxyState] = {}

    def get_session(self, entity: str) -> ProxyState:
        if entity not in self._sessions:
            self._sessions[entity] = ProxyState(entity=entity, ...)
        return self._sessions[entity]
```

Entity extracted from: `X-AEP-Entity` header → API key hash → `"default"`.

Dashboard at `/aep/` shows entity selector dropdown.

- [ ] Write tests (two entities isolated, entity from header, default fallback)
- [ ] Implement ProxyRouter + entity extraction
- [ ] Update dashboard to support entity filter
- [ ] Run tests, verify pass
- [ ] Commit

---

### Task 5: Streaming Support

**Files:**
- Create: `src/aceteam_aep/proxy/streaming.py`
- Modify: `src/aceteam_aep/proxy/app.py`
- Create: `tests/test_proxy_streaming.py`

When request has `"stream": true`:
1. Forward to target with streaming
2. Buffer chunks while streaming them through to the client
3. After stream completes, run safety checks on accumulated output
4. If safety BLOCK: append a final SSE event `data: {"aep_safety_block": true, "reason": "..."}`
5. Record cost from accumulated usage chunks

```python
async def stream_with_safety(upstream_response, session, call_id):
    accumulated = []
    async for chunk in upstream_response.aiter_lines():
        yield chunk  # pass through immediately
        accumulated.append(chunk)
    # After stream ends, check safety
    full_text = extract_text_from_sse_chunks(accumulated)
    signals = session.registry.run_all(output_text=full_text, ...)
    decision = evaluate(signals, session.policy)
    if decision.action == "block":
        yield f'data: {{"aep_blocked": true, "reason": "{decision.reason}"}}\n\n'
```

- [ ] Write tests (stream passthrough, post-stream safety, block event)
- [ ] Implement streaming handler
- [ ] Wire into proxy_handler
- [ ] Run tests, verify pass
- [ ] Commit

---

### Task 6: Prompt Injection Detector

**Files:**
- Create: `src/aceteam_aep/safety/prompt_injection.py`
- Create: `tests/test_safety/test_prompt_injection.py`

Model: `protectai/deberta-v3-base-prompt-injection-v2` (~86M params). Checks INPUT text for injection attempts.

Same pattern as PII/content detectors: lazy-load, CPU, falls back gracefully.

- [ ] Write tests
- [ ] Implement PromptInjectionDetector
- [ ] Run tests, verify pass
- [ ] Commit

---

## Milestone Checkpoints

| Milestone | Tasks | What It Unlocks |
|-----------|-------|-----------------|
| **Storage** | 1-2 | Data survives restarts, history queries |
| **Config** | 3 | Customizable without code, per-deployment settings |
| **Multi-tenant** | 4 | Multiple orgs on one proxy |
| **Streaming** | 5 | Works with real agent frameworks |
| **Prompt injection** | 6 | Input-side safety (completes the 4th detector) |
