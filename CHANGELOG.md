# Changelog

## [Unreleased]

### Added
- **Anthropic prompt caching on the static prefix.** The Anthropic client now attaches `cache_control: {"type": "ephemeral"}` breakpoints to the tool definitions and the system prompt (the last static block, which caches the tools+system prefix together). For agent/chat turns that re-send a large fixed prefix (system prompt + tool schemas) every round-trip, Anthropic now serves that prefix from cache at ~0.1x input price instead of re-billing it in full. Behavior-neutral: identical outputs, lower cost. Configurable via the `prompt_caching` param on `AnthropicClient`/`create_client()` or the `AEP_PROMPT_CACHING` env toggle (default ON; set to a false value for a byte-identical revert). Addresses aceteam-ai/aceteam#5856; follow-up to aceteam-ai/aceteam PR #5901.
- **`Usage.cache_read_input_tokens` and `Usage.cache_creation_input_tokens`.** Anthropic's cache-token counts now surface on `Usage` (default 0; summed in `__add__`) so the host app can log and bill them. Populated from `response.usage` on the non-streaming path and from the `message_start` event on the streaming path.

### Changed
### Fixed

## [0.11.6] - 2026-07-22

### Fixed
- **Safety detectors no longer rebuild their transformers model per instance.** `PiiDetector` and `ContentSafetyDetector` held the pipeline as instance state with no shared cache, so every construction re-materialized the immutable model into memory — ~0.25–0.55s each even with a warm HuggingFace disk cache, plus N duplicate copies of identical read-only weights resident at once. Consumers that build detectors per session (a reasonable thing to do, since `ProxyState` owns genuinely mutable per-session state) and test suites that reset per-session state paid that cost on every construction. The built pipeline is now cached at module level, keyed by model name, behind a lock so concurrent first-loads don't each build a copy. Only the immutable pipeline is shared; per-instance `_available`/fallback state stays per-instance, and `check()` only runs inference, so sharing is safe. Fixes #122.
- **Test suite no longer hangs at exit.** All 628 tests passed but the pytest process never returned, so CI reported a timeout rather than a result. Two non-daemon threads outlived the run and blocked interpreter shutdown: `aiosqlite.Connection` is a plain `Thread`, and four observability `store` fixtures returned a `SqliteEventStore` without ever closing it (they now yield and close); and `transformers` spawns a `Thread-auto_conversion` that calls out to the HuggingFace safetensors-conversion Space because the safety models ship only `pytorch_model.bin` (now opted out via `DISABLE_SAFETENSORS_CONVERSION`, which exists for exactly this case). Also repaired four assertions still expecting pre-rename strings — the dashboard title is "Agent Safety Net Dashboard", and the default custom policy set is `{"English only", "Social Security Number"}`.

## [0.11.5] - 2026-07-16

### Fixed
- **Streaming proxy no longer swallows upstream errors into an empty 200 SSE stream.** `handle_streaming_request` opened the upstream connection inside the response generator, after `StreamingResponse` had already committed HTTP 200 headers, and never checked the upstream status. A non-2xx upstream (e.g. Anthropic 400 rejecting a beta-gated param) produced a 200 response with a zero-byte body; clients (Claude Code / Anthropic SDK) waited indefinitely on an SSE stream that never emitted, ending in Cloudflare 524s and silent retry loops. The upstream response is now opened and status-checked BEFORE the streaming response is constructed: a pre-stream non-2xx is passed through as-is (upstream status code + verbatim JSON error body, Anthropic/OpenAI error shape preserved, no SSE wrapping), and a mid-stream failure emits a final SSE `event: error` with an Anthropic-format `{"type": "error", "error": {"type": "api_error", ...}}` payload before closing so clients terminate instead of hanging. Upstream error status + first 500 bytes of the body are logged at warning level. Successful-stream metering/telemetry is unchanged; error paths no longer fire `on_complete` with zero tokens (which recorded a bogus "unknown model" call) — the pre-stream error path instead ends the span with `status="ERROR"`, mirroring the non-streaming error branch, and mid-stream interruptions are not metered as completions. Fixes aceteam-ai/aceteam#5988.

## [0.11.4] - 2026-07-16

### Fixed
- **PII detector no longer emits every entity at `severity="high"`.** Both the NER model path and the regex path now use a per-entity-type severity map: SSN and CREDIT_CARD are "high", ID_NUM is "medium", EMAIL/PHONE/IP_ADDRESS/PERSON are "low", and unknown entity types default to "medium". Previously any email address in input (e.g. `noreply@anthropic.com` in Claude Code's system prompt) produced a "high" signal, and default enforcement policies block on "high", so the whole request was blocked. Signal emission is otherwise unchanged (spans, scores, detail strings, detector name, signal_type). Fixes aceteam-ai/aceteam#5944.

## [0.11.3] - 2026-07-11

### Fixed
- **OpenAI dotted `gpt-5.x` point releases (`gpt-5.6-terra`, `gpt-5.6-sol`, `gpt-5.6-luna`, `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.1`, ...) now correctly use `max_completion_tokens`.** These models aren't in `MODEL_REGISTRY` yet, so `_uses_max_completion_tokens()` fell back to a dash-only prefix check (`model.startswith("gpt-5-")`), which a dotted name never matches. The client sent `max_tokens` and OpenAI returned `400 Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.` Confirmed live against the OpenAI API for `gpt-5.6-terra`, `gpt-5.4`, and `gpt-5`.
- **Same dotted-name fallback bug in `_supports_temperature()`.** Dotted `gpt-5.x` models were wrongly treated as accepting a `temperature` parameter. Both fallbacks now share a single `_matches_family()` helper (tolerating `-` and `.` separators) so they can't drift apart again.

### Added
- **`uses_max_completion_tokens: bool | None` on `OpenAIClient.__init__` and `create_client()`.** Mirrors the existing `supports_temperature` override. Defaults to `None` (unchanged behavior — the registry/prefix heuristic decides); when set to `True`/`False` it forces the token-limit param regardless of model name. Gives the aceteam side a per-model lever without waiting on an aceteam-aep release. No registry entries were added for the new models; the fallback heuristic is the load-bearing fix.

## [0.6.0] - 2026-04-02

### Added
- **MCP Gateway** — 4 Tier 1 safety tools (`check_safety`, `get_safety_status`, `set_safety_policy`, `get_cost_summary`) mounted at `/mcp/` via FastMCP Streamable HTTP. Same port, same process as the LLM proxy and dashboard.
- **Dashboard setup wizard** — first-run overlay guides API key configuration with two paths: "I have API keys" or "Use AceTeam ($5 free)".
- **Dashboard policy controls** — per-detector toggle checkboxes (Agent Threat, PII, Cost Anomaly, Content Safety, Trust Engine) and per-category sub-toggles (Finance, IoT, Software, Web, Program).
- **`aceteam-aep setup` command** — detects Podman/Docker, starts proxy container, writes Claude Code MCP config, configures shell profile (bash/zsh/fish-aware).
- **`aceteam-aep connect`/`disconnect` commands** — link local proxy to AceTeam account. Saves credentials to `~/.config/aceteam-aep/credentials.json` with 0600 permissions.
- **5 R-Judge domain dimensions** — Finance, IoT, Software, Web, Program added to Trust Engine for per-category safety evaluation.
- **Trust Engine in detector builder** — `build_detectors_from_policy()` now creates TrustEngineDetector from YAML policy config.
- **Confidence response headers** — `X-AEP-Confidence`, `X-AEP-Reason`, `X-AEP-Category` on proxy responses for flagged/blocked calls.
- **`GET /aep/api/safety`** — read current safety state without POST body (was POST-only).
- **`extra` field in safety API** — detector `extra` config (e.g., Trust Engine dimensions) persists across dashboard polls.
- **FERPA detector** — education record pattern detection.
- **Budget enforcement** — per-session cost caps with `--budget` and `--budget-per-session` CLI flags.
- **Budget bar in dashboard** — visual progress bar for budget tracking.
- **Safety-off and ClawCamp policy files** — `policies/safety-off.yaml` (passthrough) and `policies/clawcamp.yaml` (5-category demo).
- **387 tests** — comprehensive unit + integration coverage across proxy, dashboard, MCP gateway, CLI, enforcement, and Trust Engine.

### Changed
- CLI banner renamed from "AEP Proxy" to "SafeClaw Gateway" with MCP endpoint URL.
- Architecture docs rewritten for gateway model (proxy + dashboard + MCP on one port).
- README reframed as SafeClaw Gateway with three-tier model documentation.

### Fixed
- `GET /aep/api/safety` returned 400 (handler always parsed JSON body).
- `threshold: 0.0` silently overridden to default (truthiness check → `is not None`).
- `check_safety` MCP tool produced duplicate call IDs (missing `call_count` increment).
- Fish shell config received bash `export` syntax (now uses `set -gx`).
- Claude Code config write was not atomic (now uses temp file + `os.replace()`).
- Credential directory created with 0o755 (now 0o700).
- Dashboard toggle POST failures silently ignored (now reverts via `.catch()`).

## [0.5.6] - 2026-03-28

Initial public release with proxy, dashboard, safety detectors, signed verdicts, and Merkle chain.
