# Changelog

## [Unreleased]

### Added
### Changed
### Fixed

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
