"""Auto-sync OpenClaw's openclaw.json from the AceTeam gateway's /v1/models.

When the SafeClaw user connects via the dashboard, an `act_*` key lands in
ProxyState. The Next.js gateway at ``${target_base_url}/v1/models`` knows
which models that org can route through (BYOK keys + platform fallbacks),
so the right design is to mirror that catalog into the OpenClaw config
instead of asking users to hand-edit JSON.

Trade-off: OpenClaw doesn't hot-reload openclaw.json. This module writes
the file but doesn't restart the gateway — the new catalog appears on the
next compose restart. Status is surfaced via ``ProxyState.openclaw_config_status``
so the dashboard can show a "restart to apply" banner.

Wiring: enabled by setting ``AEP_OPENCLAW_CONFIG_PATH=/path/to/openclaw.json``
in the proxy's environment. The SafeClaw compose stack mounts the OpenClaw
config dir into the proxy so this path resolves.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)


def get_configured_path() -> str | None:
    """Return the openclaw.json sync path from env, or None if unset."""
    raw = os.environ.get("AEP_OPENCLAW_CONFIG_PATH", "").strip()
    return raw or None


def _resolve_models_url(target_base_url: str) -> str | None:
    """Build the /v1/models URL from the proxy's target_base_url.

    target_base_url for AceTeam looks like ``https://aceteam.ai/api/gateway``
    (no /v1 suffix — proxy convention). The /v1/models endpoint is at
    ``<host>/<basepath>/v1/models``. Returns None if target_base_url is
    malformed (so we can no-op cleanly during startup).
    """
    parsed = urlparse(target_base_url)
    if not parsed.scheme or not parsed.netloc:
        return None
    base = target_base_url.rstrip("/")
    return f"{base}/v1/models"


def _transform_models_to_providers(models: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group OpenAI /v1/models entries by ``owned_by`` → OpenClaw provider stanzas.

    Each provider stanza routes through the local aep-proxy (``http://aep-proxy:8899/v1``)
    with the sentinel ``aep-proxy-managed`` key — same convention as the
    static seed in safeclaw-site/install.sh::write_openclaw_config.
    """
    providers: dict[str, dict[str, Any]] = {}
    for m in models:
        owner = (m.get("owned_by") or "unknown").lower()
        provider = providers.setdefault(
            owner,
            {
                "baseUrl": "http://aep-proxy:8899/v1",
                "apiKey": "aep-proxy-managed",
                "auth": "api-key",
                "models": [],
            },
        )
        cost = m.get("cost_per_million_tokens") or {}
        provider["models"].append(
            {
                "id": m["id"],
                "name": m["id"],
                "reasoning": False,
                "input": m.get("modalities") or ["text"],
                "cost": {
                    "input": cost.get("input", 0),
                    "output": cost.get("output", 0),
                    "cacheRead": cost.get("cache_read", 0),
                    "cacheWrite": cost.get("cache_write", 0),
                },
                "contextWindow": m.get("context_window", 128000),
                "maxTokens": m.get("max_output_tokens", 16384),
            }
        )
    return providers


def _splice_into_existing(
    existing: dict[str, Any], providers: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Insert the new providers map into an existing openclaw.json structure.

    Preserves everything outside ``models.providers`` so users keep their
    ``gateway``, ``agents.defaults``, ``plugins`` etc. ``models.mode`` is
    forced to ``"replace"`` so OpenClaw doesn't merge our list with its
    bundled 200-model default catalog (which contains models not routable
    through AceTeam — would produce silent "incomplete terminal response"
    failures).
    """
    out = dict(existing)
    models_section = dict(out.get("models") or {})
    models_section["mode"] = "replace"
    models_section["providers"] = providers
    out["models"] = models_section
    meta = dict(out.get("meta") or {})
    meta["lastTouchedAt"] = datetime.now(UTC).isoformat()
    out["meta"] = meta
    return out


def _atomic_write(path: Path, content: str) -> None:
    """Write content via tmp+rename so a crash mid-write can't leave a half file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)


async def refresh_openclaw_config(
    *,
    api_key: str,
    target_base_url: str,
    config_path: str,
) -> dict[str, Any]:
    """Fetch /v1/models, write openclaw.json, return a status record.

    The status record is what gets surfaced via the dashboard so users see
    when a sync ran, how many models landed, and whether a restart is
    needed for it to take effect.
    """
    status: dict[str, Any] = {
        "ok": False,
        "config_path": config_path,
        "checked_at": datetime.now(UTC).isoformat(),
    }

    if not api_key.startswith("act_"):
        status["error"] = "non-aceteam key (sync only supports act_* keys)"
        return status

    models_url = _resolve_models_url(target_base_url)
    if models_url is None:
        status["error"] = f"invalid target_base_url: {target_base_url!r}"
        return status

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                models_url, headers={"Authorization": f"Bearer {api_key}"}
            )
    except Exception as err:  # noqa: BLE001 — network errors are part of the contract
        log.warning("openclaw sync: gateway unreachable: %s", err)
        status["error"] = f"gateway unreachable: {err}"
        return status

    if resp.status_code != 200:
        log.warning(
            "openclaw sync: gateway %s returned %d", models_url, resp.status_code
        )
        status["error"] = f"gateway returned HTTP {resp.status_code}"
        return status

    try:
        body = resp.json()
        models = body.get("data") or []
    except (ValueError, AttributeError) as err:
        status["error"] = f"malformed gateway response: {err}"
        return status

    if not models:
        # Treat empty-catalog as a soft failure — usually means the org has
        # no usable BYOK keys, so we shouldn't blow away a working config.
        status["error"] = "gateway returned empty catalog (no usable models for this org)"
        return status

    providers = _transform_models_to_providers(models)

    path_obj = Path(config_path)
    if path_obj.exists():
        try:
            existing = json.loads(path_obj.read_text())
        except (OSError, json.JSONDecodeError) as err:
            log.warning("openclaw sync: existing config unreadable, starting fresh: %s", err)
            existing = {"gateway": {"mode": "local"}}
    else:
        existing = {"gateway": {"mode": "local"}}

    new_config = _splice_into_existing(existing, providers)

    try:
        _atomic_write(path_obj, json.dumps(new_config, indent=2) + "\n")
    except OSError as err:
        log.warning("openclaw sync: write failed: %s", err)
        status["error"] = f"write failed: {err}"
        return status

    log.info(
        "openclaw sync: wrote %d models across %d providers to %s",
        len(models),
        len(providers),
        config_path,
    )
    status.update(
        {
            "ok": True,
            "model_count": len(models),
            "providers": sorted(providers.keys()),
            "written_at": status["checked_at"],
            "restart_required": True,
        }
    )
    return status


__all__ = ["get_configured_path", "refresh_openclaw_config"]
