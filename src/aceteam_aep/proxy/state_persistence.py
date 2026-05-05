"""Persist a small slice of ProxyState across restarts.

Without this, every proxy restart wipes the API key, the connected AceTeam
account, the target base URL, and the safety toggle — all things the user
typically configured once via the dashboard. Forcing them to re-enter on
every container restart is friction we shouldn't ship.

What we persist (the "control plane" of ProxyState):

- ``api_key`` (BYOK or ``act_*`` from the connect flow)
- ``connected_account`` (org metadata so the dashboard remembers who's connected)
- ``target_base_url`` (so picking a different upstream sticks)
- ``safety_enabled`` (the toggle state — surprising if it flips on restart)

What we explicitly do NOT persist:

- Cost trackers, signals, decisions, spans — those live in :class:`EventStore`
  (already SQLite-backed via aiosqlite). Putting them here would duplicate.
- Custom policies — those have their own CRUD store and survive via that path.
- The enforcement policy ``dict`` — typically loaded from a YAML/JSON file or
  defaults; persisting and reloading would shadow that source-of-truth.
- Budget — same reason: configured at process boot via flags/env.

Storage format: a single small JSON document at ``AEP_STATE_PATH`` (default
``~/.aceteam-aep/state.json``). Atomic write via tmp+rename so a crash
mid-write can't leave a half-file. Mode 0o600 because the API key lives
inside.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

STATE_PATH_ENV = "AEP_STATE_PATH"

_DEFAULT_PATH = Path.home() / ".aceteam-aep" / "state.json"

_PERSISTED_KEYS = (
    "api_key",
    "connected_account",
    "target_base_url",
    "safety_enabled",
)


def get_configured_path() -> Path | None:
    """Return the configured state path, or None if persistence is disabled.

    Persistence is opt-in: returns ``None`` when ``AEP_STATE_PATH`` is unset
    AND the default path doesn't exist yet, so a fresh ``pip install`` user
    sees zero behavioral change. Once any state has been written (even by
    setting the env var once), the default path will exist on subsequent
    runs and the proxy will keep using it.
    """
    raw = os.environ.get(STATE_PATH_ENV, "").strip()
    if raw:
        return Path(raw).expanduser()
    if _DEFAULT_PATH.exists():
        return _DEFAULT_PATH
    return None


def get_default_path() -> Path:
    """Return the default state path. Used when the user opts in via API."""
    return _DEFAULT_PATH


def load_persisted_state(path: Path) -> dict[str, Any] | None:
    """Read a previously-saved state document; return None on any failure.

    A corrupt or missing file is treated as "no persisted state" rather than
    a hard error — the proxy should always boot, even if the file is junk.
    The next mutation will overwrite it with a clean document.
    """
    try:
        if not path.exists():
            return None
        raw = path.read_text()
    except OSError as err:
        log.warning("state persistence: cannot read %s: %s", path, err)
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as err:
        log.warning("state persistence: %s is not valid JSON: %s", path, err)
        return None

    if not isinstance(data, dict):
        log.warning("state persistence: %s is not a JSON object", path)
        return None

    return {k: data.get(k) for k in _PERSISTED_KEYS if k in data}


def save_persisted_state(path: Path, fields: dict[str, Any]) -> None:
    """Atomically write the persisted-fields document.

    Only the keys in ``_PERSISTED_KEYS`` are written, even if ``fields``
    contains more — keeps the surface area tight and prevents accidentally
    persisting cost/signal data that belongs in the EventStore.
    """
    payload = {k: fields.get(k) for k in _PERSISTED_KEYS}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(path)
        # 0o600 — the API key is in here. Parent dir gates browse access too,
        # but a tight file mode is cheap insurance against a stray world-read.
        os.chmod(path, 0o600)
    except OSError as err:
        log.warning("state persistence: cannot write %s: %s", path, err)


__all__ = [
    "STATE_PATH_ENV",
    "get_configured_path",
    "get_default_path",
    "load_persisted_state",
    "save_persisted_state",
]
