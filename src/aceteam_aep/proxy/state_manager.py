"""Per-entity ``ProxyState`` partitioning for multi-tenant proxy deployments.

The proxy historically used a single global ``ProxyState`` shared by every
request. That works for an individual developer running ``aceteam-aep proxy``
on their laptop, but breaks down when the same gateway fronts traffic for
multiple orgs / agents — they'd see each other's signals, costs, and
connected-account metadata.

This module adds opt-in per-entity isolation. Default is unchanged: the
manager hands every request the same ``ProxyState`` (the singleton path),
so a fresh ``pip install`` user experiences zero behavioral diff.

Enable by setting ``AEP_MULTI_TENANT=1``. The manager extracts an
``entity_id`` per request:

1. ``X-AEP-Entity`` request header (preferred; matches the existing AEP
   header conventions in :mod:`.headers`)
2. Fallback: a stable shard derived from the ``Authorization`` header so
   that even agents that don't yet emit ``X-AEP-Entity`` get isolated by
   API key
3. Last resort: ``"default"`` — used for the dashboard / setup / health
   endpoints where there's no agent context

Entities are stored in an LRU dict capped at ``AEP_MAX_ENTITIES`` (default
64) so a buggy caller that emits a fresh entity per request can't OOM the
proxy. Eviction is benign — the next request from that entity rebuilds a
fresh ``ProxyState`` (with persistence loading whatever was last saved).

Why explicit handler lookup rather than a contextvar:

The streaming on-complete callback fires *after* the request scope ends —
contextvar resolution would return whatever entity is "current" at the
time the callback runs, not the entity the request belonged to. Each
handler explicitly calls ``manager.get_for_request(request)`` at the top
so the per-request state is captured by closure before any deferred work
is scheduled.
"""

from __future__ import annotations

import logging
import os
import re
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from starlette.requests import Request

    from .app import ProxyState

log = logging.getLogger(__name__)

MULTI_TENANT_ENV = "AEP_MULTI_TENANT"
MAX_ENTITIES_ENV = "AEP_MAX_ENTITIES"
ENTITY_HEADER = "x-aep-entity"

_DEFAULT_MAX_ENTITIES = 64

# Filesystem-safe sanitizer for per-entity state file paths. We want a
# round-trippable shape but slugify aggressively — the original entity_id
# stays in the file's contents, so loss in the filename is acceptable.
_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def is_multi_tenant_enabled() -> bool:
    """Read the env flag. Default off for back-compat with single-tenant deploys."""
    raw = os.environ.get(MULTI_TENANT_ENV, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def get_max_entities() -> int:
    """Read the LRU cap; clamp to a sane minimum so a misconfigured value can't pin to 0."""
    raw = os.environ.get(MAX_ENTITIES_ENV, "").strip()
    if not raw:
        return _DEFAULT_MAX_ENTITIES
    try:
        n = int(raw)
    except ValueError:
        log.warning(
            "ignoring non-integer %s=%r; using %d",
            MAX_ENTITIES_ENV,
            raw,
            _DEFAULT_MAX_ENTITIES,
        )
        return _DEFAULT_MAX_ENTITIES
    return max(1, n)


def entity_for_request(request: Request) -> str:
    """Derive a stable entity id from a request.

    Order of preference:

    1. Explicit ``X-AEP-Entity`` header — the canonical way for agents to
       declare which org/tenant they belong to.
    2. ``Authorization: Bearer <key>`` prefix — uses the first 12 chars of
       the key after stripping whitespace. Two requests from the same key
       always shard to the same entity even without the header.
    3. ``"default"`` — the singleton bucket. Used by the dashboard,
       healthz, setup wizard, and anything else where there's no agent
       context.

    Returning ``"default"`` is also what's returned in non-multi-tenant
    mode (the manager will simply ignore the value), so this function is
    safe to call unconditionally.
    """
    header = request.headers.get(ENTITY_HEADER, "").strip()
    if header:
        return header
    auth = request.headers.get("authorization", "").strip()
    if auth.lower().startswith("bearer "):
        key = auth[7:].strip()
        if key:
            # Prefix is enough to keep distinct keys distinct without
            # putting full secrets into in-memory dict keys / log lines.
            return f"key:{key[:12]}"
    return "default"


def per_entity_state_path(base_path: Path | None, entity_id: str) -> Path | None:
    """Return ``base_path``'s sibling ``states/<sanitized>.json``, or None.

    The default singleton entity ("default") keeps writing to the original
    ``base_path`` so that turning multi-tenant on/off doesn't accidentally
    orphan the previously-persisted state. Only non-default entities get
    their own per-entity state file under a ``states/`` subdir next to
    the configured base path.
    """
    if base_path is None:
        return None
    if entity_id == "default":
        return base_path
    sanitized = _FILENAME_SAFE.sub("_", entity_id)
    # Collapse runs of dots so a "../../" entity_id can't survive as ".._.._".
    # We aggressively replace any "." with "_" once we've hit the safe-chars
    # filter — dots in filenames are fine, but path-traversal-shaped sequences
    # are too easy to mis-handle so we just disallow the character outright.
    sanitized = sanitized.replace(".", "_")
    sanitized = sanitized[:64].strip("_") or "entity"
    return base_path.parent / "states" / f"{sanitized}.json"


class ProxyStateManager:
    """Per-entity ``ProxyState`` registry with LRU eviction.

    In single-tenant mode (``multi_tenant=False``) every lookup returns the
    same ``default`` state — the manager is a no-op pass-through. Tests and
    existing single-tenant deployments are unaffected.

    In multi-tenant mode the manager creates a fresh ``ProxyState`` per
    distinct entity_id via the ``factory`` callable, capping the in-memory
    dict at ``max_entities`` (LRU evicts the least-recently-used bucket
    when the cap is hit).
    """

    def __init__(
        self,
        *,
        multi_tenant: bool,
        max_entities: int,
        default: ProxyState,
        factory: Callable[[str], ProxyState],
    ) -> None:
        self._multi_tenant = multi_tenant
        self._max_entities = max_entities
        self._factory = factory
        # OrderedDict gives us O(1) LRU via move_to_end + popitem(last=False).
        # The "default" entity is always present so single-tenant lookups
        # are a single dict get.
        self._states: OrderedDict[str, ProxyState] = OrderedDict()
        self._states["default"] = default

    @property
    def multi_tenant(self) -> bool:
        return self._multi_tenant

    def get_for_entity(self, entity_id: str) -> ProxyState:
        """Return the ``ProxyState`` for ``entity_id``; create + evict as needed.

        Always returns the singleton when multi-tenant is disabled, so
        single-tenant callers can ignore ``entity_id`` entirely.
        """
        if not self._multi_tenant:
            return self._states["default"]
        if entity_id in self._states:
            self._states.move_to_end(entity_id)
            return self._states[entity_id]
        # Cap reached → evict LRU before adding the new one. "default" is
        # protected from eviction since it's the bucket every non-tenant
        # request lands in (and where dashboard/health endpoints look).
        while len(self._states) >= self._max_entities:
            evicted_id, _ = next(
                ((k, v) for k, v in self._states.items() if k != "default"), (None, None)
            )
            if evicted_id is None:
                # max_entities=1 with multi-tenant on — degenerate config,
                # but don't crash; just don't create a new bucket.
                return self._states["default"]
            self._states.pop(evicted_id)
            log.info("multi-tenant LRU evicted entity_id=%r", evicted_id)
        new_state = self._factory(entity_id)
        self._states[entity_id] = new_state
        return new_state

    def get_for_request(self, request: Request) -> ProxyState:
        """Convenience: derive entity_id from the request and look up state."""
        return self.get_for_entity(entity_for_request(request))

    def all_entities(self) -> list[str]:
        """List currently-resident entity ids; for dashboard introspection."""
        return list(self._states.keys())


__all__ = [
    "ENTITY_HEADER",
    "MAX_ENTITIES_ENV",
    "MULTI_TENANT_ENV",
    "ProxyStateManager",
    "entity_for_request",
    "get_max_entities",
    "is_multi_tenant_enabled",
    "per_entity_state_path",
]
