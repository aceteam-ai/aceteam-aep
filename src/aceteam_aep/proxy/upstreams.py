"""Provider-specific proxy upstreams supplied by the container environment."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Literal
from urllib.parse import urlsplit

Provider = Literal["openai", "anthropic"]


@dataclass(frozen=True)
class UpstreamTargets:
    openai: str | None = None
    anthropic: str | None = None

    def for_provider(self, provider: Provider) -> str | None:
        return self.openai if provider == "openai" else self.anthropic


def normalize_upstream_url(value: str, *, listen_port: int) -> str:
    """Convert a client base URL to the prefix used by the reverse proxy.

    Clients include ``/v1`` in their base URL. The proxy appends the incoming
    ``/v1/...`` request path, so a terminal ``/v1`` must be removed here.
    """
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise ValueError("Upstream must be a valid HTTP or HTTPS base URL") from exc

    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Upstream must be an HTTP or HTTPS URL without credentials, query, or fragment"
        )

    local_hosts = {
        "localhost",
        "localhost.localdomain",
        "127.0.0.1",
        "::1",
        "0.0.0.0",
        "aep-proxy",
        "safeclaw-proxy",
        "host.docker.internal",
        "host.containers.internal",
    }
    try:
        address = ip_address(parsed.hostname)
        local_address = address.is_loopback or address.is_unspecified
    except ValueError:
        local_address = False
    if (parsed.hostname.lower() in local_hosts or local_address) and port == listen_port:
        raise ValueError("Upstream points back to the local proxy")

    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def load_provider_targets(
    *, listen_port: int, env: Mapping[str, str] | None = None
) -> UpstreamTargets:
    """Load each provider target, preferring explicit upstream names.

    The plain BASE_URL names are accepted for container launchers. A client
    may also set those names to the proxy URL, so that case fails the loop
    check instead of silently sending traffic back into this process.
    """
    source = os.environ if env is None else env

    def get_url(explicit: str, client: str) -> str | None:
        raw = source.get(explicit, "").strip() or source.get(client, "").strip()
        return normalize_upstream_url(raw, listen_port=listen_port) if raw else None

    return UpstreamTargets(
        openai=get_url("UPSTREAM_OPENAI_BASE_URL", "OPENAI_BASE_URL"),
        anthropic=get_url("UPSTREAM_ANTHROPIC_BASE_URL", "ANTHROPIC_BASE_URL"),
    )


def request_provider(path: str, headers: Mapping[str, str]) -> Provider:
    """Choose the wire protocol from the request path and Anthropic header."""
    if (
        path in {"/v1/messages", "/v1/complete"}
        or path.startswith("/v1/messages/")
        or headers.get("anthropic-version")
    ):
        return "anthropic"
    return "openai"
