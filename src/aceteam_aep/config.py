"""Unified AEP configuration — loads from YAML, env vars, and CLI args.

Supports a single ``aep.yaml`` file that configures the proxy, safety
detectors, enforcement policy, and budget::

    # aep.yaml
    proxy:
      port: 8080
      host: 127.0.0.1
      target: https://api.openai.com
      dashboard: true

    enforcement:
      default_action: flag
      block_on: [high]
      flag_on: [medium]
      detectors:
        pii:            { action: block, threshold: 0.8 }
        content_safety: { action: flag, threshold: 0.85 }
        agent_threat:   { action: block }
        cost_anomaly:   { action: pass, multiplier: 10 }

    budget:
      max_per_session: 10.00
      max_per_call: 1.00

Environment variables override YAML values:

- ``AEP_PORT`` — proxy port
- ``AEP_HOST`` — proxy bind address
- ``AEP_TARGET`` — upstream API URL
- ``AEP_POLICY`` — path to policy YAML (enforcement section only)
- ``AEP_LOG`` — enable verbose logging (1/true/yes)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .enforcement import EnforcementPolicy


@dataclass
class ProxyConfig:
    """Resolved proxy configuration."""

    port: int = 8899
    host: str = "127.0.0.1"
    target: str = "https://api.openai.com"
    dashboard: bool = True
    verbose: bool = False

    # Enforcement policy (built from enforcement section)
    policy: EnforcementPolicy = field(default_factory=EnforcementPolicy)

    # Budget limits
    max_per_session: float | None = None
    max_per_call: float | None = None


def load_config(
    config_path: str | Path | None = None,
    *,
    cli_overrides: dict[str, Any] | None = None,
) -> ProxyConfig:
    """Load configuration from YAML file, env vars, and CLI overrides.

    Priority (highest wins): CLI args > env vars > YAML file > defaults.
    """
    data: dict[str, Any] = {}

    # Layer 1: YAML file
    if config_path:
        data = _load_yaml(config_path)

    # Build config from YAML
    proxy_section = data.get("proxy", {})
    enforcement_section = data.get("enforcement", {})
    budget_section = data.get("budget", {})

    config = ProxyConfig(
        port=proxy_section.get("port", 8899),
        host=proxy_section.get("host", "127.0.0.1"),
        target=proxy_section.get("target", "https://api.openai.com"),
        dashboard=proxy_section.get("dashboard", True),
        verbose=proxy_section.get("verbose", False),
        max_per_session=budget_section.get("max_per_session"),
        max_per_call=budget_section.get("max_per_call"),
    )

    # Build enforcement policy from the enforcement section
    if enforcement_section:
        config.policy = EnforcementPolicy.from_dict(enforcement_section)

    # Layer 2: env var overrides
    if os.environ.get("AEP_PORT"):
        config.port = int(os.environ["AEP_PORT"])
    if os.environ.get("AEP_HOST"):
        config.host = os.environ["AEP_HOST"]
    if os.environ.get("AEP_TARGET"):
        config.target = os.environ["AEP_TARGET"]
    if os.environ.get("AEP_LOG", "") in ("1", "true", "yes"):
        config.verbose = True

    # AEP_POLICY env var overrides the enforcement section
    policy_path = os.environ.get("AEP_POLICY")
    if policy_path:
        config.policy = EnforcementPolicy.from_yaml(policy_path)

    # Layer 3: CLI overrides (highest priority)
    if cli_overrides:
        if cli_overrides.get("port") is not None:
            config.port = cli_overrides["port"]
        if cli_overrides.get("host") is not None:
            config.host = cli_overrides["host"]
        if cli_overrides.get("target") is not None:
            config.target = cli_overrides["target"]
        if cli_overrides.get("no_dashboard"):
            config.dashboard = False
        if cli_overrides.get("policy"):
            config.policy = EnforcementPolicy.from_yaml(cli_overrides["policy"])

    return config


def _load_yaml(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML config file."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "pyyaml is required for YAML config files. "
            "Install it with: pip install aceteam-aep[yaml]"
        ) from None

    text = Path(path).read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML dict in {path}, got {type(data).__name__}")
    return data


__all__ = ["ProxyConfig", "load_config"]
