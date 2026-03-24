"""Tests for the unified AEP configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest

from aceteam_aep.config import load_config


@pytest.fixture()
def yaml_config(tmp_path: Path) -> Path:
    """Write a sample aep.yaml and return its path."""
    config = tmp_path / "aep.yaml"
    config.write_text("""\
proxy:
  port: 9000
  host: 0.0.0.0
  target: https://api.anthropic.com
  dashboard: false

enforcement:
  default_action: block
  block_on: [high]
  flag_on: [medium]
  detectors:
    pii:
      action: block
      threshold: 0.8
    cost_anomaly:
      action: pass
      multiplier: 10

budget:
  max_per_session: 5.0
  max_per_call: 0.50
""")
    return config


class TestLoadConfig:
    def test_defaults_without_file(self) -> None:
        config = load_config(None)
        assert config.port == 8899
        assert config.host == "127.0.0.1"
        assert config.target == "https://api.openai.com"
        assert config.dashboard is True
        assert config.max_per_session is None

    def test_loads_from_yaml(self, yaml_config: Path) -> None:
        config = load_config(yaml_config)
        assert config.port == 9000
        assert config.host == "0.0.0.0"
        assert config.target == "https://api.anthropic.com"
        assert config.dashboard is False
        assert config.max_per_session == 5.0
        assert config.max_per_call == 0.50

    def test_enforcement_from_yaml(self, yaml_config: Path) -> None:
        config = load_config(yaml_config)
        assert config.policy.default_action == "block"
        assert "pii" in config.policy.overrides
        assert config.policy.overrides["pii"].action == "block"
        assert config.policy.overrides["pii"].threshold == 0.8
        assert config.policy.overrides["cost_anomaly"].extra["multiplier"] == 10

    def test_env_var_overrides_yaml(
        self, yaml_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AEP_PORT", "7777")
        monkeypatch.setenv("AEP_TARGET", "https://custom.api.com")
        config = load_config(yaml_config)
        assert config.port == 7777
        assert config.target == "https://custom.api.com"
        # Non-overridden values still from YAML
        assert config.host == "0.0.0.0"

    def test_cli_overrides_env_and_yaml(
        self, yaml_config: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AEP_PORT", "7777")
        config = load_config(yaml_config, cli_overrides={"port": 5555})
        assert config.port == 5555  # CLI wins over env

    def test_cli_no_dashboard_override(self, yaml_config: Path) -> None:
        # YAML has dashboard: false, but let's test the override mechanism
        config_with_dashboard = load_config(None, cli_overrides={"no_dashboard": True})
        assert config_with_dashboard.dashboard is False

    def test_env_aep_log_enables_verbose(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AEP_LOG", "1")
        config = load_config(None)
        assert config.verbose is True

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("- just a list\n- not a dict\n")
        with pytest.raises(ValueError, match="Expected YAML dict"):
            load_config(bad)

    def test_invalid_port_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AEP_PORT", "not-a-number")
        with pytest.raises(ValueError, match="AEP_PORT must be a number"):
            load_config(None)

    def test_partial_yaml(self, tmp_path: Path) -> None:
        """A YAML with only some sections should use defaults for the rest."""
        partial = tmp_path / "partial.yaml"
        partial.write_text("proxy:\n  port: 3000\n")
        config = load_config(partial)
        assert config.port == 3000
        assert config.host == "127.0.0.1"  # default
        assert config.target == "https://api.openai.com"  # default
        assert config.dashboard is True  # default
