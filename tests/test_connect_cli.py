"""Tests for the aceteam-aep connect and disconnect CLI commands."""

import json
import subprocess
import sys
from pathlib import Path


def test_connect_print_help():
    """connect --help should work."""
    result = subprocess.run(
        ["uv", "run", "aceteam-aep", "connect", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "AceTeam" in result.stdout


def test_disconnect_print_help():
    """disconnect --help should work."""
    result = subprocess.run(
        ["uv", "run", "aceteam-aep", "disconnect", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_connect_invalid_key_without_force(tmp_path):
    """connect should reject keys that don't start with act_ unless --force is given."""
    result = subprocess.run(
        [sys.executable, "-m", "aceteam_aep.proxy.cli", "connect", "--api-key", "bad_key_format"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Warning" in result.stdout
    assert "act_" in result.stdout


def test_connect_saves_credentials(tmp_path, monkeypatch):
    """connect --api-key should save credentials to ~/.config/aceteam-aep/credentials.json."""
    cred_dir = tmp_path / ".config" / "aceteam-aep"
    cred_file = cred_dir / "credentials.json"

    # Patch Path.home() by running in a subprocess with HOME overridden
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aceteam_aep.proxy.cli",
            "connect",
            "--api-key",
            "act_testkey1234567890",
        ],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "HOME": str(tmp_path)},
    )
    assert result.returncode == 0
    assert "Credentials saved" in result.stdout

    saved = json.loads(cred_file.read_text())
    assert saved["api_key"] == "act_testkey1234567890"
    assert saved["url"] == "https://aceteam.ai"


def test_connect_already_connected(tmp_path):
    """connect should report already connected when credentials exist."""
    cred_dir = tmp_path / ".config" / "aceteam-aep"
    cred_dir.mkdir(parents=True)
    cred_file = cred_dir / "credentials.json"
    cred_file.write_text(json.dumps({"api_key": "act_existing_key_abc", "url": "https://aceteam.ai"}))

    result = subprocess.run(
        [sys.executable, "-m", "aceteam_aep.proxy.cli", "connect"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "HOME": str(tmp_path)},
    )
    assert result.returncode == 0
    assert "Already connected" in result.stdout
    assert "--force" in result.stdout


def test_connect_force_overwrites(tmp_path):
    """connect --force should overwrite existing credentials."""
    cred_dir = tmp_path / ".config" / "aceteam-aep"
    cred_dir.mkdir(parents=True)
    cred_file = cred_dir / "credentials.json"
    cred_file.write_text(json.dumps({"api_key": "act_old_key", "url": "https://aceteam.ai"}))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aceteam_aep.proxy.cli",
            "connect",
            "--api-key",
            "act_new_key_xyz",
            "--force",
        ],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "HOME": str(tmp_path)},
    )
    assert result.returncode == 0
    saved = json.loads(cred_file.read_text())
    assert saved["api_key"] == "act_new_key_xyz"


def test_disconnect_removes_credentials(tmp_path):
    """disconnect should remove the credentials file."""
    cred_dir = tmp_path / ".config" / "aceteam-aep"
    cred_dir.mkdir(parents=True)
    cred_file = cred_dir / "credentials.json"
    cred_file.write_text(json.dumps({"api_key": "act_some_key", "url": "https://aceteam.ai"}))

    result = subprocess.run(
        [sys.executable, "-m", "aceteam_aep.proxy.cli", "disconnect"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "HOME": str(tmp_path)},
    )
    assert result.returncode == 0
    assert "Disconnected" in result.stdout
    assert not cred_file.exists()


def test_disconnect_when_not_connected(tmp_path):
    """disconnect should report gracefully when no credentials file exists."""
    result = subprocess.run(
        [sys.executable, "-m", "aceteam_aep.proxy.cli", "disconnect"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "HOME": str(tmp_path)},
    )
    assert result.returncode == 0
    assert "Not connected" in result.stdout
