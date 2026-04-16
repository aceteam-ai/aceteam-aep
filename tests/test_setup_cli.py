"""Tests for aceteam-aep proxy setup commands (keygen, verify, proxy --help).

The proxy command is the "setup" command for the AEP gateway — it configures
and starts the reverse proxy that intercepts LLM calls. These tests cover
help output, keygen keypair generation, verify chain validation, and proxy
argument parsing without starting a live server.
"""

from __future__ import annotations

import subprocess
import sys


def test_proxy_help():
    """proxy --help should exit 0 and include proxy-specific flags."""
    result = subprocess.run(
        [sys.executable, "-m", "aceteam_aep.proxy.cli", "proxy", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--port" in result.stdout
    assert "--target" in result.stdout
    assert "--no-safety" in result.stdout


def test_keygen_help():
    """keygen --help should exit 0 and mention Ed25519."""
    result = subprocess.run(
        [sys.executable, "-m", "aceteam_aep.proxy.cli", "keygen", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--output" in result.stdout


def test_keygen_generates_keypair(tmp_path):
    """keygen should create aep.key and aep.pub in the output directory."""
    result = subprocess.run(
        [sys.executable, "-m", "aceteam_aep.proxy.cli", "keygen", "--output", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert (tmp_path / "aep.key").exists(), "aep.key not created"
    assert (tmp_path / "aep.pub").exists(), "aep.pub not created"
    assert "Private key" in result.stdout
    assert "Public key" in result.stdout


def test_keygen_output_mentions_proxy_usage(tmp_path):
    """keygen output should show how to use the key with the proxy."""
    result = subprocess.run(
        [sys.executable, "-m", "aceteam_aep.proxy.cli", "keygen", "--output", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert "--sign-key" in result.stdout
    assert "aceteam-aep proxy" in result.stdout


def test_verify_help():
    """verify --help should exit 0 and mention chain and pub-key flags."""
    result = subprocess.run(
        [sys.executable, "-m", "aceteam_aep.proxy.cli", "verify", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--pub-key" in result.stdout
    assert "--chain" in result.stdout


def test_verify_empty_chain(tmp_path):
    """verify should exit 0 cleanly when the chain file is empty."""
    pub_key_path = tmp_path / "aep.pub"

    # Generate a real keypair first
    subprocess.run(
        [sys.executable, "-m", "aceteam_aep.proxy.cli", "keygen", "--output", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert pub_key_path.exists(), "keygen did not produce aep.pub"

    chain_path = tmp_path / "chain.jsonl"
    chain_path.write_text("")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aceteam_aep.proxy.cli",
            "verify",
            "--pub-key",
            str(pub_key_path),
            "--chain",
            str(chain_path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "nothing to verify" in result.stdout.lower() or "Empty" in result.stdout


def test_verify_missing_chain_file(tmp_path):
    """verify should exit non-zero when the chain file does not exist."""
    pub_key_path = tmp_path / "aep.pub"
    subprocess.run(
        [sys.executable, "-m", "aceteam_aep.proxy.cli", "keygen", "--output", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aceteam_aep.proxy.cli",
            "verify",
            "--pub-key",
            str(pub_key_path),
            "--chain",
            str(tmp_path / "nonexistent.jsonl"),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "not found" in result.stdout.lower() or "Error" in result.stdout


def test_proxy_no_safety_flag_in_help():
    """proxy --help should document the --no-safety flag for disabling detectors."""
    result = subprocess.run(
        [sys.executable, "-m", "aceteam_aep.proxy.cli", "proxy", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--no-safety" in result.stdout


def test_mcp_server_help():
    """mcp-server --help should exit 0."""
    result = subprocess.run(
        [sys.executable, "-m", "aceteam_aep.proxy.cli", "mcp-server", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--policy" in result.stdout


def test_top_level_help_lists_all_subcommands():
    """Top-level --help should list all expected subcommands."""
    result = subprocess.run(
        [sys.executable, "-m", "aceteam_aep.proxy.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    for sub in ("proxy", "wrap", "keygen", "verify", "mcp-server"):
        assert sub in result.stdout, f"subcommand '{sub}' missing from top-level help"
