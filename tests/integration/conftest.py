"""Shared fixtures for integration tests."""

from __future__ import annotations

import os
import socket
import subprocess
import time

import httpx
import pytest


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def require_api_key():
    """Skip tests if no OpenAI API key is available."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


@pytest.fixture(scope="module")
def aep_proxy():
    """Start an AEP proxy for integration tests. Returns (port, base_url)."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    port = _find_free_port()
    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "aceteam-aep",
            "proxy",
            "--port",
            str(port),
            "--detector",
            "aceteam_aep.safety.agent_threat:AgentThreatDetector",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for proxy to be ready
    base_url = f"http://localhost:{port}"
    for _ in range(30):
        try:
            r = httpx.get(f"{base_url}/dashboard/api/state", timeout=1)
            if r.status_code == 200:
                break
        except httpx.ConnectError:
            time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("AEP proxy failed to start")

    yield port, base_url

    proc.terminate()
    proc.wait(timeout=5)


def get_proxy_state(base_url: str) -> dict:
    """Fetch current proxy state."""
    r = httpx.get(f"{base_url}/dashboard/api/state", timeout=5)
    return r.json()
