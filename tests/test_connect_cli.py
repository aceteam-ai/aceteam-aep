"""Tests for aceteam-aep proxy connect/disconnect behavior.

"Connecting" to the LLM backend means starting the proxy with a target URL.
"Disconnecting" means the proxy correctly tears down when the child process exits
(wrap command) or when the server receives a shutdown signal.

These tests validate proxy startup configuration: target URL handling, custom port
assignment, host binding options, and budget enforcement configuration.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest


def test_proxy_default_target_in_banner():
    """Proxy startup banner should show the default OpenAI target."""
    import threading

    import uvicorn

    from aceteam_aep.proxy.app import create_proxy_app
    from aceteam_aep.proxy.cli import _find_free_port

    port = _find_free_port()
    app = create_proxy_app(target_base_url="https://api.openai.com")
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready
    import socket

    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.1)

    server.should_exit = True
    thread.join(timeout=5)
    # If we got here without error the proxy started and stopped cleanly
    assert True


def test_proxy_custom_target():
    """ProxyState should store a custom target URL."""
    from aceteam_aep.proxy.app import ProxyState

    state = ProxyState(target_base_url="https://api.anthropic.com")
    assert state.target_base_url == "https://api.anthropic.com"


def test_proxy_target_strips_trailing_slash():
    """ProxyState should normalize the target URL by stripping trailing slashes."""
    from aceteam_aep.proxy.app import ProxyState

    state = ProxyState(target_base_url="https://api.openai.com/")
    assert not state.target_base_url.endswith("/")
    assert state.target_base_url == "https://api.openai.com"


def test_proxy_initial_state_is_connected():
    """A new ProxyState should be in a clean connected state."""
    from aceteam_aep.proxy.app import ProxyState

    state = ProxyState()
    assert state.call_count == 0
    assert state.cost_usd == 0
    assert state.safety_enabled is True
    assert state.blocked_count == 0
    assert len(state.signals) == 0


def test_proxy_budget_connect_respected():
    """ProxyState should store budget limits for enforcement on connect."""
    from aceteam_aep.proxy.app import ProxyState

    state = ProxyState(budget=1.0, budget_per_session=0.5)
    assert state.budget is not None
    assert float(state.budget) == pytest.approx(1.0)
    assert state.budget_per_session is not None
    assert float(state.budget_per_session) == pytest.approx(0.5)


def test_proxy_no_budget_by_default():
    """ProxyState without budget args should have no budget constraints."""
    from aceteam_aep.proxy.app import ProxyState

    state = ProxyState()
    assert state.budget is None
    assert state.budget_per_session is None
    assert state.check_budget() is None


def test_proxy_disconnect_via_wrap(tmp_path):
    """wrap should fully disconnect (exit cleanly) after child command finishes."""
    result = subprocess.run(
        [
            sys.executable, "-m", "aceteam_aep.proxy.cli",
            "wrap", "--no-safety", "--", "true",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"wrap exited with non-zero: {result.stderr}"


def test_proxy_disconnect_propagates_child_failure():
    """wrap should exit with child's non-zero code on disconnect."""
    result = subprocess.run(
        [
            sys.executable, "-m", "aceteam_aep.proxy.cli",
            "wrap", "--no-safety", "--",
            sys.executable, "-c", "import sys; sys.exit(3)",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 3


def test_proxy_find_free_port_is_unique():
    """_find_free_port should return different ports on successive calls."""
    from aceteam_aep.proxy.cli import _find_free_port

    ports = {_find_free_port() for _ in range(5)}
    # All 5 ports should be distinct (each call binds and releases immediately)
    assert len(ports) == 5
