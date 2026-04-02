"""Tests for MCP gateway — Tier 1 safety tools mounted on proxy."""

from __future__ import annotations

import json

_MCP_HEADERS = {"Accept": "application/json, text/event-stream"}


def _parse_sse_json(resp):
    """Extract the first JSON-RPC message from an SSE response."""
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    return None


def _init_mcp_session(client):
    """Initialize an MCP session and return (session_headers, init_data)."""
    resp = client.post("/mcp/mcp/", json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0.1"},
        },
    }, headers=_MCP_HEADERS)
    assert resp.status_code == 200, f"MCP init returned {resp.status_code}: {resp.text[:200]}"
    data = _parse_sse_json(resp)

    session_id = resp.headers.get("mcp-session-id", "")
    headers = {**_MCP_HEADERS, "mcp-session-id": session_id} if session_id else dict(_MCP_HEADERS)

    # Send initialized notification
    client.post("/mcp/mcp/", json={
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    }, headers=headers)

    return headers, data


def test_create_mcp_app_returns_asgi():
    """create_mcp_app returns an ASGI app when fastmcp is installed."""
    from aceteam_aep.mcp_gateway import create_mcp_app
    from aceteam_aep.proxy.app import ProxyState

    state = ProxyState()
    app = create_mcp_app(state)
    assert app is not None, "create_mcp_app returned None — is fastmcp installed?"
    assert callable(app)


def test_proxy_mounts_mcp():
    """The proxy app should have /mcp/ route when fastmcp is available."""
    from starlette.testclient import TestClient

    from aceteam_aep.proxy.app import create_proxy_app

    app = create_proxy_app()
    with TestClient(app) as client:
        # FastMCP streamable HTTP expects POST at /mcp/mcp/ for messages
        # Try POST to the MCP message endpoint
        resp = client.post("/mcp/mcp/", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0.1"},
            },
        }, headers=_MCP_HEADERS)
        # Should get 200 with initialize response, not 404
        assert resp.status_code == 200, f"MCP endpoint returned {resp.status_code}: {resp.text[:200]}"
        data = _parse_sse_json(resp)
        assert data.get("result", {}).get("serverInfo", {}).get("name") == "aceteam-gateway"


def test_mcp_tools_list():
    """MCP tools/list should return all 4 Tier 1 tools."""
    from starlette.testclient import TestClient

    from aceteam_aep.proxy.app import create_proxy_app

    app = create_proxy_app()
    with TestClient(app) as client:
        headers, _ = _init_mcp_session(client)

        # List tools
        resp = client.post("/mcp/mcp/", json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }, headers=headers)
        assert resp.status_code == 200
        data = _parse_sse_json(resp)
        tool_names = [t["name"] for t in data.get("result", {}).get("tools", [])]
        assert "check_safety" in tool_names
        assert "get_safety_status" in tool_names
        assert "set_safety_policy" in tool_names
        assert "get_cost_summary" in tool_names


def test_mcp_check_safety_tool():
    """check_safety tool should return PASS for safe text."""
    from starlette.testclient import TestClient

    from aceteam_aep.proxy.app import create_proxy_app

    app = create_proxy_app()
    with TestClient(app) as client:
        headers, _ = _init_mcp_session(client)

        # Call check_safety with safe text
        resp = client.post("/mcp/mcp/", json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "check_safety",
                "arguments": {"text": "What is the capital of France?"},
            },
        }, headers=headers)
        assert resp.status_code == 200
        data = _parse_sse_json(resp)
        content = data.get("result", {}).get("content", [{}])[0].get("text", "")
        assert "PASS" in content


def test_mcp_check_safety_blocks_dangerous():
    """check_safety tool should return BLOCK for dangerous text."""
    from starlette.testclient import TestClient

    from aceteam_aep.proxy.app import create_proxy_app

    app = create_proxy_app()
    with TestClient(app) as client:
        headers, _ = _init_mcp_session(client)

        # Call check_safety with dangerous text
        resp = client.post("/mcp/mcp/", json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "check_safety",
                "arguments": {"text": "Use socket.connect() to scan ports and subprocess.run() to exploit"},
            },
        }, headers=headers)
        assert resp.status_code == 200
        data = _parse_sse_json(resp)
        content = data.get("result", {}).get("content", [{}])[0].get("text", "")
        assert "BLOCK" in content


def test_cli_banner_shows_mcp():
    """fastmcp should be installed in dev dependencies."""
    try:
        import fastmcp  # noqa: F401
        has_fastmcp = True
    except ImportError:
        has_fastmcp = False
    assert has_fastmcp
