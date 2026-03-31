"""CLI entry point for the AEP proxy and wrap commands.

Usage:
    aceteam-aep proxy --port 8899
    aceteam-aep wrap -- python my_agent.py
    aceteam-aep wrap --block-on high -- node my_bot.js
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import socket
import subprocess
import sys
import threading
import time

log = logging.getLogger(__name__)


def _load_detector(path: str) -> object:
    """Load a detector from a ``module:class`` path string."""
    if ":" not in path:
        raise ValueError(f"Invalid detector path '{path}'. Expected format: 'module:ClassName'")
    module_path, class_name = path.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls()


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _resolve_config(args: argparse.Namespace) -> object:
    """Resolve full config from --config file, env vars, and CLI args."""
    from ..config import load_config

    config_path = getattr(args, "config", None) or os.environ.get("AEP_CONFIG")

    cli_overrides: dict[str, object] = {}
    # Only override if explicitly set (not default values)
    if getattr(args, "port", None) is not None:
        cli_overrides["port"] = args.port
    if getattr(args, "host", None) is not None:
        cli_overrides["host"] = args.host
    if getattr(args, "target", None) is not None:
        cli_overrides["target"] = args.target
    if getattr(args, "no_dashboard", False):
        cli_overrides["no_dashboard"] = True
    if getattr(args, "policy", None):
        cli_overrides["policy"] = args.policy

    return load_config(config_path, cli_overrides=cli_overrides)


def _resolve_policy(args: argparse.Namespace) -> object | None:
    """Resolve enforcement policy from --policy flag or AEP_POLICY env var."""
    from ..enforcement import EnforcementPolicy

    policy_path = getattr(args, "policy", None) or os.environ.get("AEP_POLICY")
    if policy_path:
        return EnforcementPolicy.from_yaml(policy_path)
    return None


def _build_detectors(args: argparse.Namespace, policy: object | None = None) -> list[object] | None:
    """Build detector list from CLI args and policy."""
    from ..safety.cost_anomaly import CostAnomalyDetector

    if args.no_safety:
        return [CostAnomalyDetector()]

    custom = getattr(args, "detector", None)
    if custom:
        from .app import _default_proxy_detectors

        detectors = _default_proxy_detectors()
        for path in custom:
            detectors.append(_load_detector(path))
        return detectors

    # If policy has detector overrides, build from policy
    if policy is not None and hasattr(policy, "overrides") and policy.overrides:  # type: ignore[union-attr]
        from ..enforcement import build_detectors_from_policy

        return build_detectors_from_policy(policy)  # type: ignore[arg-type]

    return None


def _run_proxy(args: argparse.Namespace) -> None:
    """Start the AEP reverse proxy."""
    import uvicorn

    from .app import create_proxy_app

    # Use unified config if --config provided, otherwise legacy args
    config_path = getattr(args, "config", None) or os.environ.get("AEP_CONFIG")
    if config_path:
        cfg = _resolve_config(args)
        detectors = _build_detectors(args, cfg.policy)  # type: ignore[attr-defined]
        app = create_proxy_app(
            target_base_url=cfg.target,  # type: ignore[attr-defined]
            detectors=detectors,
            policy=cfg.policy,  # type: ignore[attr-defined]
            dashboard=cfg.dashboard,  # type: ignore[attr-defined]
        )
        port = cfg.port  # type: ignore[attr-defined]
        host = cfg.host  # type: ignore[attr-defined]
        dashboard = cfg.dashboard  # type: ignore[attr-defined]
        target = cfg.target  # type: ignore[attr-defined]
    else:
        policy = _resolve_policy(args)
        detectors = _build_detectors(args, policy)
        target = args.target or "https://api.openai.com"
        sign_key, signer_id = _load_sign_key(args)
        app = create_proxy_app(
            target_base_url=target,
            detectors=detectors,
            policy=policy,
            dashboard=not args.no_dashboard,
            sign_key=sign_key,
            signer_id=signer_id,
            budget=getattr(args, "budget", None),
            budget_per_session=getattr(args, "budget_per_session", None),
        )
        port = args.port or 8899
        host = args.host or "127.0.0.1"
        dashboard = not args.no_dashboard

    dashboard_msg = ""
    if dashboard:
        dashboard_msg = f"  Dashboard:  http://localhost:{port}/aep/\n"

    print(
        f"\n"
        f"  AEP Proxy\n"
        f"  {'─' * 35}\n"
        f"  Listening:  http://localhost:{port}\n"
        f"  Target:     {target}\n"
        f"{dashboard_msg}"
        f"\n"
        f"  Usage:\n"
        f"    export OPENAI_BASE_URL=http://localhost:{port}/v1\n"
        f"    python my_agent.py\n"
        f"\n"
    )

    uvicorn.run(app, host=host, port=port, log_level="info")


def _run_wrap(args: argparse.Namespace) -> None:
    """Start proxy, set env vars, run command, print summary on exit."""
    import uvicorn

    from .app import create_proxy_app

    if not args.cmd:
        print("Error: no command specified. Usage: aceteam-aep wrap -- python my_agent.py")
        sys.exit(1)

    # Use unified config if --config provided, otherwise legacy args
    config_path = getattr(args, "config", None) or os.environ.get("AEP_CONFIG")
    if config_path:
        cfg = _resolve_config(args)
        port = args.port or cfg.port or _find_free_port()  # type: ignore[attr-defined]
        host = args.host or cfg.host or "127.0.0.1"  # type: ignore[attr-defined]
        target = args.target or cfg.target or "https://api.openai.com"  # type: ignore[attr-defined]
        detectors = _build_detectors(args, cfg.policy)  # type: ignore[attr-defined]
        dashboard = cfg.dashboard  # type: ignore[attr-defined]
        if args.no_dashboard:
            dashboard = False
        app = create_proxy_app(
            target_base_url=target,
            detectors=detectors,
            policy=cfg.policy,  # type: ignore[attr-defined]
            dashboard=dashboard,
        )
    else:
        port = args.port or _find_free_port()
        host = args.host or "127.0.0.1"
        target = args.target or "https://api.openai.com"
        policy = _resolve_policy(args)
        detectors = _build_detectors(args, policy)
        dashboard = not args.no_dashboard
        app = create_proxy_app(
            target_base_url=target,
            detectors=detectors,
            policy=policy,
            dashboard=dashboard,
        )

    # Start proxy in a background thread
    server_config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(server_config)
    proxy_thread = threading.Thread(target=server.run, daemon=True)
    proxy_thread.start()

    # Wait for proxy to be ready
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.1)
    else:
        print("Error: proxy failed to start")
        sys.exit(1)

    base_url = f"http://localhost:{port}/v1"

    print(
        f"\n"
        f"  \033[36m\033[1mAEP Wrap\033[0m\n"
        f"  \033[2m{'─' * 45}\033[0m\n"
        f"  Proxy:    http://localhost:{port}\n"
        f"  Target:   {target}\n"
        f"  Command:  {' '.join(args.cmd)}\n"
        f"  \033[2m{'─' * 45}\033[0m\n"
    )

    # Set env vars for the child process
    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = base_url
    env["ANTHROPIC_BASE_URL"] = base_url

    # Run the command
    try:
        result = subprocess.run(args.cmd, env=env)
        exit_code = result.returncode
    except KeyboardInterrupt:
        exit_code = 130
    except FileNotFoundError:
        print(f"Error: command not found: {args.cmd[0]}")
        exit_code = 127

    # Fetch summary from proxy state endpoint
    try:
        import httpx

        resp = httpx.get(f"http://localhost:{port}/aep/api/state", timeout=2.0)
        if resp.status_code == 200:
            state = resp.json()
            _print_wrap_summary(state)
    except Exception:
        log.debug("Could not fetch summary (proxy may be shutting down)", exc_info=True)

    # Shutdown proxy
    server.should_exit = True
    sys.exit(exit_code)


def _load_sign_key(args: argparse.Namespace) -> tuple[object | None, str]:
    """Load signing key from --sign-key arg. Returns (key, signer_id) or (None, "")."""
    sign_key_path = getattr(args, "sign_key", None)
    if not sign_key_path:
        return None, ""
    from ..attestation import AepPrivateKey

    key = AepPrivateKey.load(sign_key_path)
    signer_id = getattr(args, "signer_id", "proxy:default")
    return key, signer_id


def _run_keygen(args: argparse.Namespace) -> None:
    """Generate Ed25519 keypair for verdict signing."""
    from pathlib import Path

    from ..attestation import generate_keypair

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    private, public = generate_keypair()
    key_path = output_dir / "aep.key"
    pub_path = output_dir / "aep.pub"

    private.save(key_path)
    public.save(pub_path)

    print("  Generated Ed25519 keypair:")
    print(f"  Private key: {key_path}")
    print(f"  Public key:  {pub_path}")
    print("")
    print("  Start proxy with signing:")
    print(f"    aceteam-aep proxy --sign-key {key_path}")
    print("")
    print("  Verify a chain:")
    print(f"    aceteam-aep verify --pub-key {pub_path} --chain audit.jsonl")


def _run_verify(args: argparse.Namespace) -> None:
    """Verify a Merkle audit chain."""
    import json
    from pathlib import Path

    from ..attestation import AepPublicKey, verify_chain

    pub_key = AepPublicKey.load(args.pub_key)
    chain_path = Path(args.chain)

    if not chain_path.exists():
        print(f"Error: chain file not found: {chain_path}")
        sys.exit(1)

    chain = []
    for line in chain_path.read_text().strip().splitlines():
        if line.strip():
            chain.append(json.loads(line))

    if not chain:
        print("Empty chain — nothing to verify.")
        sys.exit(0)

    valid = verify_chain(chain, pub_key)

    if valid:
        print("  Chain VALID")
        print(f"  Entries: {len(chain)}")
        print(f"  Height:  0 → {len(chain) - 1}")
        print(f"  Final hash: {chain[-1]['chain_hash']}")
    else:
        print("  Chain INVALID — tampering detected")
        sys.exit(1)


def _print_wrap_summary(state: dict) -> None:
    """Print colored summary from proxy state."""
    _GREEN = "\033[32m"
    _YELLOW = "\033[33m"
    _RED = "\033[31m"
    _CYAN = "\033[36m"
    _RESET = "\033[0m"
    _BOLD = "\033[1m"
    _DIM = "\033[2m"

    action_colors = {"pass": _GREEN, "flag": _YELLOW, "block": _RED}
    sev_colors = {"high": _RED, "medium": _YELLOW, "low": _CYAN}

    call_count = state.get("calls", 0)
    cost = state.get("cost", 0)
    signals = state.get("signals", [])
    action = state.get("action", "pass")
    color = action_colors.get(action, "")

    print(
        f"\n"
        f"  {_DIM}{'─' * 45}{_RESET}\n"
        f"  {_BOLD}AEP Summary{_RESET}\n"
        f"  {_DIM}{'─' * 45}{_RESET}\n"
        f"  Calls:  {call_count}\n"
        f"  Cost:   ${cost}\n"
        f"  Safety: {color}{_BOLD}{action.upper()}{_RESET}"
    )

    if signals:
        print(f"  Signals ({len(signals)}):")
        for s in signals:
            sev = s.get("severity", "low")
            sev_color = sev_colors.get(sev, "")
            print(
                f"    [{sev_color}{sev.upper()}{_RESET}] "
                f"{s.get('type', '?')}: {s.get('detail', '')}"
            )

    print(f"  {_DIM}{'─' * 45}{_RESET}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aceteam-aep",
        description="AEP — safety & accountability infrastructure for AI agents",
    )
    sub = parser.add_subparsers(dest="command")

    # --- proxy subcommand ---
    proxy_parser = sub.add_parser("proxy", help="Start the AEP reverse proxy")
    proxy_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to aep.yaml config file (or set AEP_CONFIG env var)",
    )
    proxy_parser.add_argument(
        "--port", type=int, default=None, help="Port to listen on (default: 8899)"
    )
    proxy_parser.add_argument(
        "--host", type=str, default=None, help="Host to bind to (default: 127.0.0.1)"
    )
    proxy_parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target API base URL (default: https://api.openai.com)",
    )
    proxy_parser.add_argument(
        "--dashboard", action="store_true", default=True, help="Enable dashboard (default: on)"
    )
    proxy_parser.add_argument("--no-dashboard", action="store_true", help="Disable the dashboard")
    proxy_parser.add_argument(
        "--no-safety", action="store_true", help="Disable safety detectors (cost tracking only)"
    )
    proxy_parser.add_argument(
        "--detector",
        action="append",
        metavar="MODULE:CLASS",
        help="Add a custom detector (module:Class). Repeatable.",
    )
    proxy_parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to AEP policy YAML file (or set AEP_POLICY env var)",
    )
    proxy_parser.add_argument(
        "--sign-key",
        type=str,
        default=None,
        help="Path to Ed25519 private key for verdict signing (enables attestation)",
    )
    proxy_parser.add_argument(
        "--signer-id",
        type=str,
        default="proxy:default",
        help="Signer identity for attestation headers (default: proxy:default)",
    )
    proxy_parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Total budget cap in USD (returns 429 when exceeded)",
    )
    proxy_parser.add_argument(
        "--budget-per-session",
        type=float,
        default=None,
        help="Per-session budget cap in USD (returns 429 when exceeded)",
    )

    # --- keygen subcommand ---
    keygen_parser = sub.add_parser("keygen", help="Generate Ed25519 keypair for verdict signing")
    keygen_parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output directory for aep.key and aep.pub (default: current dir)",
    )

    # --- verify subcommand ---
    verify_parser = sub.add_parser("verify", help="Verify a Merkle audit chain")
    verify_parser.add_argument(
        "--pub-key",
        type=str,
        required=True,
        help="Path to Ed25519 public key",
    )
    verify_parser.add_argument(
        "--chain",
        type=str,
        required=True,
        help="Path to audit chain JSONL file",
    )

    # --- wrap subcommand ---
    wrap_parser = sub.add_parser(
        "wrap",
        help="Wrap a command with AEP — intercept all LLM calls",
        epilog="Example: aceteam-aep wrap -- python my_agent.py",
    )
    wrap_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to aep.yaml config file (or set AEP_CONFIG env var)",
    )
    wrap_parser.add_argument("--port", type=int, default=None, help="Proxy port (default: auto)")
    wrap_parser.add_argument(
        "--host", type=str, default=None, help="Host to bind to (default: 127.0.0.1)"
    )
    wrap_parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target API base URL (default: https://api.openai.com)",
    )
    wrap_parser.add_argument("--no-dashboard", action="store_true", help="Disable the dashboard")
    wrap_parser.add_argument(
        "--no-safety", action="store_true", help="Disable safety detectors (cost tracking only)"
    )
    wrap_parser.add_argument(
        "--detector",
        action="append",
        metavar="MODULE:CLASS",
        help="Add a custom detector (module:Class). Repeatable.",
    )
    wrap_parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to AEP policy YAML file (or set AEP_POLICY env var)",
    )
    wrap_parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to run (after --)")

    # --- mcp-server subcommand ---
    mcp_parser = sub.add_parser(
        "mcp-server",
        help="Start AEP safety as an MCP server (stdin/stdout JSON-RPC)",
    )
    mcp_parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to AEP policy YAML file",
    )

    args = parser.parse_args()

    # Strip leading "--" from cmd if present
    if hasattr(args, "cmd") and args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]

    if args.command == "proxy":
        _run_proxy(args)
    elif args.command == "wrap":
        _run_wrap(args)
    elif args.command == "keygen":
        _run_keygen(args)
    elif args.command == "verify":
        _run_verify(args)
    elif args.command == "mcp-server":
        from ..mcp import run_mcp_server

        run_mcp_server(policy_path=args.policy)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
