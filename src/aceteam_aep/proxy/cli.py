"""CLI entry point for the AEP proxy and wrap commands.

Usage:
    aceteam-aep proxy --port 8899
    aceteam-aep wrap -- python my_agent.py
    aceteam-aep wrap --block-on high -- node my_bot.js
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_proxy(args: argparse.Namespace) -> None:
    """Start the AEP reverse proxy."""
    import uvicorn

    from ..safety.cost_anomaly import CostAnomalyDetector
    from .app import create_proxy_app

    detectors = None
    if args.no_safety:
        detectors = [CostAnomalyDetector()]

    app = create_proxy_app(
        target_base_url=args.target,
        detectors=detectors,
        dashboard=not args.no_dashboard,
    )

    dashboard_msg = ""
    if not args.no_dashboard:
        dashboard_msg = f"  Dashboard:  http://localhost:{args.port}/aep/\n"

    print(
        f"\n"
        f"  AEP Proxy\n"
        f"  {'─' * 35}\n"
        f"  Listening:  http://localhost:{args.port}\n"
        f"  Target:     {args.target}\n"
        f"{dashboard_msg}"
        f"\n"
        f"  Usage:\n"
        f"    export OPENAI_BASE_URL=http://localhost:{args.port}/v1\n"
        f"    python my_agent.py\n"
        f"\n"
    )

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


def _run_wrap(args: argparse.Namespace) -> None:
    """Start proxy, set env vars, run command, print summary on exit."""
    import uvicorn

    from ..safety.cost_anomaly import CostAnomalyDetector
    from .app import create_proxy_app

    if not args.cmd:
        print("Error: no command specified. Usage: aceteam-aep wrap -- python my_agent.py")
        sys.exit(1)

    port = args.port or _find_free_port()

    detectors = None
    if args.no_safety:
        detectors = [CostAnomalyDetector()]

    app = create_proxy_app(
        target_base_url=args.target,
        detectors=detectors,
        dashboard=not args.no_dashboard,
    )

    # Get proxy state for summary later
    proxy_state = None
    for route in app.routes:
        if hasattr(route, "app") and hasattr(route, "path"):
            pass
    # Access state through the app's state_getter
    # The proxy app stores state in a closure — we access it via the /aep/api/state endpoint

    # Start proxy in a background thread
    server_config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning"
    )
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
        f"  Target:   {args.target}\n"
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
        pass  # Proxy may already be shutting down

    # Shutdown proxy
    server.should_exit = True
    sys.exit(exit_code)


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
        "--port", type=int, default=8899, help="Port to listen on (default: 8899)"
    )
    proxy_parser.add_argument(
        "--target",
        type=str,
        default="https://api.openai.com",
        help="Target API base URL (default: https://api.openai.com)",
    )
    proxy_parser.add_argument(
        "--dashboard", action="store_true", default=True, help="Enable dashboard (default: on)"
    )
    proxy_parser.add_argument("--no-dashboard", action="store_true", help="Disable the dashboard")
    proxy_parser.add_argument(
        "--no-safety", action="store_true", help="Disable safety detectors (cost tracking only)"
    )

    # --- wrap subcommand ---
    wrap_parser = sub.add_parser(
        "wrap",
        help="Wrap a command with AEP — intercept all LLM calls",
        epilog="Example: aceteam-aep wrap -- python my_agent.py",
    )
    wrap_parser.add_argument(
        "--port", type=int, default=None, help="Proxy port (default: auto)"
    )
    wrap_parser.add_argument(
        "--target",
        type=str,
        default="https://api.openai.com",
        help="Target API base URL (default: https://api.openai.com)",
    )
    wrap_parser.add_argument("--no-dashboard", action="store_true", help="Disable the dashboard")
    wrap_parser.add_argument(
        "--no-safety", action="store_true", help="Disable safety detectors (cost tracking only)"
    )
    wrap_parser.add_argument(
        "cmd", nargs=argparse.REMAINDER, help="Command to run (after --)"
    )

    args = parser.parse_args()

    # Strip leading "--" from cmd if present
    if hasattr(args, "cmd") and args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]

    if args.command == "proxy":
        _run_proxy(args)
    elif args.command == "wrap":
        _run_wrap(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
