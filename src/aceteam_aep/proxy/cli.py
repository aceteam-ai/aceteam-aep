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

    mcp_msg = ""
    try:
        import fastmcp as _  # noqa: F811,F401

        mcp_msg = f"  MCP:        http://localhost:{port}/mcp/\n"
    except ImportError:
        pass

    print(
        f"\n"
        f"  SafeClaw Gateway\n"
        f"  {'─' * 35}\n"
        f"  LLM Proxy:  http://localhost:{port}/v1\n"
        f"  Target:     {target}\n"
        f"{dashboard_msg}"
        f"{mcp_msg}"
        f"\n"
        f"  Point your agent here:\n"
        f"    export OPENAI_BASE_URL=http://localhost:{port}/v1\n"
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


def _run_connect(args: argparse.Namespace) -> None:
    """Connect the local proxy to an AceTeam account."""
    import json
    import webbrowser
    from pathlib import Path

    print("\n  Connect to AceTeam")
    print(f"  {'─' * 35}\n")

    # Check if already connected
    cred_dir = Path.home() / ".config" / "aceteam-aep"
    cred_file = cred_dir / "credentials.json"

    if cred_file.exists() and not args.force:
        try:
            creds = json.loads(cred_file.read_text())
            if creds.get("api_key"):
                key_hint = creds["api_key"][:8] + "..." if len(creds.get("api_key", "")) > 8 else "***"
                print(f"  Already connected (key: {key_hint})")
                print(f"  Use --force to reconnect.\n")
                return
        except Exception:
            pass

    if args.api_key:
        # Direct API key provided
        api_key = args.api_key
        print(f"  Using provided API key: {api_key[:8]}...")
    else:
        # Interactive: open browser for auth
        aceteam_url = args.url or "https://aceteam.ai"
        auth_url = f"{aceteam_url}/settings/api-keys"
        print(f"  Opening AceTeam to generate an API key...")
        print(f"  URL: {auth_url}\n")

        try:
            webbrowser.open(auth_url)
        except Exception:
            pass

        # Prompt for the key
        try:
            api_key = input("  Paste your API key (act_...): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Cancelled.\n")
            return

        if not api_key:
            print("  No key provided. Cancelled.\n")
            return

    # Validate the key format
    if not api_key.startswith("act_") and not args.force:
        print(f"  Warning: key doesn't start with 'act_'. Use --force to save anyway.")
        return

    # Save credentials
    cred_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(cred_dir, 0o700)
    cred_data = {
        "api_key": api_key,
        "url": args.url or "https://aceteam.ai",
    }
    cred_file.write_text(json.dumps(cred_data, indent=2))
    os.chmod(cred_file, 0o600)
    print(f"  ✓ Credentials saved to {cred_file}")

    # Update proxy if running
    proxy_port = args.port or 8899
    try:
        import httpx

        r = httpx.get(f"http://localhost:{proxy_port}/aep/api/state", timeout=2)
        if r.status_code == 200:
            print(f"  ✓ Proxy detected on port {proxy_port}")
            print(f"  ℹ Restart the proxy to activate AceTeam features")
    except Exception:
        pass

    print(f"\n  Connected to AceTeam!")
    print(f"  • Workflows and 40+ node types now available")
    print(f"  • $5 free credit for LLM calls")
    print(f"  • Restart proxy to activate: aceteam-aep proxy --port {proxy_port}\n")


def _run_disconnect(args: argparse.Namespace) -> None:
    """Remove AceTeam credentials."""
    from pathlib import Path

    cred_file = Path.home() / ".config" / "aceteam-aep" / "credentials.json"
    if cred_file.exists():
        cred_file.unlink()
        print("\n  ✓ Disconnected from AceTeam. Local safety still active.")
        print("  ℹ Restart the proxy to deactivate AceTeam features.\n")
    else:
        print("\n  Not connected to AceTeam.\n")


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


def _run_setup(args: argparse.Namespace) -> None:
    """Interactive setup — detect runtime, configure proxy, write Claude Code config."""
    import json
    import shutil
    import webbrowser
    from pathlib import Path

    port = args.port or 8899

    if args.print_config:
        print(
            json.dumps(
                {
                    "shell": f"export OPENAI_BASE_URL=http://localhost:{port}/v1",
                    "claude_code": {
                        "mcpServers": {
                            "aceteam": {
                                "type": "streamable-http",
                                "url": f"http://localhost:{port}/mcp/",
                            }
                        }
                    },
                    "container": f"podman run -p {port}:{port} ghcr.io/aceteam-ai/aep-proxy",
                },
                indent=2,
            )
        )
        return

    print("\n  SafeClaw Setup")
    print(f"  {'─' * 35}\n")

    # Step 1: Detect container runtime
    container_cmd = None
    for cmd in ("podman", "docker"):
        if shutil.which(cmd):
            container_cmd = cmd
            print(f"  ✓ Found {cmd}")
            break

    # Step 2: Start the proxy
    if container_cmd and not args.no_container:
        image = "ghcr.io/aceteam-ai/aep-proxy:latest"
        print(f"  Starting proxy via {container_cmd}...")

        run_cmd = [
            container_cmd,
            "run",
            "-d",
            "--name",
            "safeclaw-proxy",
            "-p",
            f"{port}:{port}",
        ]

        # Forward API keys from environment if present
        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            val = os.environ.get(key)
            if val:
                run_cmd.extend(["-e", f"{key}={val}"])
                print(f"  ✓ Forwarding {key} from environment")

        run_cmd.append(image)

        # Remove existing container if present
        subprocess.run([container_cmd, "rm", "-f", "safeclaw-proxy"], capture_output=True)

        result = subprocess.run(run_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✓ Proxy running in container on port {port}")
        else:
            print(f"  ✗ Container failed: {result.stderr.strip()}")
            print("  Falling back to native proxy...")
            container_cmd = None

    if not container_cmd or args.no_container:
        print(f"  Starting proxy natively on port {port}...")
        print(f"  Run: aceteam-aep proxy --port {port}")
        print("  (Start this in a separate terminal)\n")

    # Step 3: Detect and configure Claude Code
    claude_configured = False
    claude_config_path = Path.home() / ".claude.json"
    claude_settings_path = Path.home() / ".claude" / "settings.json"

    mcp_entry = {
        "type": "streamable-http",
        "url": f"http://localhost:{port}/mcp/",
    }

    for config_path in (claude_settings_path, claude_config_path):
        if config_path.exists():
            try:
                import json as _json

                config = _json.loads(config_path.read_text())
                mcp_servers = config.setdefault("mcpServers", {})
                if "aceteam" not in mcp_servers:
                    mcp_servers["aceteam"] = mcp_entry
                    config_path.write_text(_json.dumps(config, indent=2))
                    print(f"  ✓ Claude Code configured: {config_path}")
                    claude_configured = True
                else:
                    print(f"  ✓ Claude Code already configured: {config_path}")
                    claude_configured = True
                break
            except Exception as exc:
                print(f"  ✗ Could not update {config_path}: {exc}")

    if not claude_configured:
        print("  ℹ Claude Code not detected. Add manually:")
        print(
            f'    {{"mcpServers": {{"aceteam": {{"type": "streamable-http",'
            f' "url": "http://localhost:{port}/mcp/"}}}}}}'
        )

    # Step 4: Configure shell profile
    shell_configured = False
    if not args.no_shell:
        export_line = f"export OPENAI_BASE_URL=http://localhost:{port}/v1"
        shell = os.environ.get("SHELL", "")
        profile_candidates: list[Path] = []
        if "zsh" in shell:
            profile_candidates = [Path.home() / ".zshrc"]
        elif "bash" in shell:
            profile_candidates = [Path.home() / ".bashrc", Path.home() / ".bash_profile"]
        elif "fish" in shell:
            profile_candidates = [Path.home() / ".config" / "fish" / "config.fish"]

        for profile in profile_candidates:
            if profile.exists():
                content = profile.read_text()
                if "OPENAI_BASE_URL" not in content:
                    with open(profile, "a") as f:
                        f.write(f"\n# SafeClaw proxy\n{export_line}\n")
                    print(f"  ✓ Added OPENAI_BASE_URL to {profile}")
                    shell_configured = True
                else:
                    print(f"  ✓ OPENAI_BASE_URL already in {profile}")
                    shell_configured = True
                break

    if not shell_configured and not args.no_shell:
        print("  ℹ Add to your shell profile:")
        print(f"    export OPENAI_BASE_URL=http://localhost:{port}/v1")

    # Step 5: Open dashboard
    dashboard_url = f"http://localhost:{port}/aep/"
    print(f"\n  {'─' * 35}")
    print(f"  Dashboard:  {dashboard_url}")
    print(f"  LLM Proxy:  http://localhost:{port}/v1")
    print(f"  MCP:        http://localhost:{port}/mcp/")
    print()

    if not args.no_browser:
        try:
            webbrowser.open(dashboard_url)
            print("  ✓ Opening dashboard in browser...")
        except Exception:
            pass

    print("\n  SafeClaw is ready. Make your first call.\n")


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

    # --- setup subcommand ---
    setup_parser = sub.add_parser(
        "setup",
        help="Interactive setup — detect runtime, configure proxy, write Claude Code config",
    )
    setup_parser.add_argument(
        "--port", type=int, default=8899, help="Proxy port (default: 8899)"
    )
    setup_parser.add_argument(
        "--no-container",
        action="store_true",
        help="Skip container detection, use native proxy",
    )
    setup_parser.add_argument(
        "--no-shell", action="store_true", help="Don't modify shell profile"
    )
    setup_parser.add_argument(
        "--no-browser", action="store_true", help="Don't open browser"
    )
    setup_parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print config as JSON without writing anything",
    )

    # --- connect subcommand ---
    connect_parser = sub.add_parser(
        "connect",
        help="Connect the local proxy to an AceTeam account",
    )
    connect_parser.add_argument("--api-key", help="AceTeam API key (act_...)")
    connect_parser.add_argument(
        "--url", default="https://aceteam.ai", help="AceTeam URL"
    )
    connect_parser.add_argument(
        "--port", type=int, default=8899, help="Proxy port"
    )
    connect_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing credentials"
    )

    # --- disconnect subcommand ---
    sub.add_parser("disconnect", help="Remove AceTeam credentials")

    # --- top subcommand ---
    top_parser = sub.add_parser("top", help="Terminal dashboard (like htop for AEP)")
    top_parser.add_argument(
        "--url", type=str, default=None, help="Proxy URL (default: http://localhost:8899)"
    )
    top_parser.add_argument(
        "--port", type=int, default=None, help="Proxy port (shorthand for --url)"
    )
    top_parser.add_argument(
        "--once", action="store_true", help="Print snapshot and exit"
    )
    top_parser.add_argument(
        "--refresh", type=float, default=2.0, help="Refresh interval in seconds (default: 2)"
    )

    # --- judge-service subcommand ---
    judge_parser = sub.add_parser(
        "judge-service",
        help="Start the local 5-category safety judge service",
    )
    judge_parser.add_argument(
        "--port", type=int, default=5000, help="Judge service port (default: 5000)"
    )
    judge_parser.add_argument(
        "--model", default="gpt-4o-mini", help="LLM model for judging (default: gpt-4o-mini)"
    )
    judge_parser.add_argument(
        "--base-url", default=None, help="OpenAI-compatible API base URL (for local models)"
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
    elif args.command == "setup":
        _run_setup(args)
    elif args.command == "connect":
        _run_connect(args)
    elif args.command == "disconnect":
        _run_disconnect(args)
    elif args.command == "top":
        from ..top import run_top

        url = args.url
        if url is None and args.port:
            url = f"http://localhost:{args.port}"
        run_top(url=url or "http://localhost:8899", once=args.once, refresh=args.refresh)
    elif args.command == "judge-service":
        from ..judge_service import run_judge_service

        run_judge_service(
            port=args.port,
            model=args.model,
            base_url=args.base_url,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
