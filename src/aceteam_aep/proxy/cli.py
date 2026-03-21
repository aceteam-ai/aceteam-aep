"""CLI entry point: aceteam-aep proxy --port 8080 --dashboard"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aceteam-aep",
        description="AEP Trust Engine — safety proxy for AI agents",
    )
    sub = parser.add_subparsers(dest="command")

    # proxy subcommand
    proxy_parser = sub.add_parser(
        "proxy",
        help="Start the AEP reverse proxy",
    )
    proxy_parser.add_argument(
        "--port", type=int, default=8080, help="Port to listen on (default: 8080)"
    )
    proxy_parser.add_argument(
        "--target",
        type=str,
        default="https://api.openai.com",
        help="Target API base URL (default: https://api.openai.com)",
    )
    proxy_parser.add_argument(
        "--dashboard",
        action="store_true",
        default=True,
        help="Enable dashboard at /aep/ (default: on)",
    )
    proxy_parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable the dashboard",
    )
    proxy_parser.add_argument(
        "--no-safety",
        action="store_true",
        help="Disable safety detectors (cost tracking only)",
    )

    args = parser.parse_args()

    if args.command != "proxy":
        parser.print_help()
        sys.exit(1)

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
        f"  ─────────────────────────────────\n"
        f"  Listening:  http://localhost:{args.port}\n"
        f"  Target:     {args.target}\n"
        f"{dashboard_msg}"
        f"\n"
        f"  Usage:\n"
        f"    export OPENAI_BASE_URL=http://localhost:{args.port}/v1\n"
        f"    openclaw run \"your task here\"\n"
        f"\n"
    )

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
