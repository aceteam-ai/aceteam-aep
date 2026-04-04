"""aep top — terminal dashboard for the AEP safety proxy.

Like htop, but for AI agent safety. Polls /aep/api/state and renders
a live terminal UI showing calls, cost, signals, and enforcement.

Usage:
    aceteam-aep top                    # polls localhost:8899
    aceteam-aep top --port 9000        # custom port
    aceteam-aep top --url http://...   # remote proxy
    aceteam-aep top --once             # print snapshot and exit
"""

from __future__ import annotations

import sys
import time
from datetime import UTC

import httpx


def _fetch_state(base_url: str) -> dict | None:
    try:
        resp = httpx.get(f"{base_url}/aep/api/state", timeout=3.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _severity_color(sev: str) -> str:
    return {"high": "red bold", "medium": "yellow", "low": "cyan"}.get(sev, "white")


def _action_color(action: str) -> str:
    return {"pass": "green", "flag": "yellow bold", "block": "red bold"}.get(action, "white")


def _action_badge(action: str) -> str:
    colors = {"pass": "on green", "flag": "on yellow", "block": "on red"}
    style = colors.get(action, "")
    return f"[{style}] {action.upper()} [/{style}]"


def _render(console: object, data: dict) -> object:
    """Build a rich renderable from proxy state."""
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    # Header metrics
    safety = data.get("safety_enabled", True)
    safety_text = Text("ON ", style="bold green") if safety else Text("OFF", style="bold red")
    calls = data.get("calls", 0)
    cost = data.get("cost", 0)
    blocked = data.get("savings", {}).get("blocked_calls", 0)
    action = data.get("action", "pass")

    header = Table.grid(padding=(0, 3))
    header.add_row(
        Text("Safety: ", style="dim") + safety_text,
        Text(f"Calls: {calls}", style="bold"),
        Text(f"Cost: ${cost:.6f}", style="bold cyan"),
        Text(f"Blocked: {blocked}", style="bold red" if blocked > 0 else "dim"),
        Text("Status: ", style="dim") + Text(action.upper(), style=_action_color(action)),
    )

    # Budget bar (if set)
    budget_section = None
    budget = data.get("budget")
    if budget and budget.get("total"):
        spent = budget.get("spent", 0)
        total = budget["total"]
        pct = min(spent / total * 100, 100) if total > 0 else 0
        bar_len = 20
        filled = int(pct / 100 * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        color = "green" if pct < 50 else "yellow" if pct < 80 else "red"
        remaining = budget.get("remaining", 0)
        budget_section = Text(
            f"  Budget: [{color}]{bar}[/{color}] ${remaining:.4f} remaining",
            style="dim",
        )

    # Spans table (latest calls)
    spans_table = Table(title="Latest Calls", expand=True, show_edge=False, padding=(0, 1))
    spans_table.add_column("Model", style="cyan", no_wrap=True, max_width=16)
    spans_table.add_column("Result", justify="center", max_width=6)
    spans_table.add_column("Cost", justify="right", style="blue", max_width=10)
    spans_table.add_column("Latency", justify="right", style="dim", max_width=8)

    spans = data.get("spans", [])
    signals = data.get("signals", [])
    # Build call_id → action lookup
    call_actions = {}
    for s in signals:
        cid = s.get("call_id", "")
        sev = s.get("severity", "low")
        if sev == "high":
            call_actions[cid] = "block"
        elif sev == "medium" and call_actions.get(cid) != "block":
            call_actions[cid] = "flag"

    for span in reversed(spans[-15:]):
        cid = span.get("call_id", "")
        act = call_actions.get(cid, "pass")
        dur = span.get("duration_ms")
        dur_str = f"{int(dur)}ms" if dur else "..."
        span_cost = span.get("cost", 0)
        spans_table.add_row(
            span.get("executor", "?"),
            Text(act.upper(), style=_action_color(act)),
            f"${span_cost:.6f}",
            dur_str,
        )

    if not spans:
        spans_table.add_row("—", "—", "—", "—")

    # Signals summary
    sig_counts: dict[str, int] = {}
    for s in signals:
        t = s.get("type", "unknown")
        sig_counts[t] = sig_counts.get(t, 0) + 1

    sig_table = Table(title="Signals", show_edge=False, padding=(0, 1))
    sig_table.add_column("Type", style="bold")
    sig_table.add_column("Count", justify="right")

    for sig_type, count in sorted(sig_counts.items(), key=lambda x: -x[1]):
        sig_table.add_row(sig_type, str(count))
    if not sig_counts:
        sig_table.add_row(Text("No signals", style="dim"), "")

    # Compliance bar
    if calls > 0:
        blocked_unique = len({s.get("call_id") for s in signals if s.get("severity") == "high"})
        compliance = (calls - blocked_unique) / calls * 100
        comp_color = "green" if compliance > 99 else "yellow" if compliance > 95 else "red"
        comp_bar = f"[{comp_color}]{compliance:.1f}%[/{comp_color}]"
    else:
        comp_bar = "[dim]—[/dim]"

    stats_table = Table(title="Stats", show_edge=False, padding=(0, 1))
    stats_table.add_column("Metric", style="dim")
    stats_table.add_column("Value", justify="right")
    stats_table.add_row("Compliance", comp_bar)
    savings = data.get("savings", {})
    if savings.get("estimated_savings_usd", 0) > 0:
        saved = savings["estimated_savings_usd"]
        stats_table.add_row("Est. savings", f"[green]${saved:.4f}[/green]")
    stats_table.add_row("Avg call cost", f"${savings.get('avg_call_cost_usd', 0):.6f}")

    # Session info
    started = data.get("session_started", "")
    if started:
        from datetime import datetime

        try:
            st = datetime.fromisoformat(started)
            elapsed = (datetime.now(UTC) - st).total_seconds()
            mins = int(elapsed // 60)
            session_str = f"{mins}m" if mins > 0 else f"{int(elapsed)}s"
        except Exception:
            session_str = "?"
    else:
        session_str = "?"

    # Assemble layout
    from rich.layout import Layout

    layout = Layout()
    layout.split_column(
        Layout(
            Panel(
                header,
                title=f"[bold cyan]AEP Top[/bold cyan] [dim]— session {session_str}[/dim]",
                border_style="cyan",
            ),
            size=3,
        ),
        Layout(name="body"),
    )

    if budget_section:
        layout["body"].split_column(
            Layout(budget_section, size=1),
            Layout(name="main"),
        )
    else:
        layout["body"].split_column(Layout(name="main"))

    layout["body"]["main" if not budget_section else "main"].split_row(
        Layout(Panel(spans_table, border_style="blue"), ratio=3),
        Layout(name="sidebar", ratio=1),
    )
    layout["body"]["main" if not budget_section else "main"]["sidebar"].split_column(
        Layout(Panel(sig_table, border_style="yellow")),
        Layout(Panel(stats_table, border_style="green")),
    )

    return layout


def run_top(
    url: str = "http://localhost:8899",
    once: bool = False,
    refresh: float = 2.0,
) -> None:
    """Run the terminal dashboard."""
    try:
        from rich.console import Console
        from rich.live import Live
    except ImportError:
        print("aep top requires 'rich'. Install with: pip install aceteam-aep[top]")
        sys.exit(1)

    console = Console()

    if once:
        data = _fetch_state(url)
        if data is None:
            console.print(f"[red]Cannot connect to {url}[/red]")
            sys.exit(1)
        console.print(_render(console, data))
        return

    console.print(f"[dim]Connecting to {url}...[/dim]")
    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            data = _fetch_state(url)
            if data is None:
                from rich.text import Text

                live.update(Text(f"Cannot connect to {url} — retrying...", style="red"))
            else:
                live.update(_render(console, data))
            time.sleep(refresh)
