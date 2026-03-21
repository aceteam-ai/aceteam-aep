"""Starlette ASGI app for the AEP dashboard."""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

if TYPE_CHECKING:
    from ..wrap import AepSession

_TEMPLATE_DIR = Path(__file__).parent / "templates"


class _DecimalEncoder(json.JSONEncoder):
    def default(self, o: object) -> Any:
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)


def _build_state(session: AepSession) -> dict[str, Any]:
    """Extract current session state as a JSON-serializable dict."""
    from ..enforcement import EnforcementPolicy, evaluate

    signals = session.safety_signals
    decision = evaluate(signals, EnforcementPolicy())

    return {
        "cost": float(session.cost_usd),
        "calls": session.call_count,
        "action": decision.action,
        "reason": decision.reason,
        "signals": [
            {
                "type": s.signal_type,
                "severity": s.severity,
                "detail": s.detail,
                "call_id": s.call_id,
                "detector": s.detector,
                "timestamp": s.timestamp,
            }
            for s in signals
        ],
        "spans": [
            {
                "id": s.span_id,
                "executor": s.executor_id,
                "status": s.status,
                "duration_ms": s.duration_ms,
                "started_at": s.started_at,
            }
            for s in session.get_spans()
        ],
    }


def create_app(
    session: AepSession | None = None,
    get_state: Callable[[], dict[str, Any]] | None = None,
) -> Starlette:
    """Create the dashboard ASGI app.

    Args:
        session: Live AepSession to monitor.
        get_state: Alternative state provider (for testing).
    """

    def _get_state() -> dict[str, Any]:
        if get_state:
            return get_state()
        assert session is not None
        return _build_state(session)

    async def homepage(request: Request) -> HTMLResponse:
        template = (_TEMPLATE_DIR / "index.html").read_text()
        return HTMLResponse(template)

    async def api_state(request: Request) -> JSONResponse:
        state = _get_state()
        return JSONResponse(
            content=json.loads(json.dumps(state, cls=_DecimalEncoder))
        )

    return Starlette(
        routes=[
            Route("/", homepage),
            Route("/api/state", api_state),
        ],
    )


__all__ = ["create_app"]
