"""AEP local dashboard — serves a web UI for session monitoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..wrap import AepSession


def serve(session: AepSession, port: int = 8899) -> None:
    """Start the dashboard web server (blocking)."""
    import uvicorn

    from .app import create_app

    app = create_app(session=session)
    print(f"\n  AEP Dashboard: http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


__all__ = ["serve"]
