"""Logging helpers for the proxy CLI."""

from __future__ import annotations

import logging
import sys

_MARK = "_aep_proxy_debug_stderr"


def configure_proxy_debug_logging() -> None:
    """Wire stderr so DEBUG from ``aceteam_aep.proxy`` is visible.

    Uvicorn only configures ``uvicorn.*`` loggers; library loggers often have no
    handlers, and the stdlib ``lastResort`` handler only prints WARNING+.

    Attaches one handler on the ``aceteam_aep.proxy`` package logger (does not
    toggle ``propagate`` on children).
    """
    proxy_pkg = logging.getLogger("aceteam_aep.proxy")
    if any(getattr(h, _MARK, False) for h in proxy_pkg.handlers):
        return
    proxy_pkg.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    setattr(handler, _MARK, True)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    proxy_pkg.addHandler(handler)
