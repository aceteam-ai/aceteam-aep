"""Global SDK instrumentation — patch openai/anthropic at the module level.

Unlike wrap() which patches a single client instance, instrument() patches
the SDK classes themselves so that ALL clients created afterwards are
automatically wrapped with AEP.

This is the integration path for frameworks like OpenClaw that create
their own clients internally.

Usage::

    from aceteam_aep import instrument

    # Call once at startup — before any client is created
    instrument()

    # Now ALL OpenAI/Anthropic clients are automatically wrapped
    import openai
    client = openai.OpenAI()  # already has client.aep
    # OpenClaw's internal clients are also wrapped
"""

from __future__ import annotations

import logging
from typing import Any

from .enforcement import EnforcementPolicy
from .wrap import AepSession, _default_detectors, _wrap_anthropic, _wrap_openai

log = logging.getLogger(__name__)

_instrumented = False


def instrument(
    *,
    entity: str = "default",
    detectors: list[Any] | None = None,
    policy: EnforcementPolicy | None = None,
) -> None:
    """Patch openai and anthropic SDK classes globally.

    Call once at startup. All clients created after this call will be
    automatically wrapped with AEP instrumentation.

    Args:
        entity: Default entity for cost attribution.
        detectors: Custom detectors. If None, uses defaults.
        policy: Custom enforcement policy. If None, uses defaults.
    """
    global _instrumented
    if _instrumented:
        log.warning("AEP already instrumented, skipping")
        return

    session = AepSession(
        entity=entity,
        _policy=policy or EnforcementPolicy(),
    )
    for det in detectors or _default_detectors():
        session._registry.add(det)

    _patch_openai(session)
    _patch_anthropic(session)

    _instrumented = True
    log.info("AEP instrumented: all new OpenAI/Anthropic clients will be wrapped")


def uninstrument() -> None:
    """Remove AEP instrumentation. Mainly for testing."""
    global _instrumented
    _restore_openai()
    _restore_anthropic()
    _instrumented = False


# ---------------------------------------------------------------------------
# OpenAI patching
# ---------------------------------------------------------------------------

_original_openai_init: Any = None
_original_async_openai_init: Any = None


def _patch_openai(session: AepSession) -> None:
    global _original_openai_init, _original_async_openai_init
    try:
        import openai

        _original_openai_init = openai.OpenAI.__init__

        def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
            _original_openai_init(self, *args, **kwargs)
            _wrap_openai(self, session)

        openai.OpenAI.__init__ = patched_init

        # Async variant
        _original_async_openai_init = openai.AsyncOpenAI.__init__

        def patched_async_init(self: Any, *args: Any, **kwargs: Any) -> None:
            _original_async_openai_init(self, *args, **kwargs)
            from .wrap import _wrap_openai_async

            _wrap_openai_async(self, session)

        openai.AsyncOpenAI.__init__ = patched_async_init

        log.debug("Patched openai.OpenAI and openai.AsyncOpenAI")
    except ImportError:
        log.debug("openai not installed, skipping")


def _restore_openai() -> None:
    global _original_openai_init, _original_async_openai_init
    try:
        import openai

        if _original_openai_init:
            openai.OpenAI.__init__ = _original_openai_init
            _original_openai_init = None
        if _original_async_openai_init:
            openai.AsyncOpenAI.__init__ = _original_async_openai_init
            _original_async_openai_init = None
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Anthropic patching
# ---------------------------------------------------------------------------

_original_anthropic_init: Any = None
_original_async_anthropic_init: Any = None


def _patch_anthropic(session: AepSession) -> None:
    global _original_anthropic_init, _original_async_anthropic_init
    try:
        import anthropic

        _original_anthropic_init = anthropic.Anthropic.__init__

        def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
            _original_anthropic_init(self, *args, **kwargs)
            _wrap_anthropic(self, session)

        anthropic.Anthropic.__init__ = patched_init

        _original_async_anthropic_init = anthropic.AsyncAnthropic.__init__

        def patched_async_init(self: Any, *args: Any, **kwargs: Any) -> None:
            _original_async_anthropic_init(self, *args, **kwargs)
            from .wrap import _wrap_anthropic_async

            _wrap_anthropic_async(self, session)

        anthropic.AsyncAnthropic.__init__ = patched_async_init

        log.debug("Patched anthropic.Anthropic and anthropic.AsyncAnthropic")
    except ImportError:
        log.debug("anthropic not installed, skipping")


def _restore_anthropic() -> None:
    global _original_anthropic_init, _original_async_anthropic_init
    try:
        import anthropic

        if _original_anthropic_init:
            anthropic.Anthropic.__init__ = _original_anthropic_init
            _original_anthropic_init = None
        if _original_async_anthropic_init:
            anthropic.AsyncAnthropic.__init__ = _original_async_anthropic_init
            _original_async_anthropic_init = None
    except ImportError:
        pass


__all__ = ["instrument", "uninstrument"]
