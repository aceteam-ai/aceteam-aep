"""Typed exceptions raised by provider clients."""

from __future__ import annotations


class StreamFailedError(Exception):
    """Raised when a provider's streaming response completes without
    producing any usable content.

    The most common trigger is an upstream silent rejection: the
    provider accepts the request, opens an SSE stream, then closes it
    without emitting any text, tool call, or finish-reason events.
    Without an explicit raise here, callers see a successful-looking
    empty stream and surface it to end-users as a blank assistant
    reply.

    Attributes:
        provider: Short slug of the provider that produced the
            response (e.g., ``"anthropic"``, ``"openai"``). Callers
            switch on this to map to provider-specific user-facing
            copy.
    """

    def __init__(self, message: str, *, provider: str) -> None:
        super().__init__(message)
        self.provider = provider


__all__ = ["StreamFailedError"]
