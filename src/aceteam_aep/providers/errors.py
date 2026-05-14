"""Typed exceptions raised by provider clients."""

from __future__ import annotations


class ProviderResponseError(Exception):
    """Raised when an LLM provider returns a structurally-valid response
    that conveys no usable content.

    The most common trigger is an upstream silent rejection: the provider
    accepts the request, opens an SSE stream, then closes it without
    emitting any text, tool call, or finish-reason events. Without an
    explicit raise here, callers see a successful-looking empty stream
    and surface it to end-users as a blank assistant reply.

    Attributes:
        provider: Short slug of the provider that produced the response
            (e.g., ``"anthropic"``, ``"openai"``). Used by callers to
            map to user-facing copy without re-parsing the message.
        user_message: Short, end-user-safe sentence. Callers may
            surface this verbatim or override it.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        user_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.user_message = user_message or (
            "The model returned an empty response. This usually means the "
            "request was rejected upstream — check your API key permissions "
            "and the selected model."
        )


__all__ = ["ProviderResponseError"]
