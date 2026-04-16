"""Extract source references from OpenAI-format message arrays.

Parses the conversation history to identify what sources contributed
to the LLM's context: tool call results, system context documents,
and declared sources from AEP headers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SourceRef:
    """A reference to a source that contributed to an LLM call."""

    source_type: str  # "tool_call", "system_context", "header_declared", "url"
    source_id: str  # Tool name, document ID, URL
    content_preview: str  # First 200 chars of content
    confidence: float = 0.0  # From Trust Engine (future)
    metadata: dict[str, Any] = field(default_factory=dict)


_URL_PATTERN = re.compile(r"https?://[^\s\)\]\"']+")


def extract_sources_from_messages(
    messages: list[dict[str, Any]],
    *,
    declared_sources: list[str] | None = None,
) -> list[SourceRef]:
    """Extract source references from an OpenAI-format messages array.

    Sources are extracted from:
    1. tool role messages → tool call results
    2. system messages with document context → RAG/retrieval sources
    3. URLs in any message content
    4. Declared sources from X-AEP-Sources header
    """
    sources: list[SourceRef] = []
    seen_ids: set[str] = set()

    for msg in messages:
        role = msg.get("role", "")
        content = _get_content_text(msg)

        # Tool call results
        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            source_id = f"tool:{tool_call_id}" if tool_call_id else "tool:unknown"
            if source_id not in seen_ids:
                seen_ids.add(source_id)
                sources.append(
                    SourceRef(
                        source_type="tool_call",
                        source_id=source_id,
                        content_preview=content[:200],
                        metadata={"tool_call_id": tool_call_id},
                    )
                )

        # System messages often contain RAG context
        elif role == "system" and len(content) > 500:
            source_id = f"system:{hash(content[:200]) & 0xFFFFFFFF:08x}"
            if source_id not in seen_ids:
                seen_ids.add(source_id)
                sources.append(
                    SourceRef(
                        source_type="system_context",
                        source_id=source_id,
                        content_preview=content[:200],
                    )
                )

        # URLs in any message
        urls = _URL_PATTERN.findall(content)
        for url in urls[:5]:  # cap at 5 per message
            source_id = f"url:{url}"
            if source_id not in seen_ids:
                seen_ids.add(source_id)
                sources.append(
                    SourceRef(
                        source_type="url",
                        source_id=source_id,
                        content_preview=url,
                    )
                )

    # Declared sources from X-AEP-Sources header
    for declared in declared_sources or []:
        source_id = declared.strip()
        if source_id and source_id not in seen_ids:
            seen_ids.add(source_id)
            # Parse type:id format
            if ":" in source_id:
                stype, _sid = source_id.split(":", 1)
            else:
                stype = "declared"
            sources.append(
                SourceRef(
                    source_type=f"header_{stype}",
                    source_id=source_id,
                    content_preview="",
                )
            )

    return sources


def _get_content_text(msg: dict[str, Any]) -> str:
    """Extract text from a message's content field."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(parts)
    return ""


__all__ = ["SourceRef", "extract_sources_from_messages"]
