"""Provenance tracker — builds citation chains across a session."""

from __future__ import annotations

from ..envelope import Citation
from .source_extractor import SourceRef


class ProvenanceTracker:
    """Tracks sources and builds citation chains for AEP envelopes."""

    def __init__(self) -> None:
        self._sources: dict[str, list[SourceRef]] = {}  # call_id -> sources

    def record_sources(self, call_id: str, sources: list[SourceRef]) -> None:
        """Record sources used in a specific call."""
        self._sources.setdefault(call_id, []).extend(sources)

    def get_sources(self, call_id: str) -> list[SourceRef]:
        """Get all sources for a specific call."""
        return self._sources.get(call_id, [])

    def get_all_sources(self) -> list[SourceRef]:
        """Get all sources across the session."""
        all_sources: list[SourceRef] = []
        for sources in self._sources.values():
            all_sources.extend(sources)
        return all_sources

    def get_citations(self, call_id: str) -> list[Citation]:
        """Convert sources for a call into AEP Citation objects."""
        return [
            Citation(
                span_id=call_id,
                source_type=src.source_type,
                content=src.content_preview,
                confidence=src.confidence if src.confidence > 0 else None,
            )
            for src in self.get_sources(call_id)
        ]

    def get_all_citations(self) -> list[Citation]:
        """Get all citations across the session."""
        citations: list[Citation] = []
        for call_id in self._sources:
            citations.extend(self.get_citations(call_id))
        return citations

    @property
    def source_count(self) -> int:
        """Total number of unique sources tracked."""
        return sum(len(s) for s in self._sources.values())


__all__ = ["ProvenanceTracker"]
