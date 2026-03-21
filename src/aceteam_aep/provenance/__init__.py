"""Provenance tracking — citation chains and source attribution for AEP."""

from .source_extractor import SourceRef, extract_sources_from_messages
from .tracker import ProvenanceTracker

__all__ = ["ProvenanceTracker", "SourceRef", "extract_sources_from_messages"]
