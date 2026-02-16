"""Simple recursive character text splitter.

Replaces LangChain's RecursiveCharacterTextSplitter with ~50 lines.
"""

from __future__ import annotations


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: list[str] | None = None,
) -> list[str]:
    """Split text into chunks using recursive separators.

    Tries each separator in order, splitting on the first one that produces
    chunks small enough. Falls back to character-level splitting.

    Args:
        text: The text to split.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Number of overlapping characters between chunks.
        separators: Ordered list of separators to try.

    Returns:
        List of text chunks.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Find the best separator
    separator = separators[-1]  # fallback to empty string
    for sep in separators:
        if sep in text:
            separator = sep
            break

    # Split by separator (or character-level if empty)
    splits = text.split(separator) if separator else list(text)

    # Merge small splits into chunks
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for split in splits:
        split_len = len(split) + (len(separator) if current_chunk else 0)

        if current_length + split_len > chunk_size and current_chunk:
            chunk_text = separator.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text)

            # Keep overlap
            overlap_target = chunk_overlap
            overlap_parts: list[str] = []
            overlap_len = 0
            for part in reversed(current_chunk):
                if overlap_len + len(part) > overlap_target:
                    break
                overlap_parts.insert(0, part)
                overlap_len += len(part) + len(separator)

            current_chunk = overlap_parts
            current_length = sum(len(p) for p in current_chunk) + len(separator) * max(
                0, len(current_chunk) - 1
            )

        current_chunk.append(split)
        current_length += split_len

    # Add remaining
    if current_chunk:
        chunk_text = separator.join(current_chunk)
        if chunk_text.strip():
            chunks.append(chunk_text)

    return chunks


__all__ = ["split_text"]
