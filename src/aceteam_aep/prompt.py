"""XML-wrapping prompt helpers (#1615).

These helpers produce XML-tagged prompt sections that improve LLM
instruction following compared to markdown code fences.
"""

from __future__ import annotations


def wrap_xml(tag: str, content: str, **attrs: str) -> str:
    """Wrap content in an XML tag.

    Args:
        tag: The XML tag name.
        content: The content to wrap.
        **attrs: Optional attributes for the tag.

    Returns:
        XML-wrapped string.

    Example:
        >>> wrap_xml("instructions", "Do the thing")
        '<instructions>\\nDo the thing\\n</instructions>'
    """
    attr_str = ""
    if attrs:
        attr_str = " " + " ".join(f'{k}="{v}"' for k, v in attrs.items())
    return f"<{tag}{attr_str}>\n{content}\n</{tag}>"


def wrap_file(content: str, filename: str, language: str = "") -> str:
    """Wrap file content in XML tags with metadata.

    Args:
        content: The file content.
        filename: The filename.
        language: Optional language identifier.

    Returns:
        XML-wrapped file content.
    """
    attrs: dict[str, str] = {"name": filename}
    if language:
        attrs["language"] = language
    return wrap_xml("file", content, **attrs)


def wrap_examples(examples: list[dict[str, str]]) -> str:
    """Wrap few-shot examples in XML tags.

    Args:
        examples: List of dicts with 'input' and 'output' keys.

    Returns:
        XML-wrapped examples.
    """
    parts: list[str] = []
    for i, ex in enumerate(examples, 1):
        example_content = wrap_xml("input", ex.get("input", ""))
        example_content += "\n" + wrap_xml("output", ex.get("output", ""))
        parts.append(wrap_xml("example", example_content, index=str(i)))
    return wrap_xml("examples", "\n".join(parts))


def wrap_context(context: str, source: str = "") -> str:
    """Wrap RAG context in XML tags.

    Args:
        context: The retrieved context.
        source: Optional source identifier.

    Returns:
        XML-wrapped context.
    """
    attrs: dict[str, str] = {}
    if source:
        attrs["source"] = source
    return wrap_xml("context", context, **attrs)


__all__ = ["wrap_context", "wrap_examples", "wrap_file", "wrap_xml"]
