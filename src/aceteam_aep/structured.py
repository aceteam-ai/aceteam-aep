"""Structured output via provider-native JSON schema."""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel

from .client import ChatClient
from .types import ChatMessage

T = TypeVar("T", bound=BaseModel)


async def structured_output(
    client: ChatClient,
    messages: list[ChatMessage],
    schema: type[T],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> T:
    """Get structured output from a chat client.

    Uses provider-native JSON schema support (response_format) when available,
    falling back to prompt-based extraction.

    Args:
        client: The chat client to use.
        messages: The conversation messages.
        schema: Pydantic model class defining the output structure.
        temperature: Optional temperature override.
        max_tokens: Optional max tokens override.

    Returns:
        Parsed instance of the schema.
    """
    json_schema = schema.model_json_schema()

    # Try response_format first (OpenAI, Anthropic support this)
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": json_schema,
            "strict": True,
        },
    }

    try:
        response = await client.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        content = response.message.text
        data = json.loads(content)
        return schema.model_validate(data)
    except Exception:
        # Fallback: add schema to system prompt and parse response
        schema_prompt = (
            f"You must respond with valid JSON matching this schema:\n"
            f"```json\n{json.dumps(json_schema, indent=2)}\n```\n"
            f"Respond ONLY with the JSON object, no other text."
        )

        augmented = [
            ChatMessage(role="system", content=schema_prompt),
            *messages,
        ]

        response = await client.chat(
            augmented,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.message.text.strip()
        # Strip markdown fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[-1].startswith("```"):
                content = "\n".join(lines[1:-1])
            else:
                content = "\n".join(lines[1:])
        data = json.loads(content)
        return schema.model_validate(data)


__all__ = ["structured_output"]
