"""Tests for Google provider - schema helpers only (no real API calls)."""

from aceteam_aep.providers.google import _strip_additional_properties


def test_strip_top_level():
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    result = _strip_additional_properties(schema)
    assert "additionalProperties" not in result
    assert result["type"] == "object"


def test_strip_nested_objects():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "address": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"city": {"type": "string"}},
            },
        },
    }
    result = _strip_additional_properties(schema)
    assert "additionalProperties" not in result
    assert "additionalProperties" not in result["properties"]["address"]
    assert result["properties"]["address"]["properties"]["city"]["type"] == "string"


def test_strip_array_items():
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"id": {"type": "integer"}},
        },
    }
    result = _strip_additional_properties(schema)
    assert "additionalProperties" not in result["items"]
    assert result["items"]["properties"]["id"]["type"] == "integer"


def test_no_additionalProperties_unchanged():
    schema = {"type": "object", "properties": {"x": {"type": "number"}}}
    result = _strip_additional_properties(schema)
    assert result == schema


def test_preserves_list_values():
    schema = {"required": ["a", "b"], "type": "object", "additionalProperties": False}
    result = _strip_additional_properties(schema)
    assert result["required"] == ["a", "b"]
