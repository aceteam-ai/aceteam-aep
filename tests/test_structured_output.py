# tests/test_structured_output.py
"""Tests that all providers properly handle response_format for structured output.

Each provider's chat() method should either:
- Pass response_format to the underlying API (OpenAI, xAI, Ollama)
- Convert response_format to provider-native format (Google)
- Inject schema into system prompt (Anthropic)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aceteam_aep.types import ChatMessage, ChatResponse, Usage

# Standard test schema
TEST_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "structured_output",
        "schema": {
            "type": "object",
            "properties": {
                "first_name": {"type": "string"},
                "school": {"type": "string"},
                "major": {"type": "string"},
            },
            "required": ["first_name", "school", "major"],
        },
        "strict": True,
    },
}

TEST_MESSAGES = [ChatMessage(role="user", content="Extract the data")]


class TestOpenAIResponseFormat:
    """OpenAI provider passes response_format directly to the API."""

    @pytest.mark.asyncio
    async def test_response_format_passed_to_api(self):
        from aceteam_aep.providers.openai import OpenAIClient

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"first_name": "John", "school": "MIT", "major": "CS"}',
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_response.model = "gpt-4o"

        with patch("openai.AsyncOpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            client = OpenAIClient(api_key="test-key", model="gpt-4o")
            await client.chat(
                TEST_MESSAGES,
                response_format=TEST_RESPONSE_FORMAT,
            )

            call_kwargs = mock_client.chat.completions.create.call_args
            assert "response_format" in call_kwargs.kwargs
            assert call_kwargs.kwargs["response_format"] == TEST_RESPONSE_FORMAT

    @pytest.mark.asyncio
    async def test_response_format_omitted_when_none(self):
        from aceteam_aep.providers.openai import OpenAIClient

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Hello!", tool_calls=None),
                finish_reason="stop",
            )
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_response.model = "gpt-4o"

        with patch("openai.AsyncOpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            client = OpenAIClient(api_key="test-key", model="gpt-4o")
            await client.chat(TEST_MESSAGES)

            call_kwargs = mock_client.chat.completions.create.call_args
            assert "response_format" not in call_kwargs.kwargs


class TestAnthropicResponseFormat:
    """Anthropic provider injects schema into system prompt."""

    @pytest.mark.asyncio
    async def test_schema_injected_into_system_prompt(self):
        from aceteam_aep.providers.anthropic import AnthropicClient

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                type="text",
                text='{"first_name": "John", "school": "MIT", "major": "CS"}',
            )
        ]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "end_turn"

        with patch("anthropic.AsyncAnthropic") as MockAnthropic:
            mock_client = MockAnthropic.return_value
            mock_client.messages.create = AsyncMock(return_value=mock_response)

            client = AnthropicClient(api_key="test-key", model="claude-sonnet-4-20250514")
            await client.chat(
                TEST_MESSAGES,
                response_format=TEST_RESPONSE_FORMAT,
            )

            call_kwargs = mock_client.messages.create.call_args
            # Schema should be in the system prompt
            assert "system" in call_kwargs.kwargs
            system = call_kwargs.kwargs["system"]
            assert "first_name" in system
            assert "school" in system
            assert "major" in system
            assert "JSON object" in system

    @pytest.mark.asyncio
    async def test_schema_appended_to_existing_system_prompt(self):
        """When messages already contain a system prompt, schema is appended."""
        from aceteam_aep.providers.anthropic import AnthropicClient

        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="{}")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "end_turn"

        messages = [
            ChatMessage(role="system", content="You are a data extractor."),
            ChatMessage(role="user", content="Extract the data"),
        ]

        with patch("anthropic.AsyncAnthropic") as MockAnthropic:
            mock_client = MockAnthropic.return_value
            mock_client.messages.create = AsyncMock(return_value=mock_response)

            client = AnthropicClient(api_key="test-key", model="claude-sonnet-4-20250514")
            await client.chat(messages, response_format=TEST_RESPONSE_FORMAT)

            call_kwargs = mock_client.messages.create.call_args
            system = call_kwargs.kwargs["system"]
            # Should contain BOTH the original system prompt and the schema
            assert "data extractor" in system
            assert "first_name" in system

    @pytest.mark.asyncio
    async def test_no_system_prompt_when_no_response_format(self):
        """Without response_format, no schema prompt is injected."""
        from aceteam_aep.providers.anthropic import AnthropicClient

        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Hello!")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "end_turn"

        with patch("anthropic.AsyncAnthropic") as MockAnthropic:
            mock_client = MockAnthropic.return_value
            mock_client.messages.create = AsyncMock(return_value=mock_response)

            client = AnthropicClient(api_key="test-key", model="claude-sonnet-4-20250514")
            await client.chat(TEST_MESSAGES)

            call_kwargs = mock_client.messages.create.call_args
            # No system prompt should be set (no system message in TEST_MESSAGES)
            assert "system" not in call_kwargs.kwargs


class TestGoogleResponseFormat:
    """Google provider converts response_format to response_mime_type + response_schema."""

    @pytest.mark.asyncio
    async def test_response_format_converted_to_google_config(self):
        # Test the helper function directly
        from google.genai import types as genai_types

        from aceteam_aep.providers.google import _apply_response_format

        config = genai_types.GenerateContentConfig(temperature=0.7, max_output_tokens=1000)
        _apply_response_format(config, TEST_RESPONSE_FORMAT)

        assert config.response_mime_type == "application/json"
        assert config.response_schema is not None
        # Schema should contain our properties
        schema = config.response_schema
        assert "properties" in schema
        assert "first_name" in schema["properties"]

    @pytest.mark.asyncio
    async def test_json_object_format_sets_mime_type_only(self):
        from google.genai import types as genai_types

        from aceteam_aep.providers.google import _apply_response_format

        config = genai_types.GenerateContentConfig(temperature=0.7, max_output_tokens=1000)
        _apply_response_format(config, {"type": "json_object"})

        assert config.response_mime_type == "application/json"
        # No schema set for json_object
        assert config.response_schema is None

    @pytest.mark.asyncio
    async def test_response_format_applied_in_chat(self):
        """Verify response_format is applied when calling chat()."""
        from aceteam_aep.providers.google import GoogleClient

        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [
            MagicMock(
                text='{"first_name": "John", "school": "MIT", "major": "CS"}',
                function_call=None,
            )
        ]
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=10, candidates_token_count=5, total_token_count=15
        )

        with patch("google.genai.Client") as MockGenAI:
            mock_genai = MockGenAI.return_value
            mock_genai.aio.models.generate_content = AsyncMock(return_value=mock_response)

            client = GoogleClient(api_key="test-key", model="gemini-2.5-flash")
            await client.chat(TEST_MESSAGES, response_format=TEST_RESPONSE_FORMAT)

            # Verify generate_content was called
            call_kwargs = mock_genai.aio.models.generate_content.call_args
            config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
            assert config.response_mime_type == "application/json"
            assert config.response_schema is not None

    @pytest.mark.asyncio
    async def test_additional_properties_stripped_from_schema(self):
        """additionalProperties must be stripped before passing schema to Gemini.

        Pydantic's model_json_schema() emits additionalProperties: false on every
        object. Gemini's response_schema uses a restricted OpenAPI 3.0 subset that
        rejects this field with a 400 INVALID_ARGUMENT error.
        """
        from google.genai import types as genai_types

        from aceteam_aep.providers.google import _apply_response_format

        schema_with_additional_props = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "schema": {
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
                    "required": ["name", "address"],
                },
            },
        }

        config = genai_types.GenerateContentConfig(temperature=0.7, max_output_tokens=1000)
        _apply_response_format(config, schema_with_additional_props)

        schema = config.response_schema
        assert "additionalProperties" not in schema
        assert "additionalProperties" not in schema["properties"]["address"]
        # Other fields are preserved
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["required"] == ["name", "address"]


class TestStructuredOutputWrapper:
    """Test the structured_output() wrapper function."""

    @pytest.mark.asyncio
    async def test_structured_output_passes_response_format(self):
        """structured_output() should pass response_format to client.chat()."""
        from pydantic import BaseModel

        from aceteam_aep.structured import structured_output

        class PersonInfo(BaseModel):
            first_name: str
            school: str
            major: str

        mock_client = MagicMock()
        mock_client.chat = AsyncMock(
            return_value=ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content='{"first_name": "John", "school": "MIT", "major": "CS"}',
                ),
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="mock-model",
            )
        )

        result = await structured_output(
            client=mock_client,
            messages=TEST_MESSAGES,
            schema=PersonInfo,
        )

        assert isinstance(result, PersonInfo)
        assert result.first_name == "John"
        assert result.school == "MIT"

        # Verify response_format was passed
        call_kwargs = mock_client.chat.call_args
        assert "response_format" in call_kwargs.kwargs
        rf = call_kwargs.kwargs["response_format"]
        assert rf["type"] == "json_schema"
