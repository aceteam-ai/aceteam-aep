"""Tests for embeddings module."""

from aceteam_aep.embeddings import EmbeddingClient, OpenAIEmbeddings


def test_openai_embeddings_is_embedding_client():
    """OpenAIEmbeddings should satisfy the EmbeddingClient protocol."""
    client = OpenAIEmbeddings(api_key="test-key", model="text-embedding-3-small")
    assert isinstance(client, EmbeddingClient)


def test_openai_embeddings_with_dimensions():
    """Should accept dimensions parameter."""
    client = OpenAIEmbeddings(
        api_key="test-key",
        model="text-embedding-3-small",
        dimensions=256,
    )
    assert client._model == "text-embedding-3-small"
    assert client._dimensions == 256
