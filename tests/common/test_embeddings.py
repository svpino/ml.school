"""Tests for the CustomEmbeddingModel class in common/embeddings.py."""

from unittest.mock import patch

from common.embeddings import CustomEmbeddingModel


class TestEmbedDocuments:
    """Test suite for the embed_documents method."""

    def test_embed_documents_calls_litellm_and_extracts_embeddings(self):
        """Should call litellm.embedding and extract each item's embedding."""
        model_name = "text-embedding-ada-002"
        embedding_model = CustomEmbeddingModel(model=model_name)

        mock_response = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ]
        }

        with patch("common.embeddings.litellm") as mock_litellm:
            mock_litellm.embedding.return_value = mock_response

            texts = ["text1", "text2"]
            result = embedding_model.embed_documents(texts)

            mock_litellm.embedding.assert_called_once_with(model=model_name, input=texts)
            assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    def test_embed_documents_with_empty_list(self):
        """An empty input list should return an empty list of embeddings."""
        embedding_model = CustomEmbeddingModel(model="text-embedding-ada-002")

        mock_response = {"data": []}

        with patch("common.embeddings.litellm") as mock_litellm:
            mock_litellm.embedding.return_value = mock_response

            result = embedding_model.embed_documents([])

            mock_litellm.embedding.assert_called_once_with(
                model="text-embedding-ada-002", input=[]
            )
            assert result == []


class TestEmbedQuery:
    """Test suite for the embed_query method."""

    def test_embed_query_wraps_text_and_returns_first_embedding(self):
        """Should wrap the query in a list, delegate to embed_documents, and unwrap."""
        embedding_model = CustomEmbeddingModel(model="text-embedding-ada-002")

        with patch.object(
            embedding_model, "embed_documents", return_value=[[0.7, 0.8, 0.9]]
        ) as mock_embed:
            result = embedding_model.embed_query("What is machine learning?")

            mock_embed.assert_called_once_with(["What is machine learning?"])
            assert result == [0.7, 0.8, 0.9]
