from unittest.mock import MagicMock, patch
from app.ai.registry import LLMServiceRegistry


def test_registry_local_embedder_routing():
    """Verify that registry.embed routes to local embedder when model='local'."""
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    with patch(
        "app.ai.local_embedder.get_local_embedder", return_value=mock_embedder
    ) as mock_get:
        registry = LLMServiceRegistry(default_embedding_model="text-embedding-ada-002")
        result = registry.embed(["hello world"], model="local")

        mock_get.assert_called_once()
        mock_embedder.embed.assert_called_once_with(["hello world"])
        assert result == [[0.1, 0.2, 0.3]]


def test_registry_local_embedder_routing_by_model_name():
    """Verify that registry.embed routes to local embedder when model='all-MiniLM-L6-v2'."""
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    with patch(
        "app.ai.local_embedder.get_local_embedder", return_value=mock_embedder
    ) as mock_get:
        registry = LLMServiceRegistry(default_embedding_model="text-embedding-ada-002")
        result = registry.embed(["hello world"], model="all-MiniLM-L6-v2")

        mock_get.assert_called_once()
        mock_embedder.embed.assert_called_once_with(["hello world"])
        assert result == [[0.1, 0.2, 0.3]]


def test_registry_local_embedder_routing_default():
    """Verify that registry.embed routes to local embedder when default model is 'local'."""
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    with patch(
        "app.ai.local_embedder.get_local_embedder", return_value=mock_embedder
    ) as mock_get:
        registry = LLMServiceRegistry(default_embedding_model="local")
        result = registry.embed(["hello world"])

        mock_get.assert_called_once()
        mock_embedder.embed.assert_called_once_with(["hello world"])
        assert result == [[0.1, 0.2, 0.3]]
