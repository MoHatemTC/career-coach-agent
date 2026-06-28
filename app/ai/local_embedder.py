"""
app/ai/local_embedder.py
========================
Local embedding engine using ``sentence-transformers``.

Runs entirely on CPU with no API key required.  The model
``all-MiniLM-L6-v2`` is downloaded once from the Hugging Face Hub on first
use and cached in the OS model cache directory.

Output dimensionality: **384** floats per text string.

Usage
-----
    from app.ai.local_embedder import get_local_embedder

    embedder = get_local_embedder()          # cached singleton
    vectors = embedder.embed(["hello world", "foo bar"])
    # → list[list[float]], each inner list has 384 elements
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Model identifier on the Hugging Face Hub.
# all-MiniLM-L6-v2: fast, 384-dim, excellent general-purpose quality.
_MODEL_NAME = "all-MiniLM-L6-v2"


class LocalEmbedder:
    """
    Thin wrapper around ``SentenceTransformer`` that matches the same
    ``embed(texts) -> list[list[float]]`` interface used by
    :class:`~app.ai.registry.LLMServiceRegistry`.

    The underlying model is loaded once and reused for the lifetime of the
    process.
    """

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        logger.info("Loading local embedding model '%s' …", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("Local embedding model loaded (dim=%d).", self.dim)

    @property
    def dim(self) -> int:
        """Dimensionality of the output vectors."""
        return self._model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text strings.

        Parameters
        ----------
        texts:
            One or more strings to embed.

        Returns
        -------
        list[list[float]]
            A list of embedding vectors (one per input string).
            Each vector has :attr:`dim` elements.
        """
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vectors]


@lru_cache(maxsize=1)
def get_local_embedder() -> LocalEmbedder:
    """
    Return the process-wide :class:`LocalEmbedder` singleton.

    The model is loaded from disk (or downloaded from the Hub) on the first
    call and reused thereafter.  Call ``get_local_embedder.cache_clear()``
    in tests if you need a fresh instance.
    """
    return LocalEmbedder()
