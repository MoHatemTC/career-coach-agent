"""
app/core/llm.py
===============
Shared service factories for the LLM and embedding model.

Uses the company LiteLLM proxy (OpenAI-compatible) so that any model
available on the proxy can be selected via environment variables without
changing any application code.

Environment variables
---------------------
LITELLM_BASE_URL
    Base URL of the LiteLLM proxy.
    Example: ``https://management.sprints.ai/litellm``

LITELLM_API_KEY
    API key for the LiteLLM proxy.

DEFAULT_LLM_MODEL
    The chat model identifier as recognised by the proxy.
    Default: ``azure/FW-Kimi-K2.6``

DEFAULT_EMBEDDING_MODEL
    The embedding model identifier as recognised by the proxy.
    Default: ``text-embedding-ada-002``
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ---------------------------------------------------------------------------
# Configuration — read once from the environment at import time
# ---------------------------------------------------------------------------

_LITELLM_BASE_URL: str = os.getenv(
    "LITELLM_BASE_URL", "https://management.sprints.ai/litellm"
).rstrip("/") + "/v1"
_LITELLM_API_KEY: str = os.getenv("LITELLM_API_KEY", "")
_DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "azure/FW-Kimi-K2.6")
_DEFAULT_EMBEDDING_MODEL: str = os.getenv(
    "DEFAULT_EMBEDDING_MODEL", "text-embedding-ada-002"
)


# ---------------------------------------------------------------------------
# Shared instance factories
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_shared_llm() -> ChatOpenAI:
    """
    Return a process-wide ``ChatOpenAI`` instance pointed at the LiteLLM proxy.

    The result is cached after the first call so that subsequent imports
    and dependency injections receive the same object (and its underlying
    HTTP connection pool).

    Returns
    -------
    ChatOpenAI
        Configured with ``temperature=0`` for deterministic, structured
        output extraction.

    Notes
    -----
    ``lru_cache`` makes this effectively a singleton per process.  In tests,
    call ``get_shared_llm.cache_clear()`` after overriding env vars if you
    need a fresh instance.
    """
    return ChatOpenAI(
        model=_DEFAULT_LLM_MODEL,
        temperature=0,
        base_url=_LITELLM_BASE_URL,
        api_key=_LITELLM_API_KEY,
    )


@lru_cache(maxsize=1)
def get_shared_embedder() -> OpenAIEmbeddings:
    """
    Return a process-wide ``OpenAIEmbeddings`` instance pointed at the
    LiteLLM proxy.

    Returns
    -------
    OpenAIEmbeddings
        Ready to call ``.embed_query()`` or ``.embed_documents()``.

    Notes
    -----
    ``lru_cache`` makes this effectively a singleton per process.  In tests,
    call ``get_shared_embedder.cache_clear()`` after overriding env vars if
    you need a fresh instance.
    """
    return OpenAIEmbeddings(
        model=_DEFAULT_EMBEDDING_MODEL,
        base_url=_LITELLM_BASE_URL,
        api_key=_LITELLM_API_KEY,
    )
