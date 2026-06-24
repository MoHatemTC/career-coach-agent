"""
app/ai/registry.py
==================
Unified LLM gateway — ``LLMServiceRegistry``.

All LLM and embedding calls across every feature in this project route through
this single registry.  It talks to the LiteLLM proxy via the ``litellm`` SDK
directly (no LangChain wrappers), giving us one consistent place to configure
models, keys, and base URLs.

Environment variables
---------------------
LITELLM_BASE_URL
    Base URL of the LiteLLM proxy.
    Example: ``https://management.sprints.ai/litellm``

LITELLM_API_KEY
    API key for the LiteLLM proxy.

DEFAULT_MODEL
    The chat/completion model identifier as recognised by the proxy.
    Default: ``azure/FW-Kimi-K2.6``

DEFAULT_EMBEDDING_MODEL
    The embedding model identifier as recognised by the proxy.
    Default: ``text-embedding-ada-002``
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Type

import litellm
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration — read once from the environment at import time
# ---------------------------------------------------------------------------

_LITELLM_BASE_URL: str = os.getenv(
    "LITELLM_BASE_URL", "https://management.sprints.ai/litellm"
).rstrip("/")
_LITELLM_API_KEY: str = os.getenv("LITELLM_API_KEY", "")
_DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "azure/FW-Kimi-K2.6")
_DEFAULT_EMBEDDING_MODEL: str = os.getenv(
    "DEFAULT_EMBEDDING_MODEL", "text-embedding-ada-002"
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class LLMServiceRegistry:
    """
    Single, unified gateway for all LLM and embedding calls.

    Uses the ``litellm`` SDK to talk directly to the LiteLLM proxy so that
    every feature in the project shares the same model configuration, API key,
    and base URL without going through LangChain wrappers.

    Obtain the process-wide singleton via :func:`get_registry` rather than
    instantiating this class directly.

    Parameters
    ----------
    base_url:
        Root URL of the LiteLLM proxy (without a trailing slash).
    api_key:
        API key passed as the ``Authorization: Bearer`` header.
    default_model:
        Model identifier used when the caller does not specify one.
    default_embedding_model:
        Embedding model identifier used when the caller does not specify one.
    """

    def __init__(
        self,
        base_url: str = _LITELLM_BASE_URL,
        api_key: str = _LITELLM_API_KEY,
        default_model: str = _DEFAULT_MODEL,
        default_embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._default_model = default_model
        self._default_embedding_model = default_embedding_model

    # ------------------------------------------------------------------
    # Chat / completion
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        response_format: Type[BaseModel] | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        """
        Send a chat completion request to the LiteLLM proxy.

        Parameters
        ----------
        messages:
            OpenAI-style message list, e.g.
            ``[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]``
        model:
            Override the default model for this call.
        response_format:
            A Pydantic ``BaseModel`` subclass.  When provided, litellm will
            request structured JSON output conforming to the model's schema and
            parse the response into an instance of that class automatically.
        temperature:
            Sampling temperature (default ``0.0`` for deterministic output).
        **kwargs:
            Any additional litellm keyword arguments.

        Returns
        -------
        Any
            If *response_format* is given, returns a parsed Pydantic model
            instance.  Otherwise returns the raw
            ``litellm.ModelResponse`` object.
        """
        resolved_model = model or self._default_model

        completion_kwargs: dict[str, Any] = dict(
            model=resolved_model,
            messages=messages,
            temperature=temperature,
            api_base=self._base_url,
            api_key=self._api_key,
            **kwargs,
        )

        if response_format is not None:
            completion_kwargs["response_format"] = response_format

        response = litellm.completion(**completion_kwargs)

        if response_format is not None:
            # litellm returns the parsed object in .choices[0].message.content
            # when a Pydantic model is passed as response_format.
            content = response.choices[0].message.content
            if isinstance(content, str):
                return response_format.model_validate_json(content)
            return content

        return response

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """
        Embed a list of text strings via the LiteLLM proxy.

        Parameters
        ----------
        texts:
            One or more strings to embed.
        model:
            Override the default embedding model for this call.
        **kwargs:
            Any additional litellm keyword arguments.

        Returns
        -------
        list[list[float]]
            A list of embedding vectors, one per input string.
        """
        resolved_model = model or self._default_embedding_model

        response = litellm.embedding(
            model=resolved_model,
            input=texts,
            api_base=self._base_url,
            api_key=self._api_key,
            **kwargs,
        )

        return [item["embedding"] for item in response.data]


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_registry() -> LLMServiceRegistry:
    """
    Return the process-wide :class:`LLMServiceRegistry` singleton.

    The instance is constructed once from environment variables and cached for
    the lifetime of the process.  In tests, call
    ``get_registry.cache_clear()`` after patching env vars if you need a
    fresh instance.
    """
    return LLMServiceRegistry()
