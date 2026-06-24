"""
app/ai/prompts.py
=================
``PromptBuilder`` — centralised factory for building LLM message payloads.

Prompt templates are stored as plain Markdown files in ``app/ai/prompts/``
and loaded once at module import time.  Callers receive standard
OpenAI-style message lists (``list[dict[str, str]]``) that can be passed
directly to :class:`~app.ai.registry.LLMServiceRegistry`.

Usage
-----
::

    from app.ai.prompts import PromptBuilder

    messages = PromptBuilder().build_role_benchmark_messages(raw_text)
    result = get_registry().complete(messages, response_format=RoleBenchmark)
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Template loading — file I/O happens once at import time, not per request
# ---------------------------------------------------------------------------

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

with open(os.path.join(_PROMPTS_DIR, "role_benchmark.md"), "r", encoding="utf-8") as _f:
    _ROLE_BENCHMARK_SYSTEM_PROMPT: str = _f.read()


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------


class PromptBuilder:
    """
    Factory for building OpenAI-style message lists from stored Markdown
    templates.

    Each ``build_*`` method returns a ``list[dict[str, str]]`` that is ready
    to pass to :meth:`~app.ai.registry.LLMServiceRegistry.complete`.

    The class is stateless; instantiate it anywhere without arguments.
    """

    # ------------------------------------------------------------------
    # Role Benchmark
    # ------------------------------------------------------------------

    @staticmethod
    def build_role_benchmark_messages(raw_text: str) -> list[dict[str, str]]:
        """
        Build the message payload for the role-benchmark extraction task.

        Combines the cached system prompt (loaded from
        ``app/ai/prompts/role_benchmark.md``) with the caller-supplied job
        description text.

        Parameters
        ----------
        raw_text:
            The raw, unstructured job description as plain text.

        Returns
        -------
        list[dict[str, str]]
            OpenAI-style messages::

                [
                    {"role": "system", "content": "<role_benchmark system prompt>"},
                    {"role": "user",   "content": "Job Description:\\n\\n<raw_text>"},
                ]
        """
        return [
            {"role": "system", "content": _ROLE_BENCHMARK_SYSTEM_PROMPT},
            {"role": "user", "content": f"Job Description:\n\n{raw_text}"},
        ]
