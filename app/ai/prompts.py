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

    messages = PromptBuilder().build_readiness_gap_analysis_messages(
        candidate_profile=profile_dict,
        benchmark=benchmark_dict,
    )
    result = get_registry().complete(messages, response_format=ReadinessGapAnalysis)
"""

from __future__ import annotations

import json
import os
from typing import Any

# ---------------------------------------------------------------------------
# Template loading — file I/O happens once at import time, not per request
# ---------------------------------------------------------------------------

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

with open(os.path.join(_PROMPTS_DIR, "role_benchmark.md"), "r", encoding="utf-8") as _f:
    _ROLE_BENCHMARK_SYSTEM_PROMPT: str = _f.read()

with open(
    os.path.join(_PROMPTS_DIR, "readiness_gap_analysis.md"), "r", encoding="utf-8"
) as _f:
    _READINESS_GAP_ANALYSIS_SYSTEM_PROMPT: str = _f.read()


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

    # ------------------------------------------------------------------
    # Readiness Gap Analysis
    # ------------------------------------------------------------------

    @staticmethod
    def build_readiness_gap_analysis_messages(
        candidate_profile: dict[str, Any],
        benchmark: dict[str, Any],
    ) -> list[dict[str, str]]:
        """
        Build the message payload for the career readiness gap-analysis task.

        Combines the cached system prompt (loaded from
        ``app/ai/prompts/readiness_gap_analysis.md``) with the caller-supplied
        candidate profile and role benchmark, serialised as JSON.

        Parameters
        ----------
        candidate_profile:
            A dict representation of the candidate's qualifications, containing
            at minimum the keys ``skills``, ``tools``, ``experience_years``, and
            ``education``.
        benchmark:
            A dict representation of the role benchmark, containing at minimum
            ``must_have_skills``, ``nice_to_have_skills``, ``required_tools``,
            ``minimum_years``, ``seniority_level``, and
            ``common_responsibilities``.

        Returns
        -------
        list[dict[str, str]]
            OpenAI-style messages::

                [
                    {"role": "system", "content": "<readiness gap-analysis system prompt>"},
                    {"role": "user",   "content": "<JSON payload with profile + benchmark>"},
                ]
        """
        user_payload = json.dumps(
            {
                "candidate_profile": candidate_profile,
                "role_benchmark": benchmark,
            },
            indent=2,
            ensure_ascii=False,
        )
        return [
            {"role": "system", "content": _READINESS_GAP_ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ]
