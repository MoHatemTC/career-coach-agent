from __future__ import annotations

import logging
from typing import Any, List, Optional, TypedDict

from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
# Langfuse is optional observability — disabled due to API incompatibility
# (LangchainCallbackHandler no longer accepts 'trace_name' kwarg).
_LANGFUSE_AVAILABLE = False
_LangfuseHandler = None
from langgraph.graph import END, StateGraph
from pydantic import ValidationError
from sqlmodel import Session

from app.core.llm import get_shared_embedder, get_shared_llm
from app.models.role_benchmark import RoleBenchmark as RoleBenchmarkModel
from app.schemas.role_benchmark import RoleBenchmark

logger = logging.getLogger(__name__)

MAX_RETRIES: int = 3


class BenchmarkState(TypedDict):
    """Shared mutable state passed between LangGraph nodes."""

    raw_text: str
    extracted_data: Optional[RoleBenchmark]
    embedding: Optional[List[float]]
    error_count: int
    validation_errors: List[str]


_SYSTEM_PROMPT = (
    "You are an expert technical recruiter. "
    "Extract structured information from the job description provided by the user. "
    "Be precise and follow the schema exactly."
)

_HUMAN_PROMPT = "Job Description:\n\n{raw_text}"

extraction_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", _HUMAN_PROMPT),
    ]
)


def make_extract_node(llm: Any | None = None):
    """Return an extract node function bound to the given LLM."""
    resolved_llm: Any = llm if llm is not None else get_shared_llm()
    chain = extraction_prompt | resolved_llm.with_structured_output(RoleBenchmark)

    def extract(state: BenchmarkState) -> BenchmarkState:
        attempt = state["error_count"] + 1
        logger.debug("extract node – attempt %d / %d", attempt, MAX_RETRIES)

        langfuse_handler = (
            _LangfuseHandler(
                trace_name="role_benchmark_extraction",
                metadata={"attempt": attempt, "max_retries": MAX_RETRIES},
            )
            if _LANGFUSE_AVAILABLE
            else None
        )

        try:
            callbacks = [langfuse_handler] if langfuse_handler is not None else []
            result: RoleBenchmark = chain.invoke(
                {"raw_text": state["raw_text"]},
                config={"callbacks": callbacks},
            )
            return {
                **state,
                "extracted_data": result,
            }
        except (ValidationError, OutputParserException) as exc:
            exc_type = type(exc).__name__
            logger.warning("%s on attempt %d: %s", exc_type, attempt, exc)
            error_msg: str = (
                ", ".join(str(e) for e in exc.errors())
                if isinstance(exc, ValidationError)
                else str(exc)
            )
            return {
                **state,
                "extracted_data": None,
                "validation_errors": state["validation_errors"] + [error_msg],
                "error_count": state["error_count"] + 1,
            }
        finally:
            if langfuse_handler is not None:
                langfuse_handler.flush()

    return extract


def make_embed_node(embedder: Any | None = None):
    """Return an embed node function bound to the given embedder."""
    resolved_embedder: Any = embedder if embedder is not None else get_shared_embedder()

    def embed(state: BenchmarkState) -> BenchmarkState:
        data: RoleBenchmark = state["extracted_data"]

        corpus_parts: List[str] = (
            data.must_have_skills
            + data.nice_to_have_skills
            + data.required_tools
        )
        corpus: str = ", ".join(corpus_parts)

        logger.debug("embed node – embedding %d tokens of combined skills/tools", len(corpus))

        vector: List[float] = resolved_embedder.embed_query(corpus)
        return {
            **state,
            "embedding": vector,
        }

    return embed


def route_after_extract(state: BenchmarkState) -> str:
    """
    Decide the next node after extract:
    - 'embed'    if extraction succeeded.
    - 'extract'  if failed with retries remaining.
    - raises ValueError after MAX_RETRIES failures.
    """
    if state["extracted_data"] is not None:
        return "embed"

    if state["error_count"] < MAX_RETRIES:
        logger.info(
            "Retrying extraction (%d / %d failures so far).",
            state["error_count"],
            MAX_RETRIES,
        )
        return "extract"

    raise ValueError(
        f"LLM extraction failed after {MAX_RETRIES} attempts. "
        f"Validation errors collected:\n" + "\n".join(state["validation_errors"])
    )


def build_graph(llm: Any | None = None, embedder: Any | None = None) -> Any:
    """Assemble and compile the LangGraph StateGraph."""
    graph = StateGraph(BenchmarkState)

    graph.add_node("extract", make_extract_node(llm))
    graph.add_node("embed", make_embed_node(embedder))

    graph.set_entry_point("extract")

    graph.add_conditional_edges(
        "extract",
        route_after_extract,
        {
            "extract": "extract",
            "embed": "embed",
        },
    )

    graph.add_edge("embed", END)

    return graph.compile()


def run_benchmark_pipeline(
    raw_text: str,
    llm: Any | None = None,
    embedder: Any | None = None,
) -> BenchmarkState:
    """Execute the full extraction + embedding pipeline for a job description."""
    initial_state: BenchmarkState = {
        "raw_text": raw_text,
        "extracted_data": None,
        "embedding": None,
        "error_count": 0,
        "validation_errors": [],
    }

    compiled_graph = build_graph(llm=llm, embedder=embedder)
    final_state: BenchmarkState = compiled_graph.invoke(initial_state)
    return final_state


def save_benchmark(state: BenchmarkState, session: Session) -> RoleBenchmarkModel:
    """Persist the result of run_benchmark_pipeline to the database."""
    extracted = state["extracted_data"]
    if extracted is None:
        raise ValueError(
            "Cannot persist benchmark: extracted_data is None. "
            "Run run_benchmark_pipeline() successfully before calling save_benchmark()."
        )

    db_record = RoleBenchmarkModel(
        must_have_skills=extracted.must_have_skills,
        nice_to_have_skills=extracted.nice_to_have_skills,
        required_tools=extracted.required_tools,
        common_responsibilities=extracted.common_responsibilities,
        minimum_years=extracted.experience_patterns.minimum_years,
        seniority_level=extracted.experience_patterns.level,
        embedding=state["embedding"],
    )

    try:
        session.add(db_record)
        session.commit()
        session.refresh(db_record)
    except Exception as exc:
        session.rollback()
        logger.exception("Database commit failed in save_benchmark()")
        raise RuntimeError("Failed to persist the benchmark to the database.") from exc

    logger.info(
        "Persisted benchmark id=%s (seniority=%s, min_years=%d)",
        db_record.id,
        db_record.seniority_level,
        db_record.minimum_years,
    )
    return db_record
