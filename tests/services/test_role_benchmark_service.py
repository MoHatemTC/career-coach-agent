"""
tests/services/test_role_benchmark_service.py
=============================================
Unit and persistence tests for the LangGraph role-benchmark extraction pipeline.

All external I/O (LLM calls, embedding calls) is mocked so tests run without
network access or API keys.

The persistence tests use an **in-memory SQLite database** via SQLModel so
they run without a running Postgres instance.  The ``db_session`` pytest
fixture sets up the schema and tears it down automatically after each test.

Test matrix
-----------
* happy path            – LLM succeeds on the first attempt; extraction data,
                          embedding, and error_count are all correct.
* retry-then-succeed    – LLM raises OutputParserException on the first call,
                          then succeeds on the second; graph loops exactly once
                          and records 1 error entry.
* exhaust retries       – LLM raises OutputParserException 3 times; the service
                          raises ValueError and the embedder is never called.
* embed failure         – extraction succeeds but the embedder raises; the error
                          propagates correctly.
* route-function tests  – white-box tests of route_after_extract in isolation.
* persistence tests     – save_benchmark() writes to and reads back from an
                          in-memory SQLite DB; all fields round-trip correctly.
"""

from __future__ import annotations

import os
from typing import List
from unittest.mock import MagicMock

import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError
from sqlmodel import Session, SQLModel, create_engine

from app.schemas.role_benchmark import ExperiencePatterns, RoleBenchmark
from app.services.role_benchmark_service import (
    MAX_RETRIES,
    BenchmarkState,
    route_after_extract,
    run_benchmark_pipeline,
    save_benchmark,
)

# ---------------------------------------------------------------------------
# Ensure DATABASE_URL is set before app.core.database is imported
# (unit tests use SQLite in-memory; this prevents the fail-fast guard from
#  raising ValueError when there is no .env in CI)
# ---------------------------------------------------------------------------
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")


# ---------------------------------------------------------------------------
# Shared fixtures & helper data
# ---------------------------------------------------------------------------

FAKE_EMBEDDING: List[float] = [0.1] * 1536

VALID_BENCHMARK = RoleBenchmark(
    must_have_skills=["System design", "Distributed systems"],
    nice_to_have_skills=["GraphQL"],
    required_tools=["Python", "FastAPI", "Docker", "PostgreSQL"],
    experience_patterns=ExperiencePatterns(minimum_years=3, level="Senior"),
    common_responsibilities=[
        "Design and implement RESTful APIs",
        "Collaborate with cross-functional teams",
        "Conduct code reviews",
    ],
)

SAMPLE_JOB_TEXT = (
    "We are hiring a Senior Backend Engineer with 3-5 years of Python experience. "
    "You will design distributed systems and RESTful APIs using FastAPI and Docker. "
    "Nice to have: GraphQL knowledge."
)


def _make_llm_mock(side_effects: list) -> MagicMock:
    """
    Build a mock LLM whose ``with_structured_output`` chain's ``.invoke``
    cycles through *side_effects*.

    Each element of *side_effects* is either:
    * A ``RoleBenchmark`` instance  → ``.invoke`` returns it.
    * An ``Exception`` instance     → ``.invoke`` raises it.

    The mock replicates the ``extraction_prompt | llm.with_structured_output(...)``
    pipe operator chain that the service builds internally.
    """
    invoke_mock = MagicMock(side_effect=side_effects)
    chain_mock = MagicMock()
    chain_mock.invoke = invoke_mock

    # extraction_prompt | llm.with_structured_output(...)  →  chain_mock
    structured_output_mock = MagicMock()
    structured_output_mock.__ror__ = MagicMock(return_value=chain_mock)

    llm_mock = MagicMock()
    llm_mock.with_structured_output = MagicMock(return_value=structured_output_mock)

    return llm_mock


def _make_embedder_mock(vector: List[float] = FAKE_EMBEDDING) -> MagicMock:
    """Return a mock embedder whose ``embed_query`` returns *vector*."""
    embedder_mock = MagicMock()
    embedder_mock.embed_query = MagicMock(return_value=vector)
    return embedder_mock


def _make_validation_error() -> ValidationError:
    """
    Construct a real Pydantic ``ValidationError`` by deliberately failing
    ``RoleBenchmark`` validation (missing all required fields).
    """
    try:
        RoleBenchmark.model_validate({})  # missing required fields → raises
    except ValidationError as exc:
        return exc
    raise RuntimeError("Expected ValidationError was not raised")  # pragma: no cover


def _make_parser_exception(message: str = "LLM output did not match schema") -> OutputParserException:
    """Construct a real ``OutputParserException`` for retry-path testing."""
    return OutputParserException(message)


# ---------------------------------------------------------------------------
# Persistence fixture (SQLite in-memory — no Postgres required)
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_session():
    """
    Provide a SQLModel ``Session`` backed by a fresh SQLite in-memory database.

    The full schema (all SQLModel ``table=True`` classes visible to the
    metadata) is created before yielding and implicitly destroyed when the
    in-memory engine is garbage-collected after the test.

    Using SQLite instead of Postgres means:
    * No running database server is required.
    * The ``Vector(1536)`` column type is not available — the ORM model stores
      the embedding as a plain JSON list in SQLite (acceptable for testing
      round-trip behaviour; pgvector features are tested separately against a
      real Postgres instance).
    """
    # Import ORM models so their metadata is registered before create_all().
    from app.models.role_benchmark import RoleBenchmark as RoleBenchmarkModel  # noqa: F401

    sqlite_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    # pgvector's Vector type is not available in SQLite; patch the column type
    # to use JSON so SQLite can store the embedding as a list.
    from sqlalchemy import JSON, Column
    from sqlmodel import Field

    RoleBenchmarkModel.__table__.c["embedding"].type = JSON()

    SQLModel.metadata.create_all(sqlite_engine)

    with Session(sqlite_engine) as session:
        yield session

    SQLModel.metadata.drop_all(sqlite_engine)


# ---------------------------------------------------------------------------
# 1. Happy-path tests
# ---------------------------------------------------------------------------


class TestHappyPath:
    """LLM succeeds on the first attempt; graph completes without any retry."""

    def test_happy_path_extraction(self):
        """
        ``test_happy_path_extraction``: assert successful extraction,
        embedding produced, and error_count == 0.
        """
        llm_mock = _make_llm_mock([VALID_BENCHMARK])
        embedder_mock = _make_embedder_mock()

        state = run_benchmark_pipeline(
            raw_text=SAMPLE_JOB_TEXT,
            llm=llm_mock,
            embedder=embedder_mock,
        )

        # Extracted data is correct
        assert state["extracted_data"] == VALID_BENCHMARK

        # Embedding is produced and has the right dimensionality
        assert state["embedding"] == FAKE_EMBEDDING
        assert len(state["embedding"]) == 1536

        # No errors recorded
        assert state["error_count"] == 0
        assert state["validation_errors"] == []

    def test_llm_invoked_exactly_once(self):
        llm_mock = _make_llm_mock([VALID_BENCHMARK])
        embedder_mock = _make_embedder_mock()

        run_benchmark_pipeline(
            raw_text=SAMPLE_JOB_TEXT,
            llm=llm_mock,
            embedder=embedder_mock,
        )

        chain = llm_mock.with_structured_output.return_value.__ror__.return_value
        assert chain.invoke.call_count == 1

    def test_embedder_called_with_combined_skills_and_tools(self):
        llm_mock = _make_llm_mock([VALID_BENCHMARK])
        embedder_mock = _make_embedder_mock()

        run_benchmark_pipeline(
            raw_text=SAMPLE_JOB_TEXT,
            llm=llm_mock,
            embedder=embedder_mock,
        )

        expected_corpus = ", ".join(
            VALID_BENCHMARK.must_have_skills
            + VALID_BENCHMARK.nice_to_have_skills
            + VALID_BENCHMARK.required_tools
        )
        embedder_mock.embed_query.assert_called_once_with(expected_corpus)


# ---------------------------------------------------------------------------
# 2. Retry-then-succeed tests
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """
    LLM raises ``OutputParserException`` on the first call, then succeeds on
    the second.  The graph must loop exactly once and record 1 error.
    """

    def setup_method(self):
        parser_exc = _make_parser_exception()
        self.llm_mock = _make_llm_mock([parser_exc, VALID_BENCHMARK])
        self.embedder_mock = _make_embedder_mock()

        self.state = run_benchmark_pipeline(
            raw_text=SAMPLE_JOB_TEXT,
            llm=self.llm_mock,
            embedder=self.embedder_mock,
        )

    def test_retry_logic_recovers(self):
        """
        ``test_retry_logic_recovers``: if the LLM raises ``OutputParserException``
        on the first call it retries, succeeds on the second, and logs 1 error.
        """
        chain = self.llm_mock.with_structured_output.return_value.__ror__.return_value

        # The chain should have been called twice (fail + succeed)
        assert chain.invoke.call_count == 2

        # Exactly one error was recorded
        assert self.state["error_count"] == 1
        assert len(self.state["validation_errors"]) >= 1

    def test_final_extraction_is_valid(self):
        assert self.state["extracted_data"] == VALID_BENCHMARK

    def test_embedding_is_produced_after_retry(self):
        assert self.state["embedding"] == FAKE_EMBEDDING

    def test_embedder_called_exactly_once(self):
        self.embedder_mock.embed_query.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Exhausted-retries tests
# ---------------------------------------------------------------------------


class TestExhaustedRetries:
    """
    LLM raises ``OutputParserException`` on all MAX_RETRIES attempts.
    The service must raise ``ValueError`` and the embedder must never be called.
    """

    def test_exhausted_retries(self):
        """
        ``test_exhausted_retries``: ``ValueError`` raised after 3 failed
        parsing attempts; embedder is never called.
        """
        parser_exc = _make_parser_exception()
        llm_mock = _make_llm_mock([parser_exc] * MAX_RETRIES)
        embedder_mock = _make_embedder_mock()

        with pytest.raises(ValueError, match="failed after"):
            run_benchmark_pipeline(
                raw_text=SAMPLE_JOB_TEXT,
                llm=llm_mock,
                embedder=embedder_mock,
            )

        # Embedder must never have been touched
        embedder_mock.embed_query.assert_not_called()

    def test_llm_invoked_exactly_max_retries_times(self):
        parser_exc = _make_parser_exception()
        llm_mock = _make_llm_mock([parser_exc] * MAX_RETRIES)
        embedder_mock = _make_embedder_mock()

        with pytest.raises(ValueError):
            run_benchmark_pipeline(
                raw_text=SAMPLE_JOB_TEXT,
                llm=llm_mock,
                embedder=embedder_mock,
            )

        chain = llm_mock.with_structured_output.return_value.__ror__.return_value
        assert chain.invoke.call_count == MAX_RETRIES

    def test_error_message_references_max_retries(self):
        parser_exc = _make_parser_exception()
        llm_mock = _make_llm_mock([parser_exc] * MAX_RETRIES)
        embedder_mock = _make_embedder_mock()

        with pytest.raises(ValueError) as exc_info:
            run_benchmark_pipeline(
                raw_text=SAMPLE_JOB_TEXT,
                llm=llm_mock,
                embedder=embedder_mock,
            )

        assert "failed after" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 4. Embed-failure test
# ---------------------------------------------------------------------------


class TestEmbedFailure:
    """Extraction succeeds but the embedder raises a RuntimeError."""

    def test_propagates_embedder_exception(self):
        llm_mock = _make_llm_mock([VALID_BENCHMARK])
        embedder_mock = MagicMock()
        embedder_mock.embed_query = MagicMock(
            side_effect=RuntimeError("OpenAI embedding endpoint unavailable")
        )

        with pytest.raises(RuntimeError, match="embedding endpoint unavailable"):
            run_benchmark_pipeline(
                raw_text=SAMPLE_JOB_TEXT,
                llm=llm_mock,
                embedder=embedder_mock,
            )


# ---------------------------------------------------------------------------
# 5. Route-function unit tests (state-machine logic in isolation)
# ---------------------------------------------------------------------------


class TestRouteAfterExtract:
    """Directly test the conditional-edge router function."""

    def test_routes_to_embed_on_success(self):
        state: BenchmarkState = {
            "raw_text": SAMPLE_JOB_TEXT,
            "extracted_data": VALID_BENCHMARK,
            "embedding": None,
            "error_count": 0,
            "validation_errors": [],
        }
        assert route_after_extract(state) == "embed"

    def test_routes_back_to_extract_on_first_failure(self):
        state: BenchmarkState = {
            "raw_text": SAMPLE_JOB_TEXT,
            "extracted_data": None,
            "embedding": None,
            "error_count": 1,
            "validation_errors": ["field required"],
        }
        assert route_after_extract(state) == "extract"

    def test_raises_after_max_retries(self):
        state: BenchmarkState = {
            "raw_text": SAMPLE_JOB_TEXT,
            "extracted_data": None,
            "embedding": None,
            "error_count": MAX_RETRIES,
            "validation_errors": ["err1", "err2", "err3"],
        }
        with pytest.raises(ValueError):
            route_after_extract(state)


# ---------------------------------------------------------------------------
# 6. Persistence tests  (SQLite in-memory — no Postgres required)
# ---------------------------------------------------------------------------


class TestPersistence:
    """
    Verify that ``save_benchmark()`` correctly writes a benchmark to the
    database and that the persisted record can be read back with all fields
    intact.

    These tests use the ``db_session`` fixture which spins up a fresh SQLite
    in-memory database for each test method.
    """

    def _make_state(
        self,
        benchmark: RoleBenchmark = VALID_BENCHMARK,
        embedding: List[float] = FAKE_EMBEDDING,
    ) -> BenchmarkState:
        return {
            "raw_text": SAMPLE_JOB_TEXT,
            "extracted_data": benchmark,
            "embedding": embedding,
            "error_count": 0,
            "validation_errors": [],
        }

    def test_save_returns_record_with_id(self, db_session: Session):
        """``save_benchmark`` must return a record with a non-None positive id."""
        state = self._make_state()
        record = save_benchmark(state, db_session)

        assert record.id is not None
        assert record.id > 0

    def test_saved_skills_match_input(self, db_session: Session):
        """must_have_skills persisted in the DB must match the schema exactly."""
        state = self._make_state()
        record = save_benchmark(state, db_session)

        assert record.must_have_skills == VALID_BENCHMARK.must_have_skills

    def test_saved_nice_to_have_skills_match_input(self, db_session: Session):
        state = self._make_state()
        record = save_benchmark(state, db_session)

        assert record.nice_to_have_skills == VALID_BENCHMARK.nice_to_have_skills

    def test_saved_tools_match_input(self, db_session: Session):
        state = self._make_state()
        record = save_benchmark(state, db_session)

        assert record.required_tools == VALID_BENCHMARK.required_tools

    def test_saved_responsibilities_match_input(self, db_session: Session):
        state = self._make_state()
        record = save_benchmark(state, db_session)

        assert record.common_responsibilities == VALID_BENCHMARK.common_responsibilities

    def test_saved_seniority_level_matches_input(self, db_session: Session):
        state = self._make_state()
        record = save_benchmark(state, db_session)

        assert record.seniority_level == VALID_BENCHMARK.experience_patterns.level

    def test_saved_minimum_years_matches_input(self, db_session: Session):
        state = self._make_state()
        record = save_benchmark(state, db_session)

        assert record.minimum_years == VALID_BENCHMARK.experience_patterns.minimum_years

    def test_embedding_round_trips_correctly(self, db_session: Session):
        """The embedding list must survive a write + read cycle intact."""
        state = self._make_state()
        record = save_benchmark(state, db_session)

        # Re-query from the DB to confirm the value is actually persisted,
        # not just held in the session identity map.
        from app.models.role_benchmark import RoleBenchmark as RoleBenchmarkModel

        db_session.expire(record)  # force reload from DB on next access
        reloaded = db_session.get(RoleBenchmarkModel, record.id)

        assert reloaded is not None
        assert len(reloaded.embedding) == 1536
        assert reloaded.embedding == FAKE_EMBEDDING

    def test_created_at_is_populated(self, db_session: Session):
        """created_at must be a non-None datetime after persistence."""
        from datetime import datetime

        state = self._make_state()
        record = save_benchmark(state, db_session)

        assert record.created_at is not None
        assert isinstance(record.created_at, datetime)

    def test_raises_when_extracted_data_is_none(self, db_session: Session):
        """save_benchmark must raise ValueError if extracted_data is None."""
        bad_state: BenchmarkState = {
            "raw_text": SAMPLE_JOB_TEXT,
            "extracted_data": None,
            "embedding": None,
            "error_count": 3,
            "validation_errors": ["err1", "err2", "err3"],
        }

        with pytest.raises(ValueError, match="extracted_data is None"):
            save_benchmark(bad_state, db_session)

    def test_multiple_records_have_distinct_ids(self, db_session: Session):
        """Each call to save_benchmark must produce a unique primary key."""
        state = self._make_state()
        record_a = save_benchmark(state, db_session)
        record_b = save_benchmark(state, db_session)

        assert record_a.id != record_b.id
