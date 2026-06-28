"""
tests/services/test_readiness_score_service.py
===============================================
Unit and persistence tests for the LangGraph career readiness scoring pipeline.

All external I/O (LLM calls) is mocked through ``LLMServiceRegistry`` so tests
run without network access or API keys.

Persistence tests use an **in-memory SQLite database** via SQLModel so they run
without a running Postgres instance.  The ``db_session`` fixture sets up the full
schema (both ``role_benchmarks`` and ``readiness_scores`` tables) and tears it
down automatically after each test.

Test matrix
-----------
* ``TestHappyPath``          — LLM succeeds on the first attempt; all output
                               fields are correct, error_count is 0.
* ``TestRetryLogic``         — Registry raises on the first call, then succeeds;
                               graph loops exactly once, records 1 error entry.
* ``TestExhaustedRetries``   — Registry raises 3× (MAX_RETRIES); the service
                               raises ValueError; no DB write occurs.
* ``TestRouteAfterScore``    — White-box tests of route_after_score in isolation.
* ``TestPersistence``        — save_readiness_score() writes all fields to SQLite
                               and they round-trip correctly.
* ``TestAPIEndpoint``        — Integration test of the FastAPI router using
                               FastAPI TestClient with mocked pipeline.
"""

from __future__ import annotations

import os
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlmodel import Session, SQLModel, create_engine

# ---------------------------------------------------------------------------
# Ensure DATABASE_URL is set before app.core.database is imported
# (unit tests use SQLite in-memory; prevents the fail-fast guard in database.py
#  from raising ValueError when there is no .env in CI)
# ---------------------------------------------------------------------------
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.schemas.readiness_score import (  # noqa: E402
    CandidateProfile,
    ReadinessGapAnalysis,
    ReadinessRequest,
    SubScores,
)
from app.services.readiness_score_service import (  # noqa: E402
    MAX_RETRIES,
    ReadinessState,
    route_after_score,
    run_readiness_pipeline,
    save_readiness_score,
)

# ---------------------------------------------------------------------------
# Shared fixtures & helper data
# ---------------------------------------------------------------------------

VALID_ANALYSIS = ReadinessGapAnalysis(
    overall_score=78,
    sub_scores=SubScores(
        must_have_skills_score=32,
        tools_score=21,
        experience_score=20,
        soft_skills_score=5,
    ),
    critical_gaps=["Kubernetes / container orchestration"],
    nice_to_have_gaps=["GraphQL"],
    strengths=[
        "Strong Python and FastAPI expertise",
        "Docker containerization experience",
        "PostgreSQL database management",
    ],
    explanation=(
        "The candidate demonstrates solid backend engineering skills with a score of 78/100. "
        "The primary blocker is the absence of Kubernetes experience, which is a must-have "
        "for this Senior Backend role. A focused Kubernetes certification would directly "
        "address this critical gap."
    ),
)

SAMPLE_CANDIDATE = CandidateProfile(
    skills=["System design", "API design principles", "Distributed systems"],
    tools=["Python", "FastAPI", "Docker", "PostgreSQL"],
    experience_years=3,
    education=["BSc Computer Science — Cairo University"],
)

SAMPLE_REQUEST = ReadinessRequest(
    benchmark_id=1,
    candidate_profile=SAMPLE_CANDIDATE,
)


def _make_mock_benchmark(
    benchmark_id: int = 1,
    must_have_skills: List[str] | None = None,
    nice_to_have_skills: List[str] | None = None,
    required_tools: List[str] | None = None,
    minimum_years: int = 3,
    seniority_level: str = "Senior",
    common_responsibilities: List[str] | None = None,
) -> MagicMock:
    """Build a mock RoleBenchmarkModel for use in service tests."""
    bm = MagicMock()
    bm.id = benchmark_id
    bm.must_have_skills = must_have_skills or [
        "System design",
        "Distributed systems",
        "API design principles",
    ]
    bm.nice_to_have_skills = nice_to_have_skills or ["GraphQL"]
    bm.required_tools = required_tools or [
        "Python", "FastAPI", "Docker", "PostgreSQL", "Kubernetes"
    ]
    bm.minimum_years = minimum_years
    bm.seniority_level = seniority_level
    bm.common_responsibilities = common_responsibilities or [
        "Design and implement RESTful APIs",
        "Collaborate with cross-functional teams",
    ]
    return bm


def _make_registry_mock(
    complete_side_effects: list,
) -> MagicMock:
    """
    Build a mock ``LLMServiceRegistry`` whose ``complete`` method cycles
    through *complete_side_effects*.

    Each element is either:
    * A ``ReadinessGapAnalysis`` instance → ``complete`` returns it.
    * An ``Exception`` instance           → ``complete`` raises it.
    """
    registry_mock = MagicMock()
    registry_mock.complete = MagicMock(side_effect=complete_side_effects)
    return registry_mock


def _make_validation_error() -> ValidationError:
    """Construct a real Pydantic ``ValidationError`` by triggering missing fields."""
    try:
        ReadinessGapAnalysis.model_validate({})
    except ValidationError as exc:
        return exc
    raise RuntimeError("Expected ValidationError was not raised")  # pragma: no cover


# ---------------------------------------------------------------------------
# Persistence fixture (SQLite in-memory — no Postgres required)
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_session():
    """
    Provide a SQLModel ``Session`` backed by a fresh SQLite in-memory database.

    Both ``role_benchmarks`` and ``readiness_scores`` tables are created before
    yielding and dropped afterwards.  The ``Vector(1536)`` column on
    ``role_benchmarks`` is patched to ``JSON`` so SQLite can store it.
    """
    # Import ORM models so their metadata is registered before create_all()
    from app.models.readiness_score import ReadinessScore as ReadinessScoreModel  # noqa: F401
    from app.models.role_benchmark import RoleBenchmark as RoleBenchmarkModel  # noqa: F401

    sqlite_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    # pgvector's Vector type is not available in SQLite; patch the column
    # to JSON so SQLite can store the embedding as a plain list.
    from sqlalchemy import JSON

    RoleBenchmarkModel.__table__.c["embedding"].type = JSON()

    SQLModel.metadata.create_all(sqlite_engine)

    with Session(sqlite_engine) as session:
        yield session

    SQLModel.metadata.drop_all(sqlite_engine)


@pytest.fixture()
def persisted_benchmark(db_session: Session):
    """Insert a real RoleBenchmarkModel row and return it."""
    from app.models.role_benchmark import RoleBenchmark as RoleBenchmarkModel

    bm = RoleBenchmarkModel(
        must_have_skills=["System design", "Distributed systems", "API design principles"],
        nice_to_have_skills=["GraphQL"],
        required_tools=["Python", "FastAPI", "Docker", "PostgreSQL", "Kubernetes"],
        common_responsibilities=["Design and implement RESTful APIs"],
        minimum_years=3,
        seniority_level="Senior",
        embedding=None,
    )
    db_session.add(bm)
    db_session.commit()
    db_session.refresh(bm)
    return bm


# ---------------------------------------------------------------------------
# 1. Happy-path tests
# ---------------------------------------------------------------------------


class TestHappyPath:
    """LLM succeeds on the first attempt; graph completes without retry."""

    def test_happy_path_returns_analysis(self):
        """``run_readiness_pipeline`` returns the LLM analysis on first try."""
        mock_bm = _make_mock_benchmark()
        registry_mock = _make_registry_mock([VALID_ANALYSIS])

        state = run_readiness_pipeline(
            request=SAMPLE_REQUEST,
            benchmark=mock_bm,
            registry=registry_mock,
        )

        assert state["analysis"] == VALID_ANALYSIS

    def test_happy_path_overall_score_correct(self):
        mock_bm = _make_mock_benchmark()
        registry_mock = _make_registry_mock([VALID_ANALYSIS])

        state = run_readiness_pipeline(
            request=SAMPLE_REQUEST,
            benchmark=mock_bm,
            registry=registry_mock,
        )

        assert state["analysis"].overall_score == 78

    def test_happy_path_sub_scores_correct(self):
        mock_bm = _make_mock_benchmark()
        registry_mock = _make_registry_mock([VALID_ANALYSIS])

        state = run_readiness_pipeline(
            request=SAMPLE_REQUEST,
            benchmark=mock_bm,
            registry=registry_mock,
        )

        sub = state["analysis"].sub_scores
        assert sub.must_have_skills_score == 32
        assert sub.tools_score == 21
        assert sub.experience_score == 20
        assert sub.soft_skills_score == 5

    def test_happy_path_error_count_zero(self):
        mock_bm = _make_mock_benchmark()
        registry_mock = _make_registry_mock([VALID_ANALYSIS])

        state = run_readiness_pipeline(
            request=SAMPLE_REQUEST,
            benchmark=mock_bm,
            registry=registry_mock,
        )

        assert state["error_count"] == 0
        assert state["validation_errors"] == []

    def test_llm_invoked_exactly_once(self):
        mock_bm = _make_mock_benchmark()
        registry_mock = _make_registry_mock([VALID_ANALYSIS])

        run_readiness_pipeline(
            request=SAMPLE_REQUEST,
            benchmark=mock_bm,
            registry=registry_mock,
        )

        assert registry_mock.complete.call_count == 1

    def test_critical_gaps_propagated(self):
        mock_bm = _make_mock_benchmark()
        registry_mock = _make_registry_mock([VALID_ANALYSIS])

        state = run_readiness_pipeline(
            request=SAMPLE_REQUEST,
            benchmark=mock_bm,
            registry=registry_mock,
        )

        assert "Kubernetes / container orchestration" in state["analysis"].critical_gaps

    def test_nice_to_have_gaps_propagated(self):
        mock_bm = _make_mock_benchmark()
        registry_mock = _make_registry_mock([VALID_ANALYSIS])

        state = run_readiness_pipeline(
            request=SAMPLE_REQUEST,
            benchmark=mock_bm,
            registry=registry_mock,
        )

        assert "GraphQL" in state["analysis"].nice_to_have_gaps

    def test_strengths_propagated(self):
        mock_bm = _make_mock_benchmark()
        registry_mock = _make_registry_mock([VALID_ANALYSIS])

        state = run_readiness_pipeline(
            request=SAMPLE_REQUEST,
            benchmark=mock_bm,
            registry=registry_mock,
        )

        assert len(state["analysis"].strengths) >= 1

    def test_explanation_is_non_empty_string(self):
        mock_bm = _make_mock_benchmark()
        registry_mock = _make_registry_mock([VALID_ANALYSIS])

        state = run_readiness_pipeline(
            request=SAMPLE_REQUEST,
            benchmark=mock_bm,
            registry=registry_mock,
        )

        assert isinstance(state["analysis"].explanation, str)
        assert len(state["analysis"].explanation) > 0


# ---------------------------------------------------------------------------
# 2. Retry-then-succeed tests
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Registry raises on the first call, then succeeds; graph loops once."""

    def setup_method(self):
        self.mock_bm = _make_mock_benchmark()
        self.registry_mock = _make_registry_mock(
            [RuntimeError("LLM output did not match schema"), VALID_ANALYSIS]
        )
        self.state = run_readiness_pipeline(
            request=SAMPLE_REQUEST,
            benchmark=self.mock_bm,
            registry=self.registry_mock,
        )

    def test_retry_logic_recovers(self):
        """complete called twice (fail + succeed); 1 error recorded."""
        assert self.registry_mock.complete.call_count == 2
        assert self.state["error_count"] == 1
        assert len(self.state["validation_errors"]) >= 1

    def test_final_analysis_is_valid(self):
        assert self.state["analysis"] == VALID_ANALYSIS

    def test_overall_score_correct_after_retry(self):
        assert self.state["analysis"].overall_score == 78


# ---------------------------------------------------------------------------
# 3. Exhausted-retries tests
# ---------------------------------------------------------------------------


class TestExhaustedRetries:
    """Registry raises on all MAX_RETRIES attempts; service raises ValueError."""

    def test_exhausted_retries_raises_value_error(self):
        """``ValueError`` raised after 3 failed attempts."""
        mock_bm = _make_mock_benchmark()
        registry_mock = _make_registry_mock(
            [RuntimeError("LLM output did not match schema")] * MAX_RETRIES
        )

        with pytest.raises(ValueError, match="failed after"):
            run_readiness_pipeline(
                request=SAMPLE_REQUEST,
                benchmark=mock_bm,
                registry=registry_mock,
            )

    def test_llm_invoked_exactly_max_retries_times(self):
        mock_bm = _make_mock_benchmark()
        registry_mock = _make_registry_mock(
            [RuntimeError("LLM error")] * MAX_RETRIES
        )

        with pytest.raises(ValueError):
            run_readiness_pipeline(
                request=SAMPLE_REQUEST,
                benchmark=mock_bm,
                registry=registry_mock,
            )

        assert registry_mock.complete.call_count == MAX_RETRIES

    def test_error_message_references_max_retries(self):
        mock_bm = _make_mock_benchmark()
        registry_mock = _make_registry_mock(
            [RuntimeError("LLM error")] * MAX_RETRIES
        )

        with pytest.raises(ValueError) as exc_info:
            run_readiness_pipeline(
                request=SAMPLE_REQUEST,
                benchmark=mock_bm,
                registry=registry_mock,
            )

        assert "failed after" in str(exc_info.value).lower()

    def test_validation_errors_listed_in_exception(self):
        mock_bm = _make_mock_benchmark()
        registry_mock = _make_registry_mock(
            [RuntimeError(f"err{i}") for i in range(MAX_RETRIES)]
        )

        with pytest.raises(ValueError) as exc_info:
            run_readiness_pipeline(
                request=SAMPLE_REQUEST,
                benchmark=mock_bm,
                registry=registry_mock,
            )

        assert "err0" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 4. Route-function unit tests (state-machine logic in isolation)
# ---------------------------------------------------------------------------


class TestRouteAfterScore:
    """Directly test the conditional-edge router function."""

    def _make_state(
        self,
        analysis=None,
        error_count: int = 0,
        validation_errors: List[str] | None = None,
    ) -> ReadinessState:
        return {
            "request": SAMPLE_REQUEST,
            "benchmark": _make_mock_benchmark(),
            "analysis": analysis,
            "error_count": error_count,
            "validation_errors": validation_errors or [],
        }

    def test_routes_to_end_on_success(self):
        """When analysis is populated, route to END."""
        from langgraph.graph import END

        state = self._make_state(analysis=VALID_ANALYSIS)
        assert route_after_score(state) == END

    def test_routes_back_to_score_on_first_failure(self):
        state = self._make_state(
            analysis=None,
            error_count=1,
            validation_errors=["field required"],
        )
        assert route_after_score(state) == "score"

    def test_routes_back_on_second_failure(self):
        state = self._make_state(
            analysis=None,
            error_count=2,
            validation_errors=["err1", "err2"],
        )
        assert route_after_score(state) == "score"

    def test_raises_after_max_retries(self):
        state = self._make_state(
            analysis=None,
            error_count=MAX_RETRIES,
            validation_errors=["err1", "err2", "err3"],
        )
        with pytest.raises(ValueError):
            route_after_score(state)

    def test_no_retry_when_analysis_present(self):
        """Even with a positive error_count, END is returned if analysis exists."""
        from langgraph.graph import END

        state = self._make_state(analysis=VALID_ANALYSIS, error_count=1)
        assert route_after_score(state) == END


# ---------------------------------------------------------------------------
# 5. Persistence tests (SQLite in-memory — no Postgres required)
# ---------------------------------------------------------------------------


class TestPersistence:
    """
    Verify that ``save_readiness_score()`` correctly writes all fields to the
    database and they round-trip correctly when re-read.
    """

    def _make_state(
        self,
        analysis: ReadinessGapAnalysis = VALID_ANALYSIS,
    ) -> ReadinessState:
        return {
            "request": SAMPLE_REQUEST,
            "benchmark": _make_mock_benchmark(),
            "analysis": analysis,
            "error_count": 0,
            "validation_errors": [],
        }

    def test_save_returns_record_with_id(self, db_session: Session, persisted_benchmark):
        """``save_readiness_score`` must return a record with a positive id."""
        req = ReadinessRequest(
            benchmark_id=persisted_benchmark.id,
            candidate_profile=SAMPLE_CANDIDATE,
        )
        state = {**self._make_state(), "request": req}
        record = save_readiness_score(state, db_session)

        assert record.id is not None
        assert record.id > 0

    def test_overall_score_persisted_correctly(self, db_session: Session, persisted_benchmark):
        req = ReadinessRequest(
            benchmark_id=persisted_benchmark.id,
            candidate_profile=SAMPLE_CANDIDATE,
        )
        state = {**self._make_state(), "request": req}
        record = save_readiness_score(state, db_session)

        assert record.overall_score == VALID_ANALYSIS.overall_score

    def test_sub_scores_persisted_correctly(self, db_session: Session, persisted_benchmark):
        req = ReadinessRequest(
            benchmark_id=persisted_benchmark.id,
            candidate_profile=SAMPLE_CANDIDATE,
        )
        state = {**self._make_state(), "request": req}
        record = save_readiness_score(state, db_session)

        assert record.must_have_skills_score == VALID_ANALYSIS.sub_scores.must_have_skills_score
        assert record.tools_score == VALID_ANALYSIS.sub_scores.tools_score
        assert record.experience_score == VALID_ANALYSIS.sub_scores.experience_score
        assert record.soft_skills_score == VALID_ANALYSIS.sub_scores.soft_skills_score

    def test_critical_gaps_persisted_correctly(self, db_session: Session, persisted_benchmark):
        req = ReadinessRequest(
            benchmark_id=persisted_benchmark.id,
            candidate_profile=SAMPLE_CANDIDATE,
        )
        state = {**self._make_state(), "request": req}
        record = save_readiness_score(state, db_session)

        assert record.critical_gaps == VALID_ANALYSIS.critical_gaps

    def test_nice_to_have_gaps_persisted_correctly(
        self, db_session: Session, persisted_benchmark
    ):
        req = ReadinessRequest(
            benchmark_id=persisted_benchmark.id,
            candidate_profile=SAMPLE_CANDIDATE,
        )
        state = {**self._make_state(), "request": req}
        record = save_readiness_score(state, db_session)

        assert record.nice_to_have_gaps == VALID_ANALYSIS.nice_to_have_gaps

    def test_strengths_persisted_correctly(self, db_session: Session, persisted_benchmark):
        req = ReadinessRequest(
            benchmark_id=persisted_benchmark.id,
            candidate_profile=SAMPLE_CANDIDATE,
        )
        state = {**self._make_state(), "request": req}
        record = save_readiness_score(state, db_session)

        assert record.strengths == VALID_ANALYSIS.strengths

    def test_explanation_persisted_correctly(self, db_session: Session, persisted_benchmark):
        req = ReadinessRequest(
            benchmark_id=persisted_benchmark.id,
            candidate_profile=SAMPLE_CANDIDATE,
        )
        state = {**self._make_state(), "request": req}
        record = save_readiness_score(state, db_session)

        assert record.explanation == VALID_ANALYSIS.explanation

    def test_candidate_audit_snapshot_persisted(
        self, db_session: Session, persisted_benchmark
    ):
        """Candidate skills, tools, and experience_years must be stored as audit trail."""
        req = ReadinessRequest(
            benchmark_id=persisted_benchmark.id,
            candidate_profile=SAMPLE_CANDIDATE,
        )
        state = {**self._make_state(), "request": req}
        record = save_readiness_score(state, db_session)

        assert record.candidate_skills == SAMPLE_CANDIDATE.skills
        assert record.candidate_tools == SAMPLE_CANDIDATE.tools
        assert record.candidate_experience_years == SAMPLE_CANDIDATE.experience_years

    def test_benchmark_id_persisted_correctly(
        self, db_session: Session, persisted_benchmark
    ):
        req = ReadinessRequest(
            benchmark_id=persisted_benchmark.id,
            candidate_profile=SAMPLE_CANDIDATE,
        )
        state = {**self._make_state(), "request": req}
        record = save_readiness_score(state, db_session)

        assert record.benchmark_id == persisted_benchmark.id

    def test_created_at_is_populated(self, db_session: Session, persisted_benchmark):
        """created_at must be a non-None datetime after persistence."""
        from datetime import datetime

        req = ReadinessRequest(
            benchmark_id=persisted_benchmark.id,
            candidate_profile=SAMPLE_CANDIDATE,
        )
        state = {**self._make_state(), "request": req}
        record = save_readiness_score(state, db_session)

        assert record.created_at is not None
        assert isinstance(record.created_at, datetime)

    def test_reviewed_at_is_none_by_default(
        self, db_session: Session, persisted_benchmark
    ):
        req = ReadinessRequest(
            benchmark_id=persisted_benchmark.id,
            candidate_profile=SAMPLE_CANDIDATE,
        )
        state = {**self._make_state(), "request": req}
        record = save_readiness_score(state, db_session)

        assert record.reviewed_at is None

    def test_raises_when_analysis_is_none(self, db_session: Session, persisted_benchmark):
        """save_readiness_score must raise ValueError if analysis is None."""
        bad_state: ReadinessState = {
            "request": ReadinessRequest(
                benchmark_id=persisted_benchmark.id,
                candidate_profile=SAMPLE_CANDIDATE,
            ),
            "benchmark": _make_mock_benchmark(),
            "analysis": None,
            "error_count": 3,
            "validation_errors": ["err1", "err2", "err3"],
        }

        with pytest.raises(ValueError, match="analysis is None"):
            save_readiness_score(bad_state, db_session)

    def test_multiple_records_have_distinct_ids(
        self, db_session: Session, persisted_benchmark
    ):
        """Each call to save_readiness_score must produce a unique primary key."""
        req = ReadinessRequest(
            benchmark_id=persisted_benchmark.id,
            candidate_profile=SAMPLE_CANDIDATE,
        )
        state_a = {**self._make_state(), "request": req}
        state_b = {**self._make_state(), "request": req}

        record_a = save_readiness_score(state_a, db_session)
        record_b = save_readiness_score(state_b, db_session)

        assert record_a.id != record_b.id

    def test_data_round_trips_from_db(self, db_session: Session, persisted_benchmark):
        """Force-reload from DB and confirm overall_score survives the round-trip."""
        from app.models.readiness_score import ReadinessScore as ReadinessScoreModel

        req = ReadinessRequest(
            benchmark_id=persisted_benchmark.id,
            candidate_profile=SAMPLE_CANDIDATE,
        )
        state = {**self._make_state(), "request": req}
        record = save_readiness_score(state, db_session)

        db_session.expire(record)  # force reload on next access
        reloaded = db_session.get(ReadinessScoreModel, record.id)

        assert reloaded is not None
        assert reloaded.overall_score == VALID_ANALYSIS.overall_score
        assert reloaded.critical_gaps == VALID_ANALYSIS.critical_gaps


# ---------------------------------------------------------------------------
# 6. FastAPI endpoint integration tests
# ---------------------------------------------------------------------------


class TestAPIEndpoint:
    """
    Integration tests for POST /api/v1/readiness/score using FastAPI TestClient.

    The LangGraph pipeline and DB session are both mocked so these tests run
    without network access or a running database.
    """

    @pytest.fixture(autouse=True)
    def _setup_client(self, db_session: Session, persisted_benchmark):
        """Set up TestClient with overridden DB session and pipeline mock."""
        from app.core.database import get_session

        def _override_session():
            yield db_session

        from main import app

        app.dependency_overrides[get_session] = _override_session
        self.client = TestClient(app)
        self.benchmark_id = persisted_benchmark.id
        self.valid_body = {
            "benchmark_id": self.benchmark_id,
            "candidate_profile": {
                "skills": ["System design", "API design principles"],
                "tools": ["Python", "FastAPI", "Docker"],
                "experience_years": 3,
                "education": ["BSc Computer Science"],
            },
        }
        yield
        app.dependency_overrides.clear()

    def test_returns_201_on_success(self):
        """Endpoint must return HTTP 201 when the pipeline succeeds."""
        with patch(
            "app.api.v1.readiness.run_readiness_pipeline",
            return_value={
                "request": ReadinessRequest(**self.valid_body),
                "benchmark": _make_mock_benchmark(self.benchmark_id),
                "analysis": VALID_ANALYSIS,
                "error_count": 0,
                "validation_errors": [],
            },
        ), patch(
            "app.api.v1.readiness.save_readiness_score",
            return_value=MagicMock(
                id=99,
                benchmark_id=self.benchmark_id,
                overall_score=78,
                must_have_skills_score=32,
                tools_score=21,
                experience_score=20,
                soft_skills_score=5,
                critical_gaps=["Kubernetes / container orchestration"],
                nice_to_have_gaps=["GraphQL"],
                strengths=["Strong Python and FastAPI expertise"],
                explanation="Assessment text.",
                candidate_skills=["System design"],
                candidate_tools=["Python"],
                candidate_experience_years=3,
                created_at=__import__("datetime").datetime.utcnow(),
                reviewed_at=None,
            ),
        ):
            response = self.client.post(
                "/api/v1/readiness/score",
                json=self.valid_body,
            )

        assert response.status_code == 201

    def test_returns_404_when_benchmark_missing(self):
        """Endpoint must return HTTP 404 if benchmark_id does not exist in DB."""
        body = {
            "benchmark_id": 999999,
            "candidate_profile": self.valid_body["candidate_profile"],
        }
        response = self.client.post("/api/v1/readiness/score", json=body)
        assert response.status_code == 404

    def test_response_body_has_overall_score(self):
        """Response JSON must contain ``overall_score``."""
        with patch(
            "app.api.v1.readiness.run_readiness_pipeline",
            return_value={
                "request": ReadinessRequest(**self.valid_body),
                "benchmark": _make_mock_benchmark(self.benchmark_id),
                "analysis": VALID_ANALYSIS,
                "error_count": 0,
                "validation_errors": [],
            },
        ), patch(
            "app.api.v1.readiness.save_readiness_score",
            return_value=MagicMock(
                id=99,
                benchmark_id=self.benchmark_id,
                overall_score=78,
                must_have_skills_score=32,
                tools_score=21,
                experience_score=20,
                soft_skills_score=5,
                critical_gaps=[],
                nice_to_have_gaps=[],
                strengths=[],
                explanation="Test.",
                candidate_skills=[],
                candidate_tools=[],
                candidate_experience_years=3,
                created_at=__import__("datetime").datetime.utcnow(),
                reviewed_at=None,
            ),
        ):
            response = self.client.post(
                "/api/v1/readiness/score",
                json=self.valid_body,
            )

        data = response.json()
        assert "overall_score" in data

    def test_response_contains_x_readiness_score_id_header(self):
        """Response must carry ``X-Readiness-Score-Id`` header."""
        with patch(
            "app.api.v1.readiness.run_readiness_pipeline",
            return_value={
                "request": ReadinessRequest(**self.valid_body),
                "benchmark": _make_mock_benchmark(self.benchmark_id),
                "analysis": VALID_ANALYSIS,
                "error_count": 0,
                "validation_errors": [],
            },
        ), patch(
            "app.api.v1.readiness.save_readiness_score",
            return_value=MagicMock(
                id=42,
                benchmark_id=self.benchmark_id,
                overall_score=78,
                must_have_skills_score=32,
                tools_score=21,
                experience_score=20,
                soft_skills_score=5,
                critical_gaps=[],
                nice_to_have_gaps=[],
                strengths=[],
                explanation="Test.",
                candidate_skills=[],
                candidate_tools=[],
                candidate_experience_years=3,
                created_at=__import__("datetime").datetime.utcnow(),
                reviewed_at=None,
            ),
        ):
            response = self.client.post(
                "/api/v1/readiness/score",
                json=self.valid_body,
            )

        assert "x-readiness-score-id" in response.headers

    def test_invalid_request_returns_422(self):
        """Missing required fields must yield 422 Unprocessable Entity."""
        response = self.client.post(
            "/api/v1/readiness/score",
            json={"benchmark_id": 1},  # missing candidate_profile
        )
        assert response.status_code == 422

    def test_pipeline_value_error_returns_502(self):
        """ValueError from the pipeline must map to HTTP 502."""
        with patch(
            "app.api.v1.readiness.run_readiness_pipeline",
            side_effect=ValueError("LLM scoring failed after 3 attempts"),
        ):
            response = self.client.post(
                "/api/v1/readiness/score",
                json=self.valid_body,
            )
        assert response.status_code == 502
