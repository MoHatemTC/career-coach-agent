"""
scripts/e2e_readiness_test.py
==============================
End-to-end demonstration of the Career Readiness Score engine.

Modes
-----
LIVE    – Makes real LLM calls via the LiteLLM proxy.
           Set LITELLM_API_KEY (and optionally LITELLM_BASE_URL / DEFAULT_MODEL)
           as environment variables or in a .env file before running.

OFFLINE – Uses mocked LLM responses (no API key required).
           Activated automatically when LITELLM_API_KEY is not set.

Usage
-----
    # LIVE mode (requires .env or exported env vars):
    d:/freelance/career-coach/.venv/Scripts/python.exe scripts/e2e_readiness_test.py

    # OFFLINE mode (always works):
    d:/freelance/career-coach/.venv/Scripts/python.exe scripts/e2e_readiness_test.py --offline

Flow
----
1.  Spin up an in-memory SQLite database (no Postgres required).
2.  Create a RoleBenchmark row directly (bypasses LLM for the benchmark step
    so the demo focuses exclusively on the readiness engine).
3.  Call run_readiness_pipeline() with a sample candidate profile.
4.  Call save_readiness_score() to persist the result.
5.  Print a rich formatted report of the assessment.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from datetime import datetime
from typing import List
from unittest.mock import MagicMock

# Force UTF-8 output on Windows so Unicode symbols render correctly
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Ensure DATABASE_URL is set before any app import ─────────────────────────
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

# ── Load .env if present ──────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; use env vars directly


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2E Career Readiness Score demo")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline mode (mocked LLM, no API key needed)",
    )
    return parser.parse_args()


# ── Determine run mode ────────────────────────────────────────────────────────
_args = _parse_args()
_LIVE = bool(os.getenv("LITELLM_API_KEY")) and not _args.offline
_MODE = "LIVE" if _LIVE else "OFFLINE"

# ── App imports (after DATABASE_URL is set) ───────────────────────────────────
from sqlalchemy import JSON
from sqlmodel import Session, SQLModel, create_engine

from app.models.readiness_score import ReadinessScore as ReadinessScoreModel
from app.models.role_benchmark import RoleBenchmark as RoleBenchmarkModel
from app.schemas.readiness_score import (
    CandidateProfile,
    ReadinessGapAnalysis,
    ReadinessRequest,
    SubScores,
)
from app.services.readiness_score_service import (
    run_readiness_pipeline,
    save_readiness_score,
)

# ── Sample data ───────────────────────────────────────────────────────────────

JOB_TITLE = "Senior Backend Engineer (Python / Cloud)"

BENCHMARK_DATA = {
    "must_have_skills": [
        "System design",
        "Distributed systems",
        "API design principles",
        "Microservices architecture",
    ],
    "nice_to_have_skills": [
        "GraphQL",
        "Event-driven architecture",
    ],
    "required_tools": [
        "Python",
        "FastAPI",
        "Docker",
        "PostgreSQL",
        "Kubernetes",
        "AWS",
    ],
    "common_responsibilities": [
        "Design and implement RESTful APIs at scale",
        "Collaborate with cross-functional teams",
        "Conduct code reviews and mentor junior engineers",
        "Maintain and improve CI/CD pipelines",
    ],
    "minimum_years": 4,
    "seniority_level": "Senior",
}

CANDIDATE_PROFILE = CandidateProfile(
    skills=[
        "System design",
        "API design principles",
        "Distributed systems",
        "Agile methodology",
    ],
    tools=[
        "Python",
        "FastAPI",
        "Docker",
        "PostgreSQL",
        "Redis",
    ],
    experience_years=3,
    education=["BSc Computer Science — Cairo University"],
)

# ── Offline mock result ───────────────────────────────────────────────────────

MOCK_ANALYSIS = ReadinessGapAnalysis(
    overall_score=71,
    sub_scores=SubScores(
        must_have_skills_score=35,
        tools_score=17,
        experience_score=14,
        soft_skills_score=5,
    ),
    critical_gaps=[
        "Kubernetes / container orchestration",
        "AWS cloud platform experience",
    ],
    nice_to_have_gaps=[
        "GraphQL",
        "Event-driven architecture",
    ],
    strengths=[
        "Strong Python and FastAPI expertise matching core backend requirements",
        "Docker containerisation experience",
        "PostgreSQL database management",
        "System design and distributed systems conceptual knowledge",
    ],
    explanation=(
        "The candidate scores 71/100 against this Senior Backend Engineer benchmark. "
        "The primary blockers are the absence of Kubernetes and AWS experience — both "
        "are must-have tools for managing containerised cloud workloads at this seniority level. "
        "Additionally, the candidate's 3 years of experience falls one year short of the "
        "4-year minimum, reducing the experience sub-score. Pursuing an AWS Solutions Architect "
        "Associate certification alongside a hands-on Kubernetes project would directly address "
        "the two most critical gaps and substantially improve hiring readiness."
    ),
)


# ── Rendering helpers ─────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
DIM = "\033[2m"


def _bar(score: int, max_score: int, width: int = 30) -> str:
    filled = int(width * score / max_score)
    bar = "█" * filled + "░" * (width - filled)
    pct = score / max_score * 100
    color = GREEN if pct >= 75 else YELLOW if pct >= 50 else RED
    return f"{color}{bar}{RESET} {score}/{max_score}"


def _score_color(score: int) -> str:
    if score >= 80:
        return GREEN
    elif score >= 60:
        return YELLOW
    else:
        return RED


def _bullet_list(items: List[str], color: str = RESET, indent: int = 4) -> str:
    if not items:
        return f"{' ' * indent}{DIM}(none){RESET}"
    return "\n".join(f"{' ' * indent}{color}•{RESET} {item}" for item in items)


def _print_header(text: str) -> None:
    width = 72
    print()
    print(f"{BOLD}{CYAN}{'─' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * width}{RESET}")


def _print_report(
    analysis: ReadinessGapAnalysis,
    db_record: ReadinessScoreModel,
    mode: str,
    elapsed_ms: float,
) -> None:
    score_color = _score_color(analysis.overall_score)
    sub = analysis.sub_scores

    _print_header("CAREER READINESS ASSESSMENT REPORT")

    print(f"\n  {BOLD}Target Role   :{RESET} {JOB_TITLE}")
    print(f"  {BOLD}Mode          :{RESET} {MAGENTA}{mode}{RESET}")
    print(f"  {BOLD}Score ID      :{RESET} {db_record.id}")
    print(f"  {BOLD}Benchmark ID  :{RESET} {db_record.benchmark_id}")
    print(f"  {BOLD}Scored At     :{RESET} {db_record.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  {BOLD}Pipeline time :{RESET} {elapsed_ms:.0f} ms")

    _print_header("OVERALL READINESS SCORE")
    print(f"\n  {score_color}{BOLD}{analysis.overall_score} / 100{RESET}\n")
    print(f"  {_bar(analysis.overall_score, 100)}\n")

    _print_header("DIMENSIONAL SUB-SCORES")
    print(f"\n  {'Must-Have Skills':22} {_bar(sub.must_have_skills_score, 40)}")
    print(f"  {'Required Tools':22} {_bar(sub.tools_score, 25)}")
    print(f"  {'Experience Level':22} {_bar(sub.experience_score, 25)}")
    print(f"  {'Soft Skills / Edu':22} {_bar(sub.soft_skills_score, 10)}")

    _print_header("CRITICAL GAPS  ⚠  (must-have skills / tools the candidate lacks)")
    print()
    print(_bullet_list(analysis.critical_gaps, color=RED))

    _print_header("NICE-TO-HAVE GAPS  (growth opportunities)")
    print()
    print(_bullet_list(analysis.nice_to_have_gaps, color=YELLOW))

    _print_header("STRENGTHS  ✓  (candidate meets or exceeds benchmark)")
    print()
    print(_bullet_list(analysis.strengths, color=GREEN))

    _print_header("ASSESSMENT EXPLANATION")
    print()
    wrapped = textwrap.fill(
        analysis.explanation,
        width=68,
        initial_indent="  ",
        subsequent_indent="  ",
    )
    print(wrapped)

    _print_header("CANDIDATE AUDIT SNAPSHOT")
    print(f"\n  Experience years : {db_record.candidate_experience_years}")
    print(f"  Skills           : {', '.join(db_record.candidate_skills)}")
    print(f"  Tools            : {', '.join(db_record.candidate_tools)}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{BOLD}  Career Readiness Score — End-to-End Demo  [{_MODE}]{RESET}")
    print(f"  {DIM}SQLite in-memory DB (no Postgres required){RESET}")

    # ── 1. Set up in-memory SQLite database ───────────────────────────────────
    print(f"\n{DIM}[1/4] Setting up in-memory SQLite database…{RESET}")
    sqlite_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    # pgvector's Vector type not available in SQLite — patch to JSON
    RoleBenchmarkModel.__table__.c["embedding"].type = JSON()
    SQLModel.metadata.create_all(sqlite_engine)

    # ── 2. Insert a benchmark row ─────────────────────────────────────────────
    print(f"{DIM}[2/4] Inserting sample RoleBenchmark row…{RESET}")
    with Session(sqlite_engine) as session:
        bm = RoleBenchmarkModel(**BENCHMARK_DATA, embedding=None)
        session.add(bm)
        session.commit()
        session.refresh(bm)
        benchmark_id = bm.id
        print(f"      Benchmark created  ->  id={benchmark_id}  ({bm.seniority_level}, {bm.minimum_years}+ yrs)")

    # ── 3. Run the readiness pipeline ─────────────────────────────────────────
    print(f"{DIM}[3/4] Running readiness pipeline [{_MODE}]…{RESET}")

    request = ReadinessRequest(
        benchmark_id=benchmark_id,
        candidate_profile=CANDIDATE_PROFILE,
    )

    registry_arg = None  # use real LLMServiceRegistry in LIVE mode

    if not _LIVE:
        # Build a mock registry that returns MOCK_ANALYSIS
        mock_registry = MagicMock()
        mock_registry.complete = MagicMock(return_value=MOCK_ANALYSIS)
        registry_arg = mock_registry

    # Load the benchmark ORM record
    with Session(sqlite_engine) as session:
        benchmark_record = session.get(RoleBenchmarkModel, benchmark_id)

        t_start = datetime.utcnow()
        final_state = run_readiness_pipeline(
            request=request,
            benchmark=benchmark_record,
            registry=registry_arg,
        )
        elapsed_ms = (datetime.utcnow() - t_start).total_seconds() * 1000

        # ── 4. Persist the result ─────────────────────────────────────────────
        print(f"{DIM}[4/4] Persisting readiness score to SQLite…{RESET}")
        db_record = save_readiness_score(state=final_state, session=session)

    # ── 5. Print report ───────────────────────────────────────────────────────
    _print_report(
        analysis=final_state["analysis"],
        db_record=db_record,
        mode=_MODE,
        elapsed_ms=elapsed_ms,
    )


if __name__ == "__main__":
    main()
