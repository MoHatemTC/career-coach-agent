from __future__ import annotations

from dotenv import load_dotenv

# Load .env before any app imports so DATABASE_URL and OPENAI_API_KEY are available.
load_dotenv()

from fastapi import FastAPI
from sqlmodel import SQLModel

from app.core.database import engine
from app.api.v1.benchmarks import router as benchmarks_router
from app.api.v1.readiness import router as readiness_router

app = FastAPI(
    title="Career Coach Agent API",
    description=(
        "AI-powered career coaching platform.\n\n"
        "**Endpoints:**\n"
        "- `POST /api/benchmarks/analyze` — Extract a structured role benchmark "
        "from a raw job description.\n"
        "- `POST /api/readiness/score` — Score a candidate's readiness against a "
        "benchmark and generate a gap analysis."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def on_startup() -> None:
    """Create all SQLModel tables if they don't exist yet.

    When running against SQLite (local dev / demo mode) the pgvector
    ``Vector`` column type is unavailable.  We patch the ``embedding``
    column to plain ``JSON`` so records with ``null`` embeddings can be
    read back without errors.  This patch is a no-op against PostgreSQL.
    """
    import os
    from app.models.role_benchmark import RoleBenchmark as RoleBenchmarkModel
    from app.models.readiness_score import ReadinessScore as ReadinessScoreModel  # noqa: F401 — registers table
    db_url: str = os.getenv("DATABASE_URL", "")
    if db_url.startswith("sqlite"):
        from sqlalchemy import JSON
        RoleBenchmarkModel.__table__.c["embedding"].type = JSON()
    SQLModel.metadata.create_all(engine)


app.include_router(benchmarks_router, prefix="/api/v1")
app.include_router(readiness_router, prefix="/api/v1")



@app.get("/health", tags=["health"])
def health_check() -> dict:
    return {"status": "ok"}
