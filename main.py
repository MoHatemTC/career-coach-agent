from __future__ import annotations

from dotenv import load_dotenv

# Load .env before any app imports so DATABASE_URL and OPENAI_API_KEY are available.
load_dotenv()

from fastapi import FastAPI
from sqlmodel import SQLModel

from app.core.database import engine
from app.api.v1.benchmarks import router as benchmarks_router

app = FastAPI(
    title="Target Role Benchmark Engine",
    description=(
        "Extracts structured skill and experience benchmarks from raw job "
        "descriptions using an LLM pipeline (GPT-4o-mini + LangGraph)."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def on_startup() -> None:
    """Create all SQLModel tables if they don't exist yet."""
    SQLModel.metadata.create_all(engine)


app.include_router(benchmarks_router, prefix="/api/v1")


@app.get("/health", tags=["health"])
def health_check() -> dict:
    return {"status": "ok"}
