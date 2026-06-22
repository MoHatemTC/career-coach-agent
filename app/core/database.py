from __future__ import annotations

import os
from collections.abc import Generator

from sqlmodel import Session, create_engine

DATABASE_URL: str | None = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL environment variable is not set. "
        "Copy .env.example to .env and fill in your connection string.\n\n"
        "Example (PostgreSQL):\n"
        "  DATABASE_URL=postgresql://user:password@localhost:5432/career_coach\n\n"
        "Example (SQLite for local dev):\n"
        "  DATABASE_URL=sqlite:///./career_coach.db"
    )

_is_sqlite = DATABASE_URL.startswith("sqlite")

_engine_kwargs: dict = dict(
    pool_pre_ping=True,
    echo=False,
)

if not _is_sqlite:
    _engine_kwargs.update(pool_size=5, max_overflow=10)

engine = create_engine(DATABASE_URL, **_engine_kwargs)


def get_session() -> Generator[Session, None, None]:
    """Yield a SQLModel Session scoped to a single HTTP request."""
    with Session(engine) as session:
        yield session
