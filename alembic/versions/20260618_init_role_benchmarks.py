"""
alembic/versions/20260618_init_role_benchmarks.py
==================================================
Initial migration: creates the ``role_benchmarks`` table.

Revision ID : 20260618_init
Revises     : (none — first migration)
Create Date : 2026-06-18

Notes
-----
* Requires the ``vector`` PostgreSQL extension (pgvector).  The upgrade
  function installs it with ``CREATE EXTENSION IF NOT EXISTS vector`` so the
  migration is idempotent and safe to run on a fresh database.
* The embedding column uses ``vector(1536)`` to match OpenAI's
  ``text-embedding-3-small`` output dimensionality.
* Downgrade removes the entire table but intentionally leaves the ``vector``
  extension in place — other tables may depend on it.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# ---------------------------------------------------------------------------
# Alembic metadata
# ---------------------------------------------------------------------------

revision: str = "20260618_init"
down_revision: str | None = None   # first migration in the chain
branch_labels: str | tuple | None = None
depends_on: str | tuple | None = None


# ---------------------------------------------------------------------------
# Migration helpers
# ---------------------------------------------------------------------------

_TABLE_NAME = "role_benchmarks"
_VECTOR_DIM = 1536   # text-embedding-3-small produces 1536-dim vectors


# ---------------------------------------------------------------------------
# upgrade
# ---------------------------------------------------------------------------


def upgrade() -> None:
    """
    Create the ``vector`` extension and the ``role_benchmarks`` table.

    Column mapping vs. the SQLModel ORM
    ------------------------------------
    id                   INTEGER   PRIMARY KEY AUTOINCREMENT
    must_have_skills     JSON      NOT NULL  DEFAULT '[]'
    nice_to_have_skills  JSON      NOT NULL  DEFAULT '[]'
    required_tools       JSON      NOT NULL  DEFAULT '[]'
    common_responsibilities JSON   NOT NULL  DEFAULT '[]'
    minimum_years        INTEGER   NOT NULL  DEFAULT 0
    seniority_level      TEXT      NOT NULL  DEFAULT ''
    embedding            vector(1536)        NULL
    created_at           TIMESTAMP NOT NULL  DEFAULT now()
    """
    # 1. Ensure the pgvector extension is available.  Using IF NOT EXISTS makes
    #    this migration safe to apply on a database that already has the
    #    extension installed (e.g. a shared dev cluster).
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 2. Create the table.  The ``vector`` type is a custom PostgreSQL type
    #    registered by the pgvector extension, so we use sa.Text() with an
    #    explicit type-string override for the column definition that Alembic
    #    sends to the database.  At the SQLAlchemy/pgvector layer this is
    #    handled transparently via pgvector.sqlalchemy.Vector, but raw DDL
    #    migrations must spell it out.
    op.create_table(
        _TABLE_NAME,
        # Primary key
        sa.Column(
            "id",
            sa.Integer(),
            primary_key=True,
            autoincrement=True,
            nullable=False,
        ),
        # JSON skill / tool arrays
        sa.Column(
            "must_have_skills",
            sa.JSON(),
            nullable=False,
            server_default="'[]'::json",
        ),
        sa.Column(
            "nice_to_have_skills",
            sa.JSON(),
            nullable=False,
            server_default="'[]'::json",
        ),
        sa.Column(
            "required_tools",
            sa.JSON(),
            nullable=False,
            server_default="'[]'::json",
        ),
        sa.Column(
            "common_responsibilities",
            sa.JSON(),
            nullable=False,
            server_default="'[]'::json",
        ),
        # Experience fields
        sa.Column(
            "minimum_years",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "seniority_level",
            sa.Text(),
            nullable=False,
            server_default=sa.text("''"),
        ),
        # pgvector embedding column — uses the custom vector(dim) type
        sa.Column(
            "embedding",
            sa.Text().with_variant(
                # Alembic renders the raw type string for the DDL statement.
                # pgvector's vector type is not natively known to SQLAlchemy,
                # so we pass the PostgreSQL type literal directly.
                sa.dialects.postgresql.base.ischema_names.get(  # type: ignore[attr-defined]
                    "vector", sa.Text
                )(),
                "postgresql",
            ),
            nullable=True,
            comment=f"pgvector {_VECTOR_DIM}-dim embedding of skills + tools",
        ),
        # Audit timestamp
        sa.Column(
            "created_at",
            sa.DateTime(timezone=False),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # Create a pgvector HNSW index for fast approximate nearest-neighbour
    # (ANN) similarity searches.  This is optional but highly recommended
    # for production workloads with > 10 k rows.
    op.execute(
        f"CREATE INDEX ix_{_TABLE_NAME}_embedding_hnsw "
        f"ON {_TABLE_NAME} USING hnsw (embedding vector_cosine_ops)"
    )


# ---------------------------------------------------------------------------
# downgrade
# ---------------------------------------------------------------------------


def downgrade() -> None:
    """
    Drop the ``role_benchmarks`` table.

    The ``vector`` extension is intentionally left intact — it may be used by
    other tables or future migrations and its removal would require a separate
    deliberate migration.
    """
    op.drop_table(_TABLE_NAME)
