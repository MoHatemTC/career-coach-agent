from typing import List, Optional
from datetime import datetime
from sqlmodel import SQLModel, Field, Column, JSON
from pgvector.sqlalchemy import Vector


class RoleBenchmark(SQLModel, table=True):
    __tablename__ = "role_benchmarks"

    id: Optional[int] = Field(default=None, primary_key=True)
    must_have_skills: List[str] = Field(
        default_factory=list, sa_column=Column(JSON)
    )
    nice_to_have_skills: List[str] = Field(
        default_factory=list, sa_column=Column(JSON)
    )
    required_tools: List[str] = Field(
        default_factory=list, sa_column=Column(JSON)
    )
    common_responsibilities: List[str] = Field(
        default_factory=list, sa_column=Column(JSON)
    )
    minimum_years: int = Field(default=0)
    seniority_level: str = Field(default="")
    embedding: Optional[List[float]] = Field(
        default=None, sa_column=Column(Vector(1536))
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
