from typing import List, Tuple, Optional
import uuid
from pydantic import BaseModel

class JobRecord(BaseModel):
    id: uuid.UUID
    title: str
    description: str
    embedding: List[float]

class UserRecord(BaseModel):
    id: uuid.UUID
    profile: dict
    embedding: List[float]

class JobRepository:
    """
    Contract for the job database repository.
    Implementations should interact with the actual database (e.g., PostgreSQL with pgvector).
    """
    
    async def get_job(self, job_id: uuid.UUID) -> Optional[JobRecord]:
        """Fetch a job by UUID."""
        raise NotImplementedError

    async def vector_search_jobs(self, embedding: List[float], limit: int = 5) -> List[Tuple[JobRecord, float]]:
        """Perform a vector search using pgvector L2/cosine distance."""
        raise NotImplementedError

    async def close(self) -> None:
        """Close the database connection/session."""
        raise NotImplementedError

import math

class InMemoryJobRepository(JobRepository):
    """
    In-memory database implementation for testing and development.
    """
    def __init__(self):
        self.jobs = {}

    async def get_job(self, job_id: uuid.UUID) -> Optional[JobRecord]:
        return self.jobs.get(job_id)

    async def vector_search_jobs(self, embedding: List[float], limit: int = 5) -> List[Tuple[JobRecord, float]]:
        results = []
        for job in self.jobs.values():
            if not job.embedding or not embedding:
                continue
            # Basic Euclidean distance (L2)
            try:
                distance = math.dist(embedding, job.embedding)
                results.append((job, distance))
            except ValueError:
                # Fallback if dimensions mismatch
                pass
                
        # Sort by distance (lower is better for L2)
        results.sort(key=lambda x: x[1])
        return results[:limit]

    async def close(self) -> None:
        """Simulate closing the database connection."""
        pass
