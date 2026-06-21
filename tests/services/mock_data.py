import uuid
from app.schemas.matching import CandidateProfile

# Pre-defined UUIDs for predictable testing
TARGET_JOB_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
CANDIDATE_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")

# --- Mock Candidates for Different Scoring Scenarios ---

# 1. Perfect Match Candidate
# Meets all hard skills, experience, soft skills, and logistics.
PERFECT_CANDIDATE = CandidateProfile(
    name="Grace Hopper",
    contact={"email": "grace@example.com"},
    skills=["Python", "FastAPI", "PostgreSQL", "LLM Orchestration", "Leadership", "Communication"],
    experience_years=6,
    education=["MSc Computer Science"],
    preferences={"remote": True, "timezone": "EST"}
)

# 2. Partial Match Candidate
# Good experience, but missing some key hard skills (e.g., FastAPI, LLM).
PARTIAL_CANDIDATE = CandidateProfile(
    name="Ada Lovelace",
    contact={"email": "ada@example.com"},
    skills=["Python", "Django", "MySQL", "Communication"],
    experience_years=5,
    education=["BSc Computer Science"],
    preferences={"remote": True, "timezone": "PST"}
)

# 3. Poor Match Candidate
# Completely different domain, low experience, logistics mismatch.
POOR_CANDIDATE = CandidateProfile(
    name="Alan Turing",
    contact={"email": "alan@example.com"},
    skills=["Java", "Spring Boot", "Oracle"],
    experience_years=1,
    education=["Bootcamp Certificate"],
    preferences={"remote": False, "timezone": "GMT"}
)

# --- Mock Job Data ---
MOCK_JOB_DATA = {
    "job_id": TARGET_JOB_ID,
    "title": "Senior AI Backend Engineer",
    "description": (
        "We are looking for a Senior AI Backend Engineer to orchestrate LLM pipelines. "
        "Must have 5+ years of experience. Mandatory skills: Python, FastAPI, PostgreSQL, and LLM Orchestration. "
        "Requires strong leadership and communication skills. Must be open to 100% remote work."
    ),
    "embedding": [0.01] * 1536
}
