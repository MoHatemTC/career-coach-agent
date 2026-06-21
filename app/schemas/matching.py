from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid

class CandidateProfile(BaseModel):
    """
    Schema for a candidate's profile, designed to be stored as JSONB.
    """
    name: str = Field(..., description="Full name of the candidate")
    contact: Dict[str, str] = Field(default_factory=dict, description="Contact information")
    skills: List[str] = Field(default_factory=list, description="List of technical and soft skills")
    experience_years: int = Field(0, description="Total years of experience")
    education: List[str] = Field(default_factory=list, description="Educational background")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="Career preferences and logistics")

class MatchRequest(BaseModel):
    """
    Request payload to initiate a job match.
    """
    candidate_id: uuid.UUID = Field(..., description="Unique identifier of the candidate (User ID)")
    job_id: uuid.UUID = Field(..., description="Unique identifier of the target job")
    candidate_profile: CandidateProfile = Field(..., description="Parsed candidate profile")

class MatchScoreDetails(BaseModel):
    """
    Internal rubric scoring breakdown.
    """
    hard_skills_score: int = Field(..., ge=0, le=40, description="Hard Skills Fit (Max 40 points)")
    experience_score: int = Field(..., ge=0, le=30, description="Experience Level Fit (Max 30 points)")
    soft_skills_score: int = Field(..., ge=0, le=20, description="Soft Skills & Domain Knowledge (Max 20 points)")
    logistics_score: int = Field(..., ge=0, le=10, description="Career Preferences & Logistics (Max 10 points)")

class JobMatchResult(BaseModel):
    """
    The structured output expected from the LLM based on the scoring rubric.
    """
    score_details: MatchScoreDetails
    total_score: int = Field(..., ge=0, le=100, description="Sum of score details")
    explanation: str = Field(..., description="Detailed explanation of the derived score")
    strengths: List[str] = Field(default_factory=list, description="Key strengths and matching skills")
    missing_skills: List[str] = Field(default_factory=list, description="Required skills the candidate lacks (weaknesses)")
    recommendation: str = Field(..., description="Actionable advice for the candidate")

class JobMatchResponse(BaseModel):
    """
    The final response payload returned to the client.
    """
    job_id: uuid.UUID
    candidate_id: uuid.UUID
    result: JobMatchResult
    vector_distance: Optional[float] = Field(None, description="Pre-filtering vector distance score (e.g., L2)")
