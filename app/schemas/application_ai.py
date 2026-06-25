from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid

from app.schemas.matching import CandidateProfile

class ApplicationRequest(BaseModel):
    """
    Request payload to initiate CV tailoring and cover letter generation.
    """
    candidate_id: uuid.UUID = Field(..., description="Unique identifier of the candidate")
    job_id: uuid.UUID = Field(..., description="Unique identifier of the target job")
    candidate_profile: CandidateProfile = Field(..., description="Candidate's current profile")
    job_description: Optional[str] = Field(None, description="Full text of the target job description. If not provided, it will be resolved from the JobRepository using job_id.")

class CVTailoringResult(BaseModel):
    """
    Structured output for CV tailoring suggestions.
    """
    tailored_summary: str = Field(..., description="A customized professional summary highlighting relevant experience for the target job")
    highlighted_skills: List[str] = Field(default_factory=list, description="List of the candidate's existing skills most relevant to the target job")
    missing_skills: List[str] = Field(default_factory=list, description="Skills mentioned in the job description that the candidate appears to lack")
    bullet_point_suggestions: List[str] = Field(default_factory=list, description="Suggestions for rephrasing experience bullet points to better align with the job requirements")

class CoverLetterResult(BaseModel):
    """
    Structured output for the generated cover letter.
    """
    draft_content: str = Field(..., description="The complete text of the generated cover letter draft")
    tone_analysis: str = Field(..., description="Brief explanation of the tone used in the cover letter (e.g., formal, enthusiastic, professional)")

class ApplicationResponse(BaseModel):
    """
    The final combined response returned to the client.
    """
    candidate_id: uuid.UUID
    job_id: uuid.UUID
    cv_tailoring: CVTailoringResult
    cover_letter: CoverLetterResult
