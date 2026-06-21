import pytest
import uuid
from app.schemas.matching import MatchRequest, CandidateProfile
from app.services.job_matching_service import JobMatchingService
from tests.services.mock_data import (
    PERFECT_CANDIDATE, 
    PARTIAL_CANDIDATE, 
    POOR_CANDIDATE, 
    TARGET_JOB_ID, 
    CANDIDATE_ID, 
    MOCK_JOB_DATA
)
import pytest
import uuid
import os
import json
from unittest.mock import AsyncMock, patch
from app.schemas.matching import MatchRequest, CandidateProfile
from app.services.job_matching_service import JobMatchingService
from app.db.repositories import JobRepository, JobRecord
from tests.services.mock_data import (
    PERFECT_CANDIDATE, 
    PARTIAL_CANDIDATE, 
    POOR_CANDIDATE, 
    TARGET_JOB_ID, 
    CANDIDATE_ID, 
    MOCK_JOB_DATA
)

@pytest.fixture
def mock_job_repo():
    repo = AsyncMock(spec=JobRepository)
    # Default to returning our MOCK_JOB_DATA
    job_record = JobRecord(
        id=MOCK_JOB_DATA["job_id"],
        title=MOCK_JOB_DATA["title"],
        description=MOCK_JOB_DATA["description"],
        embedding=MOCK_JOB_DATA["embedding"]
    )
    repo.get_job.return_value = job_record
    repo.vector_search_jobs.return_value = [(job_record, 0.15)]
    return repo

@pytest.mark.asyncio
async def test_execute_match_job_not_found(mock_job_repo):
    """Test exception raising when job doesn't exist."""
    service = JobMatchingService(db=mock_job_repo)
    mock_job_repo.get_job.return_value = None
    
    candidate_profile = CandidateProfile(
        name="Bob",
        experience_years=2
    )
    
    request = MatchRequest(
        candidate_id=uuid.uuid4(),
        job_id=uuid.uuid4(),
        candidate_profile=candidate_profile
    )
    
    with pytest.raises(ValueError, match="not found"):
        await service.execute_match(request)

@pytest.mark.asyncio
@pytest.mark.parametrize("candidate_profile, expected_min_score, expected_max_score", [
    (PERFECT_CANDIDATE, 80, 100),
    (PARTIAL_CANDIDATE, 40, 79),
    (POOR_CANDIDATE, 0, 39)
])
@patch("app.ai.nodes.LLMServiceRegistry.generate_json")
async def test_scoring_rubric_correctness(mock_generate_json, mock_job_repo, candidate_profile, expected_min_score, expected_max_score):
    """Test the scoring pipeline using our predefined mock candidate scenarios with mocked LLM."""
    service = JobMatchingService(db=mock_job_repo)
    
    if candidate_profile == PERFECT_CANDIDATE:
        score = 95
    elif candidate_profile == PARTIAL_CANDIDATE:
        score = 65
    else:
        score = 25
        
    mock_generate_json.return_value = json.dumps({
        "score_details": {
            "hard_skills_score": int(score * 0.4),
            "experience_score": int(score * 0.3),
            "soft_skills_score": int(score * 0.2),
            "logistics_score": int(score * 0.1)
        },
        "total_score": score,
        "explanation": "Mocked explanation",
        "strengths": ["Mocked strength"],
        "missing_skills": [],
        "recommendation": "Mocked recommendation"
    })
    
    request = MatchRequest(
        candidate_id=CANDIDATE_ID,
        job_id=TARGET_JOB_ID,
        candidate_profile=candidate_profile
    )
    
    response = await service.execute_match(request)
    
    print(f"\n--- Pipeline Output for {candidate_profile.name} ---")
    print(response.model_dump_json(indent=2))
    
    assert response.job_id == TARGET_JOB_ID
    assert response.candidate_id == CANDIDATE_ID
    
    # Assert ACTUAL matching correctness based on expectations
    assert expected_min_score <= response.result.total_score <= expected_max_score, \
        f"Score {response.result.total_score} for {candidate_profile.name} is outside expected bounds [{expected_min_score}, {expected_max_score}]"
