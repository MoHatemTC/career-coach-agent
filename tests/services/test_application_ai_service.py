import pytest
import uuid
import json
from unittest.mock import patch
from app.schemas.matching import CandidateProfile
from app.schemas.application_ai import ApplicationRequest
from app.services.application_ai_service import ApplicationAIService

@pytest.fixture
def sample_request():
    return ApplicationRequest(
        candidate_id=uuid.uuid4(),
        job_id=uuid.uuid4(),
        candidate_profile=CandidateProfile(
            name="John Doe",
            contact={"email": "john@example.com"},
            skills=["Python", "FastAPI"],
            experience_years=3,
            education=["BSc Computer Science"],
            preferences={"location": "Remote"}
        ),
        job_description="Looking for a Python developer with FastAPI experience."
    )

@pytest.mark.asyncio
@patch("app.services.application_ai_service.LLMServiceRegistry.generate_json")
async def test_application_ai_service_success(mock_generate_json, sample_request):
    # Mock the LLM returns for both nodes
    mock_generate_json.side_effect = [
        # First call: CV Tailoring return
        json.dumps({
            "tailored_summary": "Python Developer with 3 years experience...",
            "highlighted_skills": ["Python", "FastAPI"],
            "missing_skills": ["Docker"],
            "bullet_point_suggestions": ["Rephrase to emphasize FastAPI"]
        }),
        # Second call: Cover Letter return
        json.dumps({
            "draft_content": "Dear Hiring Manager, I am writing to apply...",
            "tone_analysis": "Professional and eager."
        })
    ]

    service = ApplicationAIService()
    response = await service.generate_application_materials(sample_request)

    assert response.candidate_id == sample_request.candidate_id
    assert response.job_id == sample_request.job_id
    
    # Assert CV result
    assert response.cv_tailoring.tailored_summary == "Python Developer with 3 years experience..."
    assert "Python" in response.cv_tailoring.highlighted_skills
    
    # Assert Cover Letter result
    assert response.cover_letter.draft_content.startswith("Dear Hiring Manager")
    
    # Check that PII was scrubbed
    # We inspect the messages passed to the first LLM call (cv_tailoring)
    first_call_args = mock_generate_json.call_args_list[0][1]
    prompt_content = first_call_args["messages"][0]["content"]
    assert "John Doe" not in prompt_content
    assert "REDACTED" in prompt_content

@pytest.mark.asyncio
async def test_application_ai_service_content_moderation_failure(sample_request):
    # Modify request to have blocked keyword
    sample_request.job_description = "We encourage hate speech in our workplace."
    
    service = ApplicationAIService()
    
    with pytest.raises(ValueError, match="blocked by content moderation"):
        await service.generate_application_materials(sample_request)
