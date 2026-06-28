import pytest
import uuid
import json
from unittest.mock import patch, AsyncMock
from app.schemas.matching import CandidateProfile
from app.schemas.application_ai import ApplicationRequest
from app.services.application_ai_service import ApplicationAIService
from app.db.repositories import JobRepository, JobRecord


# --- Fixtures ---

@pytest.fixture
def sample_candidate_profile():
    return CandidateProfile(
        name="John Doe",
        contact={"email": "john@example.com"},
        skills=["Python", "FastAPI"],
        experience_years=3,
        education=["BSc Computer Science"],
        preferences={"location": "Remote"}
    )


@pytest.fixture
def sample_request(sample_candidate_profile):
    return ApplicationRequest(
        candidate_id=uuid.uuid4(),
        job_id=uuid.uuid4(),
        candidate_profile=sample_candidate_profile,
        job_description="Looking for a Python developer with FastAPI experience."
    )


@pytest.fixture
def mock_cv_tailoring_response():
    return json.dumps({
        "tailored_summary": "Python Developer with 3 years experience...",
        "highlighted_skills": ["Python", "FastAPI"],
        "missing_skills": ["Docker"],
        "bullet_point_suggestions": ["Rephrase to emphasize FastAPI"]
    })


@pytest.fixture
def mock_cover_letter_response():
    return json.dumps({
        "draft_content": "Dear Hiring Manager, I am writing to apply...",
        "tone_analysis": "Professional and eager."
    })


# --- Existing Tests (Updated for new constructor) ---

@pytest.mark.asyncio
@patch("app.services.application_ai_service.LLMServiceRegistry.generate_json", new_callable=AsyncMock)
async def test_application_ai_service_success(
    mock_generate_json, sample_request, mock_cv_tailoring_response, mock_cover_letter_response
):
    """Test the full happy-path pipeline with job_description provided directly."""
    mock_generate_json.side_effect = [
        json.dumps({"is_safe": True}),
        mock_cv_tailoring_response,
        mock_cover_letter_response,
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
    # We check all calls to generate_json to ensure PII wasn't leaked anywhere
    all_calls_str = str(mock_generate_json.call_args_list)
    assert "john@example.com" not in all_calls_str
    assert "John Doe" not in all_calls_str
    assert "REDACTED" in all_calls_str


@pytest.mark.asyncio
@patch("app.services.application_ai_service.LLMServiceRegistry.generate_json", new_callable=AsyncMock)
async def test_application_ai_service_content_moderation_failure(mock_generate_json, sample_request):
    """Test that job descriptions with blocked keywords are rejected."""
    sample_request.job_description = "We encourage hate speech in our workplace."
    mock_generate_json.side_effect = [json.dumps({"is_safe": False, "reason": "Contains hate speech"})]

    service = ApplicationAIService()

    with pytest.raises(ValueError, match="blocked by content moderation"):
        await service.generate_application_materials(sample_request)


# --- New Tests: JobRepository Integration ---

@pytest.mark.asyncio
@patch("app.services.application_ai_service.LLMServiceRegistry.generate_json", new_callable=AsyncMock)
async def test_job_resolution_from_db(
    mock_generate_json, sample_candidate_profile, mock_cv_tailoring_response, mock_cover_letter_response
):
    """Test that the service resolves job_description from the DB when not provided in the request."""
    job_id = uuid.uuid4()

    mock_db = AsyncMock(spec=JobRepository)
    mock_db.get_job.return_value = JobRecord(
        id=job_id,
        title="Senior Python Developer",
        description="Looking for a senior Python developer with 5+ years experience.",
        embedding=[0.01] * 1536,
    )

    mock_generate_json.side_effect = [
        json.dumps({"is_safe": True}),
        mock_cv_tailoring_response,
        mock_cover_letter_response,
    ]

    # Request WITHOUT job_description — must be resolved from DB
    request = ApplicationRequest(
        candidate_id=uuid.uuid4(),
        job_id=job_id,
        candidate_profile=sample_candidate_profile,
    )

    service = ApplicationAIService(db=mock_db)
    response = await service.generate_application_materials(request)

    # Verify DB was called
    mock_db.get_job.assert_called_once_with(job_id)
    assert response.cv_tailoring.tailored_summary is not None
    assert response.cover_letter.draft_content is not None


@pytest.mark.asyncio
async def test_job_not_found_in_db(sample_candidate_profile):
    """Test that a ValueError is raised when job_id is not found in the DB and no job_description is provided."""
    job_id = uuid.uuid4()

    mock_db = AsyncMock(spec=JobRepository)
    mock_db.get_job.return_value = None

    request = ApplicationRequest(
        candidate_id=uuid.uuid4(),
        job_id=job_id,
        candidate_profile=sample_candidate_profile,
    )

    service = ApplicationAIService(db=mock_db)

    with pytest.raises(ValueError, match="not found"):
        await service.generate_application_materials(request)


@pytest.mark.asyncio
async def test_no_db_and_no_job_description(sample_candidate_profile):
    """Test that a ValueError is raised when no job_description is provided and no DB is available."""
    request = ApplicationRequest(
        candidate_id=uuid.uuid4(),
        job_id=uuid.uuid4(),
        candidate_profile=sample_candidate_profile,
    )

    service = ApplicationAIService(db=None)

    with pytest.raises(ValueError, match="No job description provided"):
        await service.generate_application_materials(request)


# --- New Tests: LLM Failure Edge Cases ---

@pytest.mark.asyncio
@patch("app.services.application_ai_service.LLMServiceRegistry.generate_json", new_callable=AsyncMock)
async def test_cv_tailoring_llm_failure(mock_generate_json, sample_request):
    """Test graceful error handling when the CV tailoring LLM call fails."""
    mock_generate_json.side_effect = [
        json.dumps({"is_safe": True}),
        Exception("LLM API timeout")
    ]

    service = ApplicationAIService()

    with pytest.raises(ValueError, match="Failed to generate tailored CV"):
        await service.generate_application_materials(sample_request)


@pytest.mark.asyncio
@patch("app.services.application_ai_service.LLMServiceRegistry.generate_json", new_callable=AsyncMock)
async def test_cover_letter_llm_failure(
    mock_generate_json, sample_request, mock_cv_tailoring_response
):
    """Test graceful error handling when CV tailoring succeeds but cover letter LLM call fails."""
    mock_generate_json.side_effect = [
        json.dumps({"is_safe": True}),
        mock_cv_tailoring_response,
        Exception("LLM API timeout"),
    ]

    service = ApplicationAIService()

    with pytest.raises(ValueError, match="Failed to generate cover letter"):
        await service.generate_application_materials(sample_request)


# --- New Tests: Edge Cases ---

@pytest.mark.asyncio
@patch("app.services.application_ai_service.LLMServiceRegistry.generate_json", new_callable=AsyncMock)
async def test_empty_skills_candidate(
    mock_generate_json, mock_cv_tailoring_response, mock_cover_letter_response
):
    """Test that the pipeline handles candidates with no skills gracefully."""
    mock_generate_json.side_effect = [
        json.dumps({"is_safe": True}),
        mock_cv_tailoring_response,
        mock_cover_letter_response,
    ]

    request = ApplicationRequest(
        candidate_id=uuid.uuid4(),
        job_id=uuid.uuid4(),
        candidate_profile=CandidateProfile(
            name="Empty Skills User",
            skills=[],
            experience_years=0,
        ),
        job_description="Looking for a Python developer.",
    )

    service = ApplicationAIService()
    response = await service.generate_application_materials(request)

    assert response.cv_tailoring is not None
    assert response.cover_letter is not None


@pytest.mark.asyncio
@patch("app.services.application_ai_service.LLMServiceRegistry.generate_json", new_callable=AsyncMock)
async def test_pii_scrubbing_removes_contact_info(
    mock_generate_json, sample_request, mock_cv_tailoring_response, mock_cover_letter_response
):
    """Test that both name and contact info are scrubbed before LLM calls."""
    mock_generate_json.side_effect = [
        json.dumps({"is_safe": True}),
        mock_cv_tailoring_response,
        mock_cover_letter_response,
    ]

    service = ApplicationAIService()
    await service.generate_application_materials(sample_request)

    # Inspect all LLM calls to verify PII is not present
    all_calls_str = str(mock_generate_json.call_args_list)
    assert "john@example.com" not in all_calls_str
    assert "John Doe" not in all_calls_str
