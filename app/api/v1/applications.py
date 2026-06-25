from fastapi import APIRouter, HTTPException, status, Depends
from app.schemas.application_ai import ApplicationRequest, ApplicationResponse
from app.services.application_ai_service import ApplicationAIService
from app.db.repositories import JobRepository, InMemoryJobRepository

router = APIRouter(prefix="/applications", tags=["Application AI"])

# --- Dependency Injection ---
# Shared singleton repository instance (same pattern as matches endpoint)
_shared_job_repository = InMemoryJobRepository()


def get_job_repository() -> JobRepository:
    """Provides the shared JobRepository instance."""
    return _shared_job_repository


def get_application_service(
    db: JobRepository = Depends(get_job_repository),
) -> ApplicationAIService:
    """Provides an ApplicationAIService instance with the shared DB dependency."""
    return ApplicationAIService(db=db)


@router.post(
    "/",
    response_model=ApplicationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Application Materials",
    description=(
        "Generates tailored CV suggestions and a cover letter draft for a candidate "
        "targeting a specific job. The job description can be provided directly in the "
        "request body, or resolved automatically from the JobRepository using job_id."
    ),
)
async def generate_application_materials(
    request: ApplicationRequest,
    service: ApplicationAIService = Depends(get_application_service),
):
    """
    Executes the four-stage Application AI pipeline:
    1. Job Resolution — Resolves job description from DB if not provided
    2. Input Validation — Content moderation & PII scrubbing
    3. CV Tailoring — LLM-powered CV improvement suggestions
    4. Cover Letter — LLM-powered cover letter generation
    """
    try:
        response = await service.generate_application_materials(request)
        return response
    except ValueError as ve:
        error_msg = str(ve)
        if "not found" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg,
            )
        elif "content moderation" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=error_msg,
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Application AI pipeline failed: {str(e)}",
        )
