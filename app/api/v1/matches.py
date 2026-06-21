from fastapi import APIRouter, HTTPException, status, Depends
from app.schemas.matching import MatchRequest, JobMatchResponse
from app.services.job_matching_service import JobMatchingService
from app.db.repositories import JobRepository

router = APIRouter(prefix="/matches", tags=["Job Matching"])

def get_job_repository() -> JobRepository:
    from app.db.repositories import InMemoryJobRepository
    return InMemoryJobRepository()

def get_matching_service(db: JobRepository = Depends(get_job_repository)) -> JobMatchingService:
    return JobMatchingService(db=db)

@router.post("/", response_model=JobMatchResponse, status_code=status.HTTP_200_OK)
async def match_candidate_to_job(request: MatchRequest, matching_service: JobMatchingService = Depends(get_matching_service)):
    """
    Evaluates a candidate's profile against a target job using a two-stage 
    vector pre-filtering and LLM re-ranking pipeline.
    """
    try:
        response = await matching_service.execute_match(request)
        return response
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Matching pipeline failed: {str(e)}"
        )
