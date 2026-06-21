import os
import logging
from app.ai.state import MatchingState
from app.schemas.matching import JobMatchResult, MatchScoreDetails
from app.core.llm import LLMServiceRegistry

logger = logging.getLogger("job_matching")

async def pre_filter_node(state: MatchingState) -> MatchingState:
    """Stage 1: Vector Database Search Node."""
    request = state["request"]
    db = state["db"]
    
    target_job = await db.get_job(request.job_id)
    if not target_job:
        # Signal the graph to halt by injecting an error
        state["error"] = f"Job with ID {request.job_id} not found."
        return state
        
    state["target_job"] = target_job
    
    # Perform vector search using actual generated embedding
    candidate_summary = f"{request.candidate_profile.skills} {request.candidate_profile.experience_years} years experience"
    try:
        candidate_embedding = await LLMServiceRegistry.generate_embedding(candidate_summary)
    except Exception as e:
        logger.error(f"Failed to generate embedding for candidate {request.candidate_id}: {str(e)}")
        candidate_embedding = [0.0] * 1536
        
    search_results = await db.vector_search_jobs(candidate_embedding, limit=5)
    
    # Extract the distance if the target job is found in the search results
    distance = 0.12
    for job, dist in search_results:
        if job.id == target_job.id:
            distance = dist
            break
            
    state["vector_distance"] = distance
    return state

async def llm_evaluation_node(state: MatchingState) -> MatchingState:
    """Stage 2: LLM Re-ranking Node."""
    request = state["request"]
    target_job = state["target_job"]
    
    prompt_path = os.path.join(
        os.path.dirname(__file__), 
        "..", "core", "prompts", "job_matching.md"
    )
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except FileNotFoundError:
        # Fallback to python string prompt if markdown file is missing
        from app.ai.prompts import JOB_MATCHING_PROMPT
        prompt_template = JOB_MATCHING_PROMPT

    # Scrub PII before inference
    safe_profile = request.candidate_profile.model_copy(deep=True)
    safe_profile.name = "REDACTED"
    safe_profile.contact = {}
    candidate_profile_json = safe_profile.model_dump_json()
    
    messages = [
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": f"Candidate Profile:\n{candidate_profile_json}\n\nJob Description:\n{target_job.description}"}
    ]

    model_name = os.getenv("DEFAULT_MODEL", "azure/FW-Kimi-K2.6")
    
    # Use the shared LLM service registry
    try:
        llm_content = await LLMServiceRegistry.generate_json(model_name=model_name, messages=messages)
        result = JobMatchResult.model_validate_json(llm_content)
        state["llm_result"] = result
        logger.info(f"Match successful | candidate_id={request.candidate_id} | job_id={request.job_id} | score={result.total_score}")
    except Exception as e:
        logger.error(f"LLM Error | candidate_id={request.candidate_id} | job_id={request.job_id} | error={str(e)}")
        state["llm_result"] = JobMatchResult(
            score_details=MatchScoreDetails(
                hard_skills_score=0,
                experience_score=0,
                soft_skills_score=0,
                logistics_score=0
            ),
            total_score=0,
            explanation=f"Error evaluating candidate (JSON parsing or LLM API failure): {str(e)}",
            strengths=[],
            missing_skills=[],
            recommendation="Evaluation aborted due to system error."
        )

    return state
