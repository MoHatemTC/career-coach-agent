import os
import json
import logging
from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, END
from langfuse.decorators import observe
from prometheus_client import Counter, Histogram, REGISTRY

from app.schemas.application_ai import (
    ApplicationRequest,
    CVTailoringResult,
    CoverLetterResult,
    ApplicationResponse
)
from app.core.llm import LLMServiceRegistry
from app.db.repositories import JobRepository

logger = logging.getLogger("application_ai")

# --- Prometheus Metrics ---
def _get_or_create(metric_cls, name, description, labels):
    try:
        return metric_cls(name, description, labels)
    except ValueError:
        return REGISTRY._names_to_collectors[name]

PIPELINE_REQUESTS = _get_or_create(Counter, "application_ai_requests_total", "Count of application AI requests", ["status"])
LLM_LATENCY = _get_or_create(Histogram, "llm_call_duration_seconds", "Duration of LLM calls", ["stage"])

# --- 1. State Definition ---
class ApplicationState(TypedDict):
    """LangGraph State for tracking application materials generation."""
    request: ApplicationRequest
    db: Any  # Database dependency injected from the service layer
    job_description: Optional[str]  # Resolved job description (from request or DB)
    cv_tailoring_result: Optional[CVTailoringResult]
    cover_letter_result: Optional[CoverLetterResult]
    error: Optional[str]

# --- 2. Node Functions ---
# (Prompt templates are now managed by app.ai.prompts.PromptBuilder)

@observe()
async def job_resolution_node(state: ApplicationState) -> ApplicationState:
    """Stage 0: Resolve job description from DB if not provided in the request."""
    request = state["request"]
    db = state["db"]

    if request.job_description:
        # Client provided the job description directly
        state["job_description"] = request.job_description
        return state

    # Resolve from the JobRepository using job_id
    if db is None:
        state["error"] = "No job description provided and no database connection available."
        return state

    target_job = await db.get_job(request.job_id)
    if not target_job:
        state["error"] = f"Job with ID {request.job_id} not found."
        return state

    state["job_description"] = target_job.description
    logger.info(f"Resolved job description from DB | job_id={request.job_id}")
    return state

@observe()
async def input_validation_node(state: ApplicationState) -> ApplicationState:
    """Stage 1: Content Moderation & PII Scrubbing"""
    job_description = state["job_description"]
    
    # LLM-based content moderation
    moderation_prompt = f"Analyze the following job description. Does it contain hate speech, promote violence, or request illegal activities? Respond with a JSON object containing a boolean 'is_safe' and a string 'reason'.\n\nJob Description:\n{job_description}"
    messages = [{"role": "system", "content": moderation_prompt}]
    model_name = os.getenv("DEFAULT_MODEL", "azure/FW-Kimi-K2.6")
    
    try:
        with LLM_LATENCY.labels(stage="content_moderation").time():
            llm_content = await LLMServiceRegistry.generate_json(model_name=model_name, messages=messages)
            
        moderation_result = json.loads(llm_content)
        if not moderation_result.get("is_safe", True):
            state["error"] = f"Job description blocked by content moderation: {moderation_result.get('reason', 'contains inappropriate content')}."
            return state
    except Exception as e:
        logger.error(f"Moderation check failed, defaulting to safe. Error: {str(e)}")
            
    # Scrub PII safely using update dict to avoid mutating frozen models
    safe_profile = state["request"].candidate_profile.model_copy(
        update={"name": "REDACTED", "contact": {}}
    )
    
    # Update request with scrubbed profile safely
    updated_request = state["request"].model_copy(
        update={"candidate_profile": safe_profile}
    )
    state["request"] = updated_request
    return state

@observe()
async def cv_tailoring_node(state: ApplicationState) -> ApplicationState:
    """Stage 2: CV Tailoring using LLM"""
    from app.ai.prompts import PromptBuilder
    request = state["request"]
    job_description = state["job_description"]
    
    candidate_profile_json = request.candidate_profile.model_dump_json()
    
    prompt = PromptBuilder.build_cv_tailoring_prompt(
        candidate_profile=candidate_profile_json,
        job_description=job_description
    )
    
    messages = [
        {"role": "system", "content": prompt}
    ]
    
    model_name = os.getenv("DEFAULT_MODEL", "azure/FW-Kimi-K2.6")
    
    try:
        with LLM_LATENCY.labels(stage="cv_tailoring").time():
            llm_content = await LLMServiceRegistry.generate_json(model_name=model_name, messages=messages)
        result = CVTailoringResult.model_validate_json(llm_content)
        state["cv_tailoring_result"] = result
        logger.info(f"CV Tailoring successful | candidate_id={request.candidate_id} | job_id={request.job_id}")
    except Exception as e:
        logger.error(f"CV Tailoring Error | candidate_id={request.candidate_id} | job_id={request.job_id} | error={str(e)}")
        state["error"] = f"Failed to generate tailored CV: {str(e)}"
        
    return state

@observe()
async def cover_letter_node(state: ApplicationState) -> ApplicationState:
    """Stage 3: Cover Letter Generation using LLM"""
    from app.ai.prompts import PromptBuilder
    request = state["request"]
    job_description = state["job_description"]
    cv_result = state["cv_tailoring_result"]
    
    cv_result_json = cv_result.model_dump_json() if cv_result else "{}"
    
    prompt = PromptBuilder.build_cover_letter_prompt(
        cv_tailoring_result=cv_result_json,
        job_description=job_description
    )
    
    messages = [
        {"role": "system", "content": prompt}
    ]
    
    model_name = os.getenv("DEFAULT_MODEL", "azure/FW-Kimi-K2.6")
    
    try:
        with LLM_LATENCY.labels(stage="cover_letter").time():
            llm_content = await LLMServiceRegistry.generate_json(model_name=model_name, messages=messages)
        result = CoverLetterResult.model_validate_json(llm_content)
        state["cover_letter_result"] = result
        logger.info(f"Cover Letter Generation successful | candidate_id={request.candidate_id} | job_id={request.job_id}")
    except Exception as e:
        logger.error(f"Cover Letter Error | candidate_id={request.candidate_id} | job_id={request.job_id} | error={str(e)}")
        state["error"] = f"Failed to generate cover letter: {str(e)}"
        
    return state

# --- 3. Graph Compilation ---
def create_application_graph():
    workflow = StateGraph(ApplicationState)
    
    workflow.add_node("job_resolution", job_resolution_node)
    workflow.add_node("input_validation", input_validation_node)
    workflow.add_node("cv_tailoring", cv_tailoring_node)
    workflow.add_node("cover_letter", cover_letter_node)
    
    workflow.set_entry_point("job_resolution")
    
    def check_resolution_error(state: ApplicationState):
        if state.get("error"):
            return END
        return "input_validation"

    def check_validation_error(state: ApplicationState):
        if state.get("error"):
            return END
        return "cv_tailoring"
        
    def check_cv_error(state: ApplicationState):
        if state.get("error"):
            return END
        return "cover_letter"
        
    workflow.add_conditional_edges("job_resolution", check_resolution_error, {"input_validation": "input_validation", END: END})
    workflow.add_conditional_edges("input_validation", check_validation_error, {"cv_tailoring": "cv_tailoring", END: END})
    workflow.add_conditional_edges("cv_tailoring", check_cv_error, {"cover_letter": "cover_letter", END: END})
    workflow.add_edge("cover_letter", END)
    
    return workflow.compile()

compiled_application_graph = create_application_graph()

# --- 4. Service Class ---
class ApplicationAIService:
    def __init__(self, db: JobRepository = None):
        """
        Initialize the Application AI Service.
        
        Args:
            db: Optional JobRepository instance. Required when the client does not
                provide a job_description in the request and expects the service to
                resolve it from the database using job_id.
        """
        self.db = db

    async def generate_application_materials(self, request: ApplicationRequest) -> ApplicationResponse:
        """
        Executes the four-stage application material generation pipeline via LangGraph orchestration.
        
        Stages:
            1. Job Resolution — Resolve job description from DB if not provided
            2. Input Validation — Content moderation & PII scrubbing
            3. CV Tailoring — LLM-powered CV improvement suggestions
            4. Cover Letter — LLM-powered cover letter generation
        """
        initial_state = {
            "request": request,
            "db": self.db,
            "job_description": request.job_description,
            "cv_tailoring_result": None,
            "cover_letter_result": None,
            "error": None
        }
        
        final_state = await compiled_application_graph.ainvoke(initial_state)
        
        if final_state.get("error"):
            PIPELINE_REQUESTS.labels(status="error").inc()
            raise ValueError(final_state["error"])
            
        PIPELINE_REQUESTS.labels(status="success").inc()
        return ApplicationResponse(
            candidate_id=request.candidate_id,
            job_id=request.job_id,
            cv_tailoring=final_state["cv_tailoring_result"],
            cover_letter=final_state["cover_letter_result"]
        )
