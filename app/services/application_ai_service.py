import os
import json
import logging
from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, END

from app.schemas.application_ai import (
    ApplicationRequest,
    CVTailoringResult,
    CoverLetterResult,
    ApplicationResponse
)
from app.core.llm import LLMServiceRegistry
from app.db.repositories import JobRepository

logger = logging.getLogger("application_ai")

# --- 1. State Definition ---
class ApplicationState(TypedDict):
    """LangGraph State for tracking application materials generation."""
    request: ApplicationRequest
    db: Any  # Database dependency injected from the service layer
    job_description: str  # Resolved job description (from request or DB)
    cv_tailoring_result: Optional[CVTailoringResult]
    cover_letter_result: Optional[CoverLetterResult]
    error: Optional[str]

# --- 2. Node Functions ---
def load_prompt(filename: str) -> str:
    prompt_path = os.path.join(
        os.path.dirname(__file__), 
        "..", "core", "prompts", filename
    )
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

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

async def input_validation_node(state: ApplicationState) -> ApplicationState:
    """Stage 1: Content Moderation & PII Scrubbing"""
    job_description = state["job_description"]
    
    # Very basic content moderation for inappropriate words in job description
    blocked_keywords = ["hate", "violence", "illegal"]
    desc_lower = job_description.lower()
    
    for kw in blocked_keywords:
        if kw in desc_lower:
            state["error"] = f"Job description blocked by content moderation: contains inappropriate content."
            return state
            
    # Scrub PII
    safe_profile = state["request"].candidate_profile.model_copy(deep=True)
    safe_profile.name = "REDACTED"
    safe_profile.contact = {}
    
    # Update request with scrubbed profile
    state["request"].candidate_profile = safe_profile
    return state

async def cv_tailoring_node(state: ApplicationState) -> ApplicationState:
    """Stage 2: CV Tailoring using LLM"""
    request = state["request"]
    job_description = state["job_description"]
    prompt_template = load_prompt("cv_tailoring.md")
    
    candidate_profile_json = request.candidate_profile.model_dump_json()
    
    prompt = prompt_template.replace("{candidate_profile}", candidate_profile_json)
    prompt = prompt.replace("{job_description}", job_description)
    
    messages = [
        {"role": "system", "content": prompt}
    ]
    
    model_name = os.getenv("DEFAULT_MODEL", "azure/FW-Kimi-K2.6")
    
    try:
        llm_content = await LLMServiceRegistry.generate_json(model_name=model_name, messages=messages)
        result = CVTailoringResult.model_validate_json(llm_content)
        state["cv_tailoring_result"] = result
        logger.info(f"CV Tailoring successful | candidate_id={request.candidate_id} | job_id={request.job_id}")
    except Exception as e:
        logger.error(f"CV Tailoring Error | candidate_id={request.candidate_id} | job_id={request.job_id} | error={str(e)}")
        state["error"] = f"Failed to generate tailored CV: {str(e)}"
        
    return state

async def cover_letter_node(state: ApplicationState) -> ApplicationState:
    """Stage 3: Cover Letter Generation using LLM"""
    request = state["request"]
    job_description = state["job_description"]
    cv_result = state["cv_tailoring_result"]
    prompt_template = load_prompt("cover_letter.md")
    
    cv_result_json = cv_result.model_dump_json() if cv_result else "{}"
    
    prompt = prompt_template.replace("{cv_tailoring_result}", cv_result_json)
    prompt = prompt.replace("{job_description}", job_description)
    
    messages = [
        {"role": "system", "content": prompt}
    ]
    
    model_name = os.getenv("DEFAULT_MODEL", "azure/FW-Kimi-K2.6")
    
    try:
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
            "job_description": request.job_description or "",
            "cv_tailoring_result": None,
            "cover_letter_result": None,
            "error": None
        }
        
        final_state = await compiled_application_graph.ainvoke(initial_state)
        
        if final_state.get("error"):
            raise ValueError(final_state["error"])
            
        return ApplicationResponse(
            candidate_id=request.candidate_id,
            job_id=request.job_id,
            cv_tailoring=final_state["cv_tailoring_result"],
            cover_letter=final_state["cover_letter_result"]
        )
