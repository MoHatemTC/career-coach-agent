import asyncio
import os
import json
from dotenv import load_dotenv

from app.schemas.application_ai import ApplicationRequest
from app.services.application_ai_service import ApplicationAIService
from tests.services.mock_data import PERFECT_CANDIDATE, MOCK_JOB_DATA

load_dotenv()

async def run_cover_letter_generation():
    print("Testing Application AI Service (CV Tailoring & Cover Letter Generator)...")
    
    # Ensure LiteLLM keys are set up
    api_key = os.getenv("LITELLM_API_KEY")
    if not api_key:
        print("Warning: LITELLM_API_KEY is not set in environment.")

    # Create an application request
    request = ApplicationRequest(
        candidate_id=MOCK_JOB_DATA["job_id"],  # using random UUID from mock
        job_id=MOCK_JOB_DATA["job_id"],
        candidate_profile=PERFECT_CANDIDATE,
        job_description=MOCK_JOB_DATA["description"]
    )

    print(f"\nTarget Job: {MOCK_JOB_DATA['title']}")
    print(f"Job Description: {request.job_description}")
    
    print("\nStarting generation... (this may take a minute as it calls the LLM twice)\n")
    
    try:
        service = ApplicationAIService()
        response = await service.generate_application_materials(request)
        
        print("\n" + "="*50)
        print("--- CV Tailoring Suggestions ---")
        print(f"Tailored Summary:\n{response.cv_tailoring.tailored_summary}\n")
        print(f"Highlighted Skills:\n{', '.join(response.cv_tailoring.highlighted_skills)}\n")
        print(f"Missing Skills:\n{', '.join(response.cv_tailoring.missing_skills)}\n")
        print("Bullet Point Suggestions:")
        for bp in response.cv_tailoring.bullet_point_suggestions:
            print(f"- {bp}")

        print("\n" + "="*50)
        print("--- Cover Letter Draft ---")
        print(response.cover_letter.draft_content)
        
        print("\n" + "="*50)
        print("--- Tone Analysis ---")
        print(response.cover_letter.tone_analysis)
        
        print("\n--- STATUS ---")
        print(response.status)
        print(response.disclaimer)
        
    except Exception as e:
        print("\n--- FAILED ---")
        print("Error details:", str(e))

import pytest

@pytest.mark.asyncio
async def test_run_cover_letter_generation(capsys):
    with capsys.disabled():
        await run_cover_letter_generation()

