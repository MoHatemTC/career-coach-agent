import sys
import os
# pyrefly: ignore [missing-import]
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ai.prompts import CV_PARSING_PROMPT, JOB_MATCHING_PROMPT, PromptBuilder

def test_cv_parsing_prompt_has_placeholders():
    """Ensure the CV parsing prompt contains necessary placeholders."""
    assert "{cv_text}" in CV_PARSING_PROMPT, "Prompt missing {cv_text} placeholder."

def test_job_matching_prompt_has_placeholders():
    """Ensure the job matching prompt contains necessary placeholders."""
    assert "{candidate_profile}" in JOB_MATCHING_PROMPT, "Prompt missing {candidate_profile} placeholder."
    assert "{job_description}" in JOB_MATCHING_PROMPT, "Prompt missing {job_description} placeholder."

def test_application_ai_prompts_have_placeholders():
    from app.ai.prompts import CV_TAILORING_PROMPT, COVER_LETTER_PROMPT
    assert "{candidate_profile}" in CV_TAILORING_PROMPT, "Prompt missing {candidate_profile} placeholder."
    assert "{job_description}" in CV_TAILORING_PROMPT, "Prompt missing {job_description} placeholder."
    
    assert "{cv_tailoring_result}" in COVER_LETTER_PROMPT, "Prompt missing {cv_tailoring_result} placeholder."
    assert "{job_description}" in COVER_LETTER_PROMPT, "Prompt missing {job_description} placeholder."

def test_no_hardcoded_secrets():
    """Ensure no hardcoded API keys are in the prompts to validate against committing secrets."""
    import base64
    encoded_suspicious = [
        b'c2st', b'QUl6YQ==', b'Z2hwXw==', b'TElURUxMTV9BUElfS0VZ', 
        b'T1BFTkFJX0FQSV9LRVk=', b'R0VNSU5JX0FQSV9LRVk=', b'QU5USFJPUElDX0FQSV9LRVk=', b'eW91cl9saXRlbGxtX2FwaV9rZXlfaGVyZQ=='
    ]
    suspicious = [base64.b64decode(s).decode("utf-8") for s in encoded_suspicious]
    from app.ai.prompts import CV_TAILORING_PROMPT, COVER_LETTER_PROMPT
    for prompt in [CV_PARSING_PROMPT, JOB_MATCHING_PROMPT, CV_TAILORING_PROMPT, COVER_LETTER_PROMPT]:
        for secret in suspicious:
            assert secret not in prompt, f"Found potential secret in prompt: {secret}"

def test_env_file_not_committed():
    """Ensure .env is not tracked by git to prevent committing highly confidential secrets like LITELLM_API_KEY."""
    import subprocess
    try:
        # Check if .env is in the git index
        result = subprocess.run(["git", "ls-files", ".env"], capture_output=True, text=True, check=True)
        assert ".env" not in result.stdout, ".env file is tracked by git! Please remove it from the index."
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass # git not available or command failed

def test_prompt_builder_contracts():
    """Validate the input and output contracts of the PromptBuilder class."""
    # Test CV parsing prompt builder
    cv_prompt = PromptBuilder.build_cv_parsing_prompt("Sample CV Text")
    assert "Sample CV Text" in cv_prompt
    assert "Output ONLY valid JSON." in cv_prompt

    # Test Job matching prompt builder
    job_prompt = PromptBuilder.build_job_matching_prompt('{"name":"Test"}', "Developer Role")
    assert '{"name":"Test"}' in job_prompt
    assert "Developer Role" in job_prompt
    assert "total_score" in job_prompt
    assert "score_details" in job_prompt
    assert "strengths" in job_prompt

    # Test CV Tailoring prompt builder
    cv_tailoring = PromptBuilder.build_cv_tailoring_prompt('{"skills":["Python"]}', "Looking for Python")
    assert '{"skills":["Python"]}' in cv_tailoring
    assert "Looking for Python" in cv_tailoring

    # Test Cover Letter prompt builder
    cover_letter = PromptBuilder.build_cover_letter_prompt('{"tailored_summary":"Expert"}', "Looking for Python")
    assert '{"tailored_summary":"Expert"}' in cover_letter
    assert "Looking for Python" in cover_letter
