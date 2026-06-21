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

def test_no_hardcoded_secrets():
    """Ensure no hardcoded API keys are in the prompts to validate against committing secrets."""
    # Included API key prefixes if any, and generic secrets.
    suspicious = ["sk-", "AIza", "ghp_", "LITELLM_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "your_litellm_api_key_here"]
    for prompt in [CV_PARSING_PROMPT, JOB_MATCHING_PROMPT]:
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
