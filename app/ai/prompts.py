"""
AI Prompt Templates and Builders for Career Coach.

This module provides the necessary prompt templates for integrating with the
LiteLLM proxy and LangGraph state nodes.
"""
import os

# Prompt template for parsing a candidate's CV document.
#
# Input parameters:
#     cv_text (str): The raw text extracted from the candidate's CV.
#
# Expected JSON output contract:
#     {
#         "name": "str",
#         "contact": {
#             "email": "str",
#             "phone": "str"
#         },
#         "skills": ["str"],
#         "experience_years": int,
#         "education": ["str"],
#         "preferences": {}
#     }

def load_prompt(filename: str) -> str:
    prompt_path = os.path.join(
        os.path.dirname(__file__), 
        "..", "core", "prompts", filename
    )
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Required prompt file '{filename}' not found at: {prompt_path}")

# Load Prompts from markdown files
JOB_MATCHING_PROMPT = load_prompt("job_matching.md")
CV_TAILORING_PROMPT = load_prompt("cv_tailoring.md")
COVER_LETTER_PROMPT = load_prompt("cover_letter.md")

# CV Parsing doesn't have an external file yet
CV_PARSING_PROMPT = """
You are an expert technical recruiter and career coach. Your task is to parse the following CV text and extract the candidate's profile into a structured JSON format.

Extract the following fields:
- "name": string
- "contact": object with email and phone
- "skills": list of strings (technical and soft skills)
- "experience_years": integer
- "education": list of degrees/certifications
- "preferences": object (e.g., job titles of interest, locations if mentioned)

CV Text:
{cv_text}

Output ONLY valid JSON.
"""

class PromptBuilder:
    """
    Builder class for constructing AI prompts with the necessary context and templates.
    """

    @staticmethod
    def build_cv_parsing_prompt(cv_text: str) -> str:
        return CV_PARSING_PROMPT.replace("{cv_text}", cv_text)

    @staticmethod
    def build_job_matching_prompt(candidate_profile: str, job_description: str) -> str:
        prompt = JOB_MATCHING_PROMPT
        prompt = prompt.replace("{candidate_profile}", candidate_profile)
        prompt = prompt.replace("{job_description}", job_description)
        return prompt

    @staticmethod
    def build_cv_tailoring_prompt(candidate_profile: str, job_description: str) -> str:
        prompt = CV_TAILORING_PROMPT
        prompt = prompt.replace("{candidate_profile}", candidate_profile)
        prompt = prompt.replace("{job_description}", job_description)
        return prompt
        
    @staticmethod
    def build_cover_letter_prompt(cv_tailoring_result: str, job_description: str) -> str:
        prompt = COVER_LETTER_PROMPT
        prompt = prompt.replace("{cv_tailoring_result}", cv_tailoring_result)
        prompt = prompt.replace("{job_description}", job_description)
        return prompt
