"""
AI Prompt Templates and Builders for Career Coach.

This module provides the necessary prompt templates for integrating with the
LiteLLM proxy and LangGraph state nodes.
"""

"""
Prompt template for parsing a candidate's CV document.

Input parameters:
    cv_text (str): The raw text extracted from the candidate's CV.

Expected JSON output contract:
    {
        "name": "str",
        "contact": {
            "email": "str",
            "phone": "str"
        },
        "skills": ["str"],
        "experience_years": int,
        "education": ["str"],
        "preferences": {}
    }
"""
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

"""
Prompt template for evaluating a candidate's fit for a specific job description.

Input parameters:
    candidate_profile (str): The structured JSON profile of the candidate.
    job_description (str): The text description of the target job.

Expected JSON output contract:
    {
        "score_details": {
            "hard_skills_score": int (0-40),
            "experience_score": int (0-30),
            "soft_skills_score": int (0-20),
            "logistics_score": int (0-10)
        },
        "total_score": int (0-100),
        "explanation": "str",
        "strengths": ["str"],
        "missing_skills": ["str"],
        "recommendation": "str"
    }
"""
JOB_MATCHING_PROMPT = """
You are an expert career coach analyzing how well a candidate fits a specific job description.

Candidate Profile (JSON):
{candidate_profile}

Job Description:
{job_description}

Based on the matching rubric, evaluate the candidate's fit for this role.
Provide your evaluation in the following JSON structure:
- "score_details": object containing scoring breakdown:
  - "hard_skills_score": integer (0 to 40)
  - "experience_score": integer (0 to 30)
  - "soft_skills_score": integer (0 to 20)
  - "logistics_score": integer (0 to 10)
- "total_score": integer from 0 to 100 (sum of the above)
- "explanation": string detailing the reasoning for the score
- "strengths": list of strings identifying key strengths and matching skills
- "missing_skills": list of strings identifying skills required by the job that the candidate lacks
- "recommendation": string with actionable advice for the candidate to improve their chances

Output ONLY valid JSON.
"""

class PromptBuilder:
    """
    Builder class for constructing AI prompts with the necessary context and templates.
    
    Provides clear input and output contracts for prompt generation, designed specifically 
    to be consumed by LangGraph orchestration nodes.
    """

    @staticmethod
    def build_cv_parsing_prompt(cv_text: str) -> str:
        """
        Build the prompt used for parsing a candidate's CV document.

        Args:
            cv_text (str): The raw text extracted from the candidate's CV document.

        Returns:
            str: A formatted string prompt instructing the LLM to extract a structured 
            JSON profile. The expected JSON output contract from the LLM contains:
            {
                "name": "str",
                "contact": {"email": "str", "phone": "str"},
                "skills": ["str"],
                "experience_years": int,
                "education": ["str"],
                "preferences": {}
            }
        """
        return CV_PARSING_PROMPT.format(cv_text=cv_text)

    @staticmethod
    def build_job_matching_prompt(candidate_profile: str, job_description: str) -> str:
        """
        Build the prompt used for evaluating a candidate's fit for a specific job.

        Args:
            candidate_profile (str): The structured JSON profile of the candidate.
            job_description (str): The text description of the target job.

        Returns:
            str: A formatted string prompt instructing the LLM to evaluate the match.
            The expected JSON output contract from the LLM contains:
            {
                "score_details": {
                    "hard_skills_score": int,
                    "experience_score": int,
                    "soft_skills_score": int,
                    "logistics_score": int
                },
                "total_score": int,
                "explanation": "str",
                "strengths": ["str"],
                "missing_skills": ["str"],
                "recommendation": "str"
            }
        """
        return JOB_MATCHING_PROMPT.format(
            candidate_profile=candidate_profile, 
            job_description=job_description
        )
