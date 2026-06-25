# Application AI Service

The Application AI Service is responsible for the CV Tailoring and Cover Letter Generation pipeline. It provides candidates with tailored profile summaries and a customized cover letter draft based on a specific target job description.

## Architecture

This feature is orchestrated using **LangGraph**, providing a resilient, multi-step pipeline that natively supports state management and early-abort semantics (e.g., if content moderation fails).

### State Management (`ApplicationState`)
- `request`: Contains the CandidateProfile and Target Job Description.
- `cv_tailoring_result`: Populated in Stage 2.
- `cover_letter_result`: Populated in Stage 3.
- `error`: Captures any validation or generation errors, causing the graph to halt and fail gracefully.

### Pipeline Nodes

1. **Input Validation Node (`input_validation_node`)**
   - **Content Moderation**: Ensures the job description does not contain inappropriate content. If flagged, it aborts the pipeline.
   - **PII Scrubbing**: Redacts the candidate's name and contact information from the profile before sending it to the LLMs.
   
2. **CV Tailoring Node (`cv_tailoring_node`)**
   - Ingests the scrubbed profile and job description.
   - Prompts the LLM (`cv_tailoring.md`) to generate a customized summary, highlight relevant skills, identify missing skills, and suggest bullet-point rewrites.
   
3. **Cover Letter Generation Node (`cover_letter_node`)**
   - Takes the output of the CV Tailoring Node and the job description.
   - Prompts the LLM (`cover_letter.md`) to write a professional cover letter draft and analyze the tone used.

## Responsible AI Guardrails

We enforce several strict guardrails through prompt engineering and pipeline design:

- **No Hallucination**: The model is explicitly instructed not to invent skills or experiences the candidate does not have.
- **Separation of Missing Skills**: Required skills the candidate lacks are segregated into a `missing_skills` array, ensuring they aren't falsely claimed in the CV or Cover Letter.
- **Professional, Bias-Free Tone**: The cover letter prompt enforces inclusive language free of demographic biases.
- **Data Privacy**: PII is scrubbed in the `input_validation_node` prior to any LLM API calls.
- **Content Moderation**: Hardcoded filters block inappropriate job descriptions before any LLM execution occurs.

## Usage

```python
from app.services.application_ai_service import ApplicationAIService
from app.schemas.application_ai import ApplicationRequest

service = ApplicationAIService()
response = await service.generate_application_materials(request_payload)

print(response.cv_tailoring.tailored_summary)
print(response.cover_letter.draft_content)
```
