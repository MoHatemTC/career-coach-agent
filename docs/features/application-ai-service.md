# Application AI Service

The Application AI Service is responsible for the CV Tailoring and Cover Letter Generation pipeline. It provides candidates with tailored profile summaries and a customized cover letter draft based on a specific target job description.

---

## Architecture

This feature is orchestrated using **LangGraph**, providing a resilient, multi-step pipeline that natively supports state management and early-abort semantics (e.g., if content moderation fails or a job is not found).

### Pipeline Flow

```
START → [job_resolution_node] → [input_validation_node] → [cv_tailoring_node] → [cover_letter_node] → END
              ↓ (error)                ↓ (error)               ↓ (error)
              END                      END                      END
```

### State Management (`ApplicationState`)

| Field | Type | Populated By |
|---|---|---|
| `request` | `ApplicationRequest` | Service (initial injection) |
| `db` | `JobRepository` | Service (dependency injection) |
| `job_description` | `str` | `job_resolution_node` |
| `cv_tailoring_result` | `CVTailoringResult` | `cv_tailoring_node` |
| `cover_letter_result` | `CoverLetterResult` | `cover_letter_node` |
| `error` | `str` | Any node (triggers early abort) |

### Pipeline Nodes

1. **Job Resolution Node (`job_resolution_node`)** — *Stage 0*
   - If `job_description` is provided in the request, uses it directly.
   - If not, resolves the job from the `JobRepository` using `job_id`.
   - Aborts with error if the job is not found and no description was provided.

2. **Input Validation Node (`input_validation_node`)** — *Stage 1*
   - **Content Moderation**: Ensures the job description does not contain inappropriate content. If flagged, it aborts the pipeline.
   - **PII Scrubbing**: Redacts the candidate's name and contact information from the profile before sending it to the LLMs.

3. **CV Tailoring Node (`cv_tailoring_node`)** — *Stage 2*
   - Ingests the scrubbed profile and job description.
   - Prompts the LLM (`cv_tailoring.md`) to generate a customized summary, highlight relevant skills, identify missing skills, and suggest bullet-point rewrites.

4. **Cover Letter Generation Node (`cover_letter_node`)** — *Stage 3*
   - Takes the output of the CV Tailoring Node and the job description.
   - Prompts the LLM (`cover_letter.md`) to write a professional cover letter draft and analyze the tone used.

---

## API Endpoint

### `POST /api/v1/applications/`

Generates tailored CV suggestions and a cover letter draft for a candidate targeting a specific job.

### Request Schema (`ApplicationRequest`)

```json
{
  "candidate_id": "550e8400-e29b-41d4-a716-446655440001",
  "job_id": "550e8400-e29b-41d4-a716-446655440002",
  "candidate_profile": {
    "name": "Ahmed Hassan",
    "contact": {
      "email": "ahmed@example.com",
      "phone": "+201234567890"
    },
    "skills": ["Python", "FastAPI", "Docker", "PostgreSQL"],
    "experience_years": 3,
    "education": ["BSc Computer Science - Cairo University"],
    "preferences": {
      "job_titles": ["Backend Developer"],
      "location": "Remote"
    }
  },
  "job_description": "We are looking for a Senior Python Backend Engineer with 5+ years of experience. Must have strong skills in FastAPI, PostgreSQL, and Docker."
}
```

> **Note:** `job_description` is optional. If omitted, the service resolves it from the `JobRepository` using `job_id`.

### Response Schema (`ApplicationResponse`)

```json
{
  "candidate_id": "550e8400-e29b-41d4-a716-446655440001",
  "job_id": "550e8400-e29b-41d4-a716-446655440002",
  "cv_tailoring": {
    "tailored_summary": "Results-driven Python Backend Developer with 3 years of hands-on experience building scalable APIs using FastAPI and PostgreSQL. Proficient in containerized deployments with Docker.",
    "highlighted_skills": ["Python", "FastAPI", "Docker", "PostgreSQL"],
    "missing_skills": ["Kubernetes", "CI/CD"],
    "bullet_point_suggestions": [
      "Rephrase 'Built APIs' to 'Designed and deployed production FastAPI services handling 10K+ RPM'",
      "Add metrics to Docker experience: 'Containerized 5+ microservices reducing deployment time by 60%'"
    ]
  },
  "cover_letter": {
    "draft_content": "Dear Hiring Manager,\n\nI am writing to express my strong interest in the Senior Python Backend Engineer position. With 3 years of dedicated experience building high-performance backend systems using Python and FastAPI, I am confident in my ability to contribute meaningfully to your team...",
    "tone_analysis": "Professional and confident, with emphasis on technical alignment and growth mindset."
  }
}
```

### Example `curl` Command

```bash
curl -X POST "http://localhost:8000/api/v1/applications/" \
     -H "Content-Type: application/json" \
     -d '{
           "candidate_id": "550e8400-e29b-41d4-a716-446655440001",
           "job_id": "550e8400-e29b-41d4-a716-446655440002",
           "candidate_profile": {
             "name": "Ahmed Hassan",
             "contact": {"email": "ahmed@example.com"},
             "skills": ["Python", "FastAPI", "Docker"],
             "experience_years": 3,
             "education": ["BSc Computer Science"],
             "preferences": {"location": "Remote"}
           },
           "job_description": "Looking for a Python developer with FastAPI experience."
         }'
```

---

## Error Handling

| Status Code | Condition | Example Error |
|---|---|---|
| `200 OK` | Pipeline completes successfully | — |
| `400 Bad Request` | Content moderation blocks the job description | `"Job description blocked by content moderation: contains inappropriate content."` |
| `404 Not Found` | `job_id` not found in DB (when no `job_description` provided) | `"Job with ID 550e8400-... not found."` |
| `422 Unprocessable Entity` | Other validation errors (e.g., no DB and no description) | `"No job description provided and no database connection available."` |
| `500 Internal Server Error` | LLM API failure or unexpected pipeline error | `"Application AI pipeline failed: LLM API timeout"` |

---

## Responsible AI Guardrails

We enforce several strict guardrails through prompt engineering and pipeline design:

- **No Hallucination**: The model is explicitly instructed not to invent skills or experiences the candidate does not have.
- **Separation of Missing Skills**: Required skills the candidate lacks are segregated into a `missing_skills` array, ensuring they aren't falsely claimed in the CV or Cover Letter.
- **Professional, Bias-Free Tone**: The cover letter prompt enforces inclusive language free of demographic biases.
- **Data Privacy**: PII is scrubbed in the `input_validation_node` prior to any LLM API calls.
- **Content Moderation**: Hardcoded filters block inappropriate job descriptions before any LLM execution occurs.

---

## Structured Output Enforcement

The service enforces deterministic, structured JSON output at two levels:

1. **LLM Level**: All LLM calls use `response_format={"type": "json_object"}` via LiteLLM, instructing the model to output only valid JSON.
2. **Validation Level**: The raw JSON string from the LLM is parsed and validated using Pydantic's `model_validate_json()`, ensuring the output strictly conforms to the `CVTailoringResult` and `CoverLetterResult` schemas. Any schema violation raises a validation error that is caught and surfaced.

---

## Integration with JobRepository

The service can operate in two modes:

| Mode | When | Behavior |
|---|---|---|
| **Direct** | `job_description` is provided in the request | Uses the provided description directly; no DB call |
| **DB Resolution** | `job_description` is `None` | Calls `db.get_job(job_id)` to fetch the job record and uses its `description` field |

This is implemented via the `job_resolution_node` at the beginning of the LangGraph pipeline. The `JobRepository` is injected into the `ApplicationAIService` constructor following the same dependency injection pattern used by `JobMatchingService`.

---

## Usage

### Python (Direct Service Call)

```python
from app.services.application_ai_service import ApplicationAIService
from app.schemas.application_ai import ApplicationRequest
from app.db.repositories import InMemoryJobRepository

# With DB connection
db = InMemoryJobRepository()
service = ApplicationAIService(db=db)
response = await service.generate_application_materials(request_payload)

print(response.cv_tailoring.tailored_summary)
print(response.cover_letter.draft_content)
```

### REST API (via FastAPI)

```bash
# Start the server
uvicorn main:app --reload

# Call the endpoint
curl -X POST http://localhost:8000/api/v1/applications/ \
     -H "Content-Type: application/json" \
     -d '{ ... }'
```

---

## Testing

The test suite (`tests/services/test_application_ai_service.py`) covers 9 test cases:

| Test | Validates |
|---|---|
| `test_application_ai_service_success` | Full happy-path pipeline with direct job_description |
| `test_application_ai_service_content_moderation_failure` | Blocked keywords rejection |
| `test_job_resolution_from_db` | Job description resolved from mocked JobRepository |
| `test_job_not_found_in_db` | ValueError when job_id not in DB |
| `test_no_db_and_no_job_description` | ValueError when no description and no DB |
| `test_cv_tailoring_llm_failure` | Graceful error on CV tailoring LLM failure |
| `test_cover_letter_llm_failure` | Graceful error on cover letter LLM failure |
| `test_empty_skills_candidate` | Handles candidates with no skills |
| `test_pii_scrubbing_removes_contact_info` | Verifies PII not leaked to LLM |

```bash
pytest tests/services/test_application_ai_service.py -v
```
