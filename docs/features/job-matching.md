# AI Job Matching Engine Architecture

## Overview
The AI Job Matching Engine evaluates candidate profiles against job postings using a hybrid Two-Stage pipeline. It balances low-latency vector pre-filtering with high-context deep semantic analysis via LangGraph orchestration.

---

## Pipeline Architecture

The engine is orchestrated as a `langgraph.StateGraph` compiled from two discrete nodes:

```
START → [pre_filter_node] → [llm_evaluation_node] → END
                    ↓ (if job not found)
                   END
```

### Stage 1: Vector Pre-filtering (`pre_filter_node`)
The first node simulates a `pgvector` cosine/L2 distance search to isolate the top candidate jobs before passing them to the expensive LLM step.

- **Dimensionality:** `1536` — Standard dimension for text embedding models (e.g., `text-embedding-ada-002`). This is a defensible assumption made in coordination with the Sprint mentor (Sarah Nader), as Sprint 1 DB specs were not available at the time of implementation.
- **Distance Metric:** L2 distance (Euclidean) — suitable for normalized embeddings.
- **Mock Layer:** `MockDB.get_job()` and `MockDB.vector_search_jobs()` simulate this layer pending upstream pgvector integration.

### Stage 2: LLM Re-ranking (`llm_evaluation_node`)
The top candidates are evaluated by a Large Language Model using a strict 40/30/20/10 weighted rubric.

- **Proxy Routing:** LiteLLM (configured via `LITELLM_BASE_URL` and `LITELLM_API_KEY`)
- **Model:** Configured dynamically via the `DEFAULT_MODEL` environment variable (default: `azure/FW-Kimi-K2.6`)
- **Context Window:** 256K tokens, supporting rich multi-job evaluation in a single inference call
- **Output Enforcement:** `response_format={"type": "json_object"}` + Pydantic `model_validate_json()` for deterministic parsing

---

## Mathematical Scoring Model

The total match score (0–100) is calculated deterministically using this weighted formula:

```
Final Score = Hard Skills (40) + Experience (30) + Soft Skills (20) + Logistics (10)
```

| Dimension | Max Points | Full Match Criteria |
|---|---|---|
| Hard Skills | 40 | All mandatory technical skills present |
| Experience | 30 | Years of experience meets or exceeds requirement |
| Soft Skills | 20 | Explicit evidence of leadership/communication |
| Logistics | 10 | Remote/timezone preferences align |

---

## ORM & Data Type Assumptions

As confirmed by mentor Sarah Nader, the following **standard, defensible assumptions** were made to unblock implementation while Sprint 1 DB schemas were pending:

| Field | Assumed Type | Rationale |
|---|---|---|
| `User.id` | `uuid.UUID` | Standard RFC 4122 UUID for distributed systems |
| `Job.id` | `uuid.UUID` | Consistent with User PK convention |
| `User.profile` | `JSONB` | Native PostgreSQL JSON enables Pydantic serialization without additional parsing |
| `Job.embedding` | `vector(1536)` | pgvector column; 1536 is the standard OpenAI embedding dimension |

> These assumptions are documented here per mentor guidance and must be reconciled with the Sprint 1 DB schema upon merge.

---

## Interface Stubs

The following mock classes simulate the database boundary:

- **`JobMock`** — Represents a normalized job entity with `id: uuid.UUID`, `title: str`, `description: str`, and `embedding: List[float]` (1536-dimensional)
- **`UserMock`** — Represents a candidate entity with `id: uuid.UUID`, `profile: dict` (JSONB-compatible), and `embedding: List[float]` (1536-dimensional)
- **`MockDB`** — Simulates async pgvector queries: `get_job()` and `vector_search_jobs()`

---

## Structured Output Format

Every match produces a `JobMatchResponse` JSON payload suitable for ranking, logging, and notifications:

```json
{
  "job_id": "11111111-1111-1111-1111-111111111111",
  "candidate_id": "22222222-2222-2222-2222-222222222222",
  "vector_distance": 0.12,
  "result": {
    "score_details": {
      "hard_skills_score": 40,
      "experience_score": 30,
      "soft_skills_score": 20,
      "logistics_score": 10
    },
    "total_score": 100,
    "explanation": "Candidate fully meets all criteria...",
    "strengths": ["Python", "FastAPI", "LLM Orchestration"],
    "missing_skills": [],
    "recommendation": "Strongly recommend moving to interview phase."
  },
  "status": "Draft - Awaiting Human Approval",
  "disclaimer": "AI-generated content. A human-in-the-loop review is required before use."
}
```

---

## Mock Test Data

Three candidate personas were created in `tests/services/mock_data.py` to test the outer bounds of the scoring logic against a standardized job posting:

| Persona | Profile | Expected Score Band |
|---|---|---|
| Grace Hopper (Perfect) | All skills, 6yr exp, remote ✅ | 80–100 |
| Ada Lovelace (Partial) | Missing FastAPI/LLM, 5yr exp | 40–79 |
| Alan Turing (Poor) | Wrong domain, 1yr exp, on-site ❌ | 0–39 |

---

## Limitations

- **Advisory Only:** The matching engine relies on probabilistic LLM evaluation. Match scores should be treated as **advisory suggestions** to assist human recruiters, rather than definitive automated decisions.
- **Context Window Limits:** While the proxy supports large context models, passing extremely large CV documents or multiple concurrent job descriptions may approach token limits, resulting in truncated evaluation.
- **Latency**: Stage 2 relies on external LLM inference, which inherently introduces latency. High-throughput scenarios should rely heavily on Stage 1 (vector pre-filtering) to reduce LLM calls.
- **Bias and Hallucination**: LLMs can occasionally hallucinate skills or implicitly favor certain phrasing over others. We continually refine our rubric prompts to mitigate this, but human oversight is recommended.

---

## Human in the Loop (HITL)

As a critical safety net, all AI-generated job matches and evaluations are explicitly marked as drafts awaiting human approval. Because Large Language Models can occasionally hallucinate, show bias, or miss subtle context, a real person—like a career coach or recruiter—must review, verify, and approve the matching results before any definitive action is taken. The API reflects this status by returning a warning flag and a disclaimer indicating that the evaluation requires human oversight.

---

## Privacy & Data Protection

Protecting candidate data is a primary directive for the AI Job Matching Engine.

- **PII Scrubbing:** All personally identifiable information (PII) such as phone numbers, physical addresses, email addresses, and full names MUST be stripped or masked prior to sending the candidate profile to the LLM. 
- **Logging Policies:** Never log raw CV text, LLM evaluation inputs, or full candidate profiles to standard output, APM tools, or centralized logging systems. Only log anonymized UUIDs and the resulting match score.
- **Data Retention (GDPR/CCPA):** Temporary cache of evaluated data must respect the platform's data retention policies, and users must have the ability to trigger a "right to be forgotten" cascade that purges their vectors from the database.

---

## Sample API Usage

To trigger the matching pipeline, you can make a `POST` request to the `/matches/` endpoint with a `MatchRequest` payload.

**Request:**
```bash
curl -X POST "http://localhost:8000/matches/" \
     -H "Content-Type: application/json" \
     -d '{
           "candidate_id": "22222222-2222-2222-2222-222222222222",
           "job_id": "11111111-1111-1111-1111-111111111111",
           "candidate_profile": {
             "name": "Grace Hopper",
             "contact": {"email": "grace@example.com"},
             "skills": ["Python", "FastAPI", "PostgreSQL", "LLM Orchestration"],
             "experience_years": 6,
             "education": ["MSc Computer Science"],
             "preferences": {"remote": true, "timezone": "EST"}
           }
         }'
```

**Response (200 OK):**
```json
{
  "job_id": "11111111-1111-1111-1111-111111111111",
  "candidate_id": "22222222-2222-2222-2222-222222222222",
  "vector_distance": 0.12,
  "result": {
    "score_details": {
      "hard_skills_score": 40,
      "experience_score": 30,
      "soft_skills_score": 20,
      "logistics_score": 10
    },
    "total_score": 100,
    "explanation": "Candidate fully meets all criteria...",
    "strengths": ["Python", "FastAPI", "LLM Orchestration"],
    "missing_skills": [],
    "recommendation": "Strongly recommend moving to interview phase."
  },
  "status": "Draft - Awaiting Human Approval",
  "disclaimer": "AI-generated content. A human-in-the-loop review is required before use."
}
```
