# AI Career Coach

Welcome to the AI Career Coach project! This repository contains the AI assistant designed to help learners and job seekers find suitable job opportunities, match their skills, and tailor their applications.

> [!CAUTION]
> **STRICT SECURITY WARNING:** The `LITELLM_API_KEY` is highly confidential. It grants access to our centralized proxy and must **ONLY** reside in your local `.env` file. Do not hardcode this key anywhere in the source code or commit `.env` to the repository.

## Sprint 1: Foundation Setup

In this phase, we have established the project's engineering foundation and AI architecture based on the mandated `fastapi-langgraph-agent-production-ready-template`. 

### Design Decisions

- **LangGraph for Orchestration**: We use LangGraph to orchestrate our AI workflows. It provides robust state management, making it easy to handle complex cyclic graphs (e.g., extracting a CV, evaluating it, and iterating on feedback).
- **LiteLLM for Routing**: Instead of relying on direct provider SDKs (OpenAI, Gemini, Anthropic), we have integrated LiteLLM as our universal gateway. This centralizes API key management via `LITELLM_API_KEY`, simplifies model routing, and enables strict cost-tracking and rate-limiting at the organizational level.

### Reasoning for Prompt Design and Rubric Scoring

- **Prompt Design Reasoning**: 
  We chose to enforce strict JSON structures in our prompt builders. This design makes the LLM output highly predictable, ensuring seamless downstream processing by LangGraph state and our internal APIs without complex regex parsing. Additionally, we split parsing and matching into separate, discrete prompts rather than a single monolithic prompt. This prevents "context-dilution," improving accuracy by ensuring the LLM is focused exclusively on one cognitive task at a time.
- **Rubric Scoring Reasoning**: 
  Our scoring rubric uses a 40/30/20/10 weighting. We prioritize Hard Skills (40%) and Experience Level (30%) because technical roles are primarily gated by technical competence and tenure. While Soft Skills (20%) and Logistics (10%) are crucial, they more often serve as secondary tie-breakers or deal-breakers during the interview process, rather than primary qualifications for the initial resume screen.

### Module Boundaries

To ensure independent workflows and high maintainability, we've structured the codebase with clear boundaries allowing the team to work concurrently:

- **API Routes (FastAPI layer)**: Exposes our LangGraph agent as a REST API. Engineers can build new endpoints without modifying the underlying agent state.
- **Agent Orchestration (LangGraph layer)**: Handles state transitions and memory. This is separate from prompts and specific API business logic.
- **Decoupled Prompt Management (`app/ai/prompts.py`)**: Prompts and their builders are decoupled from execution logic, allowing prompt engineers to refine instructions based on rigorous input/output contracts.
- **Independent Testing Layer (`tests/`)**: Structural tests and strict security validations operate independently in our CI pipeline.

## Assumptions and Limitations

To ensure transparency and manage expectations, the following assumptions and technical limitations are explicitly called out:

- **Shared LLM Registry via Proxy**: All LLM inference calls are routed through a shared `LLMServiceRegistry` using the LiteLLM proxy. It is assumed the proxy handles robust fallback and rate-limiting. A critical limitation is that if the central proxy is down, all AI features (job matching, CV tailoring, cover letter generation) will fail simultaneously.
- **Structured Output Parsing**: The service layer relies strictly on Pydantic models (`CVTailoringResult`, `CoverLetterResult`) alongside LiteLLM's `json_object` format enforcement. It is assumed the underlying model accurately follows the JSON schema. Occasional LLM hallucination of field names or malformed JSON could cause the `model_validate_json` step to raise validation errors, though the graph orchestration handles these gracefully by surfacing the error to the user.
- **Responsible AI and Human-in-the-Loop**: All AI-generated application materials (CV suggestions and Cover Letters) are explicitly returned with a `Draft - Awaiting Human Approval` status and a Responsible AI disclaimer. The AI cannot guarantee factual accuracy regarding a candidate's unspoken skills and may hallucinate experiences. A strict human-in-the-loop review is assumed and required before any submission.
- **Schema Consistency**: Typed request and response schemas (`ApplicationRequest`, `ApplicationResponse`) are strictly enforced across all API routes and tests. It is assumed that the client will provide well-formed UUIDs and complete `CandidateProfile` objects.

### User-Data Privacy Guidance

Protecting candidate data is a primary concern. To ensure compliance with privacy standards, the following guidelines are strictly enforced:
- **PII Scrubbing**: Before any data is transmitted to the LLM Service Registry, Personally Identifiable Information (PII) such as the candidate's exact name, email address, and phone numbers are explicitly scrubbed and replaced with "REDACTED" in the service layer (`ApplicationAIService`).
- **Data Retention**: The proxy and underlying LLM models are assumed to be configured with zero-data-retention policies.
- **Log Masking**: System logs (`logger.info`, `logger.error`) must never record candidate profile JSONs. Only unique UUIDs (`candidate_id`, `job_id`) and processing outcomes should be written to standard output.

## Getting Started

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables:**
   Copy `.env.example` to `.env` and fill in your proxy configuration.
   ```bash
   cp .env.example .env
   ```
   *Remember: Never commit your `.env` file!*

3. **Running Tests:**
   Use pytest to run the validation suite for our AI prompts and strict security checks.
   ```bash
   pytest tests/
   ```

## Architecture Layout

- `app/ai/`: Contains core AI logic, prompt templates, and the `PromptBuilder` (`prompts.py`). These components feed strictly-typed JSON outputs back into our LangGraph nodes.
- `tests/`: Contains automated tests ensuring codebase structural integrity and preventing secret leaks (`test_prompts.py`).
- `.env.example`: Template for required LiteLLM environment variables.
