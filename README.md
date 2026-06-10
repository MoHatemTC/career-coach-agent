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
