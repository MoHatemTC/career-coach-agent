# Development Context & State Tracker

## Overview
This file tracks the current state of the workspace, goals, completed items, and steps to recover if a failure occurs during execution.

## Goal
- Implement Option B: "Use a local model" for generating embeddings in the Career Coach backend.
- Integrate the existing `LocalEmbedder` (which uses `sentence-transformers`) into the unified `LLMServiceRegistry`.
- Make it configurable via environment variables (`DEFAULT_EMBEDDING_MODEL` and dynamic embedding dimensions).

## Current Status
- [x] Fixed router path prefix in `main.py` from `/api` to `/api/v1` to match API documentation and resolve failing integration tests (69/69 tests now passing).
- [x] Drafted implementation plan for local model integration and proposed it to the user.
- [x] Wait for user approval on the implementation plan.
- [x] Install `sentence-transformers` package in the virtual environment.
- [x] Add `sentence-transformers` dependency to `requirements.txt`.
- [x] Integrate local model option in `app/ai/registry.py` under the `embed` method when the model name starts with or matches `"local"`.
- [x] Make the database model `embedding` field column dimension dynamic based on the configuration (e.g. 384 for local MiniLM vs 1536 for Ada/OpenAI) or support 384-dimensional vectors.
- [x] Verify local model works end-to-end and run test suite.

## Virtual Environment Status
- Python interpreter: `.venv\Scripts\python.exe`
- Test runner: `pytest` (runs via `.venv\Scripts\python -m pytest` to preserve pythonpath)
- Installed dependencies: `sentence-transformers`

## How to Resume on Failure
1. If the connection fails or session restarts, read this file (`context.md`).
2. Run `.venv\Scripts\python -m pytest` to verify current tests.
3. Install dependencies: `.venv\Scripts\pip install sentence-transformers`.
