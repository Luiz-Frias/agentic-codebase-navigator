# Repository Guidelines

This repository is an agentic codebase navigator built on RLM patterns and is migrating the upstream snapshot in
`references/rlm/**` into a hexagonal modular monolith under `src/rlm/**`. Keep changes focused and aligned with
the layered architecture (ports, adapters, domain).

## Project Structure & Module Organization
- `src/rlm/`: production code
  - `domain/` (pure core), `application/` (use cases), `adapters/` (LLM/env/broker), `infrastructure/`, `api/`
  - `_legacy/` holds compatibility shims; avoid adding new features there
- `tests/`: pytest suite (unit, integration, docker-marked tests)
- `references/rlm/`: upstream snapshot for migration reference only; do not import from it
- `docs/`: supplemental documentation and examples

## Build, Test, and Development Commands
```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```
```bash
uv run --group test pytest -m unit
uv run --group test pytest -m integration
uv run --group test pytest -m docker
```
```bash
uv run ruff check --fix .
uv run ruff format .
```

## Coding Style & Naming Conventions
- Python 3.12+, Ruff formatting (double quotes, 100-char lines)
- Naming: `snake_case` functions/vars, `PascalCase` classes, `UPPER_CASE` constants
- Keep `domain/` free of adapter or infrastructure imports; depend on `ports.py` abstractions
- Prefer optional extras for non-core integrations (`.[llm-openai]`, `.[env-modal]`)

## Testing Guidelines
- Pytest is required for all behavior changes; add/adjust tests in `tests/`
- Use markers (`unit`, `integration`, `docker`) to keep suites fast and deterministic
- Mock external services; integration tests should skip cleanly when Docker is unavailable

## Commit & Pull Request Guidelines
- Use Conventional Commits: `feat(scope): ...`, `refactor(scope): ...`, `docs(README): ...`
- Keep scopes short and meaningful (e.g., `environments`, `usage`, `broker`)
- PRs should include: what changed, why, tests run, and any migration notes for `src/rlm` vs `references/rlm`

## Security & Configuration Tips
- API keys are environment variables only (e.g., `OPENAI_API_KEY`); never hardcode secrets
- Docker-backed tests require a working local Docker daemon and CLI
