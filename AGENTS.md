# Repository Guidelines

This repository is an agentic codebase navigator built on RLM patterns and is migrating the upstream snapshot in
`references/rlm/**` into a hexagonal modular monolith under `src/rlm/**`. Keep changes focused and aligned with
the layered architecture (ports, adapters, domain).

## Project Structure & Module Organization
- `src/rlm/`: production code
  - `domain/` (pure core: ports, models, services)
  - `application/` (use cases, configuration)
  - `adapters/` (LLM providers, environments, tools, policies, broker, loggers)
  - `infrastructure/` (wire protocol, logging)
  - `api/` (public facade, factories, registries)
  - `_legacy/` holds compatibility shims; avoid adding new features there
- `tests/`: pytest suite (unit, integration, e2e, packaging, performance, benchmark, docker-marked tests)
- `references/rlm/`: upstream snapshot for migration reference only; do not import from it
- `docs/`: supplemental documentation and examples (API, extending, ADRs)

## Build, Test, and Development Commands
```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv sync --group dev --group test
```
```bash
uv run --group test pytest -m unit
uv run --group test pytest -m integration
uv run --group test pytest -m e2e
uv run --group test pytest -m packaging
uv run --group test pytest -m performance
uv run --group test pytest -m benchmark
uv run --group test pytest -m docker
uv run --group test pytest
```
```bash
uv run --group dev ruff format src tests
uv run --group dev ruff check src tests --fix
uv run --group dev ty check src/rlm
```

## Coding Style & Naming Conventions
- Python 3.12+, Ruff formatting (double quotes, 100-char lines)
- Naming: `snake_case` functions/vars, `PascalCase` classes, `UPPER_CASE` constants
- Keep `domain/` free of adapter or infrastructure imports; depend on domain ports (`*_ports.py`)
- Keep dependencies flowing inward (adapters -> application -> domain); adapters implement ports
- Prefer optional extras for non-core integrations (`.[llm-openai]`, `.[llm-anthropic]`, `.[llm-gemini]`, `.[env-modal]`, `.[env-docker]`)

## Testing Guidelines
- Pytest is required for all behavior changes; add/adjust tests in `tests/`
- Use markers (`unit`, `integration`, `e2e`, `packaging`, `performance`, `benchmark`, `docker`, `live_llm`, `chaos`) to keep suites fast and deterministic
- Mock external services; integration tests should skip cleanly when Docker is unavailable
- Live provider tests are opt-in; require `RLM_RUN_LIVE_LLM_TESTS=1` and API keys

## Commit & Pull Request Guidelines
- Use Conventional Commits: `feat(scope): ...`, `refactor(scope): ...`, `docs(README): ...`
- Keep scopes short and meaningful (e.g., `environments`, `usage`, `broker`)
- PRs should include: what changed, why, tests run, and any migration notes for `src/rlm` vs `references/rlm`

## Security & Configuration Tips
- API keys are environment variables only (e.g., `OPENAI_API_KEY`); never hardcode secrets
- Docker-backed tests require a working local Docker daemon and CLI
