# Integration Gates Runbook

This repo uses **pytest markers** as milestone gates. The intent is to keep feedback fast while still providing
high-confidence release signals.

## Marker suites

- **unit**: fast, hermetic tests (no network, no docker required)
- **integration**: multi-component boundaries (wire protocol, registries, etc.)
- **e2e**: public API end-to-end flows (docker-marked tests auto-skip if Docker is unavailable)
- **packaging**: build/install/import/CLI smoke tests
- **performance**: regression/perf guard tests (opt-in by default)
- **live_llm**: opt-in real-provider smoke tests (requires `RLM_RUN_LIVE_LLM_TESTS=1`)

## Recommended workflow

During development:

```bash
uv run --group test pytest -m unit
```

Before merging:

```bash
uv run --group test pytest -m unit
uv run --group test pytest -m integration
uv run --group test pytest -m e2e
uv run --group test pytest -m packaging
```

Optional (pre-release regression guard):

```bash
uv run --group test pytest -m performance
```

## Live provider tests

These tests are **skipped by default** to avoid accidental spend:

```bash
RLM_RUN_LIVE_LLM_TESTS=1 OPENAI_API_KEY=... uv run --group test pytest -m "integration and live_llm"
RLM_RUN_LIVE_LLM_TESTS=1 ANTHROPIC_API_KEY=... uv run --group test pytest -m "integration and live_llm"
```
