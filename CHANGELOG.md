# Changelog

This project follows a lightweight changelog format. The public API lives under the `rlm` import package.

## 1.0.0

First stable release of the hexagonal architecture refactor.

### Breaking Changes

- Package renamed from `rlm` to `agentic-codebase-navigator` on PyPI (import remains `rlm`)

### Features

- **Hexagonal architecture**: Complete ports/adapters refactor with clean domain boundaries
- **Stable public API**: `RLM`, `create_rlm`, `create_rlm_from_config`, config classes
- **Multi-backend LLM support**: OpenAI, Anthropic, Gemini, Azure OpenAI, LiteLLM, Portkey
- **Execution environments**: Local (in-process) and Docker (isolated container)
- **TCP broker**: Request routing with wire protocol for nested `llm_query()` calls
- **Mock LLM adapter**: Deterministic testing without API keys
- **JSONL logging**: Versioned schema (v1) for execution tracing
- **CLI**: `rlm completion` with backend/environment options

### Infrastructure

- GitHub Actions CI: unit/integration/e2e/packaging test gates
- Comprehensive pre-commit hooks: security scanning, type checking, linting
- 90% code coverage requirement
- `uv` package manager support

### Attribution

This project is based on the [Recursive Language Models](https://github.com/alexzhang13/rlm) research by Alex L. Zhang, Tim Kraska, and Omar Khattab (MIT OASYS Lab). See [ATTRIBUTION.md](ATTRIBUTION.md) for details.

## 0.1.2

- Hexagonal modular-monolith refactor (ports/adapters) under `src/rlm/`
- Stable public API: `create_rlm`, `create_rlm_from_config`, `RLMConfig`/`LLMConfig`/`EnvironmentConfig`/`LoggerConfig`
- Deterministic test and packaging gates (unit/integration/e2e/packaging/performance)
- TCP broker with batched concurrency and safe error mapping
- Docker environment adapter with best-effort cleanup, timeouts, and host proxy for nested `llm_query`
- Versioned JSONL logging schema (v1) with console/no-op logger options
- Opt-in live provider smoke tests (OpenAI/Anthropic) gated by `RLM_RUN_LIVE_LLM_TESTS=1`
