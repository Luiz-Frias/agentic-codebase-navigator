# Changelog

This project follows a lightweight changelog format. The public API lives under the `rlm` import package.

## 1.1.0

Major feature release introducing tool calling agent capabilities and extensibility protocols.

### Features

- **Tool Calling Agent Mode**: Full agentic tool loop with native support across all LLM providers
  - Native tool calling for OpenAI, Anthropic, Gemini, Azure OpenAI, LiteLLM, and Portkey adapters
  - Tool registry with `@tool` decorator for defining callable functions
  - Automatic Pydantic model → JSON Schema conversion for structured outputs
  - Conversation management with message history and multi-turn tool execution
  - `tool_choice` parameter support (`auto`, `required`, `none`, or specific tool)
  - Prompt token counting via `count_tokens()` on all adapters

- **Extension Protocols**: Duck-typed protocols for customizing orchestrator behavior
  - `StoppingPolicy`: Control when the tool loop terminates
  - `ContextCompressor`: Compress conversation context between iterations
  - `NestedCallPolicy`: Configure handling of nested `llm_query()` calls
  - Default implementations: `DefaultStoppingPolicy`, `NoOpContextCompressor`, `SimpleNestedCallPolicy`
  - Full documentation in `docs/extending.md`

- **Performance Benchmarks**: Comprehensive profiling infrastructure
  - Frame encoding/decoding benchmarks (`tests/benchmarks/`)
  - Connection pool performance tests
  - Live LLM benchmarks gated by `RLM_LIVE_LLM=1`
  - GitHub issue templates for performance regressions

### Improvements

- **Optimized Codec**: Faster frame encoding/decoding in wire protocol
- **FINAL() Marker Search**: Optimized parsing for completion detection
- **Type Hints**: Enhanced type annotations across adapter layer
- **Docker Environment**: Host network mode for CI environments

### Fixes

- Correct async tool execution with proper `Optional`/`Union` schema handling
- Trusted OIDC publishing for PyPI releases
- Wheel installation tests now include dependencies

### Infrastructure

- Cross-platform clipboard support in justfile
- Improved commit message generation workflow
- Secrets baseline for detect-secrets v1.5.0
- Streamlined pre-commit configuration

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
