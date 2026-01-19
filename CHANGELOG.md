# Changelog

This project follows a lightweight changelog format. The public API lives under the `rlm` import package.

## 1.2.0-rc.1

Major architecture release introducing declarative state machine orchestration, subprocess-based local execution, and optional Pydantic integration.

### Breaking Changes

- **Local environment now uses subprocess workers**: Code execution moved from in-process to isolated subprocess workers. This improves reliability and timeout behavior but may affect code that relied on shared namespace state between executions. See [Migration Guide](docs/migration.md).

- **`StoppingPolicy` detection**: Pre-v1.2.0 code relying on string matching `"[Stopped by custom policy]"` should migrate to the explicit `policy_stop` flag in context.

### Features

- **State Machine Orchestration**: Complete rewrite of orchestrator control flow using declarative `StateMachine[S, E, C]`
  - Generic state machine with States (enum), Events (dataclasses), and Context (mutable dataclass)
  - Explicit transitions with optional guards and actions
  - `on_enter`/`on_exit` callbacks for state lifecycle
  - Both sync `run()` and async `arun()` execution
  - Eliminates C901 complexity violations from nested loops
  - Full documentation in `docs/internals/state-machine.md`

- **Subprocess-based Local Execution**: Improved isolation for local code execution
  - Each execution runs in a separate worker process with IPC communication
  - Reliable timeout behavior via process kill (works across all platforms)
  - `llm_query()` calls routed back to parent via broker IPC
  - New options: `execute_timeout_cap_s` for maximum timeout cap
  - Falls back to SIGALRM for in-worker timeouts (Unix main thread only)

- **Optional Pydantic Integration**: Enhanced type validation and schema generation (ADR-001)
  - `JsonSchemaMapper` with dual-path implementation (manual vs TypeAdapter)
  - `prefer_pydantic` flag for explicit path selection
  - Automatic fallback when Pydantic not installed
  - Better handling of `Optional`, `Union`, and complex nested types
  - Full documentation in `docs/extending/pydantic-integration.md`

- **Result[T, E] Pattern**: Rust-inspired error handling
  - `Ok[T]` and `Err[E]` frozen dataclasses
  - `try_call()` bridge function for exception → Result conversion
  - Used throughout tool execution and schema mapping

- **SDK Boundary Layer**: Centralized `Any` type handling for pyright compliance
  - `sdk_boundaries.py` module with typed containers
  - `ToolExecutionResult` for wrapping tool outputs
  - `execute_tool_safely()` for exception-safe tool invocation

- **SafeAccessor**: Duck-typed navigation for LLM responses
  - Chain-friendly `accessor["key"][0]["nested"]` syntax
  - `unwrap_or(default)` for safe value extraction
  - Handles malformed or missing data gracefully

- **Explicit `policy_stop` Flag**: Reliable StoppingPolicy termination detection
  - `ToolsModeContext.policy_stop: bool` field
  - Orchestrator checks flag first, eliminating string-matching heuristics
  - `_mark_policy_stop()` helper for event sources

### Improvements

- **Type Precision**: Replaced `Any` with `object` in public API boundaries
- **Malformed Response Detection**: Explicit distinction between "no tool calls" vs "malformed response"
- **Expanded Documentation**: New troubleshooting guide, state machine internals, Pydantic integration
- **Correlation ID Propagation**: End-to-end tracing through subprocess workers

### Infrastructure

- **New documentation files**:
  - `docs/internals/state-machine.md` — State machine architecture deep dive
  - `docs/extending/pydantic-integration.md` — Pydantic usage guide
  - `docs/migration.md` — Migration guide (1.1.0 → 1.2.0)
- **Expanded troubleshooting**: Quick reference table, provider-specific issues, state machine debugging
- **Updated execution environments docs**: Subprocess architecture diagram, new options table

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
