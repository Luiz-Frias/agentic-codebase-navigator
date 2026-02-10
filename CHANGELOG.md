# Changelog

This project follows a lightweight changelog format. The public API lives under the `rlm` import package.

## 1.3.1

Security patch release.

### Security

- **protobuf**: Bump `protobuf` from 6.33.2 to 6.33.5 to address CVE JSON recursion depth bypass in `ParseDict()` (CVSS 8.2, Dependabot alert #6)

### Infrastructure

- **CI**: Disable uv dependency cache in publish-only release jobs (`testpypi`, `pypi`) that lack checkout steps

## 1.3.0

Feature release introducing the **Relay Pipeline Library** — a type-safe, composable pipeline framework for orchestrating multi-step LLM workflows with conditional routing, parallel execution, and nested agent composition.

### Features

- **Relay Pipeline Library**: Full pipeline DSL for composing stateful LLM workflows (`src/rlm/domain/relay/`, `src/rlm/adapters/relay/`)
  - `StateSpec[InputT, OutputT]` descriptors with operator overloading: `>>` (sequence), `|` (parallel), `.when()` (conditional)
  - `Pipeline` builder for defining state graphs with typed edges, guards, and join groups
  - `Baton[T]` immutable request-response envelope with validation, trace events, and metadata propagation
  - Compile-time validation: type compatibility, reachability, terminal states, cycle detection via `validate_pipeline()`
  - `allow_cycles(max_iterations=...)` decorator for intentional loops
  - Full documentation in `docs/relay/`

- **Pipeline Executors**: Sync and async orchestrators for pipeline execution
  - `SyncPipelineExecutor` with `ThreadPoolExecutor` for parallel state execution
  - `AsyncPipelineExecutor` with native `asyncio` concurrency
  - Join aggregation: `"all"` (keyed dict) and `"race"` (first result) modes
  - Guard evaluation, budget checks, and trace recording at each step

- **State Executor Types**: Pluggable execution strategies for pipeline states
  - `FunctionStateExecutor` — pure Python callables with optional retry
  - `LLMStateExecutor` — LLM calls via `LLMPort` with request builder
  - `AsyncStateExecutor` — async callables with event loop detection
  - `RLMStateExecutor` — full agent orchestration as a pipeline state
  - `SyncPipelineStateExecutor` / `AsyncPipelineStateExecutor` — nested pipeline composition

- **Token Budget & Trace Tracking**: Execution observability across pipeline runs
  - `TokenBudget` with max tokens, per-state estimates, and consumption tracking
  - `PipelineTrace` immutable append-only audit log with state-level timing and status

- **Pipeline Registry & LLM-Assisted Composition**: Dynamic pipeline discovery and chaining
  - `PipelineTemplate` with name, description, types, factory, and tags
  - `InMemoryPipelineRegistry` with substring and tag-based search
  - `RootAgentComposer` — uses LLM to select and chain pipelines from registry

- **Nested Agent Policy for Relay**: Agent-as-state integration
  - `RelayNestedCallHandler` implements `NestedCallHandlerPort` + `NestedCallPolicy`
  - Registry lookup, pipeline composition, synchronous execution with depth enforcement

- **LLM Provider Resilience**: Exponential backoff retry across providers
  - `RetryStrategy` with configurable max attempts, backoff, and exception types
  - Applied to OpenAI, Azure OpenAI, LiteLLM, and Portkey adapters
  - Enhanced debug logging and error handling across all providers

- **Examples**: Three runnable examples demonstrating pipeline patterns
  - `relay_conditional_routing.py` — if/else branching with guards
  - `relay_parallel_join.py` — fan-out/fan-in with "all" join mode
  - `relay_research_pipeline.py` — sequential multi-step chain with Pydantic models

### Improvements

- **Remote Environment Adapter**: New adapter with memory profiling support
- **Enhanced Live LLM Testing**: `tests/live_llm.py` harness for opt-in provider smoke tests
- **Tooling Optimization**: Improved mise venv and Python configuration
- **Performance Thresholds**: Updated serialization benchmarks

### Infrastructure

- **New documentation files**:
  - `docs/relay/overview.md` — Relay pipeline philosophy and quick start
  - `docs/relay/composition.md` — DSL operators and composition patterns
  - `docs/relay/states.md` — All executor types with usage examples
  - `docs/relay/validation.md` — Pipeline validation rules and cycle handling
- **SOPS Encrypted Secrets**: `.sops.yaml` + `.env.enc` for secure secrets management
- **CI/CD Updates**: Bumped `actions/checkout` to v6, `actions/download-artifact` to v7, `astral-sh/setup-uv` to v7
- **Dependency Updates**: `python-multipart` 0.0.21 → 0.0.22

### Test Coverage

- **11 new test files** spanning unit, integration, and e2e layers (~1,500 LOC)
- Unit: baton, budget/trace, composition, executors, pipeline, state, validation
- Integration: pipeline execution, composition chains, nested handler, nested policy (live)
- E2E: full pipeline workflows, composition flows, nested policy integration

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
