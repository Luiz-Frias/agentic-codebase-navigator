# Changelog

This project follows a lightweight changelog format. The public API lives under the `rlm` import package.

## 0.1.2

- Hexagonal modular-monolith refactor (ports/adapters) under `src/rlm/`
- Stable public API: `create_rlm`, `create_rlm_from_config`, `RLMConfig`/`LLMConfig`/`EnvironmentConfig`/`LoggerConfig`
- Deterministic test and packaging gates (unit/integration/e2e/packaging/performance)
- TCP broker with batched concurrency and safe error mapping
- Docker environment adapter with best-effort cleanup, timeouts, and host proxy for nested `llm_query`
- Versioned JSONL logging schema (v1) with console/no-op logger options
- Opt-in live provider smoke tests (OpenAI/Anthropic) gated by `RLM_RUN_LIVE_LLM_TESTS=1`
