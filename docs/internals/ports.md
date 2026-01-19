# Ports (Phase 2)

This document describes the **domain ports** (interfaces) that isolate the hexagonal core from external concerns.

## Goals

- Keep the **domain** dependency-free and stable.
- Make integrations (LLMs, Docker/local execution, brokers, logging) swappable via adapters.
- Keep interfaces small and focused (SRP/ISP).

## Where things live

- **Ports**: `src/rlm/domain/ports.py`
- **Request/response types**: `src/rlm/domain/models/*` and `src/rlm/domain/types.py`
- **Adapter bases (optional)**: `src/rlm/adapters/base.py`

## Port contracts

### `LLMPort`

File: `src/rlm/domain/ports.py`

Responsibilities:
- Provide **sync** and **async** completion.
- Track **usage** (total + last call).

Interface shape:
- `complete(request: LLMRequest) -> ChatCompletion`
- `acomplete(request: LLMRequest) -> ChatCompletion`
- `get_usage_summary() -> UsageSummary`
- `get_last_usage() -> UsageSummary`

Notes:
- Uses explicit request/response models:
  - `LLMRequest` / `BatchedLLMRequest` in `src/rlm/domain/models/llm_request.py`
  - `ChatCompletion` in `src/rlm/domain/models/completion.py`

### `BrokerPort`

Responsibilities:
- Route requests to the correct `LLMPort` by `model_name`.
- Support batched requests while preserving **ordering**.
- Aggregate usage across routed calls.

Related docs:
- Wire protocol (Phase 3): `docs/protocol.md`

Interface shape:
- `register_llm(model_name: str, llm: LLMPort) -> None`
- `start() -> (host, port)` / `stop() -> None`
- `complete(request: LLMRequest) -> ChatCompletion`
- `complete_batched(request: BatchedLLMRequest) -> list[ChatCompletion]`
- `get_usage_summary() -> UsageSummary`

### `EnvironmentPort`

Responsibilities:
- Load a context payload into the execution environment.
- Execute a code string and return structured output.

Interface shape:
- `load_context(context_payload: ContextPayload) -> None`
- `execute_code(code: str) -> ReplResult`
- `cleanup() -> None`

### `LoggerPort`

Responsibilities:
- Record run-level metadata and iteration-level events.

Interface shape:
- `log_metadata(metadata: RunMetadata) -> None`
- `log_iteration(iteration: Iteration) -> None`

### `ClockPort` and `IdGeneratorPort`

Responsibilities:
- Provide deterministic time and IDs for tests and reproducible behavior.

Interface shape:
- `ClockPort.now() -> float`
- `IdGeneratorPort.new_id() -> str`

## Transitional note (Phase 2 bridge)

Legacy has been fully removed; all runtime code paths use native hexagonal adapters.
