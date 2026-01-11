# Architecture (Hexagonal Modular Monolith)

This repo is refactoring an upstream snapshot (`references/rlm/**`) into a **hexagonal modular monolith** under
`src/`.

## Core rule (Dependency Inversion)

Dependencies **must point inward**:

- Outer layers (adapters/infrastructure) may depend on inner layers (domain/application).
- Inner layers must **not** import outer layers.

## Layers (target)

- **Domain** (`src/rlm/domain/`)
  - Pure business logic, models, and ports (no third-party deps).
- **Application** (`src/rlm/application/`)
  - Use cases and composition contracts; orchestrates domain services.
- **Adapters** (`src/rlm/adapters/`)
  - Concrete implementations of ports (LLM providers, environments, broker, logger).
- **Infrastructure** (`src/rlm/infrastructure/`)
  - Cross-cutting technical utilities (comms protocol, filesystem/time helpers).
- **API** (`src/rlm/api/`)
  - Public entrypoints (Python API, optional CLI).

## Composition root (Phase 2)

In Phase 2, runtime orchestration flows:

- **Public facade**: `rlm.api.rlm.RLM`
  - Delegates work to the application use case.
  - Holds injected ports/factories (or uses lazy defaults).
- **Use case**: `rlm.application.use_cases.run_completion.run_completion`
  - Starts/stops the broker (for `llm_query()` inside environments).
  - Builds an environment for the run.
  - Runs the pure-domain orchestrator (`rlm.domain.services.rlm_orchestrator.RLMOrchestrator`).

Convenience builders:

- `rlm.api.factory.create_rlm(...)`
- `rlm.api.factory.create_rlm_from_config(config, llm=... | llm_registry=...)`

Registries (config → adapters) live in the API layer:

- `rlm.api.registries.DictLLMRegistry` (generic)
- `rlm.api.registries.DefaultEnvironmentRegistry` (local + docker, Phase 2)
- `rlm.api.registries.DefaultLoggerRegistry` (none + jsonl + console, Phase 2)

## Extension points (Phase 2)

You can customize wiring without touching the domain:

- Provide an `LLMPort` directly to `RLM(...)`, or build one via `LLMRegistry`.
- Override `broker_factory`, `environment_factory`, or `logger` when constructing `RLM(...)`.
- Provide custom `EnvironmentRegistry` / `LoggerRegistry` to `create_rlm_from_config(...)`.

## Goal 2 reserved namespace

Goal 2 (codebase indexing/chunking/context management) is planned to live behind domain ports in
`src/rlm/domain/goal2_ports.py`, with concrete implementations under the reserved namespace:

- `src/rlm/codebase/`

Goal 1 runtime paths must remain independent of Goal 2 (no imports or required deps).

## Scope note

`references/rlm/**` remains a read-only upstream snapshot. All runtime code must live in `src/**`.
