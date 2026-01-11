# ADR 0001: Hexagonal modular monolith structure

## Status

Accepted (Phase 06)

## Context

We are migrating an upstream snapshot (`references/rlm/**`) into a maintainable, testable codebase under `src/rlm/**`.

We need:
- clear dependency direction (inward only)
- isolated integration points (LLMs, environments, logging)
- the ability to run deterministic tests without network/Docker

## Decision

Adopt a **hexagonal modular monolith**:

- **Domain** (`src/rlm/domain/`): pure business logic + ports (Protocols) + domain models (dataclasses). No third-party deps.
- **Application** (`src/rlm/application/`): use cases and lifecycle orchestration (start broker, build env, run orchestrator).
- **Infrastructure** (`src/rlm/infrastructure/`): protocol/codec, ids, cross-cutting helpers that are still dependency-free.
- **Adapters** (`src/rlm/adapters/`): concrete implementations of ports (LLM providers, environments, broker, logger).
- **API** (`src/rlm/api/`): composition root and user-facing convenience builders.

## Consequences

- We can unit-test the domain with fake ports (fast, deterministic).
- New providers/environments are added by implementing ports (OCP/DIP).
- We enforce the dependency rule via import-boundary tests under `tests/unit/test_architecture_layering.py`.
