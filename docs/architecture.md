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

## Scope note

`references/rlm/**` remains a read-only upstream snapshot. All runtime code must live in `src/**`.
