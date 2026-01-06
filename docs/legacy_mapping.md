# Legacy → Hexagonal Mapping (Phase 1 / bridge)

This repo is migrating an upstream snapshot (`references/rlm/**`) into a `src/`-layout package and refactoring toward a **hexagonal modular monolith**.

In Phase 1, we intentionally keep **runtime behavior** delegated to the upstream loop via a **small bridge** so we can ship incremental structure + tests without a risky rewrite.

## Mapping table

| Upstream / legacy module (mirrored under `src/rlm/_legacy/**`) | Hex layer responsibility | Current state (Phase 1) | Target end-state |
|---|---|---|---|
| `rlm._legacy.core.rlm.RLM` (iteration loop) | **Application / Domain orchestration** | Still the “engine” via bridge | Re-implemented as `domain/services/rlm_orchestrator.py` using ports only |
| `rlm._legacy.core.lm_handler.LMHandler` (TCP broker) | **Adapters (broker) + Infrastructure (protocol)** | Mirrored; optional legacy adapter exists | Replace with `infrastructure/comms/*` + `adapters/broker/TcpBrokerAdapter` |
| `rlm._legacy.core.comms_utils` (wire protocol helpers) | **Infrastructure (protocol)** | Mirrored for legacy | Replace with strict DTOs/codec in `infrastructure/comms/` |
| `rlm._legacy.environments.local_repl.LocalREPL` | **Adapters (environment)** | Mirrored; used by legacy loop | Replace with native env adapter (policy-driven) |
| `rlm._legacy.environments.docker_repl.DockerREPL` | **Adapters (environment)** | Mirrored; used by legacy loop | Replace with native docker env adapter + hardened lifecycle |
| `rlm._legacy.utils.parsing` (`find_code_blocks`, `find_final_answer`, etc.) | **Domain services** | Mirrored; used by legacy loop | Replace with pure domain parsing service (+ tests) |
| `rlm._legacy.utils.prompts` | **Domain / Application** | Mirrored; used by legacy loop | Replace with injected templates + typed prompt builder |
| `rlm._legacy.logger.*` | **Adapters (logger)** | Mirrored | Replace with logger adapters + stable JSONL schema |

## The Phase 1 bridge

- **Public API**: `rlm.api.rlm.RLM` is a thin facade.
- **Bridge service**: `rlm.application.services.legacy_orchestrator.LegacyOrchestratorService` runs the legacy loop but injects an `LLMPort` by patching the legacy `get_client()` router inside a call-scoped context manager.

This keeps **dependencies pointing inward** for the new layers (`domain/` remains dependency-free), while allowing the system to stay runnable and testable during the migration.

## Guardrails

- **No runtime imports from** `references/**` (the upstream snapshot stays read-only).
- **Domain must not import outer layers** (`adapters/`, `infrastructure/`, `api/`, `application/`, `_legacy/`) or third-party deps.

