# Legacy → Hexagonal Mapping (Phase 2 / bridge)

This repo is migrating an upstream snapshot (`references/rlm/**`) into a `src/`-layout package and refactoring toward a **hexagonal modular monolith**.

In Phase 2, **runtime orchestration** is handled by the new domain orchestrator, but we still rely on legacy implementations for the broker and execution environments via adapters.

## Mapping table

| Upstream / legacy module (mirrored under `src/rlm/_legacy/**`) | Hex layer responsibility | Current state (Phase 1) | Target end-state |
|---|---|---|---|
| `rlm._legacy.core.rlm.RLM` (iteration loop) | **Application / Domain orchestration** | Kept only for regression/boundary comparisons | Re-implemented as `domain/services/rlm_orchestrator.py` using ports only |
| `rlm._legacy.core.lm_handler.LMHandler` (TCP broker) | **Adapters (broker) + Infrastructure (protocol)** | Mirrored; optional legacy adapter exists | Replace with `infrastructure/comms/*` + `adapters/broker/TcpBrokerAdapter` |
| `rlm._legacy.core.comms_utils` (wire protocol helpers) | **Infrastructure (protocol)** | Mirrored for legacy | Replace with strict DTOs/codec in `infrastructure/comms/` |
| `rlm._legacy.environments.local_repl.LocalREPL` | **Adapters (environment)** | Mirrored; used by legacy loop | Replace with native env adapter (policy-driven) |
| `rlm._legacy.environments.docker_repl.DockerREPL` | **Adapters (environment)** | Mirrored; used by legacy loop | Replace with native docker env adapter + hardened lifecycle |
| `rlm._legacy.utils.parsing` (`find_code_blocks`, `find_final_answer`, etc.) | **Domain services** | Mirrored; used by legacy loop | Replace with pure domain parsing service (+ tests) |
| `rlm._legacy.utils.prompts` | **Domain / Application** | Mirrored; used by legacy loop | Replace with injected templates + typed prompt builder |
| `rlm._legacy.logger.*` | **Adapters (logger)** | Mirrored | Replace with logger adapters + stable JSONL schema |

## The Phase 2 bridge

- **Public API**: `rlm.api.rlm.RLM` is a thin facade.
- **Use case**: `rlm.application.use_cases.run_completion.run_completion` manages broker+env lifecycle and invokes the domain orchestrator.
- **Legacy adapters (temporary)**:
  - `rlm.adapters.legacy.broker.LegacyBrokerAdapter` (legacy `LMHandler` behind `BrokerPort`)
  - `rlm.adapters.legacy.environment.LegacyEnvironmentAdapter` (legacy `LocalREPL` / `DockerREPL` behind `EnvironmentPort`)

This keeps **dependencies pointing inward** for the new layers (`domain/` remains dependency-free), while allowing the system to stay runnable and testable during the migration.

## Guardrails

- **No runtime imports from** `references/**` (the upstream snapshot stays read-only).
- **Domain must not import outer layers** (`adapters/`, `infrastructure/`, `api/`, `application/`, `_legacy/`) or third-party deps.

## Type mapping (Phase 2 / domain models)

Phase 2 introduces **domain-owned, dependency-free dataclasses** under `src/rlm/domain/models/`.
These normalize naming (drop `RLM*` prefixes, prefer clear nouns) while keeping the same core semantics.

| Legacy type (`rlm._legacy.core.types`) | Domain type (`rlm.domain.models`) | Notes |
|---|---|---|
| `ModelUsageSummary` | `ModelUsageSummary` | Same fields; domain defaults to zeroed totals. |
| `UsageSummary` | `UsageSummary` | Same field name `model_usage_summaries`. |
| `RLMChatCompletion` | `ChatCompletion` | Same payload; domain `prompt` remains `Any` until ports are tightened. |
| `REPLResult` | `ReplResult` | Same core fields; `llm_calls` becomes a list of domain `ChatCompletion`. |
| `CodeBlock` | `CodeBlock` | Same structure: `(code, result)`. |
| `RLMIteration` | `Iteration` | Same semantics; `final_answer` is `str | None` in domain. |

Serialization helpers (`to_dict`/`from_dict`) will be implemented in `p2_m1_t03`.
