# ADR 0002: Provider dependencies are lazy-imported and mostly optional

## Status

Accepted (Phase 06)

## Context

LLM provider SDKs are large, evolve quickly, and often require API keys/network to exercise.
The system needs to remain usable in CI and local development without requiring every provider SDK.

## Decision

- Keep provider SDK imports **lazy** inside the provider adapters.
- Keep most provider SDKs behind **optional extras**.
- (Project-specific) Install **OpenAI** by default so the common `openai` backend works out of the box once `OPENAI_API_KEY` is set.

## Consequences

- Default installs remain relatively small and deterministic tests remain possible using `MockLLMAdapter`.
- Selecting a provider without its dependency yields a clear error at the adapter boundary.
