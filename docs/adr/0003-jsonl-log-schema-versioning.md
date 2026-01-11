# ADR 0003: JSONL logs are line-oriented and versioned

## Status

Accepted (Phase 06)

## Context

RLM produces multi-iteration trajectories and may include nested sub-calls.
We want logs that are:
- streamable (no large in-memory buffers)
- human-inspectable
- robust to partial writes
- compatible with downstream tooling (including the upstream visualizer when possible)

## Decision

Use a **JSON Lines** format with explicit versioning:

- One JSON object per line.
- Each line includes:
  - `type` (`metadata` | `iteration`)
  - `schema_version` (currently `1`)
  - `timestamp` (UTC ISO-8601)

Nested subcalls executed inside environments are persisted under `code_blocks[*].result.rlm_calls`.

## Consequences

- Logs can be appended incrementally and processed as a stream.
- The schema can evolve with explicit migrations when needed.
