# Goal 2 (Planned): Codebase Navigation & Context Management

Goal 2 extends Goal 1 (RLM completion + execution environments) with **codebase-aware context selection**.

## Motivation

Goal 1 provides a deterministic, testable loop:

- LLM produces code blocks
- the environment executes them
- results are fed back into the LLM until a final answer is produced

Goal 2 adds the missing capability for **large codebases**: selecting and shaping the right context so the LLM can
operate efficiently and safely.

## Core capabilities (ports-first)

Goal 2 will be implemented behind domain ports (see `src/rlm/domain/goal2_ports.py`):

- **CodebaseIndexer**: produce a searchable representation of the repo (files, metadata)
- **Chunker**: turn files into semantically meaningful chunks (with line ranges)
- **ContextManager**: decide what chunks to include for a given prompt, with budgets and policies

Concrete implementations will live under `src/rlm/codebase/` and `src/rlm/adapters/`.

## Sub-agents and nested execution (future)

Longer-running tasks may benefit from sub-agents that:

- run searches/indexing in the environment
- propose context candidates
- validate assumptions (e.g., “does this symbol exist?”) before escalating to the root model

This should remain **strictly optional** and should not affect Goal 1 determinism or testability.

## Context-rot prevention

Key techniques:

- enforce context budgets (token/character/file-count)
- summarize and cache stable context
- prefer incremental retrieval over dumping large files
- track provenance (which chunks were included and why)

## Non-goals (for Goal 2)

- No mandatory network access for indexing.
- No requirement that Goal 2 is present to run Goal 1 completion flows.
