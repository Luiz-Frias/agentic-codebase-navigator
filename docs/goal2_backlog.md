# Goal 2 Backlog (Draft)

This file tracks Goal 2 follow-ups and tech debt discovered during the Goal 1 refactor.

## Backlog

- Implement a filesystem-based `CodebaseIndexer` (no network required).
- Add a semantic chunker (AST-aware for Python; fallback text chunker for other files).
- Define context budgeting policies (by tokens/bytes/chunks/files).
- Add deterministic scenario tests for Goal 2 selection logic (no provider APIs required).
- Add optional adapters for embedding/vector search backends (extras-only).
- Add UX/docs for “how to plug in your own indexer/chunker/context manager”.
