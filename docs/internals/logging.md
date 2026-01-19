# Logging (Phase 06)

RLM supports **optional run logging** through the domain `LoggerPort`.

## Logger configuration

You can inject a logger directly:

- `logger=None` (default): disables logging
- `logger=JsonlLoggerAdapter(...)`: writes JSONL logs (recommended)
- `logger=ConsoleLoggerAdapter(...)`: prints a compact per-iteration summary to stdout

If you're using config + registries (`create_rlm_from_config`), `LoggerConfig.logger` supports:

- `none`
- `jsonl` (requires `logger_kwargs["log_dir"]`)
- `console` (optional `logger_kwargs["enabled"]`)

## JSONL log schema (v1)

Each run produces a JSON Lines file with one JSON object per line.

For the normative field listing, see `docs/log_schema_v1.md`.

- **Metadata line** (first line in a run):
  - `type="metadata"`
  - `schema_version=1`
  - `timestamp` (UTC ISO-8601)
  - plus fields from `RunMetadata.to_dict()`

- **Iteration lines** (one per orchestrator iteration):
  - `type="iteration"`
  - `schema_version=1`
  - `iteration` (1-indexed)
  - `timestamp` (UTC ISO-8601)
  - plus fields from `Iteration.to_dict()`

Notes:
- Code execution output lives under `code_blocks[*].result` (stdout/stderr/locals/execution_time).
- Nested subcalls executed via `llm_query()` are stored under `code_blocks[*].result.rlm_calls`.
- Token counts are available both as:
  - `usage_summary` (new, structured) and
  - `prompt_tokens` / `completion_tokens` (legacy/visualizer compatibility).

## Visualizer compatibility strategy

We aim for **best-effort compatibility** with the upstream visualizer in `references/rlm/visualizer`:

- We keep the on-wire key name `rlm_calls` for nested sub-calls.
- We include `prompt_tokens` / `completion_tokens` in each logged chat completion.

The log also includes the richer `usage_summary` used by the refactored domain models.

## Running the upstream visualizer (optional)

The upstream visualizer is kept as a **reference-only** asset under `references/rlm/visualizer`.

Typical workflow:

```bash
cd references/rlm/visualizer
npm install
npm run dev
```

Then load a `.jsonl` file produced by `JsonlLoggerAdapter`.
