# JSONL Log Schema v1

This document defines the stable log schema written by `JsonlLoggerAdapter` when `schema_version=1`.

## Common fields

Every JSON object line includes:

- `schema_version`: `1`
- `type`: `"metadata"` or `"iteration"`
- `timestamp`: UTC ISO-8601 string

## Metadata line (`type="metadata"`)

Fields (from `RunMetadata.to_dict()`):

- `root_model`: `str`
- `max_depth`: `int`
- `max_iterations`: `int`
- `backend`: `str`
- `backend_kwargs`: `dict[str, JSON]`
- `environment_type`: `str`
- `environment_kwargs`: `dict[str, JSON]`
- `other_backends`: `list[str] | null`
- `correlation_id`: `str` (optional; omitted if null)

## Iteration line (`type="iteration"`)

Top-level fields:

- `iteration`: `int` (1-indexed)
- `correlation_id`: `str` (optional)
- `prompt`: `JSON` (serialized; string/dict/message-list)
- `response`: `str`
- `final_answer`: `str | null`
- `iteration_time`: `float`
- `iteration_usage_summary`: `UsageSummary | null`
- `cumulative_usage_summary`: `UsageSummary | null`
- `code_blocks`: `list[CodeBlock]`

### `CodeBlock`

- `code`: `str`
- `result`: `ReplResult`

### `ReplResult`

- `stdout`: `str`
- `stderr`: `str`
- `locals`: `dict[str, JSON]` (values are safely stringified if needed)
- `execution_time`: `float`
- `correlation_id`: `str` (optional)
- `rlm_calls`: `list[ChatCompletion]` (nested subcalls)

### `ChatCompletion`

- `root_model`: `str`
- `prompt`: `JSON`
- `response`: `str`
- `usage_summary`: `UsageSummary`
- `prompt_tokens`: `int` (compat convenience)
- `completion_tokens`: `int` (compat convenience)
- `execution_time`: `float`

## Notes on compatibility

- We keep the key name **`rlm_calls`** for nested calls to remain compatible with the upstream visualizer.
- `usage_summary` is the canonical structured usage signal; token counts are convenience fields.
