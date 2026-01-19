# Wire protocol (Phase 3)

This document describes the **wire protocol** used to route `llm_query()` calls from execution environments back to the host broker.

It is implemented in:

- **DTOs**: `src/rlm/infrastructure/comms/messages.py`
- **Framing (length-prefixed JSON)**: `src/rlm/infrastructure/comms/codec.py`
- **Client helpers + safe errors**: `src/rlm/infrastructure/comms/protocol.py`

## Transport framing

Each message is a single frame:

- **4-byte big-endian unsigned length prefix** (payload byte length)
- **UTF-8 JSON payload**
- Payload MUST decode to a **JSON object** (dictionary).

The default maximum accepted payload is **10MB** (`DEFAULT_MAX_MESSAGE_BYTES`) to avoid unbounded allocations.

## Data model

### Prompt payloads

The wire protocol accepts the same `Prompt` shapes as the domain:

- `str`
- `dict[str, Any]` (legacy-compatible)
- `list[dict[str, Any]]` (OpenAI-style message arrays)

### Request: `WireRequest`

Keys (JSON object):

- `correlation_id` (optional, `str`)
- `model` (optional, `str`)
- `prompt` (single prompt) **OR** `prompts` (batched prompts)

Rules:

- Exactly **one** of `prompt` / `prompts` must be provided.
- `prompts` must be a non-empty list.

### Response: `WireResponse`

Keys (JSON object):

- `correlation_id` (optional, `str`)
- `error` (`str | null`)
- `results` (`list[WireResult] | null`)

Rules:

- Response is **either**:
  - a request-level error: `error != null` and `results == null`
  - a successful response: `error == null` and `results != null`
- For successful responses:
  - `len(results)` MUST match the request cardinality (**1** for `prompt`, **N** for `prompts`)
  - ordering MUST match the request ordering

### Per-item result: `WireResult`

Keys (JSON object):

- `error` (`str | null`)
- `chat_completion` (`dict | null`)

Rules:

- Each result is **either**:
  - an item error: `error != null` and `chat_completion == null`
  - an item success: `error == null` and `chat_completion != null`

`chat_completion` uses the domain `ChatCompletion.to_dict()` shape:

- `root_model`
- `prompt`
- `response`
- `usage_summary`
- `execution_time`

## Error handling

Goals:

- Never crash the broker on malformed input.
- Never leak stack traces to clients.

Server-side:

- Parse requests via `try_parse_request()` to convert decode/validation errors into a safe `WireResponse(error=...)`.

Client-side:

- `send_request()` raises `rlm.domain.errors.BrokerError` on transport/protocol failures or request-level errors.
- For batched requests, `request_completions_batched()` returns **per-item results** so callers can decide whether to:
  - propagate errors
  - convert errors into `"Error: ..."` strings (legacy-style)

## Docker proxy expectations

The Docker execution environment uses an in-host HTTP proxy (legacy: `LLMProxyHandler`) so in-container code can call `llm_query()` without direct socket wiring.

Phase 3 target behavior:

- HTTP endpoints accept/forward:
  - `prompt` / `prompts`
  - `model` (optional)
  - `correlation_id` (optional initially; required once env scripts are updated)
- The proxy preserves batched ordering and surfaces errors deterministically.
- The proxy records structured `ChatCompletion` objects so the final `ReplResult.llm_calls` includes nested LLM calls from code execution.
