# Public API (Phase 06)

This document defines the **stable user-facing Python API** for the refactored `rlm` package.

## Primary entrypoint: `RLM`

Import:

```python
from rlm.api import RLM
```

Usage (sync):

```python
cc = rlm.completion("hello")
print(cc.response)
```

Usage (async):

```python
cc = await rlm.acompletion("hello")
print(cc.response)
```

## Convenience constructors

### `create_rlm(...)`

```python
from rlm.api import create_rlm
from rlm.adapters.llm.mock import MockLLMAdapter

rlm = create_rlm(MockLLMAdapter(model="mock", script=["FINAL(ok)"]), environment="local")
print(rlm.completion("hello").response)
```

### `create_rlm_from_config(...)`

```python
from rlm.api import create_rlm_from_config, EnvironmentConfig, LLMConfig, LoggerConfig, RLMConfig

cfg = RLMConfig(
    llm=LLMConfig(backend="mock", model_name="mock", backend_kwargs={"script": ["FINAL(ok)"]}),
    env=EnvironmentConfig(environment="local"),
    logger=LoggerConfig(logger="none"),
    max_iterations=2,
)
rlm = create_rlm_from_config(cfg)
```

## Return types

`completion(...)` and `acompletion(...)` return a domain-owned `ChatCompletion`:

- `root_model`: model used for the root call
- `prompt`: the original prompt payload
- `response`: final extracted answer (string)
- `usage_summary`: structured usage totals by model
- `execution_time`: total time spent in the orchestrator loop

## Configuration notes

### Multi-backend routing

You can register additional models for nested calls:

```python
from rlm.adapters.llm.mock import MockLLMAdapter
from rlm.api import create_rlm

root_script = "```repl\nresp = llm_query('ping', model='sub')\n```\nFINAL_VAR('resp')"
rlm = create_rlm(
    MockLLMAdapter(model="root", script=[root_script]),
    other_llms=[MockLLMAdapter(model="sub", script=["pong"])],
    environment="local",
    max_iterations=2,
)
assert rlm.completion("hello").response == "pong"
```

### Logging

See `docs/logging.md` for JSONL logging configuration and schema.

## Migration notes

The upstream snapshot remains available under `references/rlm/**` for reference only.
Runtime code lives under `src/rlm/**`; legacy internals under `src/rlm/_legacy/**` are not part of the public API.
