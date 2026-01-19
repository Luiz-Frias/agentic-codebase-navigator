# LLM providers (Phase 4)

This project keeps most **provider SDKs optional** so the default install remains lightweight and fully testable.

The OpenAI SDK is installed by default (so the `openai` backend works out of the box once `OPENAI_API_KEY` is set).

## Optional extras

Install provider adapters via extras:

- **OpenAI**: included by default (optional extra still exists): `agentic-codebase-navigator[llm-openai]`
- **Anthropic**: `agentic-codebase-navigator[llm-anthropic]`
- **Gemini (Google GenAI)**: `agentic-codebase-navigator[llm-gemini]`
- **Portkey**: `agentic-codebase-navigator[llm-portkey]`
- **LiteLLM**: `agentic-codebase-navigator[llm-litellm]`
- **Azure OpenAI**: `agentic-codebase-navigator[llm-azure-openai]`

Using `uv`:

```bash
uv pip install -e ".[llm-openai]"
```

## OpenAI

### Environment variables

- `OPENAI_API_KEY`: required

### Example: build from config

```python
from rlm.api import create_rlm_from_config
from rlm.application.config import EnvironmentConfig, LLMConfig, RLMConfig

cfg = RLMConfig(
    llm=LLMConfig(backend="openai", model_name="gpt-5-nano"),
    env=EnvironmentConfig(environment="local"),
    max_iterations=10,
)

rlm = create_rlm_from_config(cfg)
print(rlm.completion("hello").response)
```

### Example: dependency-free mock (recommended for tests)

```python
from rlm.api import create_rlm_from_config
from rlm.application.config import LLMConfig, RLMConfig

cfg = RLMConfig(
    llm=LLMConfig(
        backend="mock",
        model_name="mock-model",
        backend_kwargs={"script": ["FINAL(ok)"]},
    ),
    max_iterations=2,
)

rlm = create_rlm_from_config(cfg)
assert rlm.completion("hello").response == "ok"
```

## Multi-backend routing (root + subcalls)

You can configure **multiple models/backends** so nested environment calls can route explicitly by model name:

- `llm_query("...", model="some-model-name")`
- `llm_query_batched([...], model="some-model-name")`

Example (fully deterministic; no network):

```python
from rlm.api import create_rlm_from_config
from rlm.application.config import EnvironmentConfig, LLMConfig, RLMConfig

root_script = (
    "```repl\n"
    "resp = llm_query('ping', model='sub')\n"
    "```\n"
    "FINAL_VAR('resp')"
)

cfg = RLMConfig(
    llm=LLMConfig(backend="mock", model_name="root", backend_kwargs={"script": [root_script]}),
    other_llms=[
        LLMConfig(backend="mock", model_name="sub", backend_kwargs={"script": ["pong"]}),
    ],
    env=EnvironmentConfig(environment="local"),
    max_iterations=2,
)

cc = create_rlm_from_config(cfg).completion("hello")
assert cc.response == "pong"
assert cc.usage_summary.model_usage_summaries["root"].total_calls == 1
assert cc.usage_summary.model_usage_summaries["sub"].total_calls == 1
```

## Anthropic

### Environment variables

- `ANTHROPIC_API_KEY`: required (if not provided via config)

## Gemini (Google GenAI)

### Environment variables

- `GOOGLE_API_KEY`: required (if not provided via config)

## Portkey

### Environment variables

- `PORTKEY_API_KEY`: required (if not provided via config)

## LiteLLM

### Environment variables

LiteLLM is a **router/wrapper** over multiple providers, so environment variables depend on the
underlying provider you select (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc).

## Azure OpenAI

### Environment variables

If not provided via config, the OpenAI SDK commonly reads:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`

## Usage accounting (all providers)

RLM tracks **per-model** usage in `UsageSummary`:

- **calls**: incremented by 1 per completion request
- **tokens**: best-effort extraction from each provider response; if the provider SDK does not return usage metadata, token counts are recorded as `0`

Current extraction behavior:

- **OpenAI / Azure OpenAI / Portkey / LiteLLM**: reads `response.usage.{prompt_tokens, completion_tokens}` (or `{input_tokens, output_tokens}` when available)
- **Anthropic**: reads `response.usage.{input_tokens, output_tokens}`
- **Gemini**: reads `response.usage_metadata.{prompt_token_count, candidates_token_count}`
