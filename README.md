## agentic-codebase-navigator

This repo is being developed as an **agentic codebase navigator** built on **Recursive Language Model (RLM)** patterns.

- **PyPI / distribution name**: `agentic-codebase-navigator`
- **Python import package**: `rlm` (we are migrating the upstream snapshot from `references/rlm/**` into `src/rlm/**`)

### Dev quickstart (uv + Python 3.12)

```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
```

### Running tests

Unit tests (fast, hermetic):

```bash
uv run --group test pytest -m unit
```

Integration tests (may use Docker; skip cleanly if unavailable):

```bash
uv run --group test pytest -m integration
```

### Optional LLM adapters (providers)

Provider SDKs are installed via **optional extras** to keep the default install lightweight.

- **OpenAI**:
  - Install: `uv pip install -e ".[llm-openai]"`
  - Configure: set `OPENAI_API_KEY`
  - Use:

```python
from rlm.adapters.llm import OpenAIAdapter
from rlm.api import create_rlm

rlm = create_rlm(OpenAIAdapter(model="gpt-4o-mini"), environment="local")
print(rlm.completion("hello").response)
```

Other provider extras are available (Anthropic/Gemini/Portkey/LiteLLM/Azure OpenAI). See `docs/llm_providers.md`.

### Multi-backend routing (root + subcalls)

RLM supports **registering multiple LLM backends/models** so `repl` code blocks can route nested calls via:

- `llm_query("...", model="some-model-name")`
- `llm_query_batched([...], model="some-model-name")`

Example (dependency-free, no network):

```python
from rlm.adapters.llm.mock import MockLLMAdapter
from rlm.api import create_rlm

root_script = (
    "```repl\n"
    "resp = llm_query('ping', model='sub')\n"
    "```\n"
    "FINAL_VAR('resp')"
)

rlm = create_rlm(
    MockLLMAdapter(model="root", script=[root_script]),
    other_llms=[MockLLMAdapter(model="sub", script=["pong"])],
    environment="local",
    max_iterations=2,
)

cc = rlm.completion("hello")
assert cc.response == "pong"

# Usage is aggregated across all registered models (root + subcalls).
assert cc.usage_summary.model_usage_summaries["root"].total_calls == 1
assert cc.usage_summary.model_usage_summaries["sub"].total_calls == 1
```

### Docker execution environment (Phase 1)

The legacy Docker execution environment (`DockerREPL`) is used in Phase 1 to execute `repl` code blocks inside a
container, while proxying `llm_query()` calls back to the host.

- **Requirements**
  - A working local Docker installation (`docker` CLI available).
  - Docker daemon running and accessible (`docker info` succeeds).
  - Docker that supports `--add-host host.docker.internal:host-gateway` (Docker 20.10+).
  - Network access to install in-container deps (`pip install dill requests`) if the image doesn’t already have them.

- **Default image**
  - `python:3.12-slim`

- **Test behavior**
  - Tests that require Docker are marked with `@pytest.mark.docker`.
  - If Docker isn’t available, they’re skipped automatically by `tests/conftest.py`.
  - If the container can’t start (e.g. image pull blocked), Docker tests may skip with the raised error.
