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
