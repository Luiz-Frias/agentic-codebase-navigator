# Testing Guide

This guide covers how to run tests, understand the test organization, and write new tests for RLM.

## Quick Start

```bash
# Run all unit tests (fast, hermetic)
pytest -m unit

# Run specific test categories
pytest -m integration
pytest -m e2e
pytest -m packaging

# Run with coverage
pytest -m unit --cov=rlm --cov-report=term-missing
```

## Test Organization

```
tests/
├── conftest.py          # Global fixtures and marker enforcement
├── fakes_ports.py       # Hermetic test doubles for domain ports
│
├── unit/                # Fast, hermetic tests (~77 files)
│   ├── test_domain_*.py
│   ├── test_adapters_*.py
│   └── test_infrastructure_*.py
│
├── integration/         # Multi-component boundary tests (~8 files)
│   ├── test_integration_*.py
│   └── test_live_provider_smoke.py
│
├── e2e/                 # End-to-end API flow tests (~21 files)
│   ├── test_boundary_*.py
│   └── test_e2e_*.py
│
├── packaging/           # Build/install/import tests (~1 file)
│   └── test_build_install_import.py
│
├── performance/         # Performance regression tests (~10 files)
│   ├── test_speed_*.py
│   └── test_memory_*.py
│
└── benchmark/           # Timing/profiling tests (~10 files)
    └── *.py
```

## Test Markers

All tests must be marked with exactly one category marker:

| Marker | Purpose | Speed | Dependencies |
|--------|---------|-------|--------------|
| `@pytest.mark.unit` | Fast, hermetic unit tests | Fast | None |
| `@pytest.mark.integration` | Multi-component boundaries | Medium | None |
| `@pytest.mark.e2e` | Full public API flows | Medium | None/Docker |
| `@pytest.mark.packaging` | Build/install validation | Slow | uv |
| `@pytest.mark.performance` | Regression thresholds | Medium | None |
| `@pytest.mark.benchmark` | Timing metrics | Medium | pytest-benchmark |

### Additional Markers

| Marker | Purpose | Requirement |
|--------|---------|-------------|
| `@pytest.mark.docker` | Requires Docker daemon | Docker running |
| `@pytest.mark.live_llm` | Real provider API calls | `RLM_RUN_LIVE_LLM_TESTS=1` + API keys |
| `@pytest.mark.chaos` | Resilience/chaos tests | TBD |

### Marker Enforcement

The `conftest.py` enforces that:
- Every test has exactly one category marker
- The marker matches the directory (e.g., `@pytest.mark.unit` in `tests/unit/`)
- Tests are skipped appropriately (Docker unavailable, live LLM opt-out)

## Running Tests

### Development Workflow

```bash
# Fast feedback during development
pytest -m unit -x --tb=short

# Before committing
pytest -m unit
pytest -m integration

# Before pushing (full gate)
pytest -m unit && pytest -m integration && pytest -m e2e && pytest -m packaging
```

### Using Just (Recommended)

```bash
# Pre-commit hooks (includes unit tests)
just pc-all

# Full pre-push gate (includes integration + e2e)
just pc-full
```

### With Coverage

```bash
# Unit tests with coverage
pytest -m unit --cov=rlm --cov-report=term-missing

# HTML coverage report
pytest -m unit --cov=rlm --cov-report=html
open htmlcov/index.html

# Minimum coverage requirement: 90%
pytest -m unit --cov=rlm --cov-fail-under=90
```

## Live Provider Tests

Live tests make real API calls and are **opt-in only** to prevent accidental spend.

### Enabling Live Tests

```bash
# Enable live LLM tests
export RLM_RUN_LIVE_LLM_TESTS=1

# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Run live tests
pytest -m "integration and live_llm" -v
```

### Provider-Specific Tests

```bash
# OpenAI only
RLM_RUN_LIVE_LLM_TESTS=1 OPENAI_API_KEY=... pytest -k "openai" -m "live_llm"

# Anthropic only
RLM_RUN_LIVE_LLM_TESTS=1 ANTHROPIC_API_KEY=... pytest -k "anthropic" -m "live_llm"
```

## Docker Tests

Tests marked with `@pytest.mark.docker` require a running Docker daemon.

```bash
# Check Docker is available
docker info

# Run Docker tests
pytest -m docker -v

# Docker tests auto-skip if Docker unavailable
# You'll see: SKIPPED [1] - Docker daemon not available
```

## Performance Tests

### Regression Guards

```bash
# Run performance tests (pass/fail based on thresholds)
pytest -m performance -v

# With timing details
pytest -m performance -v --durations=0
```

### Live LLM Performance

```bash
# Use real LLMs for performance testing
export RLM_LIVE_LLM=1
export RLM_LIVE_LLM_PROVIDER=openai  # or anthropic
export OPENAI_API_KEY=...

pytest -m performance -v
```

### Benchmark Tests

```bash
# Run benchmarks with pytest-benchmark
pytest -m benchmark -v

# Save benchmark results to JSON
pytest -m benchmark --benchmark-json=.benchmarks/latest.json

# Compare against baseline
pytest -m benchmark --benchmark-compare
```

## Writing Tests

### Unit Test Pattern

```python
# tests/unit/test_my_feature.py
import pytest
from rlm.domain.models import MyModel
from tests.fakes_ports import QueueLLM, FakeClock

@pytest.mark.unit
def test_my_feature_validates_input():
    """Test description (what, not how)."""
    with pytest.raises(ValidationError, match="expected non-empty"):
        MyModel(name="")

@pytest.mark.unit
def test_my_feature_with_fake_dependencies():
    """Test with hermetic fakes."""
    llm = QueueLLM(responses=["response 1", "FINAL(done)"])
    clock = FakeClock(start=0.0, step=1.0)
    # ... test logic
```

### Integration Test Pattern

```python
# tests/integration/test_my_integration.py
import pytest
from rlm.adapters.broker.tcp import TcpBrokerAdapter
from rlm.adapters.llm.mock import MockLLMAdapter

@pytest.mark.integration
def test_broker_routes_to_correct_adapter():
    """Test multi-component interaction."""
    llm = MockLLMAdapter(model="test", script=["response"])
    broker = TcpBrokerAdapter(llm)
    addr = broker.start()
    try:
        # Test routing logic
        pass
    finally:
        broker.stop()
```

### E2E Test Pattern

```python
# tests/e2e/test_my_e2e_flow.py
import pytest
from rlm.api import create_rlm_from_config
from rlm.application.config import RLMConfig, LLMConfig, EnvironmentConfig

@pytest.mark.e2e
def test_full_flow_with_config():
    """Test complete public API flow."""
    cfg = RLMConfig(
        llm=LLMConfig(
            backend="mock",
            model_name="mock",
            backend_kwargs={"script": ["FINAL(ok)"]}
        ),
        env=EnvironmentConfig(environment="local"),
        max_iterations=2,
    )
    rlm = create_rlm_from_config(cfg)
    result = rlm.completion("test prompt")
    assert result.response == "ok"
```

### Live LLM Test Pattern

```python
# tests/integration/test_live_provider.py
import pytest

@pytest.mark.integration
@pytest.mark.live_llm
def test_live_openai_smoke():
    """Requires: RLM_RUN_LIVE_LLM_TESTS=1 + OPENAI_API_KEY."""
    from rlm.adapters.llm.openai import build_openai_adapter
    from rlm.domain.models import LLMRequest

    llm = build_openai_adapter(model="gpt-4o-mini")
    result = llm.complete(LLMRequest(prompt="Return exactly: OK"))
    assert "ok" in result.response.lower()
```

## Test Fixtures

### Global Fixtures (`conftest.py`)

```python
@pytest.fixture(scope="session")
def docker_is_available() -> bool:
    """Check if Docker daemon is available."""
    return _docker_available()
```

### Hermetic Fakes (`fakes_ports.py`)

| Fake | Purpose | Example |
|------|---------|---------|
| `QueueLLM` | Scripted LLM responses | `QueueLLM(responses=["a", "b"])` |
| `FakeClock` | Deterministic time | `FakeClock(start=0, step=1)` |
| `QueueEnvironment` | Scripted REPL results | `QueueEnvironment(results=[...])` |
| `CollectingLogger` | In-memory log capture | `CollectingLogger()` |
| `InMemoryBroker` | In-process routing | `InMemoryBroker(llm)` |

### Example: Using Fakes

```python
from tests.fakes_ports import QueueLLM, QueueEnvironment, FakeClock

@pytest.mark.unit
def test_orchestrator_completes_on_final():
    llm = QueueLLM(responses=[
        "Let me calculate...\n```repl\nresult = 42\n```",
        "FINAL(42)"
    ])
    env = QueueEnvironment(results=[
        ReplResult(stdout="", stderr="", locals={"result": 42})
    ])
    clock = FakeClock(start=0.0, step=0.1)

    orchestrator = RLMOrchestrator(llm=llm, environment=env, clock=clock)
    result = orchestrator.completion("What is 6*7?")

    assert result.response == "42"
    assert len(env.executed_code) == 1
```

## Mocking External Dependencies

### Synthetic Module Injection

For testing adapters without installing SDKs:

```python
import types
import sys

@pytest.mark.unit
def test_anthropic_adapter_without_sdk(monkeypatch):
    """Test Anthropic adapter without installing anthropic package."""

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        class messages:
            @staticmethod
            def create(**kwargs):
                return FakeResponse(content="test response")

    # Inject fake module
    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = FakeClient
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    # Now test the adapter
    from rlm.adapters.llm.anthropic import AnthropicAdapter
    adapter = AnthropicAdapter(model="claude-test", api_key="fake")
    # ...
```

## Architecture Tests

RLM enforces hexagonal architecture boundaries via AST-based tests:

```python
# tests/unit/test_architecture_layering.py

@pytest.mark.unit
def test_domain_layer_does_not_depend_on_outer_layers():
    """Domain layer must have zero external dependencies."""
    offenders = _scan_forbidden_imports(
        DOMAIN_ROOT,
        forbidden_prefixes=("rlm.adapters", "rlm.infrastructure", "rlm.api"),
    )
    assert not offenders, f"Domain layer imports outer layers: {offenders}"
```

## Environment Variables Reference

| Variable | Purpose | Values |
|----------|---------|--------|
| `RLM_RUN_LIVE_LLM_TESTS` | Enable live provider tests | `1`, `true`, `yes`, `on` |
| `OPENAI_API_KEY` | OpenAI authentication | `sk-...` |
| `OPENAI_MODEL` | OpenAI model override | Model name |
| `ANTHROPIC_API_KEY` | Anthropic authentication | `sk-ant-...` |
| `ANTHROPIC_MODEL` | Anthropic model override | Model name |
| `RLM_LIVE_LLM` | Performance test live mode | `1`, `true`, `yes`, `on` |
| `RLM_LIVE_LLM_PROVIDER` | Performance provider | `openai`, `anthropic` |
| `RLM_DOCKER_USE_HOST_NETWORK` | Docker networking mode | `1` (for CI) |

## CI Integration

### GitHub Actions

The CI workflow runs these test suites:

1. **unit job**: `pytest -m unit` + ruff format/lint
2. **integration job**: `pytest -m integration`
3. **e2e job**: `pytest -m e2e` (includes Docker tests)
4. **packaging job**: `pytest -m packaging`

### Local CI Simulation

```bash
# Simulate full CI locally
just pc-full

# Or manually:
uv run --group dev ruff format --check .
uv run --group dev ruff check .
uv run --group test pytest -m unit
uv run --group test pytest -m integration
uv run --group test pytest -m e2e
uv run --group test pytest -m packaging
```

## Troubleshooting Tests

### "Test not marked with category"

```
pytest.UsageError: Test 'test_foo' must have exactly one marker
```

Add the appropriate marker:

```python
@pytest.mark.unit  # or integration, e2e, etc.
def test_foo():
    ...
```

### "Marker doesn't match directory"

```
pytest.UsageError: Test in 'tests/unit/' must use @pytest.mark.unit
```

Move the test to the correct directory or change the marker.

### Tests Hanging

- Check for infinite loops in test code
- Verify mock responses include `FINAL(...)` for orchestrator tests
- Reduce timeout values for debugging:
  ```python
  env_kwargs={"execute_timeout_s": 5.0}
  ```

### Docker Tests Failing

```bash
# Verify Docker is running
docker info

# Check Docker permissions
docker ps

# Pull required images
docker pull python:3.12-slim
```

## See Also

- [Contributing](../contributing/development-setup.md) — Development workflow
- [Architecture](../architecture.md) — Hexagonal architecture
- [Troubleshooting](../troubleshooting.md) — Common issues
