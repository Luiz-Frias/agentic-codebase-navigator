# agentic-codebase-navigator

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An **agentic codebase navigator** built on **Recursive Language Model (RLM)** patterns. RLM enables LLMs to execute Python code iteratively, inspect results, and refine their approach until reaching a final answer.

- **PyPI / distribution name**: `agentic-codebase-navigator`
- **Python import package**: `rlm`

## Installation

### Via pip

```bash
pip install agentic-codebase-navigator
```

### Via uv (recommended)

```bash
uv pip install agentic-codebase-navigator
```

### With optional LLM providers

```bash
# OpenAI (included by default)
pip install agentic-codebase-navigator

# Anthropic
pip install "agentic-codebase-navigator[llm-anthropic]"

# Google Gemini
pip install "agentic-codebase-navigator[llm-gemini]"

# Azure OpenAI
pip install "agentic-codebase-navigator[llm-azure-openai]"

# LiteLLM (unified provider)
pip install "agentic-codebase-navigator[llm-litellm]"

# Portkey
pip install "agentic-codebase-navigator[llm-portkey]"
```

## Quick Start

### Basic usage with OpenAI

```python
from rlm import create_rlm
from rlm.adapters.llm import OpenAIAdapter

# Create RLM with OpenAI (requires OPENAI_API_KEY env var)
rlm = create_rlm(
    OpenAIAdapter(model="gpt-4o"),
    environment="local",
    max_iterations=10,
)

# Run a completion
result = rlm.completion("What is 2 + 2? Use Python to calculate.")
print(result.response)
```

### Using MockLLM for testing (no API keys required)

```python
from rlm import create_rlm
from rlm.adapters.llm import MockLLMAdapter

# Deterministic mock for testing
rlm = create_rlm(
    MockLLMAdapter(model="test", script=["```repl\nx = 42\n```\nFINAL_VAR('x')"]),
    environment="local",
    max_iterations=2,
)

result = rlm.completion("test prompt")
assert result.response == "42"
```

## Execution Environments

RLM supports multiple execution environments for running Python code blocks:

### Local Environment (default)

Executes code in-process with a persistent namespace. Fast and convenient for development.

```python
rlm = create_rlm(
    llm,
    environment="local",
    environment_kwargs={
        "execute_timeout_s": 30.0,           # Execution timeout (SIGALRM-based)
        "broker_timeout_s": 60.0,            # Timeout for nested LLM calls
        "allowed_import_roots": {"json", "math", "collections"},  # Allowed imports
    },
)
```

**Default allowed imports:** `collections`, `dataclasses`, `datetime`, `decimal`, `functools`, `itertools`, `json`, `math`, `pathlib`, `random`, `re`, `statistics`, `string`, `textwrap`, `typing`, `uuid`

### Docker Environment

Executes code in an isolated container. Recommended for untrusted code or production use.

```python
rlm = create_rlm(
    llm,
    environment="docker",
    environment_kwargs={
        "image": "python:3.12-slim",         # Docker image
        "subprocess_timeout_s": 120.0,       # Container execution timeout
        "proxy_http_timeout_s": 60.0,        # HTTP proxy timeout for LLM calls
    },
)
```

**Requirements:**
- Docker daemon running (`docker info` succeeds)
- Docker 20.10+ (for `--add-host host.docker.internal:host-gateway`)

## Multi-Backend Routing

RLM supports registering multiple LLM backends. Code blocks can route nested calls to specific models:

```python
from rlm import create_rlm
from rlm.adapters.llm import MockLLMAdapter

# Root model generates code that calls a sub-model
root_script = """```repl\nresponse = llm_query("What is the capital of France?", model="sub")```\nFINAL_VAR('response')"""

rlm = create_rlm(
    MockLLMAdapter(model="root", script=[root_script]),
    other_llms=[MockLLMAdapter(model="sub", script=["Paris"])],
    environment="local",
    max_iterations=3,
)

result = rlm.completion("hello")
assert result.response == "Paris"

# Usage is aggregated across all models
print(result.usage_summary.model_usage_summaries["root"].total_calls)  # 1
print(result.usage_summary.model_usage_summaries["sub"].total_calls)   # 1
```

### Batched LLM queries

For efficiency, code can batch multiple LLM calls:

```python
# Inside a ```repl block:
responses = llm_query_batched([
    "Question 1",
    "Question 2",
    "Question 3",
], model="fast-model")
```

## CLI Usage

```bash
# Show version
rlm --version

# Run a completion with mock backend (no API keys)
rlm completion "What is 2+2?" --backend mock --model-name test

# Run with OpenAI
rlm completion "Explain recursion" --backend openai --model-name gpt-4o

# Output full JSON response
rlm completion "Calculate pi" --backend mock --json

# Enable JSONL logging
rlm completion "Hello" --backend mock --jsonl-log-dir ./logs
```

## Configuration-Driven Usage

For complex setups, use configuration objects:

```python
from rlm import create_rlm_from_config, RLMConfig, LLMConfig, EnvironmentConfig

config = RLMConfig(
    llm=LLMConfig(backend="openai", model_name="gpt-4o"),
    other_llms=[
        LLMConfig(backend="anthropic", model_name="claude-3-5-sonnet-20241022"),
    ],
    env=EnvironmentConfig(environment="docker"),
    max_iterations=15,
    max_depth=1,
)

rlm = create_rlm_from_config(config)
result = rlm.completion("Solve this step by step...")
```

## Async Support

```python
import asyncio
from rlm import create_rlm
from rlm.adapters.llm import MockLLMAdapter

async def main():
    rlm = create_rlm(
        MockLLMAdapter(model="test", script=["FINAL('done')"]),
        environment="local",
    )
    result = await rlm.acompletion("async test")
    print(result.response)

asyncio.run(main())
```

## Tool Calling (Agent Mode)

RLM supports native tool calling across all LLM providers, enabling true agentic workflows where the model can invoke functions and use their results.

### Basic Tool Usage

```python
from rlm import create_rlm
from rlm.adapters.llm import OpenAIAdapter
from rlm.adapters.tools import tool, ToolRegistry

# Define tools using the @tool decorator
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 72°F"

@tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)

# Create a tool registry
registry = ToolRegistry()
registry.register(get_weather)
registry.register(calculate)

# Create RLM with tools
rlm = create_rlm(
    OpenAIAdapter(model="gpt-4o"),
    environment="local",
    tools=registry,
)

# The model can now call tools automatically
result = rlm.completion("What's the weather in Tokyo and what's 15 * 7?")
```

### Tool Choice Control

Control how the model uses tools:

```python
# Let model decide when to use tools (default)
result = rlm.completion("...", tool_choice="auto")

# Force tool usage
result = rlm.completion("...", tool_choice="required")

# Disable tools for this call
result = rlm.completion("...", tool_choice="none")

# Force a specific tool
result = rlm.completion("...", tool_choice="get_weather")
```

### Structured Outputs with Pydantic

Use Pydantic models for type-safe structured outputs:

```python
from pydantic import BaseModel
from rlm.adapters.tools import pydantic_to_schema

class WeatherReport(BaseModel):
    city: str
    temperature: float
    conditions: str
    humidity: int

# Pydantic models are automatically converted to JSON Schema
schema = pydantic_to_schema(WeatherReport)
```

## Extension Protocols

Customize RLM's orchestrator behavior using duck-typed protocols:

```python
from rlm import create_rlm
from rlm.domain import StoppingPolicy, ContextCompressor, NestedCallPolicy

# Custom stopping policy - stop after specific conditions
class TokenBudgetPolicy(StoppingPolicy):
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.used = 0

    def should_stop(self, iteration: int, response: str, usage: dict) -> bool:
        self.used += usage.get("total_tokens", 0)
        return self.used >= self.max_tokens

# Use custom policy
rlm = create_rlm(
    llm,
    environment="local",
    stopping_policy=TokenBudgetPolicy(max_tokens=10000),
)
```

Available protocols:
- **`StoppingPolicy`**: Control when the tool/iteration loop terminates
- **`ContextCompressor`**: Compress conversation context between iterations
- **`NestedCallPolicy`**: Configure handling of nested `llm_query()` calls

See [docs/extending.md](docs/extending.md) for detailed documentation.

## LLM Provider Configuration

| Provider | Extra | Environment Variables |
|----------|-------|----------------------|
| OpenAI | (default) | `OPENAI_API_KEY` |
| Anthropic | `llm-anthropic` | `ANTHROPIC_API_KEY` |
| Google Gemini | `llm-gemini` | `GOOGLE_API_KEY` |
| Azure OpenAI | `llm-azure-openai` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| LiteLLM | `llm-litellm` | (varies by provider) |
| Portkey | `llm-portkey` | `PORTKEY_API_KEY` |

## Architecture

RLM uses a **hexagonal (ports & adapters) architecture**:

```
src/rlm/
├── domain/          # Pure business logic, ports (protocols), models
├── application/     # Use cases, configuration
├── infrastructure/  # Wire protocol, execution policies
├── adapters/
│   ├── llm/         # LLM providers (OpenAI, Anthropic, Gemini, etc.)
│   ├── environments/# Execution environments (local, docker)
│   ├── tools/       # Tool calling infrastructure
│   ├── policies/    # Extension protocol implementations
│   ├── broker/      # TCP broker for nested LLM calls
│   └── loggers/     # Logging adapters (JSONL, console)
└── api/             # Public facade, factories, registries
```

**Key design principles:**
- Domain layer has zero external dependencies
- Adapters implement domain ports (protocols)
- Dependencies flow inward (adapters -> application -> domain)
- All LLM provider SDKs are lazy-imported (optional extras)
- Extension protocols enable customization without modifying core code

## Development

### Setup

```bash
# Clone and setup
git clone https://github.com/Luiz-Frias/agentic-codebase-navigator.git
cd agentic-codebase-navigator

# Create venv with Python 3.12
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install with dev dependencies
uv sync --group dev --group test
```

### Running Tests

```bash
# Unit tests (fast, hermetic)
uv run --group test pytest -m unit

# Integration tests (multi-component boundaries)
uv run --group test pytest -m integration

# End-to-end tests (public API flows). Docker-marked tests auto-skip if Docker isn't available.
uv run --group test pytest -m e2e

# Packaging smoke tests (build/install/import/CLI)
uv run --group test pytest -m packaging

# Performance/regression tests (opt-in)
uv run --group test pytest -m performance

# All tests
uv run --group test pytest

# With coverage
uv run --group test pytest --cov=rlm --cov-report=term-missing
```

#### Live provider smoke tests (opt-in)

These tests are skipped by default to avoid accidental spend. Enable with `RLM_RUN_LIVE_LLM_TESTS=1`
and the relevant API key:

```bash
RLM_RUN_LIVE_LLM_TESTS=1 OPENAI_API_KEY=... uv run --group test pytest -m "integration and live_llm"
RLM_RUN_LIVE_LLM_TESTS=1 ANTHROPIC_API_KEY=... uv run --group test pytest -m "integration and live_llm"
```

### Code Quality

```bash
# Format
uv run --group dev ruff format src tests

# Lint
uv run --group dev ruff check src tests --fix

# Type check
uv run --group dev ty check src/rlm
```

## API Reference

### Core Classes

- **`RLM`** - Main facade for running completions
- **`ChatCompletion`** - Result object with response, usage, iterations
- **`RLMConfig`** - Configuration dataclass for `create_rlm_from_config`

### Factory Functions

- **`create_rlm(llm, ...)`** - Create RLM with pre-built LLM adapter
- **`create_rlm_from_config(config)`** - Create RLM from configuration object

### Adapters

- **LLM**: `MockLLMAdapter`, `OpenAIAdapter`, `AnthropicAdapter`, `GeminiAdapter`, `AzureOpenAIAdapter`, `LiteLLMAdapter`, `PortkeyAdapter`
- **Environment**: `LocalEnvironmentAdapter`, `DockerEnvironmentAdapter`
- **Logger**: `JsonlLoggerAdapter`, `ConsoleLoggerAdapter`, `NoopLoggerAdapter`
- **Tools**: `ToolRegistry`, `tool` decorator, `NativeToolAdapter`

### Extension Protocols

- **`StoppingPolicy`** - Control iteration termination
- **`ContextCompressor`** - Compress context between iterations
- **`NestedCallPolicy`** - Configure nested `llm_query()` handling

## Acknowledgments

This project is built upon the excellent **Recursive Language Models (RLM)** research by Alex Zhang and colleagues from MIT OASYS Lab.

| Resource | Link |
|----------|------|
| **Original Repository** | [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm) |
| **Research Paper** | [arXiv:2512.24601](https://arxiv.org/abs/2512.24601) |
| **Authors** | Alex L. Zhang, Tim Kraska, Omar Khattab |

This repository refactors the original RLM implementation into a hexagonal/modular monolith architecture while maintaining API compatibility. See [ATTRIBUTION.md](ATTRIBUTION.md) for full details.

### Citation

```bibtex
@misc{zhang2025recursivelanguagemodels,
      title={Recursive Language Models},
      author={Alex L. Zhang and Tim Kraska and Omar Khattab},
      year={2025},
      eprint={2512.24601},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.24601},
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

- Original work: Copyright (c) 2025 Alex Zhang
- Refactored work: Copyright (c) 2026 Luiz Frias
