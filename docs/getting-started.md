# Getting Started

This guide will help you get up and running with RLM in under 5 minutes.

## Installation

### Basic Installation

```bash
# Using pip
pip install agentic-codebase-navigator

# Using uv (recommended)
uv pip install agentic-codebase-navigator
```

### With LLM Providers

Install with your preferred LLM provider:

```bash
# OpenAI (included by default)
pip install agentic-codebase-navigator

# Anthropic (Claude)
pip install "agentic-codebase-navigator[llm-anthropic]"

# Google Gemini
pip install "agentic-codebase-navigator[llm-gemini]"

# Azure OpenAI
pip install "agentic-codebase-navigator[llm-azure-openai]"

# LiteLLM (universal provider)
pip install "agentic-codebase-navigator[llm-litellm]"

# Multiple providers
pip install "agentic-codebase-navigator[llm-openai,llm-anthropic]"
```

## Quick Start

### 1. Set Your API Key

```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 2. Use the CLI

```bash
# Simple completion
rlm completion "What is 2 + 2?"

# With specific backend
rlm completion "Explain hexagonal architecture" --backend openai --model-name gpt-4

# Output as JSON
rlm completion "Hello world" --json
```

### 3. Use the Python API

```python
from rlm import create_rlm
from rlm.adapters.llm.openai import build_openai_adapter

# Create LLM adapter
llm = build_openai_adapter(model="gpt-4")

# Create RLM instance
rlm = create_rlm(llm, environment="local", max_iterations=10)

# Run completion
result = rlm.completion("What is the capital of France?")
print(result.response)
# Output: Paris

# Check token usage
print(result.usage_summary.to_dict())
```

## Code Mode vs Tools Mode

RLM supports two agent modes:

### Code Mode (Default)

The LLM generates Python code that gets executed in a sandboxed environment:

```python
from rlm import create_rlm
from rlm.adapters.llm.openai import build_openai_adapter

llm = build_openai_adapter(model="gpt-4")
rlm = create_rlm(llm, agent_mode="code", environment="local")

# The LLM will generate and execute code to answer
result = rlm.completion("Calculate the first 10 Fibonacci numbers")
print(result.response)
# The LLM generates: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

### Tools Mode (Function Calling)

Register Python functions as tools for the LLM to call:

```python
from rlm import create_rlm
from rlm.adapters.llm.openai import build_openai_adapter

def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: The city name

    Returns:
        Weather data dictionary
    """
    # Your weather API logic here
    return {"city": city, "temperature": 72, "condition": "sunny"}

def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b

llm = build_openai_adapter(model="gpt-4")
rlm = create_rlm(
    llm,
    agent_mode="tools",
    tools=[get_weather, add],
)

result = rlm.completion("What's the weather in San Francisco?")
print(result.response)
# Output: The weather in San Francisco is 72°F and sunny.
```

## Config-Based Setup

For production use, configure RLM via dataclasses:

```python
from rlm import create_rlm_from_config
from rlm.application.config import (
    RLMConfig,
    LLMConfig,
    EnvironmentConfig,
    LoggerConfig,
)

config = RLMConfig(
    llm=LLMConfig(
        backend="openai",
        model_name="gpt-4",
        backend_kwargs={"temperature": 0.7}
    ),
    env=EnvironmentConfig(
        environment="local",
        environment_kwargs={"execute_timeout_s": 30.0}
    ),
    logger=LoggerConfig(
        logger="jsonl",
        logger_kwargs={"log_dir": "./logs"}
    ),
    max_iterations=20,
    max_depth=1,
    agent_mode="code",
)

rlm = create_rlm_from_config(config)
result = rlm.completion("Analyze this data...")
```

## Async Support

All completion methods have async variants:

```python
import asyncio
from rlm import create_rlm
from rlm.adapters.llm.openai import build_openai_adapter

async def main():
    llm = build_openai_adapter(model="gpt-4")
    rlm = create_rlm(llm, environment="local")

    # Async completion
    result = await rlm.acompletion("What is machine learning?")
    print(result.response)

asyncio.run(main())
```

## Multi-Backend Routing

Use multiple LLM backends in a single RLM instance:

```python
from rlm import create_rlm
from rlm.adapters.llm.openai import build_openai_adapter
from rlm.adapters.llm.anthropic import build_anthropic_adapter

# Primary model (for main loop)
gpt4 = build_openai_adapter(model="gpt-4")

# Secondary model (for nested llm_query() calls)
claude = build_anthropic_adapter(model="claude-3-opus-20240229")

rlm = create_rlm(
    gpt4,
    other_llms=[claude],
    environment="local",
)

# Code can call: llm_query("...", model="claude-3-opus-20240229")
result = rlm.completion("Complex multi-model task...")
```

## Next Steps

- [Configuration Reference](configuration.md) — All configuration options
- [LLM Providers](providers/llm-providers.md) — Setup different providers
- [Environments](environments/execution-environments.md) — Local vs Docker execution
- [Extension Protocols](extending/extension-protocols.md) — Custom stopping policies
- [Testing Guide](testing/testing-guide.md) — How to test your RLM integrations

## Common Issues

### "API key not found"

```bash
# Ensure your API key is set
export OPENAI_API_KEY="sk-..."
```

### "Docker not available"

For Docker environment, ensure Docker is running:

```bash
docker info  # Should not error
```

### Timeout errors

Increase timeouts via environment kwargs:

```python
config = RLMConfig(
    env=EnvironmentConfig(
        environment="local",
        environment_kwargs={"execute_timeout_s": 60.0}
    ),
    ...
)
```

See [Troubleshooting](troubleshooting.md) for more solutions.
