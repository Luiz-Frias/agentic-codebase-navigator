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

## Tool Calling Mode (Function Calling)

RLM supports two agent modes:
- `"code"` (default): LLM generates Python code in \`\`\`repl blocks for execution
- `"tools"`: LLM uses function calling to invoke registered tools

### Basic Tool Calling

Define tools as Python callables with type hints and docstrings:

```python
from rlm.api import create_rlm
from rlm.adapters.llm.openai import OpenAIAdapter

def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b

def get_weather(city: str, unit: str = "celsius") -> dict:
    """Get weather for a city.

    Args:
        city: City name
        unit: Temperature unit (celsius or fahrenheit)

    Returns:
        Weather data dictionary
    """
    # Your weather API logic here
    return {"city": city, "temperature": 22, "unit": unit}

rlm = create_rlm(
    OpenAIAdapter(model="gpt-4"),
    tools=[add, get_weather],
    agent_mode="tools",
)
result = rlm.completion("What is 5 + 3?")
print(result.response)  # "The sum of 5 and 3 is 8."
```

### Config-Based Tool Calling

Use `create_rlm_from_config()` with tools injected at runtime:

```python
from rlm.api import create_rlm_from_config
from rlm.application.config import RLMConfig, LLMConfig, EnvironmentConfig

config = RLMConfig(
    llm=LLMConfig(backend="openai", model_name="gpt-4"),
    env=EnvironmentConfig(environment="local"),
    agent_mode="tools",  # Enable tool calling mode
)

# Tools are Python callables - inject at runtime (not serializable to config)
rlm = create_rlm_from_config(config, tools=[add, get_weather])
result = rlm.completion("What's the weather in Tokyo?")
```

### Async Tool Calling

Both sync and async paths support tool calling:

```python
# Sync
result = rlm.completion("Calculate 7 times 8")

# Async
result = await rlm.acompletion("Calculate 7 times 8")
```

### Tool Execution Errors

Errors from tool execution are captured and passed to the LLM for graceful handling:

```python
def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

rlm = create_rlm(llm, tools=[divide], agent_mode="tools")
# LLM will receive the error message and can respond appropriately
result = rlm.completion("Divide 10 by 0")
```

### Multi-Turn Tool Calling

The orchestrator handles multi-turn conversations where the LLM may call tools multiple times:

```text
User: "What is (4 * 5) + 10?"
  → LLM calls multiply(4, 5) → returns 20
  → LLM calls add(20, 10) → returns 30
  → LLM returns "4 times 5 is 20, plus 10 equals 30."
```

### Provider Support

Tool calling is supported across multiple LLM providers:
- **OpenAI** / **Azure OpenAI**: Native function calling
- **Anthropic**: Tool use via content blocks
- **Gemini**: FunctionDeclaration format
- **LiteLLM** / **Portkey**: Passthrough to underlying provider

## Migration notes

The upstream snapshot remains available under `references/rlm/**` for reference only.
Runtime code lives under `src/rlm/**`. Legacy has been fully removed.
