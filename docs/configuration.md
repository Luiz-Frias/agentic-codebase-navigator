# Configuration Reference

This document describes all configuration options for RLM.

## Overview

RLM can be configured in two ways:

1. **Direct instantiation** — Pass objects directly to `create_rlm()`
2. **Config-based** — Use `RLMConfig` dataclass with `create_rlm_from_config()`

## Direct Instantiation

```python
from rlm import create_rlm
from rlm.adapters.llm.openai import build_openai_adapter

llm = build_openai_adapter(model="gpt-4")
rlm = create_rlm(
    llm,
    environment="local",
    max_iterations=20,
    max_depth=1,
)
```

### `create_rlm()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `LLMPort` | Required | Primary LLM adapter |
| `other_llms` | `list[LLMPort]` | `[]` | Additional LLMs for multi-backend routing |
| `environment` | `str` | `"local"` | Execution environment name |
| `environment_kwargs` | `dict` | `{}` | Environment-specific options |
| `max_iterations` | `int` | `30` | Maximum orchestrator loop iterations |
| `max_depth` | `int` | `1` | Maximum recursion depth for nested calls |
| `agent_mode` | `str` | `"code"` | Agent mode: `"code"` or `"tools"` |
| `tools` | `list` | `None` | Tools for function calling (tools mode only) |
| `stopping_policy` | `StoppingPolicy` | `None` | Custom iteration stopping logic |
| `context_compressor` | `ContextCompressor` | `None` | Nested result compression |
| `nested_call_policy` | `NestedCallPolicy` | `None` | Sub-orchestrator spawning control |
| `logger` | `LoggerPort` | `None` | Logger for run tracking |

## Config-Based Setup

```python
from rlm import create_rlm_from_config
from rlm.application.config import (
    RLMConfig,
    LLMConfig,
    EnvironmentConfig,
    LoggerConfig,
)

config = RLMConfig(
    llm=LLMConfig(...),
    env=EnvironmentConfig(...),
    logger=LoggerConfig(...),
    max_iterations=30,
    max_depth=1,
    agent_mode="code",
)

rlm = create_rlm_from_config(config, tools=[...])  # Tools injected at runtime
```

## RLMConfig

Top-level configuration container.

```python
@dataclass
class RLMConfig:
    llm: LLMConfig
    env: EnvironmentConfig = field(default_factory=lambda: EnvironmentConfig())
    logger: LoggerConfig = field(default_factory=lambda: LoggerConfig())
    other_llms: list[LLMConfig] = field(default_factory=list)
    max_iterations: int = 30
    max_depth: int = 1
    agent_mode: AgentModeName = "code"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llm` | `LLMConfig` | Required | Primary LLM configuration |
| `env` | `EnvironmentConfig` | `local` | Execution environment config |
| `logger` | `LoggerConfig` | `none` | Logging configuration |
| `other_llms` | `list[LLMConfig]` | `[]` | Additional LLMs for routing |
| `max_iterations` | `int` | `30` | Max orchestrator iterations |
| `max_depth` | `int` | `1` | Max recursion depth |
| `agent_mode` | `str` | `"code"` | `"code"` or `"tools"` |

## LLMConfig

LLM provider configuration.

```python
@dataclass
class LLMConfig:
    backend: str
    model_name: str
    backend_kwargs: dict[str, Any] = field(default_factory=dict)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | `str` | Required | Provider name (see below) |
| `model_name` | `str` | Required | Model identifier |
| `backend_kwargs` | `dict` | `{}` | Provider-specific options |

### Supported Backends

| Backend | Extra Required | Environment Variable |
|---------|----------------|---------------------|
| `mock` | None | None |
| `openai` | `llm-openai` | `OPENAI_API_KEY` |
| `anthropic` | `llm-anthropic` | `ANTHROPIC_API_KEY` |
| `gemini` | `llm-gemini` | `GOOGLE_API_KEY` |
| `azure_openai` | `llm-azure-openai` | `AZURE_OPENAI_API_KEY` |
| `portkey` | `llm-portkey` | `PORTKEY_API_KEY` |
| `litellm` | `llm-litellm` | Varies by provider |

### Backend-Specific kwargs

#### OpenAI

```python
LLMConfig(
    backend="openai",
    model_name="gpt-4",
    backend_kwargs={
        "api_key": "sk-...",  # Or use OPENAI_API_KEY env var
        "temperature": 0.7,
        "max_tokens": 4096,
        "base_url": "https://api.openai.com/v1",  # Custom endpoint
    }
)
```

#### Anthropic

```python
LLMConfig(
    backend="anthropic",
    model_name="claude-3-opus-20240229",
    backend_kwargs={
        "api_key": "sk-ant-...",  # Or use ANTHROPIC_API_KEY env var
        "max_tokens": 4096,
    }
)
```

#### Azure OpenAI

```python
LLMConfig(
    backend="azure_openai",
    model_name="my-deployment-name",
    backend_kwargs={
        "api_key": "...",  # Or use AZURE_OPENAI_API_KEY
        "azure_endpoint": "https://xxx.openai.azure.com/",
        "api_version": "2024-02-15-preview",
    }
)
```

#### Mock (Testing)

```python
LLMConfig(
    backend="mock",
    model_name="mock-model",
    backend_kwargs={
        "script": ["Response 1", "FINAL(done)"],  # Scripted responses
    }
)
```

## EnvironmentConfig

Code execution environment configuration.

```python
@dataclass
class EnvironmentConfig:
    environment: EnvironmentName = "local"
    environment_kwargs: dict[str, Any] = field(default_factory=dict)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `environment` | `str` | `"local"` | Environment name |
| `environment_kwargs` | `dict` | `{}` | Environment-specific options |

### Supported Environments

| Environment | Description | Docker Required |
|-------------|-------------|-----------------|
| `local` | In-process Python execution | No |
| `docker` | Isolated container execution | Yes |
| `modal` | Modal cloud execution (stub) | No |
| `prime` | Prime cloud execution (stub) | No |

### Environment-Specific kwargs

#### Local Environment

```python
EnvironmentConfig(
    environment="local",
    environment_kwargs={
        "execute_timeout_s": 30.0,       # Code execution timeout
        "broker_timeout_s": 30.0,        # LLM call timeout
        "allowed_import_roots": {"/custom"},  # Additional allowed imports
        "setup_code": "import pandas",   # Code to run before each execution
    }
)
```

#### Docker Environment

```python
EnvironmentConfig(
    environment="docker",
    environment_kwargs={
        "image": "python:3.12-slim",     # Docker image
        "subprocess_timeout_s": 120.0,   # docker exec timeout
        "proxy_http_timeout_s": 60.0,    # HTTP proxy timeout
        "setup_code": "import numpy",    # Setup code
    }
)
```

## LoggerConfig

Logging configuration.

```python
@dataclass
class LoggerConfig:
    logger: LoggerName = "none"
    logger_kwargs: dict[str, Any] = field(default_factory=dict)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `logger` | `str` | `"none"` | Logger type |
| `logger_kwargs` | `dict` | `{}` | Logger-specific options |

### Supported Loggers

| Logger | Description |
|--------|-------------|
| `none` | Logging disabled |
| `jsonl` | JSONL file logging |
| `console` | Stdout logging |

### Logger-Specific kwargs

#### JSONL Logger

```python
LoggerConfig(
    logger="jsonl",
    logger_kwargs={
        "log_dir": "./logs",           # Directory for log files
        "file_name": "rlm_run",        # Base filename (timestamp added)
    }
)
```

#### Console Logger

```python
LoggerConfig(
    logger="console",
    logger_kwargs={
        "enabled": True,               # Enable/disable output
    }
)
```

## Agent Modes

### Code Mode

Default mode where the LLM generates Python code:

```python
config = RLMConfig(
    llm=LLMConfig(backend="openai", model_name="gpt-4"),
    agent_mode="code",  # Default
)
```

The LLM generates code in ` ```repl ` blocks that get executed:

```python
```repl
result = 2 + 2
print(result)
```
```

### Tools Mode

Function calling mode where the LLM invokes registered tools:

```python
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

config = RLMConfig(
    llm=LLMConfig(backend="openai", model_name="gpt-4"),
    agent_mode="tools",
)

rlm = create_rlm_from_config(config, tools=[add])
```

## Extension Protocols

### StoppingPolicy

Custom iteration loop termination:

```python
from rlm import create_rlm, DefaultStoppingPolicy

class MyStoppingPolicy:
    def should_stop(self, context: dict) -> bool:
        return context["iteration"] >= 10

    def on_iteration_complete(self, context: dict, result) -> None:
        pass

rlm = create_rlm(llm, stopping_policy=MyStoppingPolicy())
```

### ContextCompressor

Compress nested call results:

```python
from rlm import create_rlm, NoOpContextCompressor

class MyCompressor:
    def compress(self, result: str, max_tokens: int | None = None) -> str:
        if len(result) > 500:
            return result[:500] + "..."
        return result

rlm = create_rlm(llm, context_compressor=MyCompressor())
```

### NestedCallPolicy

Control sub-orchestrator spawning:

```python
from rlm import create_rlm, SimpleNestedCallPolicy

class MyNestedPolicy:
    def should_orchestrate(self, prompt: str, depth: int) -> bool:
        return depth >= 2

    def get_nested_config(self):
        return {"max_iterations": 10}

rlm = create_rlm(llm, nested_call_policy=MyNestedPolicy())
```

See [Extension Protocols](extending/extension-protocols.md) for detailed examples.

## Complete Example

```python
from rlm import create_rlm_from_config
from rlm.application.config import (
    RLMConfig,
    LLMConfig,
    EnvironmentConfig,
    LoggerConfig,
)

# Full production configuration
config = RLMConfig(
    llm=LLMConfig(
        backend="openai",
        model_name="gpt-4",
        backend_kwargs={
            "temperature": 0.7,
            "max_tokens": 4096,
        }
    ),
    other_llms=[
        LLMConfig(
            backend="anthropic",
            model_name="claude-3-opus-20240229",
            backend_kwargs={"max_tokens": 4096}
        )
    ],
    env=EnvironmentConfig(
        environment="docker",
        environment_kwargs={
            "image": "python:3.12-slim",
            "subprocess_timeout_s": 120.0,
        }
    ),
    logger=LoggerConfig(
        logger="jsonl",
        logger_kwargs={"log_dir": "./logs"}
    ),
    max_iterations=30,
    max_depth=2,
    agent_mode="code",
)

rlm = create_rlm_from_config(config)
result = rlm.completion("Complex multi-step task...")
```

## See Also

- [Getting Started](getting-started.md) — Quick start guide
- [LLM Providers](providers/llm-providers.md) — Provider setup details
- [Environments](environments/execution-environments.md) — Environment details
- [Extension Protocols](extending/extension-protocols.md) — Custom policies
