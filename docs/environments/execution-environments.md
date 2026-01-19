# Execution Environments

RLM supports multiple execution environments for running generated code. Each environment provides different trade-offs between isolation, performance, and security.

## Overview

| Environment | Isolation | Performance | Security | Requirements |
|-------------|-----------|-------------|----------|--------------|
| **Local** | Process | Fast | Medium | None |
| **Docker** | Container | Medium | High | Docker daemon |
| **Modal** | Cloud | Slow | High | Modal account |
| **Prime** | Cloud | Slow | High | Prime account |

## Local Environment

The local environment executes Python code in **isolated subprocess workers** (v1.2.0+). Each execution runs in a separate process with IPC communication to the parent orchestrator.

> **Note (v1.2.0)**: Prior to v1.2.0, local execution ran in-process with a shared namespace. The subprocess model improves isolation and reliability. See [Migration Guide](../migration.md) for upgrade details.

### Architecture

```
┌─────────────────────────────────────────────┐
│             Parent Process                   │
│  ┌─────────────┐    ┌──────────────────┐   │
│  │ RLM         │    │ Broker           │   │
│  │ Orchestrator│◄───│ (llm_query)      │   │
│  └─────────────┘    └──────────────────┘   │
│         │                   ▲               │
│         │ spawn             │ IPC           │
│         ▼                   │               │
│  ┌──────────────────────────┴──────────┐   │
│  │         Worker Subprocess            │   │
│  │  ┌─────────────────────────────┐    │   │
│  │  │     Isolated Namespace      │    │   │
│  │  │  - Code execution           │    │   │
│  │  │  - llm_query() → parent IPC │    │   │
│  │  └─────────────────────────────┘    │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### Configuration

```python
from rlm import create_rlm_from_config
from rlm.application.config import RLMConfig, LLMConfig, EnvironmentConfig

config = RLMConfig(
    llm=LLMConfig(backend="openai", model_name="gpt-4"),
    env=EnvironmentConfig(
        environment="local",
        environment_kwargs={
            "execute_timeout_s": 30.0,
            "broker_timeout_s": 30.0,
            "allowed_import_roots": {"/custom/modules"},
            "setup_code": "import pandas as pd\nimport numpy as np",
        }
    ),
)

rlm = create_rlm_from_config(config)
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `execute_timeout_s` | `float` | `30.0` | Per-execution timeout (process kill on timeout) |
| `execute_timeout_cap_s` | `float` | `300.0` | Maximum allowed timeout cap |
| `broker_timeout_s` | `float` | `30.0` | Timeout for `llm_query()` IPC calls |
| `allowed_import_roots` | `set[str]` | Standard library | Additional allowed import paths |
| `setup_code` | `str` | `None` | Code to run in worker before each execution |

### Security Model

The local environment provides **medium security** through:

1. **Process Isolation**: Each execution runs in a subprocess (v1.2.0+)
2. **Import Restrictions**: Only allowed modules can be imported
3. **Builtins Filtering**: Dangerous builtins (`eval`, `exec`, `compile`, `__import__`) are removed
4. **File I/O Restrictions**: `open()` is replaced with a restricted version
5. **Temporary Directory**: Each session gets an isolated temp directory

**Note**: While subprocess isolation improves reliability, for truly untrusted code use Docker.

### Timeout Behavior (v1.2.0+)

- Worker process is killed on timeout (reliable across platforms)
- Returns `TimeoutError` in `stderr`, triggers cleanup
- Falls back to SIGALRM for in-worker timeouts (Unix main thread only)

### Best For

- Development and testing
- Trusted code execution
- Fast iteration cycles
- Simple computations

## Docker Environment

The Docker environment executes code in an isolated container with a Python runtime.

### Configuration

```python
config = RLMConfig(
    llm=LLMConfig(backend="openai", model_name="gpt-4"),
    env=EnvironmentConfig(
        environment="docker",
        environment_kwargs={
            "image": "python:3.12-slim",
            "subprocess_timeout_s": 120.0,
            "proxy_http_timeout_s": 60.0,
            "setup_code": "import requests",
        }
    ),
)
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `image` | `str` | `python:3.12-slim` | Docker image to use |
| `subprocess_timeout_s` | `float` | `120.0` | Timeout for `docker exec` commands |
| `proxy_http_timeout_s` | `float` | `60.0` | Timeout for HTTP proxy requests |
| `setup_code` | `str` | `None` | Code to run before each execution |

### Architecture

```
┌─────────────────────────────────────────────┐
│                Host Process                 │
│                                             │
│  ┌─────────────┐    ┌──────────────────┐   │
│  │ RLM         │    │ HTTP Proxy       │   │
│  │ Orchestrator│◄───│ (llm_query)      │   │
│  └─────────────┘    └──────────────────┘   │
│         │                   ▲               │
│         │ docker exec       │ HTTP          │
│         ▼                   │               │
│  ┌──────────────────────────┴──────────┐   │
│  │           Docker Container           │   │
│  │  ┌─────────────────────────────┐    │   │
│  │  │     Python Runtime          │    │   │
│  │  │  - context variable         │    │   │
│  │  │  - llm_query() → HTTP proxy │    │   │
│  │  └─────────────────────────────┘    │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### Container Lifecycle

1. **Start**: Container created from image, proxy started
2. **Load Context**: Context serialized and injected into container
3. **Execute**: Code run via `docker exec`, results captured
4. **Cleanup**: Container stopped and removed

### LLM Queries from Container

Generated code can call `llm_query()` from within the container:

```python
# Inside container, this works:
response = llm_query("What is 2+2?")
responses = llm_query_batched(["Q1", "Q2", "Q3"])
```

The HTTP proxy forwards these calls to the host broker.

### Custom Images

For custom dependencies, build a Docker image:

```dockerfile
# Dockerfile.rlm
FROM python:3.12-slim

RUN pip install numpy pandas scikit-learn

# No CMD needed - RLM uses docker exec
```

```bash
docker build -t my-rlm-env:latest -f Dockerfile.rlm .
```

```python
config = RLMConfig(
    env=EnvironmentConfig(
        environment="docker",
        environment_kwargs={"image": "my-rlm-env:latest"}
    ),
    ...
)
```

### CI/CD Considerations

In CI environments, enable host networking:

```bash
export RLM_DOCKER_USE_HOST_NETWORK=1
```

This allows the container to reach the host proxy.

### Best For

- Untrusted code execution
- Production workloads
- Consistent environments across machines
- Complex dependencies

## Modal Environment (Stub)

Modal provides cloud-based execution with automatic scaling.

**Status**: Stub implementation — infrastructure hooks in place, full implementation pending.

### Configuration

```python
config = RLMConfig(
    env=EnvironmentConfig(
        environment="modal",
        environment_kwargs={
            # Modal-specific options (TBD)
        }
    ),
    ...
)
```

### Requirements

```bash
pip install "agentic-codebase-navigator[env-modal]"
```

### Best For (Planned)

- High-scale workloads
- GPU-accelerated computation
- Serverless execution

## Prime Environment (Stub)

Prime provides cloud-based execution optimized for AI workloads.

**Status**: Stub implementation — infrastructure hooks in place, full implementation pending.

### Configuration

```python
config = RLMConfig(
    env=EnvironmentConfig(
        environment="prime",
        environment_kwargs={
            # Prime-specific options (TBD)
        }
    ),
    ...
)
```

## Environment Comparison

### Security

| Environment | Code Isolation | Network Isolation | File System Isolation |
|-------------|---------------|-------------------|----------------------|
| Local | Process | No | Partial (restricted `open()`) |
| Docker | Container | Yes | Yes |
| Modal | VM/Container | Yes | Yes |
| Prime | VM/Container | Yes | Yes |

### Performance

| Environment | Startup Time | Execution Speed | Memory Overhead |
|-------------|-------------|-----------------|-----------------|
| Local | ~0ms | Native | Minimal |
| Docker | ~500ms | Near-native | ~50MB |
| Modal | ~2-5s | Native | Varies |
| Prime | ~2-5s | Native | Varies |

### Feature Support

| Feature | Local | Docker | Modal | Prime |
|---------|-------|--------|-------|-------|
| `llm_query()` | ✅ | ✅ | 🔜 | 🔜 |
| `llm_query_batched()` | ✅ | ✅ | 🔜 | 🔜 |
| Custom packages | Via system | Via image | 🔜 | 🔜 |
| GPU support | Via system | Via image | 🔜 | 🔜 |
| Auto-scaling | ❌ | ❌ | 🔜 | 🔜 |

## Choosing an Environment

### Decision Tree

```
Is the code trusted?
├─ YES: Use Local (fastest)
└─ NO: Is Docker available?
       ├─ YES: Use Docker (secure, portable)
       └─ NO: Use Local with caution
              (or set up Docker)
```

### Recommendations

| Use Case | Recommended Environment |
|----------|------------------------|
| Development & testing | Local |
| Production (trusted) | Local or Docker |
| Production (untrusted) | Docker |
| CI/CD pipelines | Docker |
| High-scale workloads | Modal/Prime (when available) |

## Implementing Custom Environments

To create a custom environment, implement the `EnvironmentPort` protocol:

```python
from rlm.domain.ports import EnvironmentPort
from rlm.domain.models import ReplResult

class CustomEnvironmentAdapter:
    def __init__(
        self,
        *,
        broker,
        broker_address: tuple[str, int],
        correlation_id: str,
        custom_option: str = "default",
    ):
        self._broker = broker
        self._broker_address = broker_address
        self._correlation_id = correlation_id
        self._custom_option = custom_option
        self._context = None

    def load_context(self, context_payload) -> None:
        """Load context into the execution environment."""
        self._context = context_payload

    def execute_code(self, code: str) -> ReplResult:
        """Execute code and return results."""
        # Your implementation here
        return ReplResult(
            stdout="output",
            stderr="",
            locals={},
            execution_time=0.1,
            correlation_id=self._correlation_id,
            rlm_calls=[],
        )

    def cleanup(self) -> None:
        """Release resources."""
        pass
```

Then register it in the environment registry:

```python
# In rlm/api/registries.py
case "custom":
    from rlm.adapters.environments.custom import CustomEnvironmentAdapter
    return CustomEnvironmentAdapter(
        broker=broker,
        broker_address=broker_address,
        correlation_id=correlation_id,
        **env_kwargs
    )
```

## Troubleshooting

### Local Environment

**Timeout not working**:
- SIGALRM only works on Unix and main thread
- Use Docker for reliable timeouts

**Import errors**:
- Check `allowed_import_roots` includes necessary paths
- Ensure packages are installed in the Python environment

### Docker Environment

**Container not starting**:
```bash
# Check Docker is running
docker info

# Check image exists
docker images | grep python

# Pull if needed
docker pull python:3.12-slim
```

**Proxy connection errors**:
- Ensure `RLM_DOCKER_USE_HOST_NETWORK=1` in CI
- Check firewall isn't blocking container → host traffic

**Cleanup failures**:
```bash
# Manual cleanup
docker ps -a | grep rlm | awk '{print $1}' | xargs docker rm -f
```

## See Also

- [Configuration](../configuration.md) — All configuration options
- [Troubleshooting](../troubleshooting.md) — Common issues
- [Architecture](../architecture.md) — System design
