# Architecture

RLM implements a **hexagonal modular monolith** (ports & adapters) architecture that enables testability, extensibility, and clear separation of concerns.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer                                │
│                  (RLM, create_rlm, factories)                   │
│                    Composition Root                              │
├─────────────────────────────────────────────────────────────────┤
│                     Application Layer                            │
│              (use cases, configuration DTOs)                     │
├─────────────────────────────────────────────────────────────────┤
│                       Domain Layer                               │
│        (orchestrator, ports, models — zero dependencies)         │
├─────────────────────────────────────────────────────────────────┤
│                      Adapters Layer                              │
│     (LLM providers, environments, tools, broker, logger)         │
├─────────────────────────────────────────────────────────────────┤
│                    Infrastructure Layer                          │
│            (wire protocol, comms, execution policy)              │
└─────────────────────────────────────────────────────────────────┘
```

## Core Principle: Dependency Inversion

Dependencies **must point inward**:

- Outer layers (adapters/infrastructure) may depend on inner layers (domain/application)
- Inner layers must **not** import outer layers
- This is enforced via AST-based tests in `tests/unit/test_architecture_layering.py`

## Layers

### Domain Layer (`src/rlm/domain/`)

**Purpose**: Pure business logic with zero external dependencies.

**Contents**:
- **Ports** (`ports.py`, `agent_ports.py`): Protocol definitions (interfaces)
- **Models** (`models/`): Domain value objects (`ChatCompletion`, `LLMRequest`, `Iteration`, etc.)
- **Services** (`services/`): Core orchestrator and prompt builders
- **Errors** (`errors.py`): Domain exception hierarchy
- **Types** (`types.py`): Type aliases and literals

**Key Components**:

| Component | File | Purpose |
|-----------|------|---------|
| `LLMPort` | `ports.py` | Interface for LLM providers |
| `BrokerPort` | `ports.py` | Interface for request routing |
| `EnvironmentPort` | `ports.py` | Interface for code execution |
| `LoggerPort` | `ports.py` | Interface for run logging |
| `RLMOrchestrator` | `services/rlm_orchestrator.py` | Core iteration loop |
| `StoppingPolicy` | `agent_ports.py` | Extension: custom stopping |
| `ContextCompressor` | `agent_ports.py` | Extension: result compression |
| `NestedCallPolicy` | `agent_ports.py` | Extension: sub-orchestrator control |

### Application Layer (`src/rlm/application/`)

**Purpose**: Use cases and configuration schemas.

**Contents**:
- **Config** (`config.py`): Dataclass-based configuration (`RLMConfig`, `LLMConfig`, etc.)
- **Use Cases** (`use_cases/`): Application-level orchestration

**Key Use Case**: `run_completion(request, deps)`

1. Start broker (returns address for environment)
2. Build environment (bind broker address)
3. Log run metadata
4. Execute orchestrator loop
5. Merge usage summaries
6. Cleanup (environment, broker)

### Infrastructure Layer (`src/rlm/infrastructure/`)

**Purpose**: Cross-cutting technical utilities (still dependency-free).

**Contents**:
- **Wire Protocol** (`comms/`): Length-prefixed JSON framing for TCP
- **Execution Policy** (`execution_namespace_policy.py`): Security sandbox for code execution
- **Helpers**: ID generation, logging utilities

**Wire Protocol**:
- 4-byte big-endian length prefix + UTF-8 JSON payload
- `WireRequest` → `WireResponse` with correlation IDs
- Max payload: 10MB (configurable)

### Adapters Layer (`src/rlm/adapters/`)

**Purpose**: Concrete implementations of domain ports.

**Structure**:
```
adapters/
├── llm/              # LLM provider adapters
│   ├── openai.py
│   ├── anthropic.py
│   ├── gemini.py
│   ├── azure_openai.py
│   ├── litellm.py
│   ├── portkey.py
│   └── mock.py
├── environments/     # Code execution environments
│   ├── local.py
│   ├── docker.py
│   ├── modal.py      # stub
│   └── prime.py      # stub
├── tools/            # Tool calling infrastructure
│   ├── registry.py
│   └── native.py
├── policies/         # Extension protocol implementations
│   ├── stopping.py
│   ├── compression.py
│   └── nested.py
├── broker/           # Request routing
│   └── tcp.py
└── loggers/          # Logging implementations
    ├── jsonl.py
    └── console.py
```

### API Layer (`src/rlm/api/`)

**Purpose**: Public facade and composition root.

**Contents**:
- **Facade** (`rlm.py`): `RLM` class — main entry point
- **Factory** (`factory.py`): `create_rlm()`, `create_rlm_from_config()`
- **Registries** (`registries.py`): Config → adapter mapping

**Public API**:
```python
from rlm import (
    RLM,
    create_rlm,
    create_rlm_from_config,
    ChatCompletion,
    # Config
    RLMConfig, LLMConfig, EnvironmentConfig, LoggerConfig,
    # Extension protocols
    StoppingPolicy, ContextCompressor, NestedCallPolicy,
    # Default implementations
    DefaultStoppingPolicy, NoOpContextCompressor, SimpleNestedCallPolicy,
)
```

## Agent Modes

RLM supports two agent paradigms:

### Code Mode (Default)

LLM generates Python code in ` ```repl ` blocks:

```
User: What is 6 * 7?

LLM: Let me calculate that.
```repl
result = 6 * 7
print(f"The answer is {result}")
```

LLM: FINAL(42)
```

**Flow**:
1. Load context into environment
2. Build message history (system + metadata hint)
3. Loop: LLM call → extract code blocks → execute → check for FINAL → repeat
4. Return when FINAL found or max_iterations reached

### Tools Mode (Function Calling)

LLM invokes registered tools via function calling:

```
User: What's the weather in Tokyo?

LLM: [calls get_weather(city="Tokyo")]
Tool: {"city": "Tokyo", "temperature": 72}
LLM: The weather in Tokyo is 72°F.
```

**Flow**:
1. Build conversation (system + user message)
2. Loop: LLM call → extract tool_calls → execute tools → append results → repeat
3. Auto-summarize when conversation exceeds context window threshold
4. Return when LLM stops requesting tools

## Extension Protocols

Three protocols enable customization without modifying core code:

### StoppingPolicy

Control when the iteration loop terminates:

```python
class StoppingPolicy(Protocol):
    def should_stop(self, context: dict[str, Any]) -> bool: ...
    def on_iteration_complete(self, context: dict, result: ChatCompletion) -> None: ...
```

**Use Cases**: EIG-gated stopping, entropy thresholds, budget limits

### ContextCompressor

Compress nested call results before returning to parent:

```python
class ContextCompressor(Protocol):
    def compress(self, result: str, max_tokens: int | None = None) -> str: ...
```

**Use Cases**: Summarization, truncation, token budget management

### NestedCallPolicy

Control when nested `llm_query()` calls spawn sub-orchestrators:

```python
class NestedCallPolicy(Protocol):
    def should_orchestrate(self, prompt: str, depth: int) -> bool: ...
    def get_nested_config(self) -> NestedConfig: ...
```

**Use Cases**: Depth-based orchestration, prompt-based routing

## Request Flow

```
┌──────────┐    ┌─────────┐    ┌─────────────┐    ┌─────────────┐
│  User    │───▶│   RLM   │───▶│  Use Case   │───▶│ Orchestrator│
│          │    │ (Facade)│    │             │    │             │
└──────────┘    └─────────┘    └─────────────┘    └──────┬──────┘
                                                         │
                    ┌────────────────────────────────────┼────────────────────────────────────┐
                    │                                    │                                    │
                    ▼                                    ▼                                    ▼
            ┌───────────────┐                   ┌───────────────┐                   ┌───────────────┐
            │    LLMPort    │                   │ EnvironmentPort│                  │   BrokerPort  │
            │   (Adapter)   │                   │   (Adapter)   │                   │   (TCP/IPC)   │
            └───────────────┘                   └───────────────┘                   └───────────────┘
                    │                                    │                                    │
                    ▼                                    ▼                                    ▼
            ┌───────────────┐                   ┌───────────────┐                   ┌───────────────┐
            │  OpenAI API   │                   │ Local/Docker  │                   │ Route to LLM  │
            │  Anthropic    │                   │    Modal      │                   │   by model    │
            │  Gemini, etc. │                   │               │                   │               │
            └───────────────┘                   └───────────────┘                   └───────────────┘
```

## Multi-Backend Routing

The broker routes `llm_query()` calls by model name:

```python
# Root model (for main loop)
gpt4 = build_openai_adapter(model="gpt-4")

# Sub-model (for nested calls from code)
claude = build_anthropic_adapter(model="claude-3-opus")

rlm = create_rlm(gpt4, other_llms=[claude])

# Generated code can call:
# llm_query("...", model="claude-3-opus")
```

## Testing Architecture

The architecture enables comprehensive testing:

| Test Type | What's Tested | Dependencies |
|-----------|---------------|--------------|
| **Unit** | Domain logic, individual adapters | Fake ports only |
| **Integration** | Port boundaries, multi-component | Real adapters, fake externals |
| **E2E** | Full public API flows | Real adapters, optional Docker |

**Architecture Enforcement**:
- AST-based import scanning validates layer boundaries
- Domain layer has zero third-party imports
- Tests fail if architecture rules are violated

## Directory Structure

```
src/rlm/
├── __init__.py              # Public exports
├── _meta.py                 # Version info
├── cli.py                   # CLI entry point
├── __main__.py              # python -m rlm support
│
├── domain/                  # PURE BUSINESS LOGIC
│   ├── ports.py             # Core port protocols
│   ├── agent_ports.py       # Extension protocols
│   ├── errors.py            # Exception hierarchy
│   ├── types.py             # Type aliases
│   ├── models/              # Value objects
│   │   ├── completion.py
│   │   ├── llm_request.py
│   │   ├── iteration.py
│   │   └── ...
│   └── services/
│       ├── rlm_orchestrator.py
│       └── prompts.py
│
├── application/             # USE CASES
│   ├── config.py            # Configuration DTOs
│   └── use_cases/
│       └── run_completion.py
│
├── infrastructure/          # CROSS-CUTTING
│   ├── comms/               # Wire protocol
│   │   ├── messages.py
│   │   ├── codec.py
│   │   └── protocol.py
│   └── execution_namespace_policy.py
│
├── adapters/                # IMPLEMENTATIONS
│   ├── llm/
│   ├── environments/
│   ├── tools/
│   ├── policies/
│   ├── broker/
│   └── loggers/
│
└── api/                     # PUBLIC FACADE
    ├── rlm.py               # RLM class
    ├── factory.py           # Builders
    └── registries.py        # Config → adapter
```

## Adding New Components

### New LLM Provider

1. Create `src/rlm/adapters/llm/provider.py`
2. Implement `LLMPort` protocol
3. Add builder function
4. Register in `DefaultLLMRegistry` (api/registries.py)
5. Add optional dependency in `pyproject.toml`

### New Environment

1. Create `src/rlm/adapters/environments/env.py`
2. Implement `EnvironmentPort` protocol
3. Register in `DefaultEnvironmentRegistry`
4. Add environment name to `EnvironmentName` literal

### New Extension Protocol

1. Define protocol in `src/rlm/domain/agent_ports.py`
2. Add default implementation in `src/rlm/adapters/policies/`
3. Integrate into orchestrator
4. Export from `src/rlm/__init__.py`

## Design Decisions

See [Architecture Decision Records](adr/) for rationale:

- [ADR-0001: Hexagonal Modular Monolith](adr/0001-hexagonal-modular-monolith.md)
- [ADR-0002: Optional Provider Dependencies](adr/0002-optional-provider-dependencies.md)
- [ADR-0003: JSONL Log Schema Versioning](adr/0003-jsonl-log-schema-versioning.md)

## See Also

- [API Reference](api-reference.md) — Public API documentation
- [Configuration](configuration.md) — All configuration options
- [Extension Protocols](extending/extension-protocols.md) — Custom policies
- [Internals: Ports](internals/ports.md) — Port interface details
- [Internals: Protocol](internals/protocol.md) — Wire protocol specification
