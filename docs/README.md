# RLM Documentation

Welcome to the documentation for **agentic-codebase-navigator** (`rlm`) — an agentic codebase navigator built on Recursive Language Model (RLM) patterns.

## Quick Links

| I want to... | Go to... |
|--------------|----------|
| Get started quickly | [Getting Started](getting-started.md) |
| Understand the architecture | [Architecture](architecture.md) |
| Learn the Python API | [API Reference](api-reference.md) |
| Use the CLI | [CLI Reference](cli.md) |
| Configure RLM | [Configuration](configuration.md) |
| Add a new LLM provider | [LLM Providers](providers/llm-providers.md) |
| Use different execution environments | [Environments](environments/execution-environments.md) |
| Extend with custom policies | [Extension Protocols](extending/extension-protocols.md) |
| Run tests | [Testing Guide](testing/testing-guide.md) |
| Contribute to the project | [Contributing](contributing/development-setup.md) |
| Release a new version | [Releasing](contributing/releasing.md) |
| Debug issues | [Troubleshooting](troubleshooting.md) |

## Documentation Structure

```
docs/
├── README.md                    # This file (navigation)
├── getting-started.md           # Quick start tutorial
├── architecture.md              # Hexagonal architecture overview
├── api-reference.md             # Python API documentation
├── cli.md                       # Command-line interface
├── configuration.md             # Configuration options
├── troubleshooting.md           # Common issues and solutions
│
├── providers/
│   └── llm-providers.md         # LLM provider setup (OpenAI, Anthropic, etc.)
│
├── environments/
│   └── execution-environments.md # Local, Docker, Modal environments
│
├── extending/
│   └── extension-protocols.md   # StoppingPolicy, ContextCompressor, NestedCallPolicy
│
├── testing/
│   └── testing-guide.md         # How to run and write tests
│
├── contributing/
│   ├── development-setup.md     # Local development workflow
│   ├── commit-conventions.md    # Conventional commits guide
│   └── releasing.md             # Release process
│
├── internals/
│   ├── ports.md                 # Domain port interfaces
│   ├── protocol.md              # Wire protocol (TCP broker)
│   ├── logging.md               # Logging system
│   └── log-schema-v1.md         # JSONL log schema
│
└── adr/                         # Architecture Decision Records
    ├── 0001-hexagonal-modular-monolith.md
    ├── 0002-optional-provider-dependencies.md
    └── 0003-jsonl-log-schema-versioning.md
```

## Core Concepts

### What is RLM?

RLM (Recursive Language Model) is a pattern where an LLM iteratively:
1. Generates Python code in ` ```repl ` blocks
2. Executes the code in a sandboxed environment
3. Inspects results and refines its approach
4. Repeats until reaching a final answer via `FINAL(...)` or `FINAL_VAR(...)`

### Agent Modes

RLM supports two agent modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Code** (default) | LLM generates Python code for execution | Complex computations, data analysis |
| **Tools** | LLM calls registered functions via function calling | API integrations, structured workflows |

### Architecture

RLM uses a **hexagonal (ports & adapters) architecture**:

```
┌─────────────────────────────────────────────────────────┐
│                      API Layer                          │
│              (RLM, create_rlm, factories)               │
├─────────────────────────────────────────────────────────┤
│                  Application Layer                      │
│            (use cases, configuration)                   │
├─────────────────────────────────────────────────────────┤
│                    Domain Layer                         │
│    (orchestrator, ports, models — no dependencies)      │
├─────────────────────────────────────────────────────────┤
│                   Adapters Layer                        │
│  (LLM providers, environments, tools, broker, logger)   │
├─────────────────────────────────────────────────────────┤
│                 Infrastructure Layer                    │
│          (wire protocol, comms, execution policy)       │
└─────────────────────────────────────────────────────────┘
```

## Version

Current version: **1.1.0**

See [CHANGELOG.md](../CHANGELOG.md) for release history.

## License

MIT License — see [LICENSE](../LICENSE) for details.
