# Development Setup

This guide covers how to set up a local development environment and contribute to RLM.

## Prerequisites

- **Python 3.12+**
- **uv** (recommended) or pip
- **just** (task runner)
- **mise** (optional, for environment management)
- **Docker** (for container tests)
- **Git**

## Quick Setup

```bash
# Clone the repository
git clone https://github.com/Luiz-Frias/agentic-codebase-navigator.git
cd agentic-codebase-navigator

# Install Python 3.12 (if using uv)
uv python install 3.12

# Create virtual environment
uv venv --python 3.12 .venv
source .venv/bin/activate  # or: .venv\Scripts\activate on Windows

# Install dependencies
uv sync --group dev --group test

# Install pre-commit hooks
just pc-install

# Verify setup
just pc-all
```

## Using Mise (Recommended)

[Mise](https://mise.jdx.dev/) provides unified version and environment management:

```bash
# Install mise (macOS)
brew install mise

# Activate mise in your shell
eval "$(mise activate bash)"  # or zsh/fish

# Enter the project directory (mise auto-activates)
cd agentic-codebase-navigator

# mise.toml automatically provides:
# - Python 3.12
# - uv (latest)
# - just (latest)
# - Environment variables
```

### Mise Tasks

```bash
mise run check    # Linting + type checks
mise run test     # Run tests
mise run gate     # Full pre-commit + pre-push
mise run sync     # Sync dependencies
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout dev
git pull origin dev
git checkout -b feature/my-feature
```

### 2. Make Changes

Follow the [Architecture](../architecture.md) and coding conventions.

### 3. Run Quality Checks

```bash
# Fast feedback (staged files)
just pc

# Full local validation
just pc-all

# Before pushing (full gate)
just pc-full
```

### 4. Commit Changes

```bash
# Option 1: Manual commit with conventional format
git add .
git commit -m "feat(domain): add new stopping policy"

# Option 2: AI-assisted commit message
git add .
just commit-msg  # Generates message, copies to clipboard
just commit-ai   # Generates and commits interactively
```

### 5. Push and Create PR

```bash
git push origin feature/my-feature
gh pr create --base dev --title "feat: add new stopping policy"
```

## Quality Gates

### Pre-commit Hooks

Installed via `just pc-install`, these run automatically on commit:

| Stage | Checks |
|-------|--------|
| **Security** | gitleaks, detect-secrets |
| **Format** | ruff format |
| **Lint** | ruff check |
| **Types** | mypy, pyright |
| **Tests** | pytest unit |

### Pre-push Hooks

Run automatically on push:

| Stage | Checks |
|-------|--------|
| **Integration** | pytest integration |
| **E2E** | pytest e2e |
| **Benchmarks** | pytest benchmark |
| **Security** | semgrep, bandit |

### Manual Commands

```bash
# Format code
uv run --group dev ruff format .

# Lint code
uv run --group dev ruff check . --fix

# Type check
uv run --group dev mypy src/rlm/

# Run tests
uv run --group test pytest -m unit
uv run --group test pytest -m integration
uv run --group test pytest -m e2e

# Coverage
uv run --group test pytest -m unit --cov=rlm --cov-report=term-missing
```

## Just Recipes

The `justfile` provides convenient commands:

### Quality Gates

```bash
just pc           # Pre-commit on staged files
just pc-all       # Pre-commit on all files
just pc-push      # Pre-push hooks on all files
just pc-full      # Full gate (pre-commit + pre-push)
just pc-install   # Install all hooks
just pc-update    # Update hooks to latest
```

### Git Workflow

```bash
just commit-msg   # Generate AI commit message
just commit-ai    # Generate + commit interactively
```

## Testing

See [Testing Guide](../testing/testing-guide.md) for detailed testing instructions.

### Quick Commands

```bash
# Unit tests (fast)
pytest -m unit

# Integration tests
pytest -m integration

# E2E tests (may need Docker)
pytest -m e2e

# All tests with coverage
pytest --cov=rlm --cov-report=term-missing
```

### Live LLM Tests

```bash
export RLM_RUN_LIVE_LLM_TESTS=1
export OPENAI_API_KEY="sk-..."
pytest -m "integration and live_llm"
```

## Project Structure

```
agentic-codebase-navigator/
├── src/rlm/                 # Main package
│   ├── domain/              # Core business logic (no deps)
│   ├── application/         # Use cases, config
│   ├── infrastructure/      # Wire protocol, utilities
│   ├── adapters/            # LLM, environment, etc. implementations
│   └── api/                 # Public facade, factories
├── tests/                   # Test suite
│   ├── unit/                # Fast, hermetic tests
│   ├── integration/         # Multi-component tests
│   ├── e2e/                 # End-to-end tests
│   └── ...
├── docs/                    # Documentation
├── .cursor/                 # AI assistant rules
├── .github/                 # CI/CD workflows
├── justfile                 # Task automation
├── mise.toml                # Environment config
└── pyproject.toml           # Package config
```

## Code Style

### Python

- **Python 3.12+** features encouraged (pattern matching, type unions, etc.)
- **Ruff** for formatting (100 char lines) and linting
- **Type hints** required for public APIs
- **Docstrings** for public functions (Google style)

### Imports

```python
# Standard library
from __future__ import annotations
import os
from dataclasses import dataclass

# Third-party
import pytest

# Local
from rlm.domain.models import ChatCompletion
from rlm.domain.ports import LLMPort
```

### Architecture Rules

1. **Domain layer** has zero external dependencies
2. **Dependencies point inward** (adapters → application → domain)
3. **Ports (protocols)** define interfaces, adapters implement them
4. **Tests enforce** architecture via AST scanning

## Adding New Features

### New LLM Provider

1. Create adapter in `src/rlm/adapters/llm/`
2. Implement `LLMPort` protocol
3. Add builder function
4. Register in `DefaultLLMRegistry`
5. Add optional dependency in `pyproject.toml`
6. Add tests in `tests/unit/`
7. Document in `docs/providers/`

### New Environment

1. Create adapter in `src/rlm/adapters/environments/`
2. Implement `EnvironmentPort` protocol
3. Register in `DefaultEnvironmentRegistry`
4. Add tests
5. Document in `docs/environments/`

### New Extension Protocol

1. Define protocol in `src/rlm/domain/agent_ports.py`
2. Add default implementation in `src/rlm/adapters/policies/`
3. Integrate into orchestrator
4. Export from `src/rlm/__init__.py`
5. Add tests
6. Document in `docs/extending/`

## Troubleshooting

### "Command not found: just"

```bash
# macOS
brew install just

# Or via cargo
cargo install just
```

### "Pre-commit hooks not running"

```bash
# Reinstall hooks
just pc-install

# Or manually
pre-commit install --install-hooks
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push
```

### "Tests failing with import errors"

```bash
# Ensure dev dependencies are installed
uv sync --group dev --group test

# Or reinstall
uv pip install -e ".[dev,test]"
```

### "Docker tests skipping"

```bash
# Check Docker is running
docker info

# Start Docker if needed
# (varies by platform)
```

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/Luiz-Frias/agentic-codebase-navigator/issues)
- **Discussions**: GitHub Discussions
- **Code Review**: Submit a PR for feedback

## See Also

- [Commit Conventions](commit-conventions.md) — Commit message format
- [Releasing](releasing.md) — Release process
- [Testing Guide](../testing/testing-guide.md) — How to test
- [Architecture](../architecture.md) — System design
