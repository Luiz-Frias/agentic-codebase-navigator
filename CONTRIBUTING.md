# Contributing to agentic-codebase-navigator

First off, thank you for considering a contribution to agentic-codebase-navigator! It's people like you that make RLM such a great tool.

This document provides guidelines and instructions for contributing. For detailed technical information, see [docs/contributing/](docs/contributing/).

## Code of Conduct

This project adheres to the Contributor Covenant [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to <contact.me@luizfrias.com>.

## Quick Start for Contributors

### 1. Fork & Clone

```bash
git clone https://github.com/YOUR-USERNAME/agentic-codebase-navigator.git
cd agentic-codebase-navigator
```

### 2. Setup Development Environment

```bash
# Install Python 3.12+
uv python install 3.12

# Create virtual environment
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install dependencies with dev + test groups
uv sync --group dev --group test
```

### 3. Make Your Changes

- **Architecture**: RLM uses hexagonal (ports & adapters) architecture. New code should follow this pattern.
- **Style**: Format with `ruff` and lint with `ruff check --fix`
- **Types**: All code in `src/rlm/domain/` must have full type annotations
- **Tests**: Add tests for any new functionality
- **Commits**: Follow [Conventional Commits](docs/contributing/commit-conventions.md)

### 4. Run Quality Checks

```bash
# Format
uv run ruff format src tests

# Lint
uv run ruff check src tests --fix

# Type check
uv run ty check src/rlm

# Tests (quick)
uv run pytest -m unit

# Tests (full)
uv run pytest

# Or use the quality gate recipe
just pc
```

### 5. Commit & Push

```bash
git add .
git commit -m "feat(adapters): add Bedrock LLM provider"
git push origin your-branch-name
```

### 6. Open a Pull Request

Include in your PR description:

- What problem does this solve?
- How did you test it?
- Any breaking changes?
- Related issues (#123)

## Types of Contributions

### 🐛 Bug Reports

Found a bug? Open an issue with:

- **RLM version**: `rlm --version`
- **Python version**: `python --version`
- **Environment**: local or docker?
- **LLM provider**: Which model?
- **Minimal reproduction**: Code that demonstrates the issue
- **Expected vs actual behavior**
- **Logs**: Any error messages or JSONL logs

[Open a bug report →](https://github.com/Luiz-Frias/agentic_codebase_navigator/issues/new?template=bug_report.yml)

### ✨ Feature Requests

Have an idea? Open an issue with:

- **Use case**: What problem does this solve?
- **Proposed solution**: How should this work?
- **Alternatives**: Other approaches you considered?
- **Impact**: Which layer would this change (domain/adapters/api)?

[Open a feature request →](https://github.com/Luiz-Frias/agentic_codebase_navigator/issues/new?template=feature_request.yml)

### 📚 Documentation

Good documentation is essential. You can contribute by:

- Fixing typos or unclear explanations
- Adding examples or tutorials
- Improving architecture documentation
- Translating docs to other languages

See [docs/](docs/) for the documentation structure.

### ♻️ Code Refactoring

RLM's architecture is carefully designed. Before proposing major refactors:

1. Open a discussion explaining the benefits
2. Consider how it affects the hexagonal layers
3. Ensure all tests still pass

### 🧪 Tests & Test Infrastructure

Good test coverage is critical for a code execution tool. Contributions might include:

- Adding unit tests for uncovered code
- Writing integration tests for cross-layer interactions
- E2E tests for user-facing workflows
- Performance benchmarks
- Docker environment tests

See [docs/testing/testing-guide.md](docs/testing/testing-guide.md) for the testing strategy.

## Architecture Guidelines

RLM follows a **hexagonal (ports & adapters) architecture**. When contributing:

### Know the Layers

```text
API (public facade, factories)
    ↓
Application (use cases, configuration)
    ↓
Domain (pure business logic, ports, zero dependencies) ← CORE
    ↓
Infrastructure (wire protocol, utilities)
    ↓
Adapters (LLM providers, environments, tools, etc.)
```

### Dependency Rules

- ✅ **Inward dependencies only**: Adapters → Application → Domain
- ✅ **Domain has zero external dependencies**
- ✅ **All integrations through ports (protocols)**
- ❌ **No circular dependencies**
- ❌ **No cross-layer shortcuts**

### Key Design Patterns

1. **Ports (Protocols)**: Define integration points

   ```python
   class LLMPort(Protocol):
       def complete(self, request: LLMRequest) -> ChatCompletion: ...
   ```

2. **Adapters**: Implement ports

   ```python
   class OpenAIAdapter:
       def complete(self, request: LLMRequest) -> ChatCompletion: ...
   ```

3. **Dependency Injection**: Pass adapters to core logic

   ```python
   orchestrator = RLMOrchestrator(llm=llm_adapter, env=env_adapter)
   ```

For more details, see [docs/architecture.md](docs/architecture.md).

## Testing Strategy

RLM uses **pytest markers** to organize tests:

| Marker        | Purpose                               | Speed | CI?                                    |
| ------------- | ------------------------------------- | ----- | -------------------------------------- |
| `unit`        | Fast, hermetic tests (mocks/fakes)    | ~1s   | ✅                                     |
| `integration` | Multi-component tests (real adapters) | ~10s  | ✅                                     |
| `e2e`         | Full workflow tests (public API)      | ~30s  | ✅                                     |
| `packaging`   | Build/install/import tests            | ~10s  | ✅                                     |
| `docker`      | Tests requiring Docker daemon         | ~20s  | ✅ (auto-skip if no Docker)            |
| `live_llm`    | Real LLM provider calls (opt-in)      | ~60s  | ❌ (requires RLM_RUN_LIVE_LLM_TESTS=1) |
| `performance` | Performance/load tests                | ~30s  | ❌ (opt-in)                            |
| `benchmark`   | Timing/throughput benchmarks          | ~30s  | ❌ (opt-in)                            |

### Writing Tests

**Unit Test** (use fakes):

```python
@pytest.mark.unit
def test_orchestrator_stops_on_max_iterations():
    llm = MockLLMAdapter(script=[...])
    orchestrator = RLMOrchestrator(llm=llm, max_iterations=5)
    result = orchestrator.run(...)
    assert result.iteration_count == 5
```

**Integration Test** (use real adapters):

```python
@pytest.mark.integration
def test_local_environment_executes_code():
    env = LocalEnvironmentAdapter()
    result = env.execute("print('hello')")
    assert "hello" in result.stdout
```

**E2E Test** (full API):

```python
@pytest.mark.e2e
def test_completion_with_openai():
    rlm = create_rlm(OpenAIAdapter(model="gpt-4o"))
    result = rlm.completion("What is 2+2?")
    assert "4" in result.response
```

### Run Tests

```bash
# Unit tests only (fast)
uv run pytest -m unit

# All tests except live_llm
uv run pytest -m "not live_llm"

# Live provider tests (requires API keys and RLM_RUN_LIVE_LLM_TESTS=1)
RLM_RUN_LIVE_LLM_TESTS=1 OPENAI_API_KEY=... uv run pytest -m "integration and live_llm"

# Watch mode (re-run on file change)
uv run ptw
```

## Code Quality Requirements

All contributions must pass:

### 1. Type Checking

```bash
# Check domain layer (strict)
uv run ty check src/rlm/domain

# Check all code (standard)
uv run basedpyright
```

### 2. Linting

```bash
# Auto-fix linting issues
uv run ruff check --fix src tests
```

### 3. Formatting

```bash
# Format all code
uv run ruff format src tests
```

### 4. Security Scanning

```bash
# Check for secrets
uv run pre-commit run gitleaks --all-files

# Security analysis
uv run bandit -r src/rlm
```

### 5. Test Coverage

Minimum coverage: **90%** across all modules

```bash
uv run pytest --cov=rlm --cov-report=html
# Open htmlcov/index.html to see coverage
```

### 6. Use the Quality Gate

```bash
# Run all checks at once
just pc
```

## Commit Message Convention

RLM uses **Conventional Commits** for clear, semantic commit history:

```text
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`

**Examples**:

```text
feat(adapters): add Bedrock LLM provider adapter

- Implement BedrockAdapter extending BaseLLMAdapter
- Support claude-3-opus model
- Add integration tests with mocked boto3

Implements #456
```

```text
fix(domain): handle timeout in code execution

Previously, execution timeouts could leave processes zombified.
Now we properly clean up resources in the signal handler.

Fixes #123
```

See [docs/contributing/commit-conventions.md](docs/contributing/commit-conventions.md) for full guidelines.

## Release Process

Once your PR is merged, the maintainers will:

1. Assign a semantic version bump (MAJOR.MINOR.PATCH)
2. Update CHANGELOG.md
3. Create a git tag
4. Build and publish to PyPI
5. Create a GitHub release

See [docs/contributing/releasing.md](docs/contributing/releasing.md) for details.

## Troubleshooting

### Pre-commit hooks failing?

```bash
# Install/update hooks
pre-commit install

# Run all hooks manually
pre-commit run --all-files
```

### Tests failing locally but passing in CI?

```bash
# Run with exact same configuration as CI
pytest -x -v  # Stop on first failure, verbose output
```

### Type checking errors?

```bash
# Check domain layer specifically
uv run ty check src/rlm/domain --pretty --show-error-codes
```

### Can't connect to Docker?

```bash
# Verify Docker daemon is running
docker info

# Run without Docker tests
pytest -m "not docker"
```

## Getting Help

- **Questions?** Open a [GitHub Discussion](https://github.com/Luiz-Frias/agentic_codebase_navigator/discussions)
- **Architecture confusion?** See [docs/architecture.md](docs/architecture.md)
- **Testing help?** See [docs/testing/testing-guide.md](docs/testing/testing-guide.md)
- **Development setup?** See [docs/contributing/development-setup.md](docs/contributing/development-setup.md)
- **IRC/Chat?** Not yet, but you can reach us on GitHub Discussions

## Recognition

Contributors are recognized in:

- Release notes (CHANGELOG.md)
- GitHub release pages
- Future: CONTRIBUTORS.md (coming soon)

Thank you for contributing! 🎉
