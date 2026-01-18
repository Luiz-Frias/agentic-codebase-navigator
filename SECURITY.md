# Security Policy

## Overview

This document describes the security model of agentic-codebase-navigator (RLM) and guidelines for reporting security vulnerabilities.

## Execution Security Model

### Important: RLM Executes Arbitrary Python Code

RLM enables Language Models to generate and execute Python code dynamically. This is a powerful capability that requires careful attention to security.

#### Local Environment (Default)

The **local environment** executes code in-process with the following protections:

**Isolation Mechanisms:**

- **Import Restrictions**: Only whitelisted stdlib modules can be imported
  - Default allowed: `collections`, `dataclasses`, `datetime`, `decimal`, `functools`, `itertools`, `json`, `math`, `pathlib`, `random`, `re`, `statistics`, `string`, `textwrap`, `typing`, `uuid`
  - Customizable via `allowed_import_roots` configuration
- **Execution Timeout**: Code execution is interrupted if it exceeds the timeout (default: 30 seconds, configurable)
  - Enforced via SIGALRM signal on Unix
- **Namespace Isolation**: Code runs in a controlled namespace, but has access to injected functions like `llm_query()`
- **No Filesystem Write Permission**: Relative paths only; no absolute path writes

**Limitations:**

- ⚠️ **Not suitable for untrusted code**: A sophisticated attacker could potentially escape the sandbox via signal handlers or namespace manipulation
- ✅ **Suitable for**: Internal tools, trusted LLM providers, development/testing

**Recommended for production**: Use Docker environment instead.

#### Docker Environment (Recommended for Production)

The **Docker environment** provides stronger isolation:

**Isolation Mechanisms:**

- **Process Isolation**: Code runs in a separate container
- **Filesystem Isolation**: Code cannot access the host filesystem
- **Network Isolation**: Code cannot make arbitrary network connections (LLM broker access via `host.docker.internal`)
- **Resource Limits**: CPU, memory, and timeout constraints enforced by the container runtime
- **Image Control**: Specify the exact Python image (default: `python:3.12-slim`)

**Configuration:**

```python
rlm = create_rlm(
    llm,
    environment="docker",
    environment_kwargs={
        "image": "python:3.12-slim",
        "subprocess_timeout_s": 120.0,
    },
)
```

**Requirements:**

- Docker daemon running (`docker info` succeeds)
- Docker 20.10+ (for `--add-host host.docker.internal:host-gateway`)

## LLM API Key Security

### Never Commit Secrets

All LLM providers use environment variables for API keys:

```python
# ✅ Good: Environment variables
llm = OpenAIAdapter(model="gpt-4o")  # Uses OPENAI_API_KEY from env

# ❌ Never: Hardcoded credentials
llm = OpenAIAdapter(model="gpt-4o", api_key="sk-...")  # DON'T DO THIS
```

### Safe Configuration

1. **Use `.env` files** (not tracked by git):

   ```bash
   # .env (add to .gitignore)
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-...
   ```

2. **Load with python-dotenv**:

   ```python
   from dotenv import load_dotenv
   load_dotenv()  # Reads .env
   rlm = create_rlm(OpenAIAdapter(model="gpt-4o"))
   ```

3. **Use environment variable managers** (recommended):
   - `.envrc` with `direnv` (auto-loads on directory change)
   - GitHub Actions Secrets (for CI/CD)
   - Cloud provider secret management (AWS Secrets Manager, Google Secret Manager, etc.)

### Key Rotation

If you suspect API key compromise:

1. Immediately rotate the key in your LLM provider's dashboard
2. Update environment variables
3. Audit API usage logs for unauthorized calls

## Broker Security

The internal TCP broker facilitates multi-backend LLM routing:

**Security Properties:**

- **Localhost-Only**: Broker binds to `127.0.0.1` by default (not exposed to network)
- **Message Size Limits**: Maximum payload 10MB (configurable, prevents memory exhaustion)
- **Request Correlation**: Correlation IDs prevent request/response confusion
- **Timeout Protection**: Long-running requests are terminated

**Docker Environment Specifics:**

- Broker runs in host process
- Docker container accesses via `host.docker.internal` (Docker for Mac/Windows) or host gateway
- Network access is read-only to broker port (no execution back-channel)

## Code Injection Prevention

### LLM-Generated Code Safety

Generated code runs in the same namespace as the RLM orchestrator. Mitigations:

1. **Import restrictions**: Dangerous modules are blocklisted
2. **Execution timeout**: Infinite loops are terminated
3. **Docker isolation**: Strong process boundary (recommended)

### User Input Handling

If you include user input in prompts:

```python
# ⚠️ Be careful with string interpolation
user_question = input("Enter your question: ")
result = rlm.completion(f"Answer this: {user_question}")  # Prompt injection risk

# ✅ Better: Use messages parameter with roles
result = rlm.completion(
    system="You are a helpful assistant.",
    user_input=user_question,  # Clear semantic boundary
)
```

## Dependency Security

### Supply Chain

- **Minimal dependencies**: Core package depends only on `openai>=2.14.0` and `loguru>=0.7.0`
- **Optional extras**: LLM providers are optional dependencies (only installed when needed)
- **Pre-commit scanning**: `gitleaks` and `detect-secrets` run on all commits
- **Dependency pinning**: `uv.lock` ensures reproducible builds

### Vulnerability Disclosure

RLM uses GitHub's Dependabot to monitor dependencies. If a vulnerability is discovered:

1. Dependabot creates a security alert
2. We assess the impact and severity
3. We release a patch version with the update
4. We announce in the CHANGELOG

## Vulnerability Reporting

### Private Disclosure

If you discover a security vulnerability:

**Do NOT open a public issue.** Instead:

1. **GitHub Security Advisory** (preferred):
   - Go to <https://github.com/Luiz-Frias/agentic-codebase-navigator/security/advisories>
   - Click "Report a vulnerability"
   - Fill in the details

2. **Email** (alternative):
   - Send to: <contact.me@luizfrias.com>
   - Subject: "[SECURITY] Vulnerability Report - RLM"
   - Include: description, impact, reproduction steps (if possible)

3. **Timeline**:
   - We will acknowledge receipt within 48 hours
   - We will assess and plan a fix within 7 days
   - We will release a patch and notify you before public disclosure
   - We will credit you in the release notes (unless you prefer anonymity)

### Public Disclosure

Once a fix is released, we will:

1. Publish a GitHub Security Advisory
2. Update the CHANGELOG
3. Release a new patch version
4. Credit the reporter

## Testing & Development

### Secure Development Practices

- **Pre-commit hooks**: Secrets detection via `gitleaks` and `detect-secrets`
- **Type checking**: `mypy` and `basedpyright` for type safety
- **Linting**: `ruff` for code quality
- **Security scanning**: `bandit` for security-specific issues
- **Test coverage**: 90%+ coverage requirement

### Running Tests Securely

```bash
# Run all tests
uv run pytest

# Run security-focused tests
uv run pytest -m unit -k security

# Run with coverage
uv run pytest --cov=rlm --cov-report=html
```

## Supported Versions

Security updates are provided for:

- **Current version** (1.1.0): All fixes
- **Previous minor version** (1.0.x): Critical fixes only
- **Older versions**: No support

## Best Practices for Users

### 1. Use Docker for Untrusted Code

```python
# For LLM-generated code, use Docker
rlm = create_rlm(
    llm,
    environment="docker",
)
```

### 2. Limit Import Roots

```python
# Restrict imports to essential modules only
rlm = create_rlm(
    llm,
    environment="local",
    environment_kwargs={
        "allowed_import_roots": {"json", "math", "re"},  # Minimal
    },
)
```

### 3. Timeout Protection

```python
# Set reasonable execution timeouts
rlm = create_rlm(
    llm,
    environment="local",
    environment_kwargs={
        "execute_timeout_s": 10.0,  # Short timeout
    },
)
```

### 4. Monitor Execution

```python
# Log all executions for audit trails
from rlm.adapters.logger import JsonlLoggerAdapter

rlm = create_rlm(
    llm,
    environment="docker",
    logger=JsonlLoggerAdapter(log_dir="./logs"),
)

# Later, review logs for suspicious patterns
```

### 5. Validate Model Responses

```python
# Don't blindly trust LLM-generated code
result = rlm.completion("Generate Python code to...")
# Review result.response before using in production
```

## Contact & Questions

For security-related questions or concerns:

- **Email**: <contact.me@luizfrias.com>
- **GitHub Discussions**: <https://github.com/Luiz-Frias/agentic-codebase-navigator/discussions>
- **Security Advisory**: <https://github.com/Luiz-Frias/agentic-codebase-navigator/security/advisories>

## Version History

| Date       | Version | Change                  |
| ---------- | ------- | ----------------------- |
| 2026-01-18 | 1.0     | Initial Security Policy |
