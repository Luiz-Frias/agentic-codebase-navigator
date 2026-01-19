# Migration Guide: 1.1.0 → 1.2.0

This guide covers breaking changes and migration steps when upgrading from RLM 1.1.0 to 1.2.0.

## Quick Summary

| Change | Impact | Action Required |
|--------|--------|-----------------|
| Local env → subprocess | Medium | Review shared namespace usage |
| `StoppingPolicy` detection | Low | Replace string matching with `policy_stop` |
| State machine internals | None | No action (internal refactor) |

---

## Local Environment: Subprocess Workers

### What Changed

**Before (1.1.0)**: Code executed in-process with a persistent shared namespace.

```python
# 1.1.0: In-process execution
# Variables persisted between execute_code() calls
env.execute_code("x = 42")
result = env.execute_code("print(x)")  # Worked: x still in namespace
```

**After (1.2.0)**: Code executes in isolated subprocess workers with IPC communication.

```python
# 1.2.0: Subprocess execution
# Each execution runs in a fresh subprocess
env.execute_code("x = 42")
result = env.execute_code("print(x)")  # NameError: x not defined
```

### Why This Changed

1. **Reliability**: Process isolation prevents runaway code from affecting the parent
2. **Timeout Behavior**: Process kill works reliably across all platforms (not just Unix main thread)
3. **Memory Safety**: Each execution gets a clean memory space

### Migration Steps

#### If you relied on shared namespace state:

**Option 1**: Use `setup_code` to pre-initialize each execution:

```python
# Configure shared state via setup_code
rlm = create_rlm(
    llm,
    environment="local",
    environment_kwargs={
        "setup_code": "import pandas as pd\nx = 42",  # Runs before each execution
    },
)
```

**Option 2**: Use `load_context()` for dynamic state:

```python
# Pass context that's loaded into each worker
env.load_context({"x": 42, "data": my_dataframe})
result = env.execute_code("print(x)")  # Works: x loaded from context
```

**Option 3**: Switch to Docker environment for persistent sessions:

```python
# Docker maintains state within the container lifecycle
rlm = create_rlm(
    llm,
    environment="docker",
    environment_kwargs={"image": "python:3.12-slim"},
)
```

#### Timeout Behavior Changes

The new subprocess model improves timeout reliability:

| Scenario | 1.1.0 Behavior | 1.2.0 Behavior |
|----------|---------------|----------------|
| Unix main thread | SIGALRM | Process kill (faster) |
| Unix non-main thread | No timeout | Process kill ✓ |
| Windows | No timeout | Process kill ✓ |
| macOS | SIGALRM | Process kill (more reliable) |

If you were handling timeout exceptions, the error message format has changed:

```python
# 1.1.0
# stderr: "TimeoutError: Code execution timed out after 30.0 seconds"

# 1.2.0
# stderr: "TimeoutError: Execution timed out after 30.0s"
```

---

## StoppingPolicy Detection

### What Changed

**Before (1.1.0)**: Policy stop detected via string matching in the response.

```python
# 1.1.0: String matching (fragile)
if "[Stopped by custom policy]" in result.last_response:
    print("Policy stopped execution")
```

**After (1.2.0)**: Explicit `policy_stop` flag in context.

```python
# 1.2.0: Explicit flag (reliable)
if result.context.policy_stop:
    print("Policy stopped execution")
```

### Why This Changed

String matching was fragile — LLMs could accidentally include the marker string in their responses, or internationalization could break detection.

### Migration Steps

Update any code that checks for policy stops:

```python
# Before (1.1.0)
def check_result(result):
    if "[Stopped by custom policy]" in str(result):
        return "stopped_by_policy"
    return "completed"

# After (1.2.0)
def check_result(result):
    if result.context.policy_stop:
        return "stopped_by_policy"
    return "completed"
```

If you have custom `StoppingPolicy` implementations, no changes are needed — the framework now sets the flag automatically when your policy returns `True`.

---

## State Machine Internals

### What Changed

The orchestrator's internal control flow was rewritten from nested `while True` loops to a declarative state machine pattern.

### Impact

**None for most users**. This is an internal refactor that:

- Reduces cyclomatic complexity (C901 violations eliminated)
- Makes control flow explicit and testable
- Enables future extensions (custom states, transitions)

### If You Extended Orchestrator Internals

If you subclassed or monkey-patched orchestrator internals, review the new architecture:

```python
# Old pattern (1.1.0)
class MyOrchestrator(RLMOrchestrator):
    def _run_code_mode(self, ...):
        while True:  # Nested loop
            ...

# New pattern (1.2.0)
# Orchestrator uses StateMachine[S, E, C]
# Extend via event sources, guards, or custom machines
from rlm.domain.models.state_machine import StateMachine
from rlm.domain.models.orchestration_types import CodeModeState, CodeModeEvent

def my_custom_event_source(state: CodeModeState, ctx: CodeModeContext):
    # Custom logic here
    ...
```

See [State Machine Architecture](./internals/state-machine.md) for details.

---

## New Optional Dependencies

### Pydantic Integration

v1.2.0 adds optional Pydantic support for enhanced type validation:

```bash
# Install with Pydantic support
pip install "agentic-codebase-navigator[pydantic]"
```

This is **optional** — RLM works without Pydantic using the manual schema implementation.

### When to Use Pydantic

| Use Case | Recommendation |
|----------|---------------|
| Simple tools with basic types | Manual (default) |
| Complex nested structures | Pydantic |
| Recursive types (e.g., tree nodes) | Pydantic |
| Strict schema validation needed | Pydantic |
| Minimal dependencies preferred | Manual |

See [Pydantic Integration](./extending/pydantic-integration.md) for details.

---

## Configuration Changes

### New Local Environment Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `execute_timeout_cap_s` | `float` | `300.0` | Maximum allowed timeout cap |
| `broker_timeout_s` | `float` | `30.0` | Timeout for `llm_query()` IPC calls |

### Example: Updated Configuration

```python
# 1.1.0 configuration
config = RLMConfig(
    llm=LLMConfig(backend="openai", model_name="gpt-4"),
    env=EnvironmentConfig(
        environment="local",
        environment_kwargs={
            "execute_timeout_s": 30.0,
        }
    ),
)

# 1.2.0 configuration (with new options)
config = RLMConfig(
    llm=LLMConfig(backend="openai", model_name="gpt-4"),
    env=EnvironmentConfig(
        environment="local",
        environment_kwargs={
            "execute_timeout_s": 30.0,
            "execute_timeout_cap_s": 300.0,  # New: max timeout cap
            "broker_timeout_s": 30.0,         # New: IPC timeout
            "setup_code": "import pandas",    # Pre-execution setup
        }
    ),
)
```

---

## Troubleshooting Migration Issues

### "NameError: name 'x' is not defined"

**Cause**: Code relies on shared namespace state between executions.

**Solution**: Use `setup_code` or `load_context()`. See [Subprocess Workers](#local-environment-subprocess-workers).

### Timeout behavior changed

**Cause**: v1.2.0 uses process kill instead of SIGALRM.

**Solution**: This is expected — timeouts are now more reliable. Update error message parsing if needed.

### "policy_stop not found"

**Cause**: Accessing `policy_stop` on pre-1.2.0 context object.

**Solution**: Ensure you're using v1.2.0+ and access via `result.context.policy_stop`.

### Import errors for Pydantic

**Cause**: Code assumes Pydantic is installed.

**Solution**: Either install with `pip install "agentic-codebase-navigator[pydantic]"` or check availability:

```python
from rlm.domain.models.json_schema_mapper import has_pydantic

if has_pydantic():
    # Use Pydantic features
    pass
else:
    # Fall back to manual implementation
    pass
```

---

## Getting Help

If you encounter migration issues not covered here:

1. Check [Troubleshooting](./troubleshooting.md) for common issues
2. Review [Execution Environments](./environments/execution-environments.md) for environment-specific behavior
3. Open an issue with:
   - Your v1.1.0 code
   - The error you're seeing in v1.2.0
   - Your environment (OS, Python version, Docker availability)

---

## See Also

- [Execution Environments](./environments/execution-environments.md) — Full environment documentation
- [State Machine Architecture](./internals/state-machine.md) — Internal orchestration details
- [Pydantic Integration](./extending/pydantic-integration.md) — Optional Pydantic features
- [Troubleshooting](./troubleshooting.md) — Common issues and solutions
