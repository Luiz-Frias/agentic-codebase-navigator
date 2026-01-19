# Troubleshooting

This guide covers common issues, their symptoms, and solutions.

## Quick Reference

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Completion hangs during `llm_query_batched()` | Broker timeout | Reduce `broker_timeout_s` |
| Docker env never returns | `docker exec` stuck | Lower `subprocess_timeout_s` |
| Orphaned Docker containers | Partial cleanup | Run manual cleanup |
| "No tool calls" but expected tools | Malformed LLM response | Check provider compatibility |
| Policy stop not detected | Missing `policy_stop` flag | Update to v1.2.0+ |
| Import errors in local env | Restricted imports | Add to `allowed_import_roots` |
| Timeout not working (local) | SIGALRM limitations | Use Docker environment |

---

## Timeouts and Hangs

### Completion hangs during `llm_query_batched(...)`

**Symptoms**: Orchestration appears frozen during batched nested calls.

**Cause**: A batched subcall is stuck (provider call never returns) or broker tasks aren't making forward progress.

**Solutions**:

1. Reduce broker batched timeout:
   ```python
   from rlm.domain.policies.timeouts import BrokerTimeouts

   # Configure shorter timeouts
   timeouts = BrokerTimeouts(
       batch_completion_timeout_s=30.0,  # Default: 60.0
       cancellation_grace_s=5.0,         # Default: 10.0
   )
   ```

2. Enable debug logging to identify stuck calls:
   ```python
   import logging
   logging.getLogger("rlm").setLevel(logging.DEBUG)
   ```

### Docker env hangs / never returns

**Symptoms**: `execute_code()` never completes when using Docker environment.

**Cause**: `docker exec` is blocked or stuck inside the container.

**Solutions**:

1. Lower subprocess timeout:
   ```python
   rlm = create_rlm(
       llm,
       environment="docker",
       environment_kwargs={
           "subprocess_timeout_s": 60.0,  # Default: 120.0
       },
   )
   ```

2. Check Docker daemon health:
   ```bash
   docker info
   docker ps  # Should respond quickly
   ```

3. Kill stuck containers manually:
   ```bash
   docker ps -a | grep rlm | awk '{print $1}' | xargs docker rm -f
   ```

### Local execution timeout not working

**Symptoms**: Code runs indefinitely despite `execute_timeout_s` being set.

**Cause**: SIGALRM only works on Unix, in the main thread.

**Solutions**:

1. Use Docker environment for reliable timeouts:
   ```python
   rlm = create_rlm(llm, environment="docker")
   ```

2. Ensure you're running in the main thread (not from a thread pool)

3. On Windows, timeouts are not supported — use Docker

---

## Tool Calling Issues

### "No tool calls" when tools expected

**Symptoms**: LLM returns text without invoking registered tools.

**Possible Causes**:

1. **Tool schema mismatch**: LLM doesn't understand tool format
2. **Malformed response**: LLM returned tool calls in unexpected format
3. **Tool choice setting**: `tool_choice="none"` prevents tool calls

**Solutions**:

1. Check tool definitions are valid:
   ```python
   from rlm.adapters.tools import ToolRegistry

   registry = ToolRegistry()
   registry.register(my_tool)

   # Inspect generated schema
   print(registry.get_tool_definitions())
   ```

2. Force tool usage:
   ```python
   result = rlm.completion(prompt, tool_choice="required")
   ```

3. Check provider compatibility:
   ```python
   # Some providers have specific tool format requirements
   # OpenAI: function calling format
   # Anthropic: tool_use blocks
   # Gemini: function declarations
   ```

### Malformed tool call responses (v1.2.0+)

**Symptoms**: Error distinguishing malformed responses from missing tool calls.

**Background**: v1.2.0 introduced explicit detection of malformed LLM responses vs. intentionally empty tool calls.

**Check the error message**:
- `"No tool calls in response"` — LLM chose not to call tools (normal)
- `"Malformed tool call response"` — LLM returned invalid structure

**Solution for malformed responses**:
```python
# Enable SafeAccessor fallback for parsing
from rlm.domain.models.safe_accessor import SafeAccessor

response = llm_response.get("tool_calls", [])
if not response:
    # Check if it's truly empty or malformed
    accessor = SafeAccessor(llm_response)
    raw_calls = accessor["tool_calls"].unwrap_or([])
```

### Policy stop not being detected

**Symptoms**: Custom `StoppingPolicy` returns `True` but orchestration continues.

**Cause (pre-v1.2.0)**: Relied on string matching `"[Stopped by custom policy]"`.

**Solution**: Update to v1.2.0+ which uses explicit `policy_stop` flag:
```python
# v1.2.0+: Context has explicit flag
if ctx.policy_stop:
    # Policy requested stop
    pass
```

---

## Environment Issues

### Import errors in local environment

**Symptoms**: `ImportError` or `ModuleNotFoundError` when code tries to import packages.

**Cause**: Local environment restricts imports for security.

**Default allowed imports**:
- `collections`, `dataclasses`, `datetime`, `decimal`
- `functools`, `itertools`, `json`, `math`
- `pathlib`, `random`, `re`, `statistics`
- `string`, `textwrap`, `typing`, `uuid`

**Solution**: Add custom import roots:
```python
rlm = create_rlm(
    llm,
    environment="local",
    environment_kwargs={
        "allowed_import_roots": {"numpy", "pandas", "requests"},
    },
)
```

### Docker container not starting

**Symptoms**: Error during environment initialization.

**Diagnostic steps**:
```bash
# 1. Check Docker is running
docker info

# 2. Check image exists
docker images | grep python

# 3. Pull if needed
docker pull python:3.12-slim

# 4. Test container creation
docker run --rm python:3.12-slim python -c "print('ok')"
```

### Docker proxy connection errors

**Symptoms**: `llm_query()` fails inside container with connection refused.

**Cause**: Container can't reach host proxy server.

**Solutions**:

1. In CI environments, enable host networking:
   ```bash
   export RLM_DOCKER_USE_HOST_NETWORK=1
   ```

2. Check firewall isn't blocking container → host traffic

3. Verify proxy is listening:
   ```python
   # Debug: Check broker address
   print(f"Broker listening on: {broker.address}")
   ```

### Cleanup failures / orphaned containers

**Symptoms**: Docker containers remain after RLM exits.

**Cause**: Partial initialization, exceptions during cleanup, or transient Docker failures.

**Manual cleanup**:
```bash
# List RLM containers
docker ps -a | grep rlm

# Remove all RLM containers
docker ps -a | grep rlm | awk '{print $1}' | xargs docker rm -f

# Remove dangling volumes (if any)
docker volume prune -f
```

---

## Provider-Specific Issues

### OpenAI: Rate limiting

**Symptoms**: `RateLimitError` or 429 status codes.

**Solutions**:
1. Implement exponential backoff
2. Use `max_iterations` to limit API calls
3. Consider using a gateway like Portkey or LiteLLM

### Anthropic: Context length exceeded

**Symptoms**: Error about maximum context length.

**Solutions**:
1. Use `ContextCompressor` to reduce message history
2. Lower `max_iterations`
3. Implement custom truncation policy

### Gemini: Function calling format

**Symptoms**: Tools not recognized or malformed responses.

**Note**: Gemini uses a different function declaration format.

**Solution**: RLM handles format conversion automatically — ensure you're using `GeminiAdapter`.

### Azure OpenAI: Deployment not found

**Symptoms**: 404 errors or deployment name issues.

**Solution**:
```python
from rlm.adapters.llm import AzureOpenAIAdapter

adapter = AzureOpenAIAdapter(
    model="your-deployment-name",  # Not the model name!
    azure_endpoint="https://your-resource.openai.azure.com/",
)
```

---

## Debugging with Correlation IDs

Every `run_completion()` generates a **correlation ID** for end-to-end tracing.

### Where correlation IDs appear

| Location | Access |
|----------|--------|
| Run metadata | `LoggerPort.log_metadata(...)` |
| Iterations | `Iteration.correlation_id` |
| REPL results | `ReplResult.correlation_id` |
| Docker env | `RLM_CORRELATION_ID` env var |

### Using correlation IDs

```python
result = rlm.completion("My prompt")

# Get correlation ID from result
correlation_id = result.metadata.get("correlation_id")

# Search logs for this ID
grep -r "$correlation_id" ./logs/
```

### JSONL log analysis

```python
import json

with open("logs/run.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        if entry.get("correlation_id") == target_id:
            print(entry)
```

---

## Protocol and Wire Issues

### TCP protocol timeouts

When using TCP protocol client helpers:

```python
from rlm.infrastructure.comms.protocol import request_completion

result = request_completion(
    address=("localhost", 8000),
    request=my_request,
    timeout_s=30.0,  # Socket timeout
    max_message_bytes=10_000_000,  # Payload limit
)
```

### Large payload errors

**Symptoms**: Message rejected or truncated.

**Solution**: Increase `max_message_bytes` or compress payloads.

---

## State Machine Issues (v1.2.0+)

### Unexpected terminal state

**Symptoms**: State machine ends in unexpected state.

**Debug approach**:
```python
# Add logging to event source
def debug_event_source(state, ctx):
    print(f"State: {state}, Iteration: {ctx.iteration}")
    event = original_event_source(state, ctx)
    print(f"Event: {event}")
    return event
```

### Guard always returns False

**Symptoms**: Transition never fires.

**Check guard logic**:
```python
def my_guard(event, ctx):
    result = some_condition(ctx)
    print(f"Guard result: {result}, ctx: {ctx}")  # Debug
    return result
```

---

## Getting Help

### Information to include in bug reports

1. **RLM version**: `python -c "import rlm; print(rlm.__version__)"`
2. **Python version**: `python --version`
3. **Environment**: local, docker, modal
4. **LLM provider**: openai, anthropic, gemini, etc.
5. **Minimal reproduction code**
6. **Full error traceback**
7. **Correlation ID** (if available)

### Log collection

```bash
# Enable debug logging
export RLM_LOG_LEVEL=DEBUG

# Run with JSONL logging
rlm completion "test" --jsonl-log-dir ./debug-logs/

# Collect Docker logs
docker logs $(docker ps -q --filter "name=rlm") 2>&1 > docker.log
```

---

## See Also

- [Configuration](./configuration.md) — All configuration options
- [Execution Environments](./environments/execution-environments.md) — Environment details
- [Extension Protocols](./extending/extension-protocols.md) — Custom policies
- [State Machine Architecture](./internals/state-machine.md) — Orchestrator internals
