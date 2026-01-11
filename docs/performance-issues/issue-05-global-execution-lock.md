# perf: Global process execution lock limits concurrency

## Summary

The local environment uses a process-global lock (`_PROCESS_EXEC_LOCK`) that serializes all code executions, preventing concurrent execution across multiple environments.

## Location

`src/rlm/adapters/environments/local.py` line 31 (definition) and line 158 (usage)

## Problem

```python
# Line 31
_PROCESS_EXEC_LOCK = threading.Lock()

# Lines 158-172
with _PROCESS_EXEC_LOCK:
    old_stdout, old_stderr = sys.stdout, sys.stderr
    old_cwd = os.getcwd()
    try:
        sys.stdout, sys.stderr = stdout_buf, stderr_buf
        os.chdir(self._session_dir)
        # ... execute code ...
    finally:
        os.chdir(old_cwd)
        sys.stdout, sys.stderr = old_stdout, old_stderr
```

The lock is necessary to protect global state (stdout/stderr, cwd) but it prevents any parallelism for local execution.

## Impact

- **Throughput**: Only one code block can execute at a time across all environments
- **Latency**: Parallel orchestrators must wait for each other
- **Scalability**: Cannot scale to multiple concurrent runs

## Suggested Fixes

1. **Per-Environment Isolation**: Use subprocess or threading with isolated I/O
2. **Contextvar for I/O**: Use contextvars for per-coroutine stdout/stderr capture
3. **Don't Redirect**: Use explicit output parameters instead of stdout capture
4. **Subprocess Execution**: Execute code in subprocess with separate I/O

## Example: Contextvar Approach

```python
import contextvars
_stdout_capture = contextvars.ContextVar('stdout_capture')

class CapturingWriter:
    def __init__(self):
        self.buffer = io.StringIO()
    def write(self, s):
        capture = _stdout_capture.get(None)
        if capture:
            capture.write(s)
        else:
            sys.__stdout__.write(s)
```

## Considerations

- **Global State**: Some user code may rely on sys.stdout
- **CWD**: Working directory is inherently process-global
- **Compatibility**: Docker/Modal environments don't have this limitation

## Severity

**High** - Limits concurrent local execution to single-threaded
