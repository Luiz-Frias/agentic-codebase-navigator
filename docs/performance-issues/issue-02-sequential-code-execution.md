# perf: Sequential code block execution blocks parallelization

## Summary

Code blocks within an iteration are executed sequentially, even when they could potentially be parallelized.

## Location

`src/rlm/domain/services/rlm_orchestrator.py` lines 139-142 (sync) and 275-278 (async)

## Problem

```python
# Lines 139-142
for code in code_block_strs:
    repl_result = self.environment.execute_code(code)
    repl_result.correlation_id = correlation_id
    code_blocks.append(CodeBlock(code=code, result=repl_result))
```

When an LLM response contains multiple code blocks, they are executed one after another. This is intentional for state-dependent code, but unnecessarily slow for independent blocks.

## Impact

- **Latency**: Execution time scales linearly with number of code blocks
- **Throughput**: Cannot utilize parallel execution for independent computations
- **Resource Utilization**: Single-threaded execution underutilizes available resources

## Suggested Fixes

1. **Dependency Analysis**: Detect independent code blocks and execute in parallel
2. **Parallel Flag**: Allow LLM to mark blocks as parallelizable with a pragma
3. **Async Execution**: Use asyncio.gather() for blocks that don't share state
4. **Opt-in Parallelism**: Add configuration option to enable parallel block execution

## Considerations

- **State Dependencies**: Many code blocks depend on variables from previous blocks
- **Namespace Isolation**: Would need separate namespaces for parallel blocks
- **Error Handling**: How to handle partial failures in parallel execution
- **Determinism**: Sequential execution is more predictable

## Severity

**Critical** - Limits throughput for multi-block responses
