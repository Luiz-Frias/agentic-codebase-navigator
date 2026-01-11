# perf: Message history O(n) growth causes quadratic token usage

## Summary

The orchestrator's message history grows O(n) per iteration, leading to quadratic token usage over the course of a run.

## Location

`src/rlm/domain/services/rlm_orchestrator.py` lines 126-189 (sync) and 253-321 (async)

## Problem

```python
# Line 189 / 321
message_history.extend(format_iteration(iteration))
```

Each iteration appends previous iteration results to the message history, which is then sent to the LLM. For a run with N iterations:
- Iteration 1: sends ~1 message worth of context
- Iteration 2: sends ~2 messages worth of context
- Iteration N: sends ~N messages worth of context

Total tokens ≈ 1 + 2 + ... + N = O(N²)

## Impact

- **Token Cost**: Quadratic increase in API costs for long runs
- **Latency**: Each iteration slower than the last due to larger prompts
- **Context Window**: May hit context limits before hitting iteration limits

## Suggested Fixes

1. **Sliding Window**: Keep only last K iterations in history
2. **Summarization**: Periodically summarize older iterations
3. **Selective History**: Only include relevant previous iterations
4. **Compression**: Store execution results more compactly

## Benchmarks

See `tests/performance/test_speed_orchestrator.py::test_orchestrator_message_history_growth_linear`

## Severity

**Critical** - Affects cost and performance for all multi-iteration runs
