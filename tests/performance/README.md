# RLM Performance Profiling Suite

This directory contains performance tests designed to profile, benchmark, and identify hotspots across the RLM package for speed, memory, and reliability tuning.

## Profiling Categories

### 1. Speed Benchmarks (`test_speed_*.py`)

Tests that measure execution time of critical paths:

- **Orchestrator Loop**: Message history growth, iteration overhead
- **Parsing Operations**: Code block extraction, final answer parsing
- **Serialization**: Value serialization for nested structures
- **Wire Protocol**: Codec framing, JSON encoding/decoding
- **Usage Tracking**: Lock contention, snapshot operations

### 2. Memory Profiling (`test_memory_*.py`)

Tests that identify memory hotspots:

- **Message Accumulation**: O(n) history growth per iteration
- **Recursive Serialization**: Stack depth and object creation
- **Namespace Snapshots**: Dictionary copying overhead
- **Large Payload Handling**: Context payload and REPL results

### 3. Reliability Stress Tests (`test_stress_*.py`)

Tests that validate system behavior under load:

- **Concurrent Batched Requests**: Barrier-synchronized parallel execution
- **High Iteration Counts**: Memory stability over many iterations
- **Error Recovery**: Graceful degradation under failure conditions
- **Timeout Behavior**: Cancellation and cleanup correctness

### 4. Live LLM Benchmarks (`test_live_llm_performance.py`)

Optional tests that hit real provider APIs (OpenAI, Anthropic):

- **Single Completion Latency**: Baseline LLM call timing
- **Usage Tracking Accuracy**: Verify token counting
- **Orchestrator Timing**: Full stack with real LLM
- **Async Performance**: Async completion benchmarks
- **Provider Comparison**: Side-by-side latency comparison

## Running Performance Tests

```bash
# Run all performance tests (excludes live LLM by default)
pytest -m performance tests/performance/

# Run with timing output
pytest -m performance tests/performance/ -v --durations=0

# Run with memory profiling (requires pytest-memray or similar)
pytest -m performance tests/performance/test_memory_*.py

# Run specific benchmark
pytest -m performance tests/performance/test_speed_orchestrator.py -v
```

### Running Live LLM Tests

Live LLM tests are **opt-in** to avoid API costs. Enable with environment variables:

```bash
# Enable live LLM tests
export RLM_RUN_LIVE_LLM_TESTS=1

# Set API keys for providers you want to test
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# Optional: specify models (defaults to cost-efficient models)
export OPENAI_MODEL=gpt-4o-mini
export ANTHROPIC_MODEL=claude-3-5-haiku-20241022

# Run live LLM performance tests
pytest -m "performance and live_llm" tests/performance/ -v -s

# Run only OpenAI tests
pytest -m "performance and live_llm" tests/performance/ -v -s -k openai

# Run only Anthropic tests
pytest -m "performance and live_llm" tests/performance/ -v -s -k anthropic
```

## Identified Hotspots (from codebase analysis)

### Critical (High Impact)

| Location | Issue | Impact |
|----------|-------|--------|
| `rlm_orchestrator.py:126-189` | Message history O(n) growth | Quadratic token usage |
| `rlm_orchestrator.py:139-142` | Sequential code block execution | Blocks parallelization |
| `serialization.py:7-22` | Recursive serialization (no depth limit) | Stack overflow risk |

### High (Significant Impact)

| Location | Issue | Impact |
|----------|-------|--------|
| `query_metadata.py:30-98` | Multiple json.dumps() for length measurement | Redundant serialization |
| `local.py:158` | Global process execution lock | Single-threaded exec |
| `parsing.py:21-22` | Regex with DOTALL on large text | CPU overhead |

### Medium (Moderate Impact)

| Location | Issue | Impact |
|----------|-------|--------|
| `jsonl.py:93-126` | Synchronous file I/O per entry | Stalls per iteration |
| `rlm_orchestrator.py:62` | sorted() in usage snapshots | O(n log n) per snapshot |
| `provider_base.py:188` | Thread lock on every usage record | Lock contention |
| `codec.py:22-26` | Byte-by-byte recv with extend() | Buffer reallocation |

## Performance Test Utilities

The `conftest.py` provides shared fixtures:

- `perf_timer`: Context manager for timing code blocks
- `memory_tracker`: Tracks memory allocations
- `large_context`: Generates large context payloads
- `deep_nested_data`: Creates deeply nested structures for serialization tests
- `many_code_blocks`: Generates responses with many code blocks

## Adding New Performance Tests

1. Add tests to appropriate file based on category (speed/memory/stress)
2. Use `@pytest.mark.performance` marker
3. Use provided fixtures for consistent benchmarking
4. Document expected performance characteristics
5. Include assertions for regression detection

## Interpreting Results

Performance tests should:

1. **Pass/Fail**: Validate performance doesn't regress beyond thresholds
2. **Timing Output**: Use `--durations=0` to see all test timings
3. **Memory Output**: Check peak memory usage for memory tests
4. **Concurrency**: Verify parallel execution via barrier tests (deadlock = sequential)
