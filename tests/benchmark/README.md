# RLM Benchmark Suite

This directory contains pytest-benchmark tests that capture timing metrics for
RLM hotspots. It complements `tests/performance/`, which remains the pass/fail
regression guardrail suite.

## Running Benchmarks

```bash
# Run all benchmarks
pytest -m benchmark tests/benchmark/

# Run a specific benchmark file
pytest -m benchmark tests/benchmark/test_benchmark_parsing.py -v

# Emit benchmark JSON
pytest -m benchmark tests/benchmark/ --benchmark-json=.benchmarks/latest.json
```

## Guidelines

- Keep `tests/performance/` intact for deterministic pass/fail thresholds.
- Use the `benchmark` fixture (or `benchmark.pedantic`) to ensure timings are
  recorded.
- Favor correctness assertions, not timing thresholds, in benchmarks.
