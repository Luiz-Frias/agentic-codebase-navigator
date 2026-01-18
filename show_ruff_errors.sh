#!/usr/bin/env bash
set -euo pipefail

echo "=== Factory.py ==="
ruff check src/rlm/api/factory.py --output-format=text 2>&1 || true

echo ""
echo "=== run_completion.py ==="
ruff check src/rlm/application/use_cases/run_completion.py --output-format=text 2>&1 || true

echo ""
echo "=== config.py ==="
ruff check src/rlm/application/config.py --output-format=text 2>&1 || true
