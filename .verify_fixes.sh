#!/usr/bin/env bash
set -euo pipefail

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Verifying fixes for tools and logger adapters"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

FILES=(
  "src/rlm/adapters/tools/__init__.py"
  "src/rlm/adapters/tools/native.py"
  "src/rlm/adapters/tools/pydantic_output.py"
  "src/rlm/adapters/tools/registry.py"
  "src/rlm/adapters/logger/__init__.py"
  "src/rlm/adapters/logger/console.py"
  "src/rlm/adapters/logger/jsonl.py"
)

echo ""
echo "→ Running ruff check on target files..."
ruff check "${FILES[@]}" && echo "✓ Ruff passed" || {
  echo "❌ Ruff found issues"
  exit 1
}

echo ""
echo "→ Running basedpyright on target files..."
basedpyright "${FILES[@]}" && echo "✓ Basedpyright passed" || {
  echo "❌ Basedpyright found issues"
  exit 1
}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ All checks passed!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
