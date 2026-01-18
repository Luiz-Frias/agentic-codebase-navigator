#!/usr/bin/env bash
# Verification script for ruff and basedpyright fixes

set -euo pipefail

echo "==================================="
echo "Verification Script for Linting Fixes"
echo "==================================="
echo ""

FILES=(
    "src/rlm/adapters/llm/openai.py"
    "src/rlm/cli.py"
)

echo "Files to verify:"
for file in "${FILES[@]}"; do
    echo "  - $file"
done
echo ""

# Run ruff check
echo "→ Running ruff check..."
echo ""
for file in "${FILES[@]}"; do
    echo "  Checking: $file"
    if ruff check "$file" --output-format=text; then
        echo "    ✓ No ruff errors"
    else
        echo "    ✗ Ruff errors found"
        exit 1
    fi
done
echo ""

# Run basedpyright check
echo "→ Running basedpyright check..."
echo ""
for file in "${FILES[@]}"; do
    echo "  Checking: $file"
    if basedpyright "$file" 2>&1 | tee /tmp/pyright_output.txt; then
        # Check if there are any reportAny errors
        if grep -q "reportAny" /tmp/pyright_output.txt; then
            echo "    ✗ reportAny errors found:"
            grep "reportAny" /tmp/pyright_output.txt
            exit 1
        else
            echo "    ✓ No basedpyright errors"
        fi
    else
        echo "    ✗ Basedpyright errors found"
        exit 1
    fi
done
echo ""

echo "==================================="
echo "✓ All checks passed!"
echo "==================================="
