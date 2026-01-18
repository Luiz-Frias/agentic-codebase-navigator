#!/usr/bin/env python3
"""Temporary script to check ruff errors."""
import subprocess
import sys

files = [
    "src/rlm/api/factory.py",
    "src/rlm/application/use_cases/run_completion.py",
    "src/rlm/application/config.py",
]

for file in files:
    print(f"\n{'='*80}")
    print(f"Checking {file}")
    print('='*80)
    result = subprocess.run(
        ["ruff", "check", file, "--output-format=text"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr, file=sys.stderr)
