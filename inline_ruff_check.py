#!/usr/bin/env python3
import sys
from ruff.__main__ import find_ruff_bin
from pathlib import Path
import subprocess

files = [
    "src/rlm/api/factory.py",
    "src/rlm/application/use_cases/run_completion.py",
    "src/rlm/application/config.py",
]

for f in files:
    print(f"\n{'='*80}\n{f}\n{'='*80}")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", f, "--output-format=text"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
        )
        print(result.stdout if result.stdout else "✓ No errors")
        if result.stderr:
            print(f"STDERR: {result.stderr}")
    except Exception as e:
        print(f"Error: {e}")
