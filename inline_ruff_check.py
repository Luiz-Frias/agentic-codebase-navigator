#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

files = [
    "src/rlm/api/factory.py",
    "src/rlm/application/use_cases/run_completion.py",
    "src/rlm/application/config.py",
]

for f in files:
    print(f"\n{'=' * 80}\n{f}\n{'=' * 80}")
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
