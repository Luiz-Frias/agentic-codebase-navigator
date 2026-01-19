#!/usr/bin/env python3
"""Run ruff check and display errors."""

import subprocess
import sys


def main():
    files = [
        "src/rlm/api/factory.py",
        "src/rlm/application/use_cases/run_completion.py",
        "src/rlm/application/config.py",
    ]

    for filepath in files:
        print(f"\n{'=' * 80}")
        print(f"FILE: {filepath}")
        print("=" * 80)

        try:
            result = subprocess.run(
                ["ruff", "check", filepath, "--output-format=text"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"STDERR: {result.stderr}", file=sys.stderr)
            if result.returncode == 0:
                print("✓ No errors found")

        except subprocess.TimeoutExpired:
            print(f"ERROR: Timeout checking {filepath}")
        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
