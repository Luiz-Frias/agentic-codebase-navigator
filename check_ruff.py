#!/usr/bin/env python3
"""Quick script to check ruff errors in specific files."""
import subprocess
import sys

files = [
    "src/rlm/domain/ports.py",
    "src/rlm/domain/agent_ports.py",
    "src/rlm/domain/goal2_ports.py",
    "src/rlm/domain/policies/__init__.py",
]

all_clean = True
for file in files:
    print(f"\n{'='*60}")
    print(f"Checking: {file}")
    print('='*60)
    result = subprocess.run(
        ["ruff", "check", file, "--output-format=text"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("✅ No errors found")
    else:
        print("❌ Errors found:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        all_clean = False

print(f"\n{'='*60}")
if all_clean:
    print("✅ All files passed ruff check!")
    sys.exit(0)
else:
    print("❌ Some files have errors")
    sys.exit(1)
