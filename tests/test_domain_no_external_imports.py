from __future__ import annotations

import ast
from pathlib import Path

import pytest


@pytest.mark.unit
def test_domain_layer_has_no_third_party_or_outer_layer_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    domain_root = repo_root / "src" / "rlm" / "domain"

    # We keep this allowlist intentionally small and focused on preventing
    # accidental coupling during the migration. It is OK for domain code to
    # import stdlib modules and other domain modules.
    forbidden_prefixes = (
        # Outer layers (hexagonal rule: dependencies point inward)
        "rlm.adapters",
        "rlm.infrastructure",
        "rlm.api",
        "rlm.application",
        # Transitional/legacy code should not leak into domain.
        "rlm._legacy",
        "references",
        # Known third-party deps (we want domain to stay dependency-free)
        "openai",
        "anthropic",
        "google",
        "google_genai",
        "portkey",
        "portkey_ai",
        "litellm",
        "modal",
        "dill",
    )

    offenders: list[str] = []
    for py_file in domain_root.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            match node:
                case ast.Import(names=names):
                    for alias in names:
                        if alias.name.startswith(forbidden_prefixes):
                            offenders.append(f"{py_file}: import {alias.name}")
                case ast.ImportFrom(module=module) if module is not None:
                    if module.startswith(forbidden_prefixes):
                        offenders.append(f"{py_file}: from {module} import ...")

    assert not offenders, "Domain layer imports forbidden modules:\n" + "\n".join(offenders)
