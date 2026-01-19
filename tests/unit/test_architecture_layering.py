from __future__ import annotations

import ast
from pathlib import Path

import pytest


def _scan_forbidden_imports(
    root: Path,
    *,
    forbidden_prefixes: tuple[str, ...],
    forbidden_rlm_children: tuple[str, ...] = (),
    allowlist_files: set[Path] | None = None,
) -> list[str]:
    """
    Return a list of "file: import ..." strings for any forbidden imports under `root`.

    Notes:
    - We scan both `import x.y` and `from x import y` forms.
    - `from rlm import adapters` is handled via `forbidden_rlm_children`.

    """
    allowlist_files = allowlist_files or set()

    offenders: list[str] = []
    for py_file in root.rglob("*.py"):
        if py_file in allowlist_files:
            continue

        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            match node:
                case ast.Import(names=names):
                    for alias in names:
                        if alias.name.startswith(forbidden_prefixes):
                            offenders.append(f"{py_file}: import {alias.name}")
                case ast.ImportFrom(module=module, names=names) if module is not None:
                    if module.startswith(forbidden_prefixes):
                        offenders.append(f"{py_file}: from {module} import ...")
                        continue
                    if module == "rlm" and forbidden_rlm_children:
                        for alias in names:
                            if alias.name in forbidden_rlm_children:
                                offenders.append(f"{py_file}: from rlm import {alias.name}")
    return offenders


@pytest.mark.unit
def test_adapters_layer_does_not_depend_on_application_or_api() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    adapters_root = repo_root / "src" / "rlm" / "adapters"

    offenders = _scan_forbidden_imports(
        adapters_root,
        forbidden_prefixes=("rlm.api", "rlm.application", "references"),
        forbidden_rlm_children=("api", "application"),
    )
    assert not offenders, "Adapters layer imports forbidden modules:\n" + "\n".join(offenders)


@pytest.mark.unit
def test_application_layer_does_not_depend_on_adapters_or_infra_except_bridge() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    application_root = repo_root / "src" / "rlm" / "application"

    offenders = _scan_forbidden_imports(
        application_root,
        forbidden_prefixes=(
            "rlm.adapters",
            "rlm.infrastructure",
            "rlm.api",
            "references",
        ),
        forbidden_rlm_children=("adapters", "infrastructure", "api"),
    )
    assert not offenders, "Application layer imports forbidden modules:\n" + "\n".join(offenders)


@pytest.mark.unit
def test_infrastructure_layer_does_not_depend_on_api_application_or_adapters() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    infra_root = repo_root / "src" / "rlm" / "infrastructure"

    offenders = _scan_forbidden_imports(
        infra_root,
        forbidden_prefixes=(
            "rlm.api",
            "rlm.application",
            "rlm.adapters",
            "references",
        ),
        forbidden_rlm_children=("api", "application", "adapters"),
    )
    assert not offenders, "Infrastructure layer imports forbidden modules:\n" + "\n".join(offenders)


@pytest.mark.unit
def test_domain_layer_does_not_depend_on_outer_layers() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    domain_root = repo_root / "src" / "rlm" / "domain"

    offenders = _scan_forbidden_imports(
        domain_root,
        forbidden_prefixes=(
            "rlm.api",
            "rlm.application",
            "rlm.adapters",
            "rlm.infrastructure",
            "references",
        ),
        forbidden_rlm_children=("api", "application", "adapters", "infrastructure"),
    )
    assert not offenders, "Domain layer imports forbidden modules:\n" + "\n".join(offenders)
