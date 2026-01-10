from __future__ import annotations

import ast
import inspect
import json
import textwrap

import pytest


@pytest.mark.unit
def test_old_json_loads_string_embedding_breaks_on_single_quotes() -> None:
    # This mirrors the legacy (buggy) pattern:
    #   f"import json; context = json.loads('{json.dumps(payload)}')"
    payload = {"name": "O'Brien"}
    code = f"import json; context = json.loads('{json.dumps(payload)}')"

    with pytest.raises(SyntaxError):
        ast.parse(code)


@pytest.mark.unit
def test_docker_repl_load_context_does_not_embed_json_in_string_literal() -> None:
    from rlm._legacy.environments.docker_repl import DockerREPL

    src = inspect.getsource(DockerREPL.load_context)
    tree = ast.parse(textwrap.dedent(src))

    # Regression guard: ensure we are not calling json.loads/json.dumps in the
    # function body (docstring text is ignored by AST walk below).
    forbidden: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id != "json":
            continue
        if node.func.attr in {"loads", "dumps"}:
            forbidden.append(node.func.attr)

    assert not forbidden, f"Unexpected json calls in DockerREPL.load_context: {forbidden}"

    # Ensure we load via mounted file path instead.
    assert "/workspace/context.json" in src
