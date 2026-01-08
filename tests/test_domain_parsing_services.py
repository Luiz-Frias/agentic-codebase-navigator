from __future__ import annotations

import pytest

from rlm.domain.models import Iteration, ReplResult
from rlm.domain.models.iteration import CodeBlock
from rlm.domain.services.parsing import (
    afind_final_answer,
    find_code_blocks,
    find_final_answer,
    format_execution_result,
    format_iteration,
)


@pytest.mark.unit
def test_find_code_blocks_multiple_and_ignores_non_repl_fences() -> None:
    text = """
Some text
```repl
print("a")
```

```python
print("should_not_run")
```

More text

```repl
x = 1
print(x)
```
"""
    blocks = find_code_blocks(text)
    assert blocks == ['print("a")', "x = 1\nprint(x)"]


@pytest.mark.unit
def test_find_code_blocks_ignores_malformed_fences() -> None:
    # Missing closing fence -> should be ignored.
    text = "before\n```repl\nprint('a')\n"
    assert find_code_blocks(text) == []

    # Missing newline before closing fence -> does not match our strict pattern.
    text2 = "```repl\nprint('a')```"
    assert find_code_blocks(text2) == []


@pytest.mark.unit
def test_find_final_answer_final() -> None:
    text = "noise\nFINAL(hello world)\nmore"
    assert find_final_answer(text) == "hello world"


@pytest.mark.unit
def test_find_final_answer_final_var_requires_env() -> None:
    assert find_final_answer("FINAL_VAR(answer)") is None


@pytest.mark.unit
def test_find_final_answer_final_var_executes_env() -> None:
    class _FakeEnv:
        def __init__(self) -> None:
            self.last_code: str | None = None

        def execute_code(self, code: str, /) -> ReplResult:
            self.last_code = code
            return ReplResult(stdout="42\n", stderr="", locals={}, execution_time=0.0)

    env = _FakeEnv()
    assert find_final_answer("FINAL_VAR(answer)", environment=env) == "42"
    assert env.last_code == "print(FINAL_VAR('answer'))"


@pytest.mark.unit
async def test_afind_final_answer_final_var_executes_env() -> None:
    class _FakeEnv:
        def __init__(self) -> None:
            self.last_code: str | None = None

        def execute_code(self, code: str, /) -> ReplResult:
            self.last_code = code
            return ReplResult(stdout="42\n", stderr="", locals={}, execution_time=0.0)

    env = _FakeEnv()
    assert await afind_final_answer("FINAL_VAR(answer)", environment=env) == "42"
    assert env.last_code == "print(FINAL_VAR('answer'))"


@pytest.mark.unit
def test_format_execution_result_includes_stdout_stderr_and_simple_locals() -> None:
    rr = ReplResult(
        stdout="out\n",
        stderr="err\n",
        locals={"x": 1, "_internal": 2, "__name__": "__main__"},
        execution_time=0.0,
    )
    formatted = format_execution_result(rr)
    assert "out" in formatted
    assert "err" in formatted
    assert "REPL variables:" in formatted
    assert "x" in formatted
    assert "_internal" not in formatted


@pytest.mark.unit
def test_format_iteration_truncates_long_repl_output() -> None:
    rr = ReplResult(stdout="x" * 50, stderr="", locals={}, execution_time=0.0)
    it = Iteration(
        prompt="p",
        response="r",
        code_blocks=[CodeBlock(code="print('hi')", result=rr)],
        final_answer=None,
        iteration_time=0.0,
    )

    msgs = format_iteration(it, max_character_length=10)
    # assistant + one user message for code execution
    assert len(msgs) == 2
    assert msgs[0]["role"] == "assistant"
    assert "Code executed:" in msgs[1]["content"]
    assert "... + [" in msgs[1]["content"]
