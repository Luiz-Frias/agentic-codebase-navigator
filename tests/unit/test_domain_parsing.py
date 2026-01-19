from __future__ import annotations

import pytest

from rlm.domain.models.iteration import CodeBlock, Iteration
from rlm.domain.models.repl import ReplResult
from rlm.domain.services.parsing import (
    afind_final_answer,
    find_code_blocks,
    find_final_answer,
    format_iteration,
)


class _DummyEnv:
    def __init__(self, result: ReplResult) -> None:
        self._result = result

    def execute_code(self, code: str) -> ReplResult:
        return self._result


@pytest.mark.unit
def test_find_code_blocks_only_repl_blocks() -> None:
    text = "```python\nx=1\n```\n```repl\nprint('hi')\n```\n```repl\nx=2\n```"
    assert find_code_blocks(text) == ["print('hi')", "x=2"]


@pytest.mark.unit
def test_find_final_answer_handles_escaped_quotes_and_nested_parentheses() -> None:
    text = 'some text\nFINAL("a\\"b (c)")\nmore'
    assert find_final_answer(text) == '"a\\"b (c)"'


@pytest.mark.unit
def test_find_final_answer_unclosed_call_returns_none() -> None:
    assert find_final_answer("FINAL(") is None


@pytest.mark.unit
def test_find_final_answer_final_var_requires_environment() -> None:
    assert find_final_answer("FINAL_VAR('x')") is None


@pytest.mark.unit
def test_find_final_answer_final_var_uses_stderr_when_stdout_empty() -> None:
    env = _DummyEnv(ReplResult(stdout="", stderr="err"))
    assert find_final_answer("FINAL_VAR('x')", environment=env) == "err"


@pytest.mark.unit
async def test_afind_final_answer_final_var_uses_environment_and_stderr_when_stdout_empty() -> None:
    env = _DummyEnv(ReplResult(stdout="", stderr="err"))
    assert await afind_final_answer("FINAL_VAR('x')", environment=env) == "err"


@pytest.mark.unit
async def test_afind_final_answer_final_var_requires_environment() -> None:
    assert await afind_final_answer("FINAL_VAR('x')") is None


@pytest.mark.unit
async def test_afind_final_answer_handles_escaped_quotes_and_nested_parentheses() -> None:
    text = "FINAL('a\\'b (c)')"
    assert await afind_final_answer(text) == "'a\\'b (c)'"


@pytest.mark.unit
async def test_afind_final_answer_handles_double_quotes() -> None:
    assert await afind_final_answer('FINAL("x")') == '"x"'


@pytest.mark.unit
async def test_afind_final_answer_handles_nested_parentheses() -> None:
    assert await afind_final_answer("FINAL((1+2))") == "(1+2)"


@pytest.mark.unit
async def test_afind_final_answer_unclosed_call_returns_none() -> None:
    assert await afind_final_answer("FINAL((1+2)") is None


@pytest.mark.unit
def test_format_iteration_truncates_long_repl_output() -> None:
    long_out = "x" * 50
    it = Iteration(
        prompt="p",
        response="r",
        code_blocks=[CodeBlock(code="print('x')", result=ReplResult(stdout=long_out))],
    )
    msgs = format_iteration(it, max_character_length=10)
    assert len(msgs) == 2
    assert "... + [" in msgs[1]["content"]
    assert "chars...]" in msgs[1]["content"]
