from __future__ import annotations

import pytest

from rlm._legacy.core.types import REPLResult
from rlm._legacy.utils.parsing import find_code_blocks, find_final_answer


@pytest.mark.unit
def test_find_code_blocks_multiple() -> None:
    text = """
Some text
```repl
print("a")
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
def test_find_final_answer_final() -> None:
    text = "noise\nFINAL(hello world)\nmore"
    assert find_final_answer(text) == "hello world"


@pytest.mark.unit
def test_find_final_answer_final_var_requires_env() -> None:
    assert find_final_answer("FINAL_VAR(answer)") is None


@pytest.mark.unit
def test_find_final_answer_final_var_executes_env() -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.last_code: str | None = None

        def execute_code(self, code: str) -> REPLResult:
            self.last_code = code
            return REPLResult(stdout="42\n", stderr="", locals={}, execution_time=0.0)

    env = FakeEnv()
    assert find_final_answer("FINAL_VAR(answer)", environment=env) == "42"
    assert env.last_code == "print(FINAL_VAR('answer'))"
