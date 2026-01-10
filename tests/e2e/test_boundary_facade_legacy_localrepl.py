from __future__ import annotations

import pytest

from rlm.api import create_rlm
from rlm.domain.models import ChatCompletion, LLMRequest, ModelUsageSummary, UsageSummary
from rlm.domain.ports import LLMPort
from rlm.domain.types import Prompt


def _prompt_to_text(prompt: Prompt) -> str:
    match prompt:
        case str() as s:
            return s
        case dict() as d:
            return str(d)
        case list() as items:
            # Most legacy calls pass OpenAI-style message dicts; join `content`.
            parts: list[str] = []
            for it in items:
                if isinstance(it, dict) and "content" in it:
                    parts.append(str(it.get("content", "")))
                else:
                    parts.append(str(it))
            return "\n".join(parts)
        case _:
            return str(prompt)


class _ScriptedLLM:
    """
    Boundary-test LLM (no network).

    - Call 1: returns a `repl` code block (no FINAL) so the legacy loop executes it.
    - Call 2: asserts the next prompt includes evidence of code execution, then FINALs.
    """

    # TODO(phase4/phase5): Add a parallel boundary test that uses a real provider adapter
    # (opt-in via env vars) to validate the same facade wiring without scripted responses.

    def __init__(self) -> None:
        self.model_name = "dummy"
        self._calls = 0
        self._usage = UsageSummary(model_usage_summaries={"dummy": ModelUsageSummary(1, 0, 0)})

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        prompt = request.prompt
        self._calls += 1
        if self._calls == 1:
            return ChatCompletion(
                root_model=request.model or self.model_name,
                prompt=prompt,
                response='Let\'s run code first.\n\n```repl\nprint("HELLO_FROM_REPL")\nx = 123\n```\n',
                usage_summary=self._usage,
                execution_time=0.0,
            )
        if self._calls == 2:
            text = _prompt_to_text(prompt)
            assert "Code executed:" in text
            assert "HELLO_FROM_REPL" in text
            return ChatCompletion(
                root_model=request.model or self.model_name,
                prompt=prompt,
                response="FINAL(ok)",
                usage_summary=self._usage,
                execution_time=0.0,
            )
        return ChatCompletion(
            root_model=request.model or self.model_name,
            prompt=prompt,
            response="FINAL(unexpected)",
            usage_summary=self._usage,
            execution_time=0.0,
        )

    async def acomplete(self, request: LLMRequest, /) -> ChatCompletion:
        return self.complete(request)

    def get_usage_summary(self):
        return self._usage

    def get_last_usage(self):
        return self._usage


@pytest.mark.e2e
def test_facade_boundary_runs_legacy_loop_and_executes_local_repl_code_block() -> None:
    llm: LLMPort = _ScriptedLLM()
    rlm = create_rlm(llm, environment="local", max_iterations=3, verbose=False)
    cc = rlm.completion("hello")
    assert isinstance(cc, ChatCompletion)
    assert cc.response == "ok"
    assert cc.root_model == "dummy"
    assert cc.prompt == "hello"
    # The legacy loop should require two LLM calls here:
    # 1) produce a repl code block (no FINAL)
    # 2) after execution evidence is appended, return FINAL(...)
    assert cc.usage_summary.model_usage_summaries["dummy"].total_calls == 2
