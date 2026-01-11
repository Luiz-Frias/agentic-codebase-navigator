from __future__ import annotations

import pytest

from rlm.api import create_rlm
from rlm.domain.models import (
    ChatCompletion,
    LLMRequest,
    ModelUsageSummary,
    UsageSummary,
)
from rlm.domain.ports import LLMPort
from rlm.domain.types import Prompt


def _prompt_to_text(prompt: Prompt) -> str:
    match prompt:
        case str() as s:
            return s
        case dict() as d:
            return str(d)
        case list() as items:
            parts: list[str] = []
            for it in items:
                if isinstance(it, dict) and "content" in it:
                    parts.append(str(it.get("content", "")))
                else:
                    parts.append(str(it))
            return "\n".join(parts)
        case _:
            return str(prompt)


class _DockerLLMQueryBatchedLLM:
    """LLM script that triggers an in-container `llm_query_batched()` subcall."""

    # TODO(phase4/phase5): Add a live-provider version that validates ordering against a
    # real adapter (opt-in) while keeping this scripted test as the deterministic baseline.

    def __init__(self) -> None:
        self.model_name = "dummy"
        self.root_calls = 0
        self.sub_calls = 0
        self._usage = UsageSummary(model_usage_summaries={"dummy": ModelUsageSummary(1, 0, 0)})

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        prompt = request.prompt
        # Root calls are message lists/dicts; subcalls from the container are strings.
        if isinstance(prompt, str):
            self.sub_calls += 1
            return ChatCompletion(
                root_model=request.model or self.model_name,
                prompt=prompt,
                response=f"sub:{prompt}",
                usage_summary=self._usage,
                execution_time=0.0,
            )

        self.root_calls += 1
        if self.root_calls == 1:
            return ChatCompletion(
                root_model=request.model or self.model_name,
                prompt=prompt,
                response=(
                    "```repl\n"
                    "prompts = ['a', 'b', 'c']\n"
                    "resps = llm_query_batched(prompts)\n"
                    "expected = [f'sub:{p}' for p in prompts]\n"
                    "print(resps)\n"
                    "print('ORDER_OK' if resps == expected else f'ORDER_BAD:{resps!r}')\n"
                    "```\n"
                ),
                usage_summary=self._usage,
                execution_time=0.0,
            )
        if self.root_calls == 2:
            text = _prompt_to_text(prompt)
            assert "ORDER_OK" in text, text
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
@pytest.mark.docker
def test_docker_env_llm_query_batched_preserves_order() -> None:
    llm = _DockerLLMQueryBatchedLLM()
    llm_port: LLMPort = llm

    try:
        rlm = create_rlm(
            llm_port,
            environment="docker",
            environment_kwargs={"image": "python:3.12-slim"},
            max_iterations=3,
            verbose=False,
        )
        cc = rlm.completion("hello")
        assert cc.response == "ok"
    except RuntimeError as e:
        if "Failed to start container" in str(e):
            pytest.skip(str(e))
        raise

    assert llm.sub_calls >= 3
