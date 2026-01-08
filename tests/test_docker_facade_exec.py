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
            parts: list[str] = []
            for it in items:
                if isinstance(it, dict) and "content" in it:
                    parts.append(str(it.get("content", "")))
                else:
                    parts.append(str(it))
            return "\n".join(parts)
        case _:
            return str(prompt)


class _DockerScriptedLLM:
    """LLM script that forces one REPL code execution, then FINALs."""

    # TODO(phase4/phase5): Add an opt-in variant that uses a real provider adapter so we
    # exercise actual network calls across the docker proxy boundary (skipped by default).

    def __init__(self) -> None:
        self.model_name = "dummy"
        self._calls = 0
        self._usage = UsageSummary(model_usage_summaries={"dummy": ModelUsageSummary(1, 0, 0)})

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        prompt = request.prompt
        self._calls += 1
        if self._calls == 1:
            # Produce stdout + stderr so we can verify both are captured.
            return ChatCompletion(
                root_model=request.model or self.model_name,
                prompt=prompt,
                response='```repl\nprint("HELLO_DOCKER")\n1/0\n```\n',
                usage_summary=self._usage,
                execution_time=0.0,
            )
        if self._calls == 2:
            text = _prompt_to_text(prompt)
            assert "Code executed:" in text
            assert "HELLO_DOCKER" in text
            assert "ZeroDivisionError" in text
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


@pytest.mark.integration
@pytest.mark.docker
def test_facade_runs_docker_env_and_captures_stdout_and_stderr() -> None:
    llm: LLMPort = _DockerScriptedLLM()

    try:
        rlm = create_rlm(
            llm,
            environment="docker",
            environment_kwargs={"image": "python:3.12-slim"},
            max_iterations=3,
            verbose=False,
        )
        cc = rlm.completion("hello")
        assert cc.response == "ok"
    except RuntimeError as e:
        # Docker can be present but image pulls may fail in restricted environments.
        if "Failed to start container" in str(e):
            pytest.skip(str(e))
        raise
