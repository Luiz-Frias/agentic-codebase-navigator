from __future__ import annotations

import pytest

from rlm._legacy.core.types import ModelUsageSummary, UsageSummary
from rlm.api import create_rlm
from rlm.domain.ports import LLMPort, Prompt


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

    def __init__(self) -> None:
        self.model_name = "dummy"
        self._calls = 0
        self._usage = UsageSummary(model_usage_summaries={"dummy": ModelUsageSummary(1, 0, 0)})

    def completion(self, prompt: Prompt, /, *, model: str | None = None) -> str:
        self._calls += 1
        if self._calls == 1:
            # Produce stdout + stderr so we can verify both are captured.
            return '```repl\nprint("HELLO_DOCKER")\n1/0\n```\n'
        if self._calls == 2:
            text = _prompt_to_text(prompt)
            assert "Code executed:" in text
            assert "HELLO_DOCKER" in text
            assert "ZeroDivisionError" in text
            return "FINAL(ok)"
        return "FINAL(unexpected)"

    async def acompletion(self, prompt: Prompt, /, *, model: str | None = None) -> str:
        return self.completion(prompt, model=model)

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
        assert rlm.completion("hello") == "ok"
    except RuntimeError as e:
        # Docker can be present but image pulls may fail in restricted environments.
        if "Failed to start container" in str(e):
            pytest.skip(str(e))
        raise
