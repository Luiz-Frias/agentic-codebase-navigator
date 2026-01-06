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


class _DockerLLMQueryLLM:
    """LLM script that triggers an in-container `llm_query()` subcall."""

    def __init__(self) -> None:
        self.model_name = "dummy"
        self.root_calls = 0
        self.sub_calls = 0
        self._usage = UsageSummary(model_usage_summaries={"dummy": ModelUsageSummary(1, 0, 0)})

    def completion(self, prompt: Prompt, /, *, model: str | None = None) -> str:
        # Root calls are message lists; subcalls from the container are strings.
        if isinstance(prompt, str):
            self.sub_calls += 1
            return f"subcall:{prompt}"

        self.root_calls += 1
        if self.root_calls == 1:
            return "```repl\nresp = llm_query('ping')\nprint(resp)\n```\n"
        if self.root_calls == 2:
            text = _prompt_to_text(prompt)
            assert "subcall:ping" in text
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
def test_docker_env_llm_query_routes_to_llm_port() -> None:
    llm = _DockerLLMQueryLLM()
    llm_port: LLMPort = llm

    try:
        rlm = create_rlm(
            llm_port,
            environment="docker",
            environment_kwargs={"image": "python:3.12-slim"},
            max_iterations=3,
            verbose=False,
        )
        assert rlm.completion("hello") == "ok"
    except RuntimeError as e:
        if "Failed to start container" in str(e):
            pytest.skip(str(e))
        raise

    assert llm.sub_calls >= 1
