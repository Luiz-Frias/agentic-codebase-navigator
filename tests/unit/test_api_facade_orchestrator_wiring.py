from __future__ import annotations

import pytest

from rlm.api import create_rlm
from rlm.domain.models import ChatCompletion, ReplResult
from rlm.domain.ports import LLMPort
from tests.fakes_ports import QueueLLM


@pytest.mark.unit
def test_facade_uses_domain_orchestrator_and_can_run_with_fake_local_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_envs: list[_FakeLocalEnv] = []

    class _FakeLocalEnv:
        def __init__(self, *args, **kwargs) -> None:  # accepts ctor kwargs
            created_envs.append(self)
            self.loaded_contexts: list[object] = []
            self.executed_code: list[str] = []
            self.cleaned = False
            self._results: list[ReplResult] = [
                ReplResult(stdout="HELLO_FROM_FAKE\n", stderr="", locals={}, execution_time=0.0)
            ]

        def load_context(self, context_payload) -> None:
            self.loaded_contexts.append(context_payload)

        def execute_code(self, code: str) -> ReplResult:
            self.executed_code.append(code)
            assert self._results, "FakeLocalREPL: no scripted REPL results left"
            return self._results.pop(0)

        def cleanup(self) -> None:
            self.cleaned = True

    # Patch the local env constructor used in the default factory.
    import rlm.adapters.environments.local as local_env_mod

    monkeypatch.setattr(local_env_mod, "LocalEnvironmentAdapter", _FakeLocalEnv)

    llm: LLMPort = QueueLLM(
        model_name="mock",
        responses=[
            "```repl\nprint('HELLO_FROM_FAKE')\n```",
            "FINAL(ok)",
        ],
    )

    rlm = create_rlm(llm, environment="local", max_iterations=3, verbose=False)
    cc = rlm.completion("hello")

    assert isinstance(cc, ChatCompletion)
    assert cc.response == "ok"
    assert cc.root_model == "mock"
    assert cc.prompt == "hello"

    assert len(created_envs) == 1
    env = created_envs[0]
    assert env.loaded_contexts == ["hello"]
    assert env.executed_code == ["print('HELLO_FROM_FAKE')"]
    assert env.cleaned is True
