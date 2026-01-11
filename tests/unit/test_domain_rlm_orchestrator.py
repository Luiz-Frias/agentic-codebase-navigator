from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from rlm.domain.models import ChatCompletion
from rlm.domain.models.llm_request import LLMRequest
from rlm.domain.models.repl import ReplResult
from rlm.domain.models.usage import ModelUsageSummary, UsageSummary
from rlm.domain.services.rlm_orchestrator import RLMOrchestrator
from tests.fakes_ports import CollectingLogger, QueueEnvironment, QueueLLM


class _FakeEnv:
    def __init__(self, repl_result: ReplResult) -> None:
        self._repl_result = repl_result
        self.loaded: list[Any] = []

    def load_context(self, context: Any) -> None:
        self.loaded.append(context)

    def execute_code(self, code: str) -> ReplResult:  # noqa: ARG002
        # Return a fresh result so tests don't accidentally share mutated objects.
        return ReplResult(
            correlation_id=self._repl_result.correlation_id,
            stdout=self._repl_result.stdout,
            stderr=self._repl_result.stderr,
            locals=dict(self._repl_result.locals),
            llm_calls=list(self._repl_result.llm_calls),
            execution_time=self._repl_result.execution_time,
        )


class _FakeLLM:
    def __init__(self, *, model_name: str, responses: list[str]) -> None:
        self.model_name = model_name
        self._responses = list(responses)

    def _next(self) -> str:
        assert self._responses, "test bug: no more fake responses"
        return self._responses.pop(0)

    def complete(self, req: LLMRequest) -> ChatCompletion:  # noqa: ARG002
        usage = UsageSummary(model_usage_summaries={self.model_name: ModelUsageSummary(1, 1, 1)})
        return ChatCompletion(
            root_model=self.model_name,
            prompt=req.prompt,
            response=self._next(),
            usage_summary=usage,
            execution_time=0.0,
        )

    async def acomplete(self, req: LLMRequest) -> ChatCompletion:  # noqa: ARG002
        usage = UsageSummary(model_usage_summaries={self.model_name: ModelUsageSummary(1, 1, 1)})
        return ChatCompletion(
            root_model=self.model_name,
            prompt=req.prompt,
            response=self._next(),
            usage_summary=usage,
            execution_time=0.0,
        )


@dataclass(slots=True)
class _CapturingLogger:
    iterations: list[Any]

    def log_iteration(self, iteration: Any) -> None:
        self.iterations.append(iteration)


@pytest.mark.unit
def test_orchestrator_includes_sub_llm_usage_in_iteration_and_cumulative_when_logger_present() -> (
    None
):
    sub_usage = UsageSummary(model_usage_summaries={"sub": ModelUsageSummary(1, 10, 20)})
    sub_cc = ChatCompletion(
        root_model="sub",
        prompt="p",
        response="r",
        usage_summary=sub_usage,
        execution_time=0.0,
    )
    env = _FakeEnv(ReplResult(llm_calls=[sub_cc]))
    logger = _CapturingLogger(iterations=[])
    llm = _FakeLLM(model_name="root", responses=["```repl\nprint('x')\n```\nFINAL(ok)"])

    orch = RLMOrchestrator(llm=llm, environment=env, logger=logger)
    cc = orch.completion("hi", correlation_id="cid", max_iterations=1)
    assert cc.response == "ok"
    assert len(logger.iterations) == 1

    it = logger.iterations[0]
    assert it.iteration_usage_summary is not None
    assert it.cumulative_usage_summary is not None
    assert set(it.iteration_usage_summary.model_usage_summaries) == {"root", "sub"}
    assert set(it.cumulative_usage_summary.model_usage_summaries) == {"root", "sub"}


@pytest.mark.unit
async def test_orchestrator_acompletion_falls_back_to_plain_llm_call_at_max_depth() -> None:
    env = _FakeEnv(ReplResult())
    llm = _FakeLLM(model_name="root", responses=["FINAL(ok)"])
    orch = RLMOrchestrator(llm=llm, environment=env, logger=None)

    cc = await orch.acompletion("hi", depth=1, max_depth=1)
    assert cc.response == "ok"


@pytest.mark.unit
async def test_orchestrator_acompletion_includes_sub_llm_usage_when_logger_present() -> None:
    sub_usage = UsageSummary(model_usage_summaries={"sub": ModelUsageSummary(1, 10, 20)})
    sub_cc = ChatCompletion(
        root_model="sub",
        prompt="p",
        response="r",
        usage_summary=sub_usage,
        execution_time=0.0,
    )
    env = _FakeEnv(ReplResult(llm_calls=[sub_cc]))
    logger = _CapturingLogger(iterations=[])
    llm = _FakeLLM(model_name="root", responses=["```repl\nprint('x')\n```\nFINAL(ok)"])

    orch = RLMOrchestrator(llm=llm, environment=env, logger=logger)
    cc = await orch.acompletion("hi", correlation_id="cid", max_iterations=1)
    assert cc.response == "ok"
    assert len(logger.iterations) == 1
    it = logger.iterations[0]
    assert it.iteration_usage_summary is not None
    assert set(it.iteration_usage_summary.model_usage_summaries) == {"root", "sub"}


@pytest.mark.unit
async def test_orchestrator_acompletion_out_of_iterations_final_prompt() -> None:
    env = _FakeEnv(ReplResult())
    llm = _FakeLLM(model_name="root", responses=["FINAL(last)"])
    orch = RLMOrchestrator(llm=llm, environment=env, logger=None)

    cc = await orch.acompletion("hi", max_iterations=0)
    assert cc.response == "last"


class _CapturingQueueLLM(QueueLLM):
    """QueueLLM variant that captures prompts for assertions."""

    def __init__(self, *, model_name: str = "mock", responses: list[str | Exception] | None = None):
        super().__init__(model_name=model_name, responses=responses)
        self.prompts: list[object] = []

    def complete(self, request: LLMRequest, /):
        self.prompts.append(request.prompt)
        return super().complete(request)

    async def acomplete(self, request: LLMRequest, /):
        # NOTE: QueueLLM.acomplete delegates to self.complete(), so we only capture there
        # to avoid double-counting prompts.
        return await super().acomplete(request)


def _prompt_contains(prompt: object, needle: str) -> bool:
    if isinstance(prompt, str):
        return needle in prompt
    if isinstance(prompt, dict):
        return needle in str(prompt.get("content", ""))
    if isinstance(prompt, list):
        for it in prompt:
            if isinstance(it, dict) and needle in str(it.get("content", "")):
                return True
            if needle in str(it):
                return True
    return needle in str(prompt)


@pytest.mark.unit
def test_orchestrator_sync_executes_code_and_carries_repl_output_into_next_prompt() -> None:
    env = QueueEnvironment(results=[ReplResult(stdout="HELLO\n")])
    llm = _CapturingQueueLLM(
        responses=[
            '```repl\nprint("HELLO")\n```\n',
            "FINAL(ok)",
        ]
    )
    logger = CollectingLogger()
    orch = RLMOrchestrator(llm=llm, environment=env, logger=logger)

    cc = orch.completion("ctx", max_iterations=3)

    assert cc.response == "ok"
    assert env.loaded_contexts == ["ctx"]
    assert env.executed_code == ['print("HELLO")']
    assert cc.usage_summary.model_usage_summaries["mock"].total_calls == 2
    assert len(logger.iterations) == 2
    assert logger.iterations[0].final_answer is None
    assert logger.iterations[1].final_answer == "ok"

    assert len(llm.prompts) == 2
    first_prompt, second_prompt = llm.prompts
    assert not _prompt_contains(first_prompt, "Code executed:")
    assert _prompt_contains(second_prompt, "Code executed:")
    assert _prompt_contains(second_prompt, "HELLO")
    assert _prompt_contains(second_prompt, '```repl\nprint("HELLO")\n```')


@pytest.mark.unit
async def test_orchestrator_async_executes_code_and_returns_final_answer() -> None:
    env = QueueEnvironment(results=[ReplResult(stdout="HELLO\n")])
    llm = _CapturingQueueLLM(
        responses=[
            '```repl\nprint("HELLO")\n```\n',
            "FINAL(ok)",
        ]
    )
    logger = CollectingLogger()
    orch = RLMOrchestrator(llm=llm, environment=env, logger=logger)

    cc = await orch.acompletion("ctx", max_iterations=3)

    assert cc.response == "ok"
    assert env.loaded_contexts == ["ctx"]
    assert env.executed_code == ['print("HELLO")']
    assert cc.usage_summary.model_usage_summaries["mock"].total_calls == 2
    assert len(logger.iterations) == 2
    assert logger.iterations[0].final_answer is None
    assert logger.iterations[1].final_answer == "ok"

    assert len(llm.prompts) == 2
    _first_prompt, second_prompt = llm.prompts
    assert _prompt_contains(second_prompt, "Code executed:")
    assert _prompt_contains(second_prompt, "HELLO")


@pytest.mark.unit
def test_orchestrator_max_depth_falls_back_to_plain_llm_call() -> None:
    env = QueueEnvironment()
    llm = _CapturingQueueLLM(responses=["FINAL(depth_ok)"])
    orch = RLMOrchestrator(llm=llm, environment=env)

    cc = orch.completion("ctx", depth=1, max_depth=1, max_iterations=3)

    assert cc.response == "depth_ok"
    assert env.loaded_contexts == []
    assert env.executed_code == []
    assert cc.usage_summary.model_usage_summaries["mock"].total_calls == 1
    assert llm.prompts == ["ctx"]


@pytest.mark.unit
def test_orchestrator_sync_handles_final_answer_without_code_blocks() -> None:
    env = QueueEnvironment()
    llm = _CapturingQueueLLM(responses=["FINAL(done)"])
    logger = CollectingLogger()
    orch = RLMOrchestrator(llm=llm, environment=env, logger=logger)

    cc = orch.completion("ctx", max_iterations=3)

    assert cc.response == "done"
    assert env.loaded_contexts == ["ctx"]
    assert env.executed_code == []
    assert len(logger.iterations) == 1
    assert logger.iterations[0].code_blocks == []
    assert logger.iterations[0].final_answer == "done"


@pytest.mark.unit
def test_orchestrator_sync_resolves_final_var_via_environment() -> None:
    # FINAL_VAR triggers an extra environment.execute_code() to resolve the variable.
    env = QueueEnvironment(results=[ReplResult(stdout="resolved\n")])
    llm = _CapturingQueueLLM(responses=["FINAL_VAR(answer)"])
    orch = RLMOrchestrator(llm=llm, environment=env)

    cc = orch.completion("ctx", max_iterations=3)

    assert cc.response == "resolved"
    assert env.loaded_contexts == ["ctx"]
    assert env.executed_code == ["print(FINAL_VAR('answer'))"]


@pytest.mark.unit
def test_orchestrator_sync_out_of_iterations_asks_for_final_answer() -> None:
    env = QueueEnvironment()
    llm = _CapturingQueueLLM(
        responses=[
            "no final here",
            "FINAL(done)",
        ]
    )
    logger = CollectingLogger()
    orch = RLMOrchestrator(llm=llm, environment=env, logger=logger)

    cc = orch.completion("ctx", max_iterations=1)

    assert cc.response == "done"
    assert env.loaded_contexts == ["ctx"]
    assert env.executed_code == []
    assert len(logger.iterations) == 1
    assert logger.iterations[0].final_answer is None
    assert llm.prompts and _prompt_contains(llm.prompts[-1], "Please provide a final answer")


@pytest.mark.unit
def test_orchestrator_sync_propagates_llm_errors() -> None:
    env = QueueEnvironment()
    llm = _CapturingQueueLLM(responses=[RuntimeError("llm_down")])
    orch = RLMOrchestrator(llm=llm, environment=env)

    with pytest.raises(RuntimeError, match="llm_down"):
        orch.completion("ctx", max_iterations=3)

    # Context load happens before the first LLM call in the sync path.
    assert env.loaded_contexts == ["ctx"]


@pytest.mark.unit
def test_orchestrator_sync_propagates_environment_execution_errors() -> None:
    env = QueueEnvironment(results=[ValueError("exec_failed")])
    llm = _CapturingQueueLLM(responses=['```repl\nprint("x")\n```\n'])
    orch = RLMOrchestrator(llm=llm, environment=env)

    with pytest.raises(ValueError, match="exec_failed"):
        orch.completion("ctx", max_iterations=3)

    assert env.loaded_contexts == ["ctx"]
    assert env.executed_code == ['print("x")']
