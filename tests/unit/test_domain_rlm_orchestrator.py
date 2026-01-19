from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pytest

from rlm.adapters.tools.registry import InMemoryToolRegistry
from rlm.domain.agent_ports import ToolCallRequest
from rlm.domain.errors import ToolNotFoundError
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

    def execute_code(self, code: str) -> ReplResult:
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

    def complete(self, req: LLMRequest) -> ChatCompletion:
        usage = UsageSummary(model_usage_summaries={self.model_name: ModelUsageSummary(1, 1, 1)})
        return ChatCompletion(
            root_model=self.model_name,
            prompt=req.prompt,
            response=self._next(),
            usage_summary=usage,
            execution_time=0.0,
        )

    async def acomplete(self, req: LLMRequest) -> ChatCompletion:
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

    def __init__(
        self,
        *,
        model_name: str = "mock",
        responses: list[str | Exception] | None = None,
    ):
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
        ],
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
        ],
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
        ],
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


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL CALLING MODE TESTS (Phase 2.4)
# ═══════════════════════════════════════════════════════════════════════════════


class _ToolQueueLLM:
    """LLM fake that can return tool_calls in responses.

    Script items can be:
    - str: text response (no tool calls)
    - dict: {"tool_calls": [...], "response": "...", "finish_reason": "..."}
    """

    def __init__(
        self,
        *,
        model_name: str = "tool-mock",
        script: list[str | dict[str, Any]] | None = None,
    ):
        self._model_name = model_name
        self._script: list[str | dict[str, Any]] = list(script or [])
        self._calls: list[LLMRequest] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supports_tools(self) -> bool:
        return True

    def _pop_next(self) -> tuple[str, list[ToolCallRequest] | None, str | None]:
        """Pop next script item and return (response, tool_calls, finish_reason)."""
        if not self._script:
            raise AssertionError("_ToolQueueLLM: no scripted responses left")
        item = self._script.pop(0)

        if isinstance(item, str):
            return item, None, "stop"

        tool_calls = item.get("tool_calls")
        response = str(item.get("response", ""))
        finish_reason = item.get("finish_reason")
        if finish_reason is None:
            finish_reason = "tool_calls" if tool_calls else "stop"
        return response, tool_calls, finish_reason

    def complete(self, request: LLMRequest, /) -> ChatCompletion:
        self._calls.append(request)
        response_text, tool_calls, finish_reason = self._pop_next()
        usage = UsageSummary(
            model_usage_summaries={
                self._model_name: ModelUsageSummary(
                    total_calls=1,
                    total_input_tokens=10,
                    total_output_tokens=10,
                ),
            },
        )
        return ChatCompletion(
            root_model=self._model_name,
            prompt=request.prompt,
            response=response_text,
            usage_summary=usage,
            execution_time=0.01,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    async def acomplete(self, request: LLMRequest, /) -> ChatCompletion:
        return self.complete(request)

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(
            model_usage_summaries={
                self._model_name: ModelUsageSummary(
                    total_calls=len(self._calls),
                    total_input_tokens=10 * len(self._calls),
                    total_output_tokens=10 * len(self._calls),
                ),
            },
        )

    def get_last_usage(self) -> UsageSummary:
        return UsageSummary(
            model_usage_summaries={
                self._model_name: ModelUsageSummary(
                    total_calls=1,
                    total_input_tokens=10,
                    total_output_tokens=10,
                ),
            },
        )


def _make_tool_call(tool_id: str, name: str, arguments: dict[str, Any]) -> ToolCallRequest:
    """Helper to create a ToolCallRequest."""
    return ToolCallRequest(id=tool_id, name=name, arguments=arguments)


def _extract_tool_messages(prompt: object) -> list[dict[str, Any]]:
    if isinstance(prompt, list):
        return [m for m in prompt if isinstance(m, dict) and m.get("role") == "tool"]
    return []


def simple_add(a: int, b: int) -> int:
    """Simple add function for testing."""
    return a + b


def simple_multiply(x: int, y: int) -> int:
    """Simple multiply function for testing."""
    return x * y


def failing_tool(msg: str) -> None:
    """Tool that always raises an error."""
    raise ValueError(f"Tool error: {msg}")


@pytest.mark.unit
def test_orchestrator_tool_mode_requires_registry() -> None:
    """Tool mode raises ValueError if tool_registry is not provided."""
    env = QueueEnvironment()
    llm = _ToolQueueLLM(script=["ignored"])

    orch = RLMOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        environment=env,
        agent_mode="tools",
        tool_registry=None,
    )

    with pytest.raises(ValueError, match="tool_registry"):
        orch.completion("What is 2 + 2?")


@pytest.mark.unit
def test_orchestrator_tool_mode_rejects_adapters_without_tool_support() -> None:
    """Tool mode rejects adapters that report supports_tools=False."""

    class _NoToolsLLM(_ToolQueueLLM):
        @property
        def supports_tools(self) -> bool:  # type: ignore[override]
            return False

    env = QueueEnvironment()
    registry = InMemoryToolRegistry()
    registry.register(simple_add)

    llm = _NoToolsLLM(script=["ignored"])

    orch = RLMOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        environment=env,
        agent_mode="tools",
        tool_registry=registry,
    )

    with pytest.raises(ValueError, match="supports tool calling"):
        orch.completion("What is 2 + 2?")


@pytest.mark.unit
def test_orchestrator_tool_mode_happy_path_single_tool_call() -> None:
    """Tool mode executes tool and returns final answer."""
    env = QueueEnvironment()
    registry = InMemoryToolRegistry()
    registry.register(simple_add)

    llm = _ToolQueueLLM(
        script=[
            # First: LLM calls the add tool
            {
                "tool_calls": [_make_tool_call("call_1", "simple_add", {"a": 2, "b": 3})],
                "response": "",
                "finish_reason": "tool_calls",
            },
            # Second: LLM returns final answer after seeing result
            "The answer is 5.",
        ],
    )

    orch = RLMOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        environment=env,
        agent_mode="tools",
        tool_registry=registry,
    )

    cc = orch.completion("What is 2 + 3?")

    assert cc.response == "The answer is 5."
    assert cc.finish_reason == "stop"
    # Usage should account for both LLM calls
    assert cc.usage_summary.model_usage_summaries["tool-mock"].total_calls == 2


@pytest.mark.unit
def test_orchestrator_tool_mode_propagates_tool_choice() -> None:
    env = QueueEnvironment()
    registry = InMemoryToolRegistry()
    registry.register(simple_add)

    llm = _ToolQueueLLM(script=["Direct answer"])
    orch = RLMOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        environment=env,
        agent_mode="tools",
        tool_registry=registry,
    )

    orch.completion("Hello", tool_choice="required")

    assert llm._calls
    assert llm._calls[0].tool_choice == "required"


@pytest.mark.unit
def test_orchestrator_tool_mode_serializes_dataclass_tool_results() -> None:
    @dataclass
    class _Payload:
        total: int
        note: str

    def produce_payload() -> _Payload:
        return _Payload(total=5, note="ok")

    env = QueueEnvironment()
    registry = InMemoryToolRegistry()
    registry.register(produce_payload)

    llm = _ToolQueueLLM(
        script=[
            {"tool_calls": [_make_tool_call("call_1", "produce_payload", {})]},
            "Done.",
        ],
    )

    orch = RLMOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        environment=env,
        agent_mode="tools",
        tool_registry=registry,
    )

    orch.completion("Run tool")

    tool_messages = _extract_tool_messages(llm._calls[1].prompt)
    assert len(tool_messages) == 1
    payload = json.loads(tool_messages[0]["content"])
    assert payload == {"total": 5, "note": "ok"}


@pytest.mark.unit
def test_orchestrator_tool_mode_serializes_common_types() -> None:
    def produce_payload() -> dict[str, Any]:
        return {
            "when": datetime(2024, 1, 2, 3, 4, 5),
            "day": date(2024, 1, 2),
            "amount": Decimal("12.34"),
            "blob": b"hello",
        }

    env = QueueEnvironment()
    registry = InMemoryToolRegistry()
    registry.register(produce_payload)

    llm = _ToolQueueLLM(
        script=[
            {"tool_calls": [_make_tool_call("call_1", "produce_payload", {})]},
            "Done.",
        ],
    )

    orch = RLMOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        environment=env,
        agent_mode="tools",
        tool_registry=registry,
    )

    orch.completion("Run tool")

    tool_messages = _extract_tool_messages(llm._calls[1].prompt)
    assert len(tool_messages) == 1
    payload = json.loads(tool_messages[0]["content"])

    assert payload["when"] == "2024-01-02T03:04:05"
    assert payload["day"] == "2024-01-02"
    assert payload["amount"] == "12.34"
    assert payload["blob"] == {"__bytes__": base64.b64encode(b"hello").decode("ascii")}


@pytest.mark.unit
def test_orchestrator_tool_mode_serialization_error_is_reported() -> None:
    class _Unserializable:
        def __init__(self) -> None:
            self.value = object()

    def produce_unserializable() -> _Unserializable:
        return _Unserializable()

    env = QueueEnvironment()
    registry = InMemoryToolRegistry()
    registry.register(produce_unserializable)

    llm = _ToolQueueLLM(
        script=[
            {"tool_calls": [_make_tool_call("call_1", "produce_unserializable", {})]},
            "Done.",
        ],
    )

    orch = RLMOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        environment=env,
        agent_mode="tools",
        tool_registry=registry,
    )

    orch.completion("Run tool")

    tool_messages = _extract_tool_messages(llm._calls[1].prompt)
    assert len(tool_messages) == 1
    payload = json.loads(tool_messages[0]["content"])
    assert "error" in payload
    assert "serialization failed" in payload["error"]


@pytest.mark.unit
def test_orchestrator_tool_mode_summarizes_long_conversations() -> None:
    env = QueueEnvironment()
    registry = InMemoryToolRegistry()
    registry.register(simple_add)

    llm = _ToolQueueLLM(
        script=[
            "Summary text.",
            {"tool_calls": [_make_tool_call("call_1", "simple_add", {"a": 1, "b": 2})]},
            "Final.",
        ],
    )

    orch = RLMOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        environment=env,
        agent_mode="tools",
        tool_registry=registry,
        system_prompt="sys",
        context_window_tokens=500,
        tool_summary_trigger_ratio=0.5,
        tool_summary_keep_last_messages=0,
        tool_summary_min_messages=2,
    )

    long_prompt = "x" * 2000
    orch.completion(long_prompt)

    assert len(llm._calls) == 3
    assert _prompt_contains(llm._calls[1].prompt, "Summary of prior conversation")


@pytest.mark.unit
def test_orchestrator_tool_mode_uses_adapter_token_count() -> None:
    class _CountingLLM(_ToolQueueLLM):
        def count_prompt_tokens(self, request: LLMRequest, /) -> int | None:  # type: ignore[override]
            return 1

    env = QueueEnvironment()
    registry = InMemoryToolRegistry()
    registry.register(simple_add)

    llm = _CountingLLM(
        script=[
            {"tool_calls": [_make_tool_call("call_1", "simple_add", {"a": 1, "b": 2})]},
            "Final.",
        ],
    )

    orch = RLMOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        environment=env,
        agent_mode="tools",
        tool_registry=registry,
        context_window_tokens=100,
        tool_summary_trigger_ratio=0.5,
        tool_summary_keep_last_messages=0,
        tool_summary_min_messages=2,
    )

    long_prompt = "x" * 2000
    cc = orch.completion(long_prompt)

    assert cc.response == "Final."
    assert len(llm._calls) == 2


@pytest.mark.unit
def test_orchestrator_tool_mode_multi_turn_tool_calls() -> None:
    """Tool mode can handle multiple tool calls in sequence."""
    env = QueueEnvironment()
    registry = InMemoryToolRegistry()
    registry.register(simple_add)
    registry.register(simple_multiply)

    llm = _ToolQueueLLM(
        script=[
            # First: LLM calls add
            {
                "tool_calls": [_make_tool_call("call_1", "simple_add", {"a": 2, "b": 3})],
            },
            # Second: LLM calls multiply with result
            {
                "tool_calls": [_make_tool_call("call_2", "simple_multiply", {"x": 5, "y": 4})],
            },
            # Third: Final answer
            "2+3=5, then 5*4=20.",
        ],
    )

    orch = RLMOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        environment=env,
        agent_mode="tools",
        tool_registry=registry,
    )

    cc = orch.completion("What is (2+3)*4?")

    assert cc.response == "2+3=5, then 5*4=20."
    assert cc.usage_summary.model_usage_summaries["tool-mock"].total_calls == 3


@pytest.mark.unit
def test_orchestrator_tool_mode_multiple_parallel_tool_calls() -> None:
    """Tool mode can handle multiple tool calls in a single response."""
    env = QueueEnvironment()
    registry = InMemoryToolRegistry()
    registry.register(simple_add)
    registry.register(simple_multiply)

    llm = _ToolQueueLLM(
        script=[
            # LLM calls both tools at once
            {
                "tool_calls": [
                    _make_tool_call("call_1", "simple_add", {"a": 1, "b": 2}),
                    _make_tool_call("call_2", "simple_multiply", {"x": 3, "y": 4}),
                ],
            },
            # Final answer
            "1+2=3 and 3*4=12.",
        ],
    )

    orch = RLMOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        environment=env,
        agent_mode="tools",
        tool_registry=registry,
    )

    cc = orch.completion("What is 1+2 and 3*4?")

    assert cc.response == "1+2=3 and 3*4=12."


@pytest.mark.unit
def test_orchestrator_tool_mode_tool_not_found_raises() -> None:
    """Tool mode raises ToolNotFoundError for unknown tools."""
    env = QueueEnvironment()
    registry = InMemoryToolRegistry()
    # Registry is empty - no tools registered

    llm = _ToolQueueLLM(
        script=[
            {
                "tool_calls": [_make_tool_call("call_1", "nonexistent_tool", {"arg": "value"})],
            },
        ],
    )

    orch = RLMOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        environment=env,
        agent_mode="tools",
        tool_registry=registry,
    )

    with pytest.raises(ToolNotFoundError) as exc_info:
        orch.completion("Call the mystery tool")

    assert exc_info.value.tool_name == "nonexistent_tool"


@pytest.mark.unit
def test_orchestrator_tool_mode_tool_execution_error_captured() -> None:
    """Tool execution errors are captured and returned to LLM, not raised."""
    env = QueueEnvironment()
    registry = InMemoryToolRegistry()
    registry.register(failing_tool)

    llm = _ToolQueueLLM(
        script=[
            # LLM calls the failing tool
            {
                "tool_calls": [_make_tool_call("call_1", "failing_tool", {"msg": "oops"})],
            },
            # LLM sees the error and responds accordingly
            "The tool failed with an error.",
        ],
    )

    orch = RLMOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        environment=env,
        agent_mode="tools",
        tool_registry=registry,
    )

    # Should NOT raise - error is captured and passed to LLM
    cc = orch.completion("Do the failing thing")

    assert cc.response == "The tool failed with an error."
    # Verify the LLM received the error message (check conversation history)
    assert len(llm._calls) == 2


@pytest.mark.unit
def test_orchestrator_tool_mode_max_iterations_enforced() -> None:
    """Tool mode respects max_tool_iterations limit."""
    env = QueueEnvironment()
    registry = InMemoryToolRegistry()
    registry.register(simple_add)

    # LLM keeps calling tools for exactly max_tool_iterations calls
    # After max_tool_iterations=3 iterations, orchestrator asks for final answer
    llm = _ToolQueueLLM(
        script=[
            # First 3 calls: LLM returns tool_calls
            {"tool_calls": [_make_tool_call("call_0", "simple_add", {"a": 1, "b": 1})]},
            {"tool_calls": [_make_tool_call("call_1", "simple_add", {"a": 1, "b": 1})]},
            {"tool_calls": [_make_tool_call("call_2", "simple_add", {"a": 1, "b": 1})]},
            # 4th call: orchestrator forces final answer request
            "Forced final answer",
        ],
    )

    orch = RLMOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        environment=env,
        agent_mode="tools",
        tool_registry=registry,
        max_tool_iterations=3,  # Only allow 3 iterations
    )

    cc = orch.completion("Keep adding forever")

    assert cc.response == "Forced final answer"
    assert cc.finish_reason == "max_iterations"
    # 3 tool iterations + 1 final answer request = 4 calls
    assert len(llm._calls) == 4


@pytest.mark.unit
def test_orchestrator_tool_mode_immediate_final_answer() -> None:
    """Tool mode handles LLM responding without tool calls."""
    env = QueueEnvironment()
    registry = InMemoryToolRegistry()
    registry.register(simple_add)

    llm = _ToolQueueLLM(
        script=[
            # LLM decides to answer directly without using tools
            "I already know the answer is 42.",
        ],
    )

    orch = RLMOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        environment=env,
        agent_mode="tools",
        tool_registry=registry,
    )

    cc = orch.completion("What is the meaning of life?")

    assert cc.response == "I already know the answer is 42."
    assert cc.finish_reason == "stop"
    assert len(llm._calls) == 1


@pytest.mark.unit
async def test_orchestrator_tool_mode_async_happy_path() -> None:
    """Async tool mode works correctly."""
    env = QueueEnvironment()
    registry = InMemoryToolRegistry()
    registry.register(simple_add)

    llm = _ToolQueueLLM(
        script=[
            {
                "tool_calls": [_make_tool_call("call_1", "simple_add", {"a": 10, "b": 20})],
            },
            "10 + 20 = 30",
        ],
    )

    orch = RLMOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        environment=env,
        agent_mode="tools",
        tool_registry=registry,
    )

    cc = await orch.acompletion("What is 10 + 20?")

    assert cc.response == "10 + 20 = 30"
    assert cc.finish_reason == "stop"
    assert cc.usage_summary.model_usage_summaries["tool-mock"].total_calls == 2


@pytest.mark.unit
async def test_orchestrator_tool_mode_async_requires_registry() -> None:
    """Async tool mode also requires registry."""
    env = QueueEnvironment()
    llm = _ToolQueueLLM(script=["ignored"])

    orch = RLMOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        environment=env,
        agent_mode="tools",
        tool_registry=None,
    )

    with pytest.raises(ValueError, match="tool_registry"):
        await orch.acompletion("What is 2 + 2?")


@pytest.mark.unit
def test_orchestrator_tool_mode_handles_string_prompt() -> None:
    """Tool mode handles plain string prompts."""
    env = QueueEnvironment()
    registry = InMemoryToolRegistry()
    registry.register(simple_add)

    llm = _ToolQueueLLM(script=["Direct answer"])

    orch = RLMOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        environment=env,
        agent_mode="tools",
        tool_registry=registry,
    )

    cc = orch.completion("Simple string prompt")
    assert cc.response == "Direct answer"


@pytest.mark.unit
def test_orchestrator_tool_mode_handles_dict_prompt() -> None:
    """Tool mode handles dict prompts."""
    env = QueueEnvironment()
    registry = InMemoryToolRegistry()
    registry.register(simple_add)

    llm = _ToolQueueLLM(script=["Direct answer"])

    orch = RLMOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        environment=env,
        agent_mode="tools",
        tool_registry=registry,
    )

    cc = orch.completion({"role": "user", "content": "Dict prompt"})
    assert cc.response == "Direct answer"


@pytest.mark.unit
def test_orchestrator_tool_mode_handles_list_prompt() -> None:
    """Tool mode handles list of messages prompts."""
    env = QueueEnvironment()
    registry = InMemoryToolRegistry()
    registry.register(simple_add)

    llm = _ToolQueueLLM(script=["Direct answer"])

    orch = RLMOrchestrator(
        llm=llm,  # type: ignore[arg-type]
        environment=env,
        agent_mode="tools",
        tool_registry=registry,
    )

    cc = orch.completion([{"role": "user", "content": "List prompt"}])
    assert cc.response == "Direct answer"
