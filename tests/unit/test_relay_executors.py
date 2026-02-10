from __future__ import annotations

import pytest

from rlm.adapters.relay import (
    AsyncStateExecutor,
    FunctionStateExecutor,
    LLMStateExecutor,
    RLMStateExecutor,
)
from rlm.domain.models import ChatCompletion, LLMRequest
from rlm.domain.models.result import Err, Ok
from rlm.domain.relay import Baton, StateError, StateSpec, has_pydantic
from tests.fakes_ports import QueueLLM


@pytest.mark.unit
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_function_state_executor_success() -> None:
    state = StateSpec(name="fn", input_type=int, output_type=int)
    executor = FunctionStateExecutor(lambda x: x + 1)

    baton_result = Baton.create(1, int)
    assert isinstance(baton_result, Ok)

    result = executor.execute(state, baton_result.value)

    assert isinstance(result, Ok)
    assert result.value.payload == 2


@pytest.mark.unit
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_function_state_executor_error() -> None:
    def _boom(_: int) -> int:
        raise ValueError("boom")

    state = StateSpec(name="fn", input_type=int, output_type=int)
    executor = FunctionStateExecutor(_boom)
    baton_result = Baton.create(1, int)
    assert isinstance(baton_result, Ok)

    result = executor.execute(state, baton_result.value)

    assert isinstance(result, Err)
    assert isinstance(result.error, StateError)


@pytest.mark.unit
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_llm_state_executor_with_llm_request_payload() -> None:
    llm = QueueLLM(responses=["hello"])
    executor = LLMStateExecutor(llm)
    state = StateSpec(name="llm", input_type=LLMRequest, output_type=ChatCompletion)

    baton_result = Baton.create(LLMRequest(prompt="hi"), LLMRequest)
    assert isinstance(baton_result, Ok)

    result = executor.execute(state, baton_result.value)
    assert isinstance(result, Ok)
    assert isinstance(result.value.payload, ChatCompletion)
    assert result.value.payload.response == "hello"


@pytest.mark.unit
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_llm_state_executor_with_builder() -> None:
    llm = QueueLLM(responses=["hello"])
    executor = LLMStateExecutor(llm, request_builder=lambda text: LLMRequest(prompt=text))
    state = StateSpec(name="llm", input_type=str, output_type=ChatCompletion)

    baton_result = Baton.create("hi", str)
    assert isinstance(baton_result, Ok)

    result = executor.execute(state, baton_result.value)
    assert isinstance(result, Ok)
    assert result.value.payload.response == "hello"


@pytest.mark.unit
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_rlm_state_executor_uses_orchestrator() -> None:
    class FakeOrchestrator:
        def completion(self, prompt: str, **_kwargs: object) -> ChatCompletion:
            return ChatCompletion(
                root_model="fake",
                prompt=prompt,
                response="ok",
                usage_summary=QueueLLM().get_usage_summary(),
                execution_time=0.0,
            )

    executor = RLMStateExecutor(orchestrator=FakeOrchestrator())
    state = StateSpec(name="rlm", input_type=str, output_type=ChatCompletion)

    baton_result = Baton.create("hello", str)
    assert isinstance(baton_result, Ok)

    result = executor.execute(state, baton_result.value)

    assert isinstance(result, Ok)
    assert result.value.payload.response == "ok"


@pytest.mark.unit
@pytest.mark.skipif(not has_pydantic(), reason="pydantic not installed")
def test_async_state_executor_executes() -> None:
    async def _work(value: int) -> int:
        return value + 3

    executor = AsyncStateExecutor(_work)
    state = StateSpec(name="async", input_type=int, output_type=int)

    baton_result = Baton.create(4, int)
    assert isinstance(baton_result, Ok)

    result = executor.execute(state, baton_result.value)
    assert isinstance(result, Ok)
    assert result.value.payload == 7
