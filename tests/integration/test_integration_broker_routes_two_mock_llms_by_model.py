from __future__ import annotations

import pytest

from rlm.adapters.broker.tcp import TcpBrokerAdapter
from rlm.adapters.llm.mock import MockLLMAdapter
from rlm.domain.errors import BrokerError
from rlm.infrastructure.comms.protocol import request_completion
from tests.live_llm import LiveLLMSettings


@pytest.mark.integration
def test_tcp_broker_routes_between_two_mock_llms_by_model_name(
    live_llm_settings: LiveLLMSettings | None,
) -> None:
    if live_llm_settings is not None:
        from rlm.adapters.llm.openai import OpenAIAdapter

        llm_a = OpenAIAdapter(
            model=live_llm_settings.model,
            api_key=live_llm_settings.api_key,
            base_url=live_llm_settings.base_url,
            default_request_kwargs={"temperature": 0, "max_tokens": 32},
        )
        llm_b = OpenAIAdapter(
            model=live_llm_settings.model_sub,
            api_key=live_llm_settings.api_key,
            base_url=live_llm_settings.base_url,
            default_request_kwargs={"temperature": 0, "max_tokens": 32},
        )
    else:
        llm_a = MockLLMAdapter(model="a", script=["A"])
        llm_b = MockLLMAdapter(model="b", script=["B"])

    broker = TcpBrokerAdapter(llm_a)
    broker.register_llm(llm_b.model_name, llm_b)

    addr = broker.start()
    try:
        cc_a = request_completion(addr, "hi")
        assert cc_a.root_model == llm_a.model_name
        assert cc_a.response.strip()

        cc_b = request_completion(addr, "hi", model=llm_b.model_name)
        assert cc_b.root_model == llm_b.model_name
        assert cc_b.response.strip()

        with pytest.raises(BrokerError, match="Unknown model"):
            request_completion(addr, "hi", model="unknown")
    finally:
        broker.stop()
