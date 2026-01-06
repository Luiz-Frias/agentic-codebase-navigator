from __future__ import annotations

from typing import Any

from rlm._legacy.core.lm_handler import LMHandler
from rlm.adapters.legacy.llm import _as_legacy_client
from rlm.domain.ports import LLMPort, Prompt


class LegacyBrokerAdapter:
    """Adapter: legacy `LMHandler` -> domain `BrokerPort`."""

    def __init__(
        self,
        default_llm: LLMPort,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ):
        self._handler = LMHandler(_as_legacy_client(default_llm), host=host, port=port)

    def register_llm(self, model_name: str, llm: LLMPort, /) -> None:
        self._handler.register_client(model_name, _as_legacy_client(llm))

    def start(self) -> tuple[str, int]:
        return self._handler.start()

    def stop(self) -> None:
        self._handler.stop()

    def completion(self, prompt: Prompt, /, *, model: str | None = None) -> str:
        return self._handler.completion(prompt, model=model)  # type: ignore[arg-type]

    def get_usage_summary(self) -> Any:
        return self._handler.get_usage_summary()
