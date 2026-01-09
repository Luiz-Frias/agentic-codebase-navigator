from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rlm.domain.models import ChatCompletion
from rlm.domain.types import Prompt


def _is_prompt(value: object) -> bool:
    if isinstance(value, str):
        return True
    if isinstance(value, dict):
        # Legacy payloads allow arbitrary JSON-y dicts.
        return all(isinstance(k, str) for k in value.keys())
    if isinstance(value, list):
        # OpenAI-style: list[dict[str, Any]]
        return all(isinstance(item, dict) for item in value)
    return False


@dataclass(frozen=True, slots=True)
class WireRequest:
    """
    Wire DTO: request from an environment/process to the broker.

    Supports both:
    - single prompt: `prompt`
    - batched prompts: `prompts`
    """

    correlation_id: str | None = None
    prompt: Prompt | None = None
    prompts: list[Prompt] | None = None
    model: str | None = None

    @property
    def is_batched(self) -> bool:
        return self.prompts is not None and len(self.prompts) > 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.correlation_id is not None:
            d["correlation_id"] = self.correlation_id
        if self.prompt is not None:
            d["prompt"] = self.prompt
        if self.prompts is not None:
            d["prompts"] = self.prompts
        if self.model is not None:
            d["model"] = self.model
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WireRequest:
        allowed = {"correlation_id", "prompt", "prompts", "model"}
        unknown = set(data.keys()) - allowed
        if unknown:
            raise ValueError(f"Unknown keys in WireRequest: {sorted(unknown)!r}")

        correlation_id = data.get("correlation_id")
        if correlation_id is not None and not isinstance(correlation_id, str):
            raise TypeError("WireRequest.correlation_id must be a string when present")

        model = data.get("model")
        if model is not None and not isinstance(model, str):
            raise TypeError("WireRequest.model must be a string when present")

        prompt = data.get("prompt")
        prompts = data.get("prompts")

        if prompt is not None and prompts is not None:
            raise ValueError("WireRequest must include only one of 'prompt' or 'prompts'")

        if prompt is not None:
            if not _is_prompt(prompt):
                raise TypeError("WireRequest.prompt must be a valid Prompt")
            return cls(correlation_id=correlation_id, prompt=prompt, model=model)

        if prompts is not None:
            if not isinstance(prompts, list):
                raise TypeError("WireRequest.prompts must be a list when present")
            if len(prompts) == 0:
                raise ValueError("WireRequest.prompts must not be empty")
            for i, p in enumerate(prompts):
                if not _is_prompt(p):
                    raise TypeError(f"WireRequest.prompts[{i}] must be a valid Prompt")
            return cls(correlation_id=correlation_id, prompts=prompts, model=model)

        raise ValueError("WireRequest missing 'prompt' or 'prompts'")


@dataclass(frozen=True, slots=True)
class WireResult:
    """Wire DTO: result for a single prompt (success or error)."""

    error: str | None = None
    chat_completion: ChatCompletion | None = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.error,
            "chat_completion": self.chat_completion.to_dict() if self.chat_completion else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WireResult:
        allowed = {"error", "chat_completion"}
        unknown = set(data.keys()) - allowed
        if unknown:
            raise ValueError(f"Unknown keys in WireResult: {sorted(unknown)!r}")

        error = data.get("error")
        if error is not None and not isinstance(error, str):
            raise TypeError("WireResult.error must be a string when present")

        raw_cc = data.get("chat_completion")
        if raw_cc is None:
            if error is None:
                raise ValueError("WireResult must include either 'error' or 'chat_completion'")
            return cls(error=error, chat_completion=None)

        if error is not None:
            raise ValueError("WireResult cannot include both 'error' and 'chat_completion'")
        if not isinstance(raw_cc, dict):
            raise TypeError("WireResult.chat_completion must be a dict when present")
        return cls(error=None, chat_completion=ChatCompletion.from_dict(raw_cc))


@dataclass(frozen=True, slots=True)
class WireResponse:
    """
    Wire DTO: broker response.

    - For request-level failures (invalid payload): `error` is set and `results` is None.
    - For successful routing: `results` is set with per-item success/error.
    """

    correlation_id: str | None = None
    error: str | None = None
    results: list[WireResult] | None = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "error": self.error,
            "results": [r.to_dict() for r in self.results] if self.results is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WireResponse:
        allowed = {"correlation_id", "error", "results"}
        unknown = set(data.keys()) - allowed
        if unknown:
            raise ValueError(f"Unknown keys in WireResponse: {sorted(unknown)!r}")

        correlation_id = data.get("correlation_id")
        if correlation_id is not None and not isinstance(correlation_id, str):
            raise TypeError("WireResponse.correlation_id must be a string when present")

        error = data.get("error")
        if error is not None and not isinstance(error, str):
            raise TypeError("WireResponse.error must be a string when present")

        raw_results = data.get("results")
        if raw_results is None:
            return cls(correlation_id=correlation_id, error=error, results=None)

        if error is not None:
            raise ValueError("WireResponse cannot include both 'error' and 'results'")
        if not isinstance(raw_results, list):
            raise TypeError("WireResponse.results must be a list when present")
        return cls(
            correlation_id=correlation_id,
            error=None,
            results=[WireResult.from_dict(r) for r in raw_results],
        )
