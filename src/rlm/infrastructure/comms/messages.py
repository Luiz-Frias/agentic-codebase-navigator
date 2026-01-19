from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rlm.domain.errors import ValidationError
from rlm.domain.models import ChatCompletion
from rlm.domain.models.validation import Validator

if TYPE_CHECKING:
    from rlm.domain.models.serialization import SerializedValue
    from rlm.domain.types import Prompt


# =============================================================================
# Validators (using domain Validator pattern)
# =============================================================================


def _is_prompt(value: object) -> bool:
    """Check if value is a valid Prompt type (str | dict | list[dict])."""
    if isinstance(value, str):
        return True
    if isinstance(value, dict):
        # Legacy payloads allow arbitrary JSON-y dicts.
        return all(isinstance(k, str) for k in value)
    if isinstance(value, list):
        # OpenAI-style: list[dict[str, Any]]
        return all(isinstance(item, dict) for item in value)
    return False


# Pre-built validators for wire message fields
_optional_str_validator: Validator[object] = Validator[object]().satisfies(
    lambda v: v is None or isinstance(v, str),
    "must be a string when present",
)

_prompt_validator: Validator[object] = Validator[object]().satisfies(
    _is_prompt,
    "must be a valid Prompt (str | dict | list[dict])",
)

_prompts_list_validator: Validator[object] = (
    Validator[object]()
    .is_type(list, "must be a list")
    .satisfies(lambda v: len(v) > 0 if isinstance(v, list) else False, "must not be empty")
    .satisfies(
        lambda v: all(_is_prompt(p) for p in v) if isinstance(v, list) else False,
        "each item must be a valid Prompt",
    )
)


def _validate_unknown_keys(
    data: dict[str, object],
    allowed: set[str],
    type_name: str,
) -> None:
    """Raise ValidationError if data contains keys not in allowed set."""
    unknown = set(data.keys()) - allowed
    if unknown:
        raise ValidationError(f"Unknown keys in {type_name}: {sorted(unknown)!r}")


def _extract_optional_str(data: dict[str, object], key: str, type_name: str) -> str | None:
    """Extract and validate an optional string field using Validator."""
    value = data.get(key)
    if value is None:
        return None
    try:
        _optional_str_validator.validate(value)
    except ValidationError as e:
        raise ValidationError(f"{type_name}.{key} {e}") from None
    # After validation, we know it's a string
    return value  # type: ignore[return-value]


def _extract_prompt(data: dict[str, object], type_name: str) -> Prompt | None:
    """Extract and validate the 'prompt' field using Validator."""
    prompt = data.get("prompt")
    if prompt is None:
        return None
    try:
        _prompt_validator.validate(prompt)
    except ValidationError as e:
        raise ValidationError(f"{type_name}.prompt {e}") from None
    # After validation, we know it's a valid Prompt
    return prompt  # type: ignore[return-value]


def _extract_prompts(data: dict[str, object], type_name: str) -> list[Prompt] | None:
    """Extract and validate the 'prompts' field using Validator."""
    prompts = data.get("prompts")
    if prompts is None:
        return None
    try:
        _prompts_list_validator.validate(prompts)
    except ValidationError as e:
        raise ValidationError(f"{type_name}.prompts {e}") from None
    # After validation, we know it's a valid list[Prompt]
    return prompts  # type: ignore[return-value]


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

    def to_dict(self) -> dict[str, SerializedValue]:
        d: dict[str, SerializedValue] = {}
        if self.correlation_id is not None:
            d["correlation_id"] = self.correlation_id
        if self.prompt is not None:
            d["prompt"] = self.prompt  # type: ignore[assignment]  # Prompt is JSON-compatible
        if self.prompts is not None:
            d["prompts"] = self.prompts  # type: ignore[assignment]  # list[Prompt] is JSON-compatible
        if self.model is not None:
            d["model"] = self.model
        return d

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> WireRequest:
        """
        Parse wire data into a WireRequest.

        Type-driven boundary: accepts dict[str, object], validates internally
        using domain Validator pattern.
        """
        _validate_unknown_keys(
            data, {"correlation_id", "prompt", "prompts", "model"}, "WireRequest"
        )

        correlation_id = _extract_optional_str(data, "correlation_id", "WireRequest")
        model = _extract_optional_str(data, "model", "WireRequest")
        prompt = _extract_prompt(data, "WireRequest")
        prompts = _extract_prompts(data, "WireRequest")

        if prompt is not None and prompts is not None:
            raise ValidationError("WireRequest must include only one of 'prompt' or 'prompts'")
        if prompt is None and prompts is None:
            raise ValidationError("WireRequest missing 'prompt' or 'prompts'")

        return cls(correlation_id=correlation_id, prompt=prompt, prompts=prompts, model=model)


@dataclass(frozen=True, slots=True)
class WireResult:
    """Wire DTO: result for a single prompt (success or error)."""

    error: str | None = None
    chat_completion: ChatCompletion | None = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, SerializedValue]:
        return {
            "error": self.error,
            "chat_completion": self.chat_completion.to_dict() if self.chat_completion else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> WireResult:
        """
        Parse wire data into a WireResult.

        Type-driven boundary: accepts dict[str, object], validates internally
        using domain Validator pattern.
        """
        _validate_unknown_keys(data, {"error", "chat_completion"}, "WireResult")

        error = _extract_optional_str(data, "error", "WireResult")
        raw_cc = data.get("chat_completion")

        if raw_cc is None:
            if error is None:
                raise ValidationError("WireResult must include either 'error' or 'chat_completion'")
            return cls(error=error, chat_completion=None)

        if error is not None:
            raise ValidationError("WireResult cannot include both 'error' and 'chat_completion'")
        if not isinstance(raw_cc, dict):
            raise ValidationError("WireResult.chat_completion must be a dict when present")
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

    def to_dict(self) -> dict[str, SerializedValue]:
        return {
            "correlation_id": self.correlation_id,
            "error": self.error,
            "results": [r.to_dict() for r in self.results] if self.results is not None else None,  # type: ignore[misc]  # list[dict] is SerializedValue-compatible
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> WireResponse:
        """
        Parse wire data into a WireResponse.

        Type-driven boundary: accepts dict[str, object], validates internally
        using domain Validator pattern.
        """
        _validate_unknown_keys(data, {"correlation_id", "error", "results"}, "WireResponse")

        correlation_id = _extract_optional_str(data, "correlation_id", "WireResponse")
        error = _extract_optional_str(data, "error", "WireResponse")

        raw_results = data.get("results")
        if raw_results is None:
            return cls(correlation_id=correlation_id, error=error, results=None)

        if error is not None:
            raise ValidationError("WireResponse cannot include both 'error' and 'results'")
        if not isinstance(raw_results, list):
            raise ValidationError("WireResponse.results must be a list when present")

        # Validate each result dict before parsing
        results: list[WireResult] = []
        for i, r in enumerate(raw_results):
            if not isinstance(r, dict):
                raise ValidationError(f"WireResponse.results[{i}] must be a dict")
            results.append(WireResult.from_dict(r))

        return cls(
            correlation_id=correlation_id,
            error=None,
            results=results,
        )
