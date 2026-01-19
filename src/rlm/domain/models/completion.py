from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rlm.domain.models.serialization import SerializedValue, serialize_value
from rlm.domain.models.usage import ModelUsageSummary, UsageSummary

if TYPE_CHECKING:
    from rlm.domain.agent_ports import ToolCallRequest
    from rlm.domain.types import Prompt


@dataclass(slots=True)
class ChatCompletion:
    """
    A single LLM call result.

    Mirrors the shape of the legacy `RLMChatCompletion`, but is dependency-free
    and owned by the domain layer.

    Attributes:
        root_model: The model that generated this completion.
        prompt: The prompt that was sent to the LLM (typed as domain Prompt).
        response: The text response from the LLM (may be empty if tool_calls present).
        usage_summary: Token usage statistics.
        execution_time: Time taken for the API call in seconds.
        tool_calls: List of tool call requests from the LLM (None if no tools called).
        finish_reason: Why the LLM stopped generating (e.g., "stop", "tool_calls").

    """

    root_model: str
    prompt: Prompt
    response: str
    usage_summary: UsageSummary
    execution_time: float
    tool_calls: list[ToolCallRequest] | None = field(default=None)
    finish_reason: str | None = field(default=None)

    def to_dict(self) -> dict[str, SerializedValue]:
        """Serialize to dict with JSON-compatible values."""
        mus = self.usage_summary.model_usage_summaries.get(self.root_model)
        prompt_tokens = mus.total_input_tokens if mus is not None else 0
        completion_tokens = mus.total_output_tokens if mus is not None else 0
        result: dict[str, SerializedValue] = {
            "root_model": self.root_model,
            "prompt": serialize_value(self.prompt),
            "response": self.response,
            "usage_summary": serialize_value(self.usage_summary.to_dict()),
            # Back-compat / visualizer convenience: legacy schema expects flat token counts.
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "execution_time": self.execution_time,
        }
        # Only include tool_calls if present (backward compatibility)
        if self.tool_calls is not None:
            result["tool_calls"] = serialize_value(self.tool_calls)
        if self.finish_reason is not None:
            result["finish_reason"] = self.finish_reason
        return result

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ChatCompletion:
        """
        Create ChatCompletion from dict.

        Type-driven boundary: accepts dict[str, object], validates internally.
        """
        raw_usage = data.get("usage_summary")
        usage = (
            UsageSummary.from_dict(raw_usage)
            if isinstance(raw_usage, dict)
            else UsageSummary(model_usage_summaries={})
        )
        # Back-compat: legacy logs may store flat token counts instead of a UsageSummary.
        if not usage.model_usage_summaries:
            pt = data.get("prompt_tokens")
            ct = data.get("completion_tokens")
            if isinstance(pt, (int, float)) and isinstance(ct, (int, float)):
                raw_model = data.get("root_model", "")
                model = str(raw_model) if raw_model else "unknown"
                usage = UsageSummary(
                    model_usage_summaries={
                        model: ModelUsageSummary(
                            total_calls=1,
                            total_input_tokens=int(pt),
                            total_output_tokens=int(ct),
                        ),
                    },
                )
        # Parse tool_calls if present (list of ToolCallRequest dicts)
        raw_tool_calls = data.get("tool_calls")
        tool_calls: list[dict[str, object]] | None = None
        if isinstance(raw_tool_calls, list):
            tool_calls = raw_tool_calls  # Already in ToolCallRequest format

        # Parse finish_reason if present
        raw_finish_reason = data.get("finish_reason")
        finish_reason = str(raw_finish_reason) if raw_finish_reason is not None else None

        # Parse prompt - narrow to Prompt type (str | dict | list)
        raw_prompt = data.get("prompt")
        prompt: Prompt
        if isinstance(raw_prompt, (str, dict, list)):
            prompt = raw_prompt
        else:
            prompt = str(raw_prompt) if raw_prompt is not None else ""

        # Parse root_model
        raw_root_model = data.get("root_model", "")
        root_model = str(raw_root_model) if raw_root_model else ""

        # Parse response
        raw_response = data.get("response", "")
        response = str(raw_response) if raw_response else ""

        # Parse execution_time
        raw_execution_time = data.get("execution_time", 0.0)
        execution_time: float
        if isinstance(raw_execution_time, (int, float)):
            execution_time = float(raw_execution_time)
        else:
            execution_time = 0.0

        return cls(
            root_model=root_model,
            prompt=prompt,
            response=response,
            usage_summary=usage,
            execution_time=execution_time,
            tool_calls=tool_calls,  # type: ignore[arg-type]  # TypedDict ≈ dict
            finish_reason=finish_reason,
        )
