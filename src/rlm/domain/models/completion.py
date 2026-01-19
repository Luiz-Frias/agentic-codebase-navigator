from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rlm.domain.models.serialization import serialize_value
from rlm.domain.models.usage import ModelUsageSummary, UsageSummary

if TYPE_CHECKING:
    from rlm.domain.agent_ports import ToolCallRequest


@dataclass(slots=True)
class ChatCompletion:
    """A single LLM call result.

    Mirrors the shape of the legacy `RLMChatCompletion`, but is dependency-free
    and owned by the domain layer.

    Attributes:
        root_model: The model that generated this completion.
        prompt: The prompt that was sent to the LLM.
        response: The text response from the LLM (may be empty if tool_calls present).
        usage_summary: Token usage statistics.
        execution_time: Time taken for the API call in seconds.
        tool_calls: List of tool call requests from the LLM (None if no tools called).
        finish_reason: Why the LLM stopped generating (e.g., "stop", "tool_calls").

    """

    root_model: str
    prompt: Any
    response: str
    usage_summary: UsageSummary
    execution_time: float
    tool_calls: list[ToolCallRequest] | None = field(default=None)
    finish_reason: str | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        mus = self.usage_summary.model_usage_summaries.get(self.root_model)
        prompt_tokens = mus.total_input_tokens if mus is not None else 0
        completion_tokens = mus.total_output_tokens if mus is not None else 0
        result: dict[str, Any] = {
            "root_model": self.root_model,
            "prompt": serialize_value(self.prompt),
            "response": self.response,
            "usage_summary": self.usage_summary.to_dict(),
            # Back-compat / visualizer convenience: legacy schema expects flat token counts.
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "execution_time": self.execution_time,
        }
        # Only include tool_calls if present (backward compatibility)
        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls
        if self.finish_reason is not None:
            result["finish_reason"] = self.finish_reason
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatCompletion:
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
                model = str(data.get("root_model", "") or "unknown")
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
        tool_calls: list[dict[str, Any]] | None = None
        if isinstance(raw_tool_calls, list):
            tool_calls = raw_tool_calls  # Already in ToolCallRequest format

        # Parse finish_reason if present
        raw_finish_reason = data.get("finish_reason")
        finish_reason = str(raw_finish_reason) if raw_finish_reason is not None else None

        return cls(
            root_model=str(data.get("root_model", "")),
            prompt=data.get("prompt"),
            response=str(data.get("response", "")),
            usage_summary=usage,
            execution_time=float(data.get("execution_time", 0.0)),
            tool_calls=tool_calls,  # type: ignore[arg-type]  # TypedDict ≈ dict
            finish_reason=finish_reason,
        )
