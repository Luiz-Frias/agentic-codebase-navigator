"""
Legacy core types (ported from upstream).

NOTE: This module is a temporary mirror of `references/rlm/rlm/core/types.py`.
We keep it importable during the Phase 1 legacy port, then progressively
migrate callers toward the hexagonal layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Literal

ClientBackend = Literal[
    "openai",
    "portkey",
    "openrouter",
    "vllm",
    "litellm",
    "anthropic",
    "azure_openai",
    "gemini",
]

# Upstream had mismatched literals vs actual supported environments. We widen
# this union to include the environments that exist in the upstream repo.
EnvironmentType = Literal["local", "docker", "modal", "prime"]


def _serialize_value(value: Any) -> Any:
    """Convert a value to a JSON-serializable representation."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, ModuleType):
        return f"<module '{value.__name__}'>"
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if callable(value):
        return f"<{type(value).__name__} '{getattr(value, '__name__', repr(value))}'>"
    # Try to convert to string for other types
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__}>"


########################################################
########    Types for LM Cost Tracking         #########
########################################################


@dataclass
class ModelUsageSummary:
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int

    def to_dict(self) -> dict[str, int]:
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelUsageSummary:
        return cls(
            total_calls=data.get("total_calls"),
            total_input_tokens=data.get("total_input_tokens"),
            total_output_tokens=data.get("total_output_tokens"),
        )


@dataclass
class UsageSummary:
    model_usage_summaries: dict[str, ModelUsageSummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_usage_summaries": {
                model: usage_summary.to_dict()
                for model, usage_summary in self.model_usage_summaries.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsageSummary:
        return cls(
            model_usage_summaries={
                model: ModelUsageSummary.from_dict(usage_summary)
                for model, usage_summary in data.get("model_usage_summaries", {}).items()
            },
        )


########################################################
########   Types for REPL and RLM Iterations   #########
########################################################
@dataclass
class RLMChatCompletion:
    """Record of a single LLM call made from within the environment."""

    root_model: str
    prompt: str | dict[str, Any]
    response: str
    usage_summary: UsageSummary
    execution_time: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_model": self.root_model,
            "prompt": self.prompt,
            "response": self.response,
            "usage_summary": self.usage_summary.to_dict(),
            "execution_time": self.execution_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RLMChatCompletion:
        usage_summary_data = data.get("usage_summary", {}) or {}
        if not isinstance(usage_summary_data, dict):
            raise TypeError(
                "RLMChatCompletion.usage_summary must be a dict when present "
                f"(got {type(usage_summary_data).__name__})"
            )
        return cls(
            root_model=data.get("root_model"),
            prompt=data.get("prompt"),
            response=data.get("response"),
            usage_summary=UsageSummary.from_dict(usage_summary_data),
            execution_time=data.get("execution_time"),
        )


@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals: dict[str, Any]
    execution_time: float | None
    llm_calls: list[RLMChatCompletion] = field(default_factory=list)

    def __init__(
        self,
        stdout: str,
        stderr: str,
        locals: dict[str, Any],
        execution_time: float | None = None,
        llm_calls: list[RLMChatCompletion] | None = None,
        rlm_calls: list[RLMChatCompletion] | None = None,
    ):
        # Upstream bug: the field was annotated as `llm_calls`, but runtime stored
        # `self.rlm_calls`. We make `llm_calls` canonical and keep a compat alias.
        if llm_calls is not None and rlm_calls is not None:
            raise ValueError("Pass only one of `llm_calls` or `rlm_calls`.")

        self.stdout = stdout
        self.stderr = stderr
        self.locals = locals
        self.execution_time = execution_time
        self.llm_calls = (llm_calls if llm_calls is not None else rlm_calls) or []

    @property
    def rlm_calls(self) -> list[RLMChatCompletion]:
        """Backward-compatible alias used throughout upstream code."""

        return self.llm_calls

    @rlm_calls.setter
    def rlm_calls(self, value: list[RLMChatCompletion]) -> None:
        self.llm_calls = value

    def __str__(self) -> str:
        return (
            "REPLResult("
            f"stdout={self.stdout}, "
            f"stderr={self.stderr}, "
            f"locals={self.locals}, "
            f"execution_time={self.execution_time}, "
            f"llm_calls={len(self.llm_calls)})"
        )

    def to_dict(self) -> dict[str, Any]:
        # Keep upstream key name `rlm_calls` for log/schema compatibility.
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "locals": {k: _serialize_value(v) for k, v in self.locals.items()},
            "execution_time": self.execution_time,
            "rlm_calls": [call.to_dict() for call in self.llm_calls],
        }


@dataclass
class CodeBlock:
    code: str
    result: REPLResult

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "result": self.result.to_dict()}


@dataclass
class RLMIteration:
    prompt: str | dict[str, Any]
    response: str
    code_blocks: list[CodeBlock]
    final_answer: str | None = None
    iteration_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "code_blocks": [code_block.to_dict() for code_block in self.code_blocks],
            "final_answer": self.final_answer,
            "iteration_time": self.iteration_time,
        }


########################################################
########   Types for RLM Metadata   #########
########################################################


@dataclass
class RLMMetadata:
    """Metadata about the RLM configuration."""

    root_model: str
    max_depth: int
    max_iterations: int
    backend: str
    backend_kwargs: dict[str, Any]
    environment_type: str
    environment_kwargs: dict[str, Any]
    other_backends: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_model": self.root_model,
            "max_depth": self.max_depth,
            "max_iterations": self.max_iterations,
            "backend": self.backend,
            "backend_kwargs": {k: _serialize_value(v) for k, v in self.backend_kwargs.items()},
            "environment_type": self.environment_type,
            "environment_kwargs": {
                k: _serialize_value(v) for k, v in self.environment_kwargs.items()
            },
            "other_backends": self.other_backends,
        }


########################################################
########   Types for RLM Prompting   #########
########################################################


@dataclass
class QueryMetadata:
    context_lengths: list[int]
    context_total_length: int
    context_type: str

    def __init__(self, prompt: str | list[str] | dict[Any, Any] | list[dict[Any, Any]]):
        if isinstance(prompt, str):
            self.context_lengths = [len(prompt)]
            self.context_type = "str"
        elif isinstance(prompt, dict):
            self.context_type = "dict"
            self.context_lengths = []
            for chunk in prompt.values():
                if isinstance(chunk, str):
                    self.context_lengths.append(len(chunk))
                    continue
                try:
                    import json

                    self.context_lengths.append(len(json.dumps(chunk, default=str)))
                except Exception:
                    self.context_lengths.append(len(repr(chunk)))
            self.context_type = "dict"
        elif isinstance(prompt, list):
            self.context_type = "list"
            if len(prompt) == 0:
                self.context_lengths = [0]
            elif isinstance(prompt[0], dict):
                if "content" in prompt[0]:
                    self.context_lengths = [len(str(chunk.get("content", ""))) for chunk in prompt]
                else:
                    self.context_lengths = []
                    for chunk in prompt:
                        try:
                            import json

                            self.context_lengths.append(len(json.dumps(chunk, default=str)))
                        except Exception:
                            self.context_lengths.append(len(repr(chunk)))
            else:
                self.context_lengths = [len(chunk) for chunk in prompt]
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        self.context_total_length = sum(self.context_lengths)
