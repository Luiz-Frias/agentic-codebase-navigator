from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlm.domain.agent_ports import ToolDefinition
    from rlm.domain.types import Prompt

# Tool choice options for LLM requests:
# - "auto": LLM decides whether to call tools
# - "required": LLM must call at least one tool
# - "none": LLM will not call any tools
# - "<tool_name>": Force a specific tool by name
type ToolChoice = str | None


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """
    A typed LLM request.

    This is a Phase 2 bridge model used by ports/adapters and the new orchestrator.

    Attributes:
        prompt: The prompt to send to the LLM (string, dict, or list of messages).
        model: Optional model override (uses adapter's default if None).
        tools: Optional list of tool definitions for function calling mode.
        tool_choice: Controls how the LLM selects tools:
            - "auto": LLM decides whether to call tools (default)
            - "required": LLM must call at least one tool
            - "none": LLM will not call any tools
            - str: Force a specific tool by name

    """

    prompt: Prompt
    model: str | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoice = None


@dataclass(frozen=True, slots=True)
class BatchedLLMRequest:
    """A typed batched LLM request (ordered)."""

    prompts: list[Prompt]
    model: str | None = None
