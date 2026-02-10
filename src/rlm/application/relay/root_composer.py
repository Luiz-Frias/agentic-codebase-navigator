from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rlm.domain.errors import ValidationError
from rlm.domain.models import LLMRequest

if TYPE_CHECKING:
    from rlm.domain.ports import LLMPort
    from rlm.domain.relay.pipeline import Pipeline
    from rlm.domain.relay.registry import PipelineRegistry, PipelineTemplate
    from rlm.domain.types import Prompt


@dataclass(frozen=True, slots=True)
class RootAgentComposer:
    registry: PipelineRegistry
    llm: LLMPort

    def compose(self, task: Prompt, /) -> Pipeline:
        return compose_from_registry(task, registry=self.registry, llm=self.llm)


def compose_from_registry(
    task: Prompt,
    *,
    registry: PipelineRegistry,
    llm: LLMPort,
) -> Pipeline:
    candidates = registry.search(str(task))
    if not candidates:
        raise ValidationError("No pipeline candidates found for task.")
    if len(candidates) == 1:
        return candidates[0].factory()

    selected = _select_pipeline_names(task, candidates, llm)
    if not selected:
        return candidates[0].factory()

    pipelines: list[Pipeline] = []
    by_name = {template.name: template for template in candidates}
    for name in selected:
        template = by_name.get(name)
        if template is None:
            continue
        pipelines.append(template.factory())

    if not pipelines:
        return candidates[0].factory()
    return _chain_pipelines(pipelines)


def _select_pipeline_names(
    task: Prompt,
    candidates: tuple[PipelineTemplate[object, object], ...],
    llm: LLMPort,
) -> tuple[str, ...]:
    choices = "\n".join(f"- {template.name}: {template.description}" for template in candidates)
    prompt = (
        "You are selecting pipeline templates for a task. "
        "Return a comma-separated list of pipeline names from the choices. "
        "If only one is relevant, return just that name.\n\n"
        f"Task: {task}\n\nChoices:\n{choices}\n\nNames:"
    )
    response = llm.complete(LLMRequest(prompt=prompt)).response
    lowered = response.lower()
    selected = [template.name for template in candidates if template.name.lower() in lowered]
    return tuple(selected)


def _chain_pipelines(pipelines: list[Pipeline]) -> Pipeline:
    base = pipelines[0]
    for next_pipeline in pipelines[1:]:
        base = _merge_pipeline(base, next_pipeline)
    return base


def _merge_pipeline(left: Pipeline, right: Pipeline) -> Pipeline:
    if left.entry_state is None or right.entry_state is None:
        raise ValidationError("Cannot chain pipeline without entry state.")
    left_terminals = left.terminal_states
    if not left_terminals:
        raise ValidationError("Cannot chain pipeline without terminal state.")
    if len(left_terminals) != 1:
        raise ValidationError("Pipeline chaining requires a single terminal state.")

    left_terminal = left_terminals[0]
    right_entry = right.entry_state

    if left_terminal.output_type is not right_entry.input_type:
        raise ValidationError(
            "Type mismatch when chaining pipelines: "
            f"{left_terminal.output_type} -> {right_entry.input_type}",
        )

    for state in right.states:
        left.add_state(state)
    for edge in right.edges:
        left.add_edge(edge.from_state, edge.to_state, guard=edge.guard)
    for group in right.join_groups:
        left.add_join_group(group.sources, group.target, group.join_spec)

    left.add_edge(left_terminal, right_entry)
    return left
