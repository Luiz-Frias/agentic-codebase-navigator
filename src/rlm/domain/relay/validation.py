from __future__ import annotations

from typing import TYPE_CHECKING

from rlm.domain.relay.errors import PipelineDefinitionError

if TYPE_CHECKING:
    from collections.abc import Callable

    from rlm.domain.relay.pipeline import Edge, Pipeline
    from rlm.domain.relay.state import StateSpec


def allow_cycles(*, max_iterations: int) -> Callable[[Pipeline], Pipeline]:
    def _apply(pipeline: Pipeline) -> Pipeline:
        pipeline.set_cycle_policy(allow_cycles=True, max_iterations=max_iterations)
        return pipeline

    return _apply


def validate_pipeline(pipeline: Pipeline) -> None:
    errors: list[str] = []

    if not pipeline.states:
        errors.append("Pipeline must define at least one state.")
    if pipeline.entry_state is None:
        errors.append("Pipeline must define an entry state.")

    errors.extend(_validate_edges(pipeline.edges))
    errors.extend(_validate_joins(pipeline.join_groups))

    if pipeline.entry_state is not None:
        errors.extend(_validate_reachability(pipeline.entry_state, pipeline.states, pipeline.edges))

    if not pipeline.terminal_states:
        errors.append("Pipeline must define at least one terminal state.")

    if pipeline.allow_cycles:
        if pipeline.max_cycle_iterations is None or pipeline.max_cycle_iterations <= 0:
            errors.append("Cycle allowance requires max_cycle_iterations > 0.")
    elif _has_cycle(pipeline.states, pipeline.edges):
        errors.append("Cycle detected. Use allow_cycles(max_iterations=...) if intentional.")

    if errors:
        raise PipelineDefinitionError(tuple(errors))


def _validate_edges(edges: tuple[Edge[object, object, object], ...]) -> list[str]:
    errors: list[str] = []
    for edge in edges:
        output_type = edge.from_state.output_type
        input_type = edge.to_state.input_type
        if not _is_type_compatible(output_type, input_type):
            errors.append(
                "Type mismatch: "
                f"{edge.from_state.name} outputs {output_type}, "
                f"but {edge.to_state.name} expects {input_type}."
            )
    return errors


def _validate_joins(join_groups: tuple) -> list[str]:
    errors: list[str] = []
    for group in join_groups:
        if group.join_spec.mode == "all":
            if group.target.input_type is not dict:
                errors.append(
                    "Type mismatch: "
                    f"{group.target.name} expects {group.target.input_type}, "
                    "but join(all) aggregates into dict.",
                )
            continue
        mismatches = [
            (
                "Type mismatch: "
                f"{source.name} outputs {source.output_type}, "
                f"but {group.target.name} expects {group.target.input_type}."
            )
            for source in group.sources
            if not _is_type_compatible(source.output_type, group.target.input_type)
        ]
        errors.extend(mismatches)
    return errors


def _is_type_compatible(output_type: type, input_type: type) -> bool:
    if output_type is input_type:
        return True
    if isinstance(output_type, type) and isinstance(input_type, type):
        return issubclass(output_type, input_type)
    return False


def _validate_reachability(
    entry_state: StateSpec[object, object],
    states: tuple[StateSpec[object, object], ...],
    edges: tuple[Edge[object, object, object], ...],
) -> list[str]:
    reachable = _reachable_from(entry_state, edges)
    unreachable = [state.name for state in states if state.name not in reachable]
    if unreachable:
        return [f"Unreachable states: {sorted(unreachable)}"]
    return []


def _reachable_from(
    entry_state: StateSpec[object, object],
    edges: tuple[Edge[object, object, object], ...],
) -> set[str]:
    adjacency = _build_adjacency(edges)
    visited: set[str] = set()
    stack: list[str] = [entry_state.name]

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(adjacency.get(current, ()))

    return visited


def _has_cycle(
    states: tuple[StateSpec[object, object], ...],
    edges: tuple[Edge[object, object, object], ...],
) -> bool:
    adjacency = _build_adjacency(edges)
    visiting: set[str] = set()
    visited: set[str] = set()

    def _visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for neighbor in adjacency.get(node, ()):  # pragma: no branch - tight loop
            if _visit(neighbor):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(_visit(state.name) for state in states)


def _build_adjacency(
    edges: tuple[Edge[object, object, object], ...],
) -> dict[str, tuple[str, ...]]:
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.from_state.name, []).append(edge.to_state.name)
    return {key: tuple(values) for key, values in adjacency.items()}
