from __future__ import annotations

import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rlm.domain.models.result import Err
from rlm.domain.relay.errors import StateError
from rlm.domain.relay.execution import PipelineStep
from rlm.domain.relay.trace import PipelineTrace, TraceEntry

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from rlm.domain.relay.baton import Baton
    from rlm.domain.relay.pipeline import Edge, JoinGroup, Pipeline
    from rlm.domain.relay.ports import StateResult
    from rlm.domain.relay.state import StateSpec


@dataclass(slots=True)
class SyncPipelineExecutor:
    pipeline: Pipeline
    _queue: deque[tuple[StateSpec[object, object], Baton[object]]]
    _pending_result: StateResult[object] | None
    _failed: StateError | None
    _join_state: dict[int, dict[str, Baton[object]]]
    _completed_joins: set[int]
    _current_state: StateSpec[object, object] | None
    _current_baton: Baton[object] | None
    _trace: PipelineTrace

    def __init__(self, pipeline: Pipeline, initial_baton: Baton[object]) -> None:
        self.pipeline = pipeline
        self._queue = deque()
        if pipeline.entry_state is None:
            raise ValueError("Pipeline has no entry state.")
        seeded = False
        for group in pipeline.join_groups:
            if any(source.name == pipeline.entry_state.name for source in group.sources):
                for source in group.sources:
                    self._queue.append((source, initial_baton))
                seeded = True
                break
        if not seeded:
            self._queue.append((pipeline.entry_state, initial_baton))
        self._pending_result = None
        self._failed = None
        self._join_state = {}
        self._completed_joins = set()
        self._current_state = None
        self._current_baton = None
        self._trace = PipelineTrace()

    def __iter__(self) -> SyncPipelineExecutor:
        return self

    def __next__(self) -> PipelineStep[object, object]:
        if self._failed is not None:
            raise StopIteration
        if self._pending_result is not None:
            self._advance_from_result(self._pending_result)
            self._pending_result = None
        if not self._queue:
            raise StopIteration
        state, baton = self._queue.popleft()
        if not self._has_budget(state, baton):
            self._failed = StateError(
                error_type="fatal",
                message="Token budget exhausted for state.",
                retry_hint="Increase TokenBudget or adjust per-state estimate.",
            )
            raise StopIteration
        self._current_state = state
        self._current_baton = baton
        next_states = self._predict_next_states(state)
        return PipelineStep(state=state, baton=baton, next_states=next_states)

    def advance(self, result: StateResult[object]) -> None:
        self._pending_result = result

    @property
    def failed(self) -> StateError | None:
        return self._failed

    @property
    def trace(self) -> PipelineTrace:
        return self._trace

    def _record_and_route(self, state: StateSpec[object, object], baton: Baton[object]) -> None:
        join_targets = self._handle_join_groups(state, baton)
        outgoing = tuple(
            edge for edge in self._outgoing_edges(state) if edge.to_state.name not in join_targets
        )
        if self._route_guarded(outgoing, baton):
            return
        for edge in outgoing:
            self._queue.append((edge.to_state, baton))

    def _advance_from_result(self, result: StateResult[object]) -> None:
        if isinstance(result, Err):
            self._failed = result.error
            self._trace = self._trace.add(
                TraceEntry(
                    state_name=self._current_state.name if self._current_state else "<unknown>",
                    status="error",
                    finished_at=time.time(),
                    error=str(result.error),
                )
            )
            self._queue.clear()
            return
        if self._current_state is None:
            return
        self._trace = self._trace.add(
            TraceEntry(
                state_name=self._current_state.name,
                status="completed",
                finished_at=time.time(),
            )
        )
        self._record_and_route(self._current_state, result.value)
        self._current_state = None
        self._current_baton = None

    def _handle_join_groups(
        self,
        state: StateSpec[object, object],
        baton: Baton[object],
    ) -> set[str]:
        join_targets: set[str] = set()
        join_groups = self._join_groups_for_state(state)
        for group_id, group in join_groups:
            join_targets.add(group.target.name)
            if group_id in self._completed_joins:
                continue
            group_state = self._join_state.setdefault(group_id, {})
            group_state[state.name] = baton
            if group.join_spec.mode == "race" or all(
                source.name in group_state for source in group.sources
            ):
                self._completed_joins.add(group_id)
                self._queue.append((group.target, baton))
        return join_targets

    def _route_guarded(
        self,
        outgoing: tuple[Edge[object, object, object], ...],
        baton: Baton[object],
    ) -> bool:
        guarded = [edge for edge in outgoing if edge.guard is not None]
        if not guarded:
            return False
        for edge in guarded:
            if edge.guard is not None and edge.guard(baton):
                self._queue.append((edge.to_state, baton))
                return True
        defaults = [edge for edge in outgoing if edge.guard is None]
        if len(defaults) == 1:
            self._queue.append((defaults[0].to_state, baton))
        else:
            for edge in defaults:
                self._queue.append((edge.to_state, baton))
        return True

    def run_parallel(
        self,
        states: Iterable[StateSpec[object, object]],
        baton: Baton[object],
        executor: Callable[[StateSpec[object, object], Baton[object]], StateResult[object]],
        /,
        *,
        mode: str = "all",
        timeout_seconds: float | None = None,
    ) -> list[StateResult[object]]:
        """
        Run multiple states in parallel using a thread pool.

        Caller provides the executor function; this helper only manages
        concurrency and result collection.
        """
        state_list = list(states)
        if not state_list:
            return []
        with ThreadPoolExecutor() as pool:
            futures = [pool.submit(executor, state, baton) for state in state_list]
            if mode == "race":
                done, _ = wait(futures, timeout=timeout_seconds, return_when=FIRST_COMPLETED)
                return [future.result() for future in done]
            done, _ = wait(futures, timeout=timeout_seconds)
            return [future.result() for future in done]

    def _outgoing_edges(
        self,
        state: StateSpec[object, object],
    ) -> tuple[Edge[object, object, object], ...]:
        return tuple(edge for edge in self.pipeline.edges if edge.from_state.name == state.name)

    def _has_budget(self, state: StateSpec[object, object], baton: Baton[object]) -> bool:
        if baton.metadata.budget is None:
            return True
        estimate = baton.metadata.budget.estimate_for(state.name)
        return baton.metadata.budget.can_consume(estimate)

    def _join_groups_for_state(
        self,
        state: StateSpec[object, object],
    ) -> list[tuple[int, JoinGroup[object, object, object]]]:
        groups: list[tuple[int, JoinGroup[object, object, object]]] = []
        for idx, group in enumerate(self.pipeline.join_groups):
            if any(source.name == state.name for source in group.sources):
                groups.append((idx, group))
        return groups

    def _predict_next_states(
        self, state: StateSpec[object, object]
    ) -> tuple[StateSpec[object, object], ...]:
        outgoing = self._outgoing_edges(state)
        return tuple(edge.to_state for edge in outgoing)
