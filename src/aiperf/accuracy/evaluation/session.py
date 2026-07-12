# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neutral evaluator session seam and supervised background runtime."""

from __future__ import annotations

import asyncio
import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol

from aiperf.accuracy.evaluation.canonical import canonical_dumps
from aiperf.accuracy.evaluation.contracts import (
    CaseOccurrenceDescriptor,
    CaseOutcome,
    CaseOutcomeKind,
    EvaluationBundle,
    EvaluationError,
    EvaluationIdentity,
    EvaluationPlan,
    EvaluationQueueCredits,
    EvaluationUnitPage,
    ExecutionUnitOccurrence,
    HostOperationEvent,
    JsonValue,
    SchedulingMode,
    UnitOccurrenceRequest,
)
from aiperf.accuracy.evaluation.host import PipeEvaluationHost


class EvaluationSession(Protocol):
    """Provider-neutral frozen evaluation session."""

    session_id: str
    identity: EvaluationIdentity
    plan: EvaluationPlan
    finite_units: tuple[ExecutionUnitOccurrence, ...]

    async def instantiate_units(
        self, requests: Sequence[UnitOccurrenceRequest]
    ) -> Sequence[ExecutionUnitOccurrence]:
        """Materialize Rust-scheduled occurrences deterministically."""

    async def run_unit(
        self, unit_id: str, host: PipeEvaluationHost
    ) -> Sequence[CaseOutcome]:
        """Run the provider's complete semantic lifecycle for one unit."""

    async def cancel_unit(self, unit_id: str) -> None:
        """Cancel provider-local work for one unit idempotently."""

    def record_outcomes(self, outcomes: Sequence[CaseOutcome]) -> None:
        """Record canonical terminals for provider aggregation/finalization."""

    async def finalize(self) -> EvaluationBundle:
        """Aggregate and write the canonical provider bundle candidate."""

    async def close(self) -> None:
        """Tear down every local semantic task and writer idempotently."""


class BaseEvaluationSession(ABC):
    """Invariant-heavy base for stock provider adapters.

    Concrete providers own case selection, prompts, solving, scoring, reducers,
    and artifacts.  This class owns only occurrence identity and terminal
    uniqueness, leaving the next provider implementation replaceable.
    """

    def __init__(
        self,
        session_id: str,
        identity: EvaluationIdentity,
        plan: EvaluationPlan,
        finite_units: Sequence[ExecutionUnitOccurrence],
    ) -> None:
        self.session_id = session_id
        self.identity = identity
        self.plan = plan
        self.finite_units = tuple(finite_units)
        self._units: dict[str, ExecutionUnitOccurrence] = {
            item.unit_id: item for item in self.finite_units
        }
        if len(self._units) != len(self.finite_units):
            raise ValueError("finite provider session contains duplicate unit IDs")
        self._outcomes: dict[str, CaseOutcome] = {}
        self._closed = False
        self._cancelled_units: set[str] = set()

    @property
    def units(self) -> tuple[ExecutionUnitOccurrence, ...]:
        """Return every currently materialized unit in canonical order."""
        return tuple(self._units.values())

    @property
    def outcomes(self) -> tuple[CaseOutcome, ...]:
        """Return recorded outcomes in canonical case order."""
        return tuple(
            self._outcomes[case.case_id]
            for unit in self._units.values()
            for case in unit.cases
            if case.case_id in self._outcomes
        )

    async def instantiate_units(
        self, requests: Sequence[UnitOccurrenceRequest]
    ) -> Sequence[ExecutionUnitOccurrence]:
        """Materialize idempotent deterministic Rust-scheduled occurrences."""
        if self.plan.scheduling_mode is not SchedulingMode.RUST_OCCURRENCES:
            raise ValueError("finite evaluation session rejects instantiate_units")
        templates = {
            template.unit_template_id: template
            for template in self.identity.unit_templates
        }
        cases = {
            template.template_id: template for template in self.identity.case_templates
        }
        result: list[ExecutionUnitOccurrence] = []
        for request in requests:
            template = templates.get(request.unit_template_id)
            if template is None:
                raise ValueError(f"unknown unit template {request.unit_template_id!r}")
            identity_value = {
                "session_id": self.session_id,
                "unit_template_id": request.unit_template_id,
                "phase_id": request.phase_id,
                "issue_ordinal": request.issue_ordinal,
                "cycle_index": request.cycle_index,
            }
            digest = hashlib.sha256(canonical_dumps(identity_value)).hexdigest()
            unit_id = f"unit-{digest}"
            concrete_cases = tuple(
                CaseOccurrenceDescriptor(
                    case_id=f"case-{hashlib.sha256(canonical_dumps({**identity_value, 'template_id': template_id})).hexdigest()}",
                    template_id=template_id,
                    issue_ordinal=request.issue_ordinal,
                    phase_id=request.phase_id,
                    cycle_index=request.cycle_index,
                )
                for template_id in template.case_template_ids
                if template_id in cases
            )
            occurrence = ExecutionUnitOccurrence(
                unit_id=unit_id,
                unit_template_id=request.unit_template_id,
                cases=concrete_cases,
            )
            prior = self._units.get(unit_id)
            if prior is not None and prior != occurrence:
                raise RuntimeError("deterministic occurrence identity collision")
            self._units.setdefault(unit_id, occurrence)
            result.append(self._units[unit_id])
        return tuple(result)

    def record_outcomes(self, outcomes: Sequence[CaseOutcome]) -> None:
        """Record exactly one terminal for each known concrete case."""
        known = {case.case_id for unit in self._units.values() for case in unit.cases}
        for outcome in outcomes:
            if outcome.case_id not in known:
                raise ValueError(f"outcome references unknown case {outcome.case_id!r}")
            if outcome.case_id in self._outcomes:
                raise ValueError(f"duplicate terminal for case {outcome.case_id!r}")
            self._outcomes[outcome.case_id] = outcome

    async def cancel_unit(self, unit_id: str) -> None:
        """Mark provider-local cancellation; concrete sessions may extend it."""
        if unit_id not in self._units:
            raise ValueError(f"unknown unit {unit_id!r}")
        self._cancelled_units.add(unit_id)

    async def close(self) -> None:
        """Idempotently mark the provider session closed."""
        self._closed = True

    @abstractmethod
    async def run_unit(
        self, unit_id: str, host: PipeEvaluationHost
    ) -> Sequence[CaseOutcome]:
        """Run one provider-owned unit lifecycle."""

    @abstractmethod
    async def finalize(self) -> EvaluationBundle:
        """Write and return the provider's complete final candidate."""


class SessionRuntime:
    """Background task, event, credit, and cancellation runtime for a session."""

    def __init__(self, session: EvaluationSession) -> None:
        self.session = session
        self._events: asyncio.Queue[dict[str, JsonValue]] = asyncio.Queue(
            maxsize=session.plan.queue_credits.stream_events
        )
        self.host = PipeEvaluationHost(self._enqueue_event, session.plan.queue_credits)
        self._sequence = 0
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._terminal_units: set[str] = set()
        self._cancel_requested: set[str] = set()
        self._fatal: BaseException | None = None
        self._finalized = False
        self._closed = False

    @property
    def units(self) -> tuple[ExecutionUnitOccurrence, ...]:
        """Return materialized units from the provider session."""
        return (
            self.session.finite_units
            if not isinstance(self.session, BaseEvaluationSession)
            else self.session.units
        )

    async def next_units(self, offset: int, limit: int) -> EvaluationUnitPage:
        """Return one bounded page for a finite provider plan."""
        if self.session.plan.scheduling_mode is not SchedulingMode.FINITE:
            raise ValueError("rust_occurrences session rejects next_units")
        if offset < 0 or limit <= 0:
            raise ValueError("next_units offset/limit is invalid")
        items = self.units[offset : offset + limit]
        next_offset = offset + len(items)
        return EvaluationUnitPage(
            items=items,
            next_offset=next_offset,
            done=next_offset == len(self.units),
        )

    async def instantiate_units(
        self, requests: Sequence[UnitOccurrenceRequest]
    ) -> tuple[ExecutionUnitOccurrence, ...]:
        """Delegate deterministic occurrence instantiation."""
        units = tuple(await self.session.instantiate_units(requests))
        if len({item.unit_id for item in units}) != len(units):
            raise ValueError("instantiate_units returned duplicate unit IDs")
        return units

    async def start_units(self, unit_ids: Sequence[str]) -> tuple[str, ...]:
        """Create background unit tasks and acknowledge without awaiting them."""
        self._raise_fatal()
        if len(unit_ids) > self.session.plan.queue_credits.units:
            raise ValueError("start_units exceeds negotiated unit credit")
        known = {item.unit_id for item in self.units}
        started: list[str] = []
        for unit_id in unit_ids:
            if unit_id not in known:
                raise ValueError(f"start_units references unknown unit {unit_id!r}")
            if unit_id in self._tasks or unit_id in self._terminal_units:
                raise ValueError(f"unit {unit_id!r} was already started")
            active = sum(not task.done() for task in self._tasks.values())
            if active >= self.session.plan.queue_credits.units:
                raise ValueError("active units exceed negotiated credit")
            self._tasks[unit_id] = asyncio.create_task(
                self._run_unit(unit_id), name=f"evaluation-unit:{unit_id}"
            )
            started.append(unit_id)
        return tuple(started)

    async def submit_host_events(
        self, events: Sequence[HostOperationEvent]
    ) -> tuple[str, ...]:
        """Apply typed Rust host events to the pipe ledger."""
        self._raise_fatal()
        return await self.host.submit_events(events)

    async def cancel_units(self, unit_ids: Sequence[str]) -> tuple[str, ...]:
        """Idempotently cancel units and keep host cancellations live to ack."""
        known = {item.unit_id: item for item in self.units}
        cancelled: list[str] = []
        for unit_id in unit_ids:
            unit = known.get(unit_id)
            if unit is None:
                raise ValueError(f"cancel_units references unknown unit {unit_id!r}")
            if unit_id in self._cancel_requested:
                cancelled.append(unit_id)
                continue
            self._cancel_requested.add(unit_id)
            await self.session.cancel_unit(unit_id)
            task = self._tasks.get(unit_id)
            if task is None:
                outcomes = tuple(
                    CaseOutcome(
                        case_id=case.case_id,
                        kind=CaseOutcomeKind.CANCELLED,
                        cancellation_stage="before_start",
                        cancellation_reason="rust_cancelled_unit",
                    )
                    for case in unit.cases
                )
                self.session.record_outcomes(outcomes)
                for outcome in outcomes:
                    await self._enqueue_event(
                        {"kind": "case_terminal", "outcome": outcome.to_wire()}
                    )
                self._terminal_units.add(unit_id)
            elif not task.done():
                task.cancel()
            cancelled.append(unit_id)
        return tuple(cancelled)

    async def poll_events(
        self, limit: int, wait_ms: int
    ) -> tuple[tuple[dict[str, JsonValue], ...], int, bool, EvaluationQueueCredits]:
        """Long-poll a bounded ordered semantic event batch."""
        self._raise_fatal()
        if limit <= 0 or limit > self.session.plan.queue_credits.stream_events:
            raise ValueError("poll_events limit exceeds negotiated credit")
        if wait_ms < 0:
            raise ValueError("poll_events wait_ms must be non-negative")
        events: list[dict[str, JsonValue]] = []
        if self._events.empty() and wait_ms:
            try:
                first = await asyncio.wait_for(
                    self._events.get(), timeout=wait_ms / 1000.0
                )
                events.append(first)
            except TimeoutError:
                pass
        while len(events) < limit and not self._events.empty():
            events.append(self._events.get_nowait())
        self._raise_fatal()
        return (
            tuple(events),
            self._sequence,
            self.is_drained,
            self.host.remaining_credits(),
        )

    @property
    def is_drained(self) -> bool:
        """True only when all materialized units and host operations terminal."""
        return (
            len(self._terminal_units) == len(self.units)
            and self.host.is_drained
            and all(task.done() for task in self._tasks.values())
        )

    async def finalize(self) -> EvaluationBundle:
        """Finalize only after exact unit/operation drainage."""
        self._raise_fatal()
        if self._finalized:
            raise ValueError("finalize_session may be called exactly once")
        if not self.is_drained:
            raise RuntimeError(
                "cannot finalize before every unit/operation is terminal"
            )
        candidate = await self.session.finalize()
        self._finalized = True
        return candidate

    async def close(self) -> None:
        """Quiesce the complete provider task tree idempotently."""
        if self._closed:
            return
        if any(not task.done() for task in self._tasks.values()):
            raise RuntimeError("cannot shut down with live evaluation unit tasks")
        if not self.host.is_drained:
            raise RuntimeError("cannot shut down with live host operations")
        await self.host.close()
        await self.session.close()
        self._closed = True

    async def _run_unit(self, unit_id: str) -> None:
        unit = next(item for item in self.units if item.unit_id == unit_id)
        try:
            outcomes = tuple(await self.session.run_unit(unit_id, self.host))
            expected = {case.case_id for case in unit.cases}
            actual = {outcome.case_id for outcome in outcomes}
            if actual != expected or len(actual) != len(outcomes):
                raise RuntimeError(
                    "provider unit omitted, duplicated, or added case outcomes"
                )
        except asyncio.CancelledError:
            await self.host.cancel_unit(unit_id, "rust_cancelled_unit")
            existing = (
                {item.case_id for item in self.session.outcomes}
                if isinstance(self.session, BaseEvaluationSession)
                else set()
            )
            outcomes = tuple(
                CaseOutcome(
                    case_id=case.case_id,
                    kind=CaseOutcomeKind.CANCELLED,
                    cancellation_stage="running",
                    cancellation_reason="rust_cancelled_unit",
                )
                for case in unit.cases
                if case.case_id not in existing
            )
        except BaseException as error:
            self._fatal = error
            return
        try:
            self.session.record_outcomes(outcomes)
            for outcome in outcomes:
                await self._enqueue_event(
                    {"kind": "case_terminal", "outcome": outcome.to_wire()}
                )
            self._terminal_units.add(unit_id)
        except BaseException as error:
            self._fatal = error

    async def _enqueue_event(self, event: dict[str, JsonValue]) -> None:
        self._sequence += 1
        key = hashlib.sha256(
            canonical_dumps(
                {
                    "session": self.session.session_id,
                    "sequence": self._sequence,
                    "event": event,
                }
            )
        ).hexdigest()
        await self._events.put(
            {
                "sequence": self._sequence,
                "idempotency_key": f"event-{key}",
                "event": event,
            }
        )

    def _raise_fatal(self) -> None:
        if self._fatal is not None:
            raise RuntimeError(
                f"evaluation session failed: {type(self._fatal).__name__}"
            ) from self._fatal


def infrastructure_outcome(case_id: str, stage: str, error_kind: str) -> CaseOutcome:
    """Construct a redacted infrastructure terminal for provider-owned use."""
    return CaseOutcome(
        case_id=case_id,
        kind=CaseOutcomeKind.INFRASTRUCTURE_ERROR,
        error=EvaluationError(
            stage=stage,
            error_kind=error_kind,
            retryable=False,
            message="provider infrastructure prevented semantic completion",
        ),
    )
