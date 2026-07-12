# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed pipe-backed evaluator host with a strict operation ledger.

``PipeEvaluationHost`` has no networking, subprocess, Docker, credential, or
endpoint implementation.  It converts evaluator-owned semantic work into
events for Rust and resolves the corresponding futures only from validated
``HostOperationEvent`` values returned over the supervised control pipe.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from aiperf.accuracy.evaluation.contracts import (
    EvaluationQueueCredits,
    HostOperationCancelRequest,
    HostOperationDisposition,
    HostOperationEvent,
    HostOperationRequest,
    HostOperationTerminal,
    HostOperationUsage,
    JsonValue,
)

EvaluationEventEmitter = Callable[[dict[str, JsonValue]], Awaitable[None]]


class EvaluationHost(Protocol):
    """Replaceable typed evaluator host boundary."""

    async def execute(self, request: HostOperationRequest) -> HostOperationTerminal:
        """Execute one terminal or buffered semantic host operation."""

    def stream(
        self, request: HostOperationRequest
    ) -> AsyncIterator[HostOperationEvent]:
        """Execute one true-streaming semantic host operation."""

    async def cancel_operation(
        self, operation_id: str, semantic_attempt_id: str, reason: str
    ) -> HostOperationTerminal:
        """Request cancellation and await Rust's terminal acknowledgement."""


@dataclass
class _OperationState:
    request: HostOperationRequest
    terminal: asyncio.Future[HostOperationTerminal]
    stream_queue: asyncio.Queue[HostOperationEvent] = field(
        default_factory=asyncio.Queue
    )
    next_stream_sequence: int = 0
    usage_seen: bool = False
    cancel_emitted: bool = False
    cancellation_ack: asyncio.Future[bool] | None = None
    terminal_value: HostOperationTerminal | None = None


class PipeEvaluationHost:
    """Bounded, cancellation-safe implementation of the evaluator host seam."""

    def __init__(
        self,
        emit: EvaluationEventEmitter,
        credits: EvaluationQueueCredits,
    ) -> None:
        self._emit = emit
        self._credits = credits
        self._condition = asyncio.Condition()
        self._operations: dict[str, _OperationState] = {}
        self._active_by_unit: Counter[str] = Counter()
        self._active_count = 0
        self._buffered_stream_events = 0
        self._closed = False

    @property
    def outstanding_operation_ids(self) -> tuple[str, ...]:
        """Return active operation IDs in deterministic insertion order."""
        return tuple(
            operation_id
            for operation_id, state in self._operations.items()
            if state.terminal_value is None
        )

    @property
    def is_drained(self) -> bool:
        """Return whether every emitted host operation is terminal."""
        return self._active_count == 0

    @property
    def producer_capacity(self) -> int:
        """Return the negotiated Python producer-operation ceiling."""
        return self._credits.host_operations

    def remaining_credits(self) -> EvaluationQueueCredits:
        """Snapshot worker-side remaining credits for ``poll_events``."""
        return EvaluationQueueCredits(
            units=self._credits.units,
            host_operations=self._credits.host_operations - self._active_count,
            host_operations_per_unit=min(
                self._credits.host_operations_per_unit,
                self._credits.host_operations - self._active_count,
            ),
            stream_events=self._credits.stream_events - self._buffered_stream_events,
            sandboxes=self._credits.sandboxes,
            processes=self._credits.processes,
            artifacts=self._credits.artifacts,
            artifact_bytes=self._credits.artifact_bytes,
        )

    async def execute(self, request: HostOperationRequest) -> HostOperationTerminal:
        """Emit one operation and await exactly one terminal result."""
        state = await self._begin(request)
        try:
            terminal = await state.terminal
        except asyncio.CancelledError:
            await self._cancel_after_local_cancellation(state)
            raise
        if request.response_mode.value == "terminal" and state.next_stream_sequence:
            raise RuntimeError("terminal-mode host operation received stream deltas")
        return terminal

    async def _stream(
        self, request: HostOperationRequest
    ) -> AsyncIterator[HostOperationEvent]:
        if request.response_mode.value != "streaming":
            raise ValueError("stream() requires response_mode='streaming'")
        state = await self._begin(request)
        try:
            while True:
                event = await state.stream_queue.get()
                if event.kind == "stream_delta":
                    async with self._condition:
                        self._buffered_stream_events -= 1
                        self._condition.notify_all()
                    yield event
                    continue
                if event.kind == "usage":
                    yield event
                    continue
                assert event.terminal is not None
                yield event
                return
        except asyncio.CancelledError:
            await self._cancel_after_local_cancellation(state)
            raise

    def stream(
        self, request: HostOperationRequest
    ) -> AsyncIterator[HostOperationEvent]:
        """Emit one streaming operation and yield typed deltas/usage/terminal."""
        return self._stream(request)

    async def submit_events(
        self, events: Sequence[HostOperationEvent]
    ) -> tuple[str, ...]:
        """Validate and apply a bounded Rust-to-provider event batch."""
        if len(events) > self._credits.stream_events:
            raise ValueError("host event batch exceeds negotiated stream-event credit")
        accepted: list[str] = []
        for event in events:
            state = self._operations.get(event.operation_id)
            if state is None:
                raise ValueError(
                    f"host event references unknown operation {event.operation_id!r}"
                )
            if (
                state.terminal_value is not None
                and event.kind != "cancellation_acknowledged"
            ):
                raise ValueError(
                    f"late/duplicate event for terminal operation {event.operation_id!r}"
                )
            if event.kind == "stream_delta":
                if state.request.response_mode.value != "streaming":
                    raise ValueError(
                        "stream delta delivered to terminal-mode operation"
                    )
                if event.stream_sequence != state.next_stream_sequence:
                    raise ValueError(
                        "host stream sequence regression/gap: expected "
                        f"{state.next_stream_sequence}, got {event.stream_sequence}"
                    )
                async with self._condition:
                    while self._buffered_stream_events >= self._credits.stream_events:
                        await self._condition.wait()
                    self._buffered_stream_events += 1
                state.next_stream_sequence += 1
                state.stream_queue.put_nowait(event)
            elif event.kind == "usage":
                if state.usage_seen:
                    raise ValueError("host operation received duplicate usage event")
                state.usage_seen = True
                if state.request.response_mode.value == "streaming":
                    state.stream_queue.put_nowait(event)
            elif event.kind == "terminal":
                terminal = event.terminal
                assert terminal is not None
                if (
                    terminal.semantic_attempt_id
                    != state.request.context.semantic_attempt_id
                ):
                    raise ValueError("host terminal semantic-attempt identity drift")
                state.terminal_value = terminal
                if not state.terminal.done():
                    state.terminal.set_result(terminal)
                if state.request.response_mode.value == "streaming":
                    state.stream_queue.put_nowait(event)
                await self._release(state)
            elif event.kind == "cancellation_acknowledged":
                if not state.cancel_emitted or state.cancellation_ack is None:
                    raise ValueError("unsolicited host cancellation acknowledgement")
                if (
                    event.semantic_attempt_id
                    != state.request.context.semantic_attempt_id
                ):
                    raise ValueError(
                        "cancellation acknowledgement attempt identity drift"
                    )
                if state.cancellation_ack.done():
                    raise ValueError("duplicate host cancellation acknowledgement")
                assert event.already_terminal is not None
                state.cancellation_ack.set_result(event.already_terminal)
            else:  # pragma: no cover - constructor/parser already closes this set.
                raise ValueError(f"unknown host event kind {event.kind!r}")
            accepted.append(event.operation_id)
        return tuple(accepted)

    async def cancel_operation(
        self, operation_id: str, semantic_attempt_id: str, reason: str
    ) -> HostOperationTerminal:
        """Emit one idempotent cancellation request and await its terminal ack."""
        state = self._operations.get(operation_id)
        if state is None:
            raise ValueError(f"cannot cancel unknown host operation {operation_id!r}")
        if state.request.context.semantic_attempt_id != semantic_attempt_id:
            raise ValueError("host cancellation semantic-attempt identity drift")
        if state.terminal_value is not None:
            return state.terminal_value
        if not state.cancel_emitted:
            state.cancel_emitted = True
            state.cancellation_ack = asyncio.get_running_loop().create_future()
            request = HostOperationCancelRequest(
                operation_id=operation_id,
                semantic_attempt_id=semantic_attempt_id,
                reason=reason,
            )
            await self._emit(
                {
                    "kind": "host_operation_cancel_requested",
                    "request": request.to_wire(),
                }
            )
        terminal = await asyncio.shield(state.terminal)
        if terminal.disposition is HostOperationDisposition.CANCELLED:
            return terminal
        assert state.cancellation_ack is not None
        await asyncio.shield(state.cancellation_ack)
        return terminal

    async def cancel_unit(self, unit_id: str, reason: str) -> None:
        """Cancel every active operation owned by a unit and await all acks."""
        states = [
            state
            for state in self._operations.values()
            if state.request.context.unit_id == unit_id and state.terminal_value is None
        ]
        if not states:
            return
        await asyncio.gather(
            *(
                self.cancel_operation(
                    state.request.operation_id,
                    state.request.context.semantic_attempt_id,
                    reason,
                )
                for state in states
            )
        )

    async def close(self) -> None:
        """Reject new work and require all previously emitted work to be terminal."""
        self._closed = True
        if not self.is_drained:
            raise RuntimeError(
                "cannot close PipeEvaluationHost with outstanding operations"
            )

    async def _begin(self, request: HostOperationRequest) -> _OperationState:
        if self._closed:
            raise RuntimeError("PipeEvaluationHost is closed")
        if request.operation_id in self._operations:
            raise ValueError(f"duplicate host operation ID {request.operation_id!r}")
        unit_id = request.context.unit_id
        async with self._condition:
            while (
                self._active_count >= self._credits.host_operations
                or self._active_by_unit[unit_id]
                >= self._credits.host_operations_per_unit
            ):
                await self._condition.wait()
                if self._closed:
                    raise RuntimeError(
                        "PipeEvaluationHost closed while waiting for credit"
                    )
            loop = asyncio.get_running_loop()
            state = _OperationState(request=request, terminal=loop.create_future())
            self._operations[request.operation_id] = state
            self._active_count += 1
            self._active_by_unit[unit_id] += 1
        try:
            await self._emit(
                {"kind": "host_operation_requested", "request": request.to_wire()}
            )
        except BaseException:
            self._operations.pop(request.operation_id, None)
            await self._release(state)
            raise
        return state

    async def _release(self, state: _OperationState) -> None:
        async with self._condition:
            self._active_count -= 1
            unit_id = state.request.context.unit_id
            self._active_by_unit[unit_id] -= 1
            if self._active_by_unit[unit_id] == 0:
                del self._active_by_unit[unit_id]
            self._condition.notify_all()

    async def _cancel_after_local_cancellation(self, state: _OperationState) -> None:
        if state.terminal_value is not None:
            return
        await self.cancel_operation(
            state.request.operation_id,
            state.request.context.semantic_attempt_id,
            "provider_task_cancelled",
        )


def terminal_result_payload(terminal: HostOperationTerminal) -> JsonValue:
    """Return completed payload or raise a stable host infrastructure exception."""
    if terminal.disposition is HostOperationDisposition.COMPLETED:
        return terminal.result
    if terminal.error is not None:
        raise HostOperationFailure(terminal.error.stage, terminal.error.error_kind)
    raise HostOperationFailure("host_operation", terminal.disposition.value)


class HostOperationFailure(RuntimeError):
    """Restricted provider-facing host failure without secret diagnostics."""

    def __init__(self, stage: str, error_kind: str) -> None:
        self.stage = stage
        self.error_kind = error_kind
        super().__init__(f"host operation failed at {stage}: {error_kind}")


def empty_usage() -> HostOperationUsage:
    """Construct an empty authoritative usage projection."""
    return HostOperationUsage()
