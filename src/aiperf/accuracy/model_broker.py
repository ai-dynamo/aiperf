# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harness-neutral bridge from canonical Python agents to Rust inference.

Agent harnesses expose both asynchronous model APIs (Harbor) and synchronous
ones that run in an evaluator-owned thread (AgentLab).  :class:`ModelCallBroker`
normalizes both forms into the same JSONL ``model_call`` event.  It never opens
a socket or invokes a model client: only Rust can resolve the returned future.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from aiperf.accuracy.agentic import (
    AgenticEvent,
    AgenticModelCall,
    AgenticModelResult,
    EventQueue,
)


class RustInferenceError(RuntimeError):
    """A Rust-owned inference call failed before producing a usable response."""


class ModelCallBroker:
    """Correlate harness model calls with terminal results submitted by Rust."""

    def __init__(
        self,
        events: EventQueue,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._events = events
        self._loop = loop or _running_loop_or_none()
        self._pending: dict[str, tuple[str, asyncio.Future[AgenticModelResult]]] = {}
        self._turns: dict[str, int] = {}
        self._closed = False

    def model_call_count(self, episode_id: str) -> int:
        """Return the number of calls authored for one episode."""
        return self._turns.get(episode_id, 0)

    async def call(
        self,
        *,
        episode_id: str,
        model: str | None,
        prompt: str,
        messages: list[dict[str, Any]],
        generation: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        response_format: Any | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> AgenticModelResult:
        """Publish one model request and await its Rust-owned terminal result."""
        loop = asyncio.get_running_loop()
        self._bind_loop(loop)
        if self._closed:
            raise RuntimeError("agentic model-call broker is closed")
        turn_index = self._turns.get(episode_id, 0)
        self._turns[episode_id] = turn_index + 1
        call_id = f"{episode_id}:call:{turn_index:08d}"
        if call_id in self._pending:
            raise RuntimeError(f"duplicate agentic model call id {call_id!r}")
        future = loop.create_future()
        self._pending[call_id] = (episode_id, future)
        await self._events.put(
            AgenticEvent.call(
                AgenticModelCall(
                    episode_id=episode_id,
                    call_id=call_id,
                    turn_index=turn_index,
                    model=model,
                    prompt=prompt,
                    messages=messages,
                    generation=generation,
                    tools=list(tools or []),
                    tool_choice=tool_choice,
                    response_format=response_format,
                    extra_body=dict(extra_body or {}),
                )
            )
        )
        try:
            return await future
        finally:
            self._pending.pop(call_id, None)

    def call_sync(self, **kwargs: Any) -> AgenticModelResult:
        """Block a harness thread on the same event-loop-owned broker call.

        The worker event loop must have created or previously bound the broker.
        Calling this method from that loop's own thread would deadlock and is
        rejected explicitly.
        """
        loop = self._loop
        if loop is None:
            raise RuntimeError(
                "synchronous agentic model call requires an event-loop-bound broker"
            )
        if _running_loop_or_none() is loop:
            raise RuntimeError(
                "synchronous agentic model call cannot block its broker event loop"
            )
        return asyncio.run_coroutine_threadsafe(self.call(**kwargs), loop).result()

    def submit(self, result: AgenticModelResult) -> None:
        """Resolve one outstanding callback from a Rust terminal result."""
        self._require_bound_loop()
        pending = self._pending.get(result.call_id)
        if pending is None:
            raise KeyError(f"unknown or already completed call_id {result.call_id!r}")
        episode_id, future = pending
        if result.episode_id != episode_id:
            raise ValueError(
                f"call {result.call_id!r} belongs to episode {episode_id!r}, "
                f"not {result.episode_id!r}"
            )
        if future.done():
            raise ValueError(f"call {result.call_id!r} was submitted more than once")
        future.set_result(result)

    def fail_episode(self, episode_id: str, error: BaseException) -> None:
        """Fail every outstanding model call for a cancelled episode."""
        self._require_bound_loop()
        for pending_episode, future in self._pending.values():
            if pending_episode == episode_id and not future.done():
                future.set_exception(error)

    def close(self) -> None:
        """Fail outstanding calls so no harness remains blocked on shutdown."""
        self._require_bound_loop()
        self._closed = True
        for _, future in self._pending.values():
            if not future.done():
                future.set_exception(RuntimeError("agentic model-call broker closed"))

    def _bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("agentic model-call broker cannot cross event loops")

    def _require_bound_loop(self) -> None:
        loop = self._loop
        if loop is None:
            raise RuntimeError(
                "agentic model-call broker is not bound to an event loop"
            )
        if _running_loop_or_none() is not loop:
            raise RuntimeError(
                "agentic model-call broker mutation must run on its event loop"
            )


_BROKERS: dict[str, ModelCallBroker] = {}


def register_broker(broker: ModelCallBroker) -> str:
    """Register a process-local broker and return its opaque lookup id."""
    broker_id = uuid.uuid4().hex
    _BROKERS[broker_id] = broker
    return broker_id


def unregister_broker(broker_id: str) -> None:
    """Remove a broker after all harness episodes have drained."""
    _BROKERS.pop(broker_id, None)


def broker_for_id(broker_id: str) -> ModelCallBroker:
    """Resolve a process-local broker used by a picklable harness adapter."""
    try:
        return _BROKERS[broker_id]
    except KeyError as error:
        raise RuntimeError(f"unknown AIPerf model-call broker {broker_id!r}") from error


def _running_loop_or_none() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None
