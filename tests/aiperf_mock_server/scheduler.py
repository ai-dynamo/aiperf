# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step-based batched scheduler for the mock server.

Models the dominant first-order behavior of a continuous-batching LLM server:
a global decode loop ticking every `step_ms`, admitting up to `max_batch_size`
decoders per step, plus a separate prefill chunk pool with bounded
`max_prefill_chunks_per_step`. Produces a real throughput-vs-concurrency
saturation knee at concurrency ~= max_batch_size.

Out of scope (would require design C): KV-block budget, preemption, swap.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from dataclasses import dataclass, field

from aiperf_mock_server.config import MockServerConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _DecodeWaiter:
    """A request waiting to have its next decode token admitted."""

    request_id: str
    event: asyncio.Event = field(default_factory=asyncio.Event)
    admitted_step: int = -1


@dataclass(slots=True)
class _PrefillWaiter:
    """A prefill chunk waiting for a slot."""

    request_id: str
    event: asyncio.Event = field(default_factory=asyncio.Event)
    admitted_step: int = -1


class BatchScheduler:
    """Step-based scheduler instance used as the mock server's process-wide scheduler.

    Use ``init_scheduler`` / ``get_scheduler`` / ``shutdown_scheduler`` for the
    process singleton. Direct instances own one background tick task and wake all
    queued decode/prefill waiters on ``stop()`` or per-request ``cancel()``.
    """

    def __init__(self, cfg: MockServerConfig) -> None:
        self._cfg = cfg
        self._step_index = 0
        self._tick_task: asyncio.Task | None = None
        self._decode_queue: deque[_DecodeWaiter] = deque()
        self._prefill_queue: deque[_PrefillWaiter] = deque()
        self._stopped = False

    @property
    def step_index(self) -> int:
        return self._step_index

    async def start(self) -> None:
        if not self._cfg.scheduler_enabled:
            raise RuntimeError(
                "BatchScheduler.start called but scheduler is not enabled"
            )
        if self._tick_task is not None:
            return
        self._tick_task = asyncio.create_task(
            self._tick_loop(), name="batch-scheduler-tick"
        )
        logger.info(
            "BatchScheduler started: step_ms=%.3f max_batch=%d max_prefill_chunks=%d "
            "prefill_chunk_tokens=%d",
            self._cfg.scheduler_step_ms,
            self._cfg.scheduler_max_batch_size,
            self._cfg.scheduler_max_prefill_chunks_per_step,
            self._cfg.scheduler_prefill_chunk_tokens,
        )

    async def stop(self) -> None:
        self._stopped = True
        if self._tick_task is not None:
            self._tick_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._tick_task
            self._tick_task = None
        for w in list(self._decode_queue):
            w.event.set()
        for w in list(self._prefill_queue):
            w.event.set()
        self._decode_queue.clear()
        self._prefill_queue.clear()

    async def next_decode_step(self, request_id: str) -> int:
        """Block until this request's next decode token is admitted."""
        if self._stopped:
            return self._step_index
        waiter = _DecodeWaiter(request_id=request_id)
        self._decode_queue.append(waiter)
        await waiter.event.wait()
        return waiter.admitted_step

    async def run_prefill(self, request_id: str, prompt_tokens: int) -> int:
        """Block until all prefill chunks for this prompt have been admitted."""
        if self._stopped or prompt_tokens <= 0:
            return 0
        chunks = max(
            1, _ceil_div(prompt_tokens, self._cfg.scheduler_prefill_chunk_tokens)
        )
        for _ in range(chunks):
            waiter = _PrefillWaiter(request_id=request_id)
            self._prefill_queue.append(waiter)
            await waiter.event.wait()
        return chunks

    def cancel(self, request_id: str) -> None:
        """Drop all pending waiters for this request and wake any awaiters.

        Used when a streaming client disconnects: we set the events so any
        currently-awaiting coroutines unblock cleanly, and remove the waiters
        from the queues so they don't artificially inflate queue depth (which
        would otherwise drive goodput-collapse and oversubscription accounting).
        """
        for queue in (self._decode_queue, self._prefill_queue):
            keep: deque = deque()
            for w in queue:
                if w.request_id == request_id:
                    w.event.set()
                else:
                    keep.append(w)
            queue.clear()
            queue.extend(keep)

    async def _tick_loop(self) -> None:
        step_seconds = self._cfg.scheduler_step_ms * 0.001
        loop = asyncio.get_running_loop()
        next_tick = loop.time() + step_seconds
        while not self._stopped:
            sleep_for = next_tick - loop.time()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            self._step_index += 1
            self._admit_prefill()
            self._admit_decode()
            next_tick += step_seconds

    def _admit_decode(self) -> None:
        budget = self._effective_decode_budget()
        while budget > 0 and self._decode_queue:
            w = self._decode_queue.popleft()
            w.admitted_step = self._step_index
            w.event.set()
            budget -= 1

    def _effective_decode_budget(self) -> int:
        """Per-step decode admit budget after goodput-collapse adjustment.

        When `scheduler_goodput_collapse_enabled`, the budget shrinks linearly
        once the queue grows past `threshold * max_batch_size`. At full
        collapse it bottoms at `floor * max_batch_size` (>=1). When disabled,
        always returns `max_batch_size`.
        """
        cfg = self._cfg
        max_batch = cfg.scheduler_max_batch_size
        if not cfg.scheduler_goodput_collapse_enabled:
            return max_batch
        ratio = len(self._decode_queue) / max_batch
        overload = ratio - cfg.scheduler_goodput_collapse_threshold
        if overload <= 0:
            return max_batch
        shrink = min(
            1.0 - cfg.scheduler_goodput_collapse_floor,
            overload * cfg.scheduler_goodput_collapse_slope,
        )
        return max(1, int(max_batch * (1.0 - shrink)))

    def _admit_prefill(self) -> None:
        budget = self._cfg.scheduler_max_prefill_chunks_per_step
        while budget > 0 and self._prefill_queue:
            w = self._prefill_queue.popleft()
            w.admitted_step = self._step_index
            w.event.set()
            budget -= 1


def _ceil_div(n: int, d: int) -> int:
    return -(-n // d)


_scheduler: BatchScheduler | None = None


def get_scheduler() -> BatchScheduler | None:
    return _scheduler


async def init_scheduler(cfg: MockServerConfig) -> BatchScheduler | None:
    """Lifespan hook: start the scheduler if enabled."""
    global _scheduler
    if not cfg.scheduler_enabled:
        _scheduler = None
        return None
    _scheduler = BatchScheduler(cfg)
    await _scheduler.start()
    return _scheduler


async def shutdown_scheduler() -> None:
    """Lifespan hook: stop the scheduler if running."""
    global _scheduler
    if _scheduler is not None:
        await _scheduler.stop()
        _scheduler = None
