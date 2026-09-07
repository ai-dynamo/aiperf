# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker in-flight credit tracking must survive seamless phase overlap.

``Credit.id`` restarts at 0 for every phase (``CreditCounter._dispatch_seq`` is
per-phase), so a bare-int ``credit_tasks`` key lets a profiling credit clobber a
still-draining warmup credit with the same id.
"""

from __future__ import annotations

import asyncio

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.credit.messages import CancelCredits
from aiperf.credit.structs import Credit, CreditContext
from aiperf.workers.worker import Worker, _credit_task_key


def _credit(
    credit_id: int,
    phase: CreditPhase = CreditPhase.PROFILING,
    phase_index: int | None = None,
) -> Credit:
    return Credit(
        id=credit_id,
        phase=phase,
        phase_index=phase_index,
        conversation_id=f"conv-{phase}-{credit_id}",
        x_correlation_id=f"corr-{phase}-{credit_id}",
        turn_index=0,
        num_turns=1,
        issued_at_ns=1_000_000,
    )


@pytest.fixture
async def mock_worker(benchmark_run, fake_tokenizer, skip_service_registration):
    """A started Worker with no SystemController behind it."""
    worker = Worker(run=benchmark_run, service_id="credit-task-key-worker")
    await worker.initialize()
    await worker.start()
    yield worker
    await worker.stop()


@pytest.mark.parametrize(
    "left,right",
    [
        param(
            _credit(3, CreditPhase.WARMUP),
            _credit(3, CreditPhase.PROFILING),
            id="same-id-different-phase",
        ),
        param(
            _credit(3, CreditPhase.PROFILING, 0),
            _credit(3, CreditPhase.PROFILING, 1),
            id="same-id-different-phase-index",
        ),
    ],
)  # fmt: skip
def test_credit_task_key_same_id_across_phases_returns_distinct_keys(
    left: Credit, right: Credit
) -> None:
    assert _credit_task_key(left) != _credit_task_key(right)


def test_credit_task_key_same_credit_returns_equal_keys() -> None:
    assert _credit_task_key(_credit(7)) == _credit_task_key(_credit(7))


@pytest.mark.asyncio
async def test_on_credit_drop_message_overlapping_phase_ids_keeps_both_tasks(
    mock_worker,
) -> None:
    """Fails on the bare-int key: the profiling credit overwrote the warmup entry."""
    started = asyncio.Event()

    async def _never_finishes(credit_context: CreditContext) -> None:
        started.set()
        await asyncio.Event().wait()

    mock_worker._on_credit_drop_message_task = _never_finishes

    mock_worker._schedule_credit_drop_task(_credit(0, CreditPhase.WARMUP))
    mock_worker._schedule_credit_drop_task(_credit(0, CreditPhase.PROFILING))
    await started.wait()

    assert len(mock_worker.credit_tasks) == 2
    assert set(mock_worker.credit_tasks) == {
        (CreditPhase.WARMUP, None, 0),
        (CreditPhase.PROFILING, None, 0),
    }
    for task in list(mock_worker.credit_tasks.values()):
        task.cancel()


@pytest.mark.asyncio
async def test_on_credit_drop_message_task_done_only_pops_its_own_phase(
    mock_worker,
) -> None:
    """A warmup completion must not evict the live profiling task with the same id."""

    async def _never_finishes(credit_context: CreditContext) -> None:
        await asyncio.Event().wait()

    mock_worker._on_credit_drop_message_task = _never_finishes

    warmup = _credit(0, CreditPhase.WARMUP)
    profiling = _credit(0, CreditPhase.PROFILING)
    mock_worker._schedule_credit_drop_task(warmup)
    mock_worker._schedule_credit_drop_task(profiling)

    warmup_task = mock_worker.credit_tasks[_credit_task_key(warmup)]
    warmup_task.cancel()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert _credit_task_key(profiling) in mock_worker.credit_tasks
    assert not mock_worker.credit_tasks[_credit_task_key(profiling)].done()
    mock_worker.credit_tasks[_credit_task_key(profiling)].cancel()


@pytest.mark.asyncio
async def test_on_cancel_credits_message_cancels_every_phase_holding_the_id(
    mock_worker,
) -> None:
    """Router-side cancellation is global, so a bare id cancels across phases."""

    async def _never_finishes(credit_context: CreditContext) -> None:
        await asyncio.Event().wait()

    mock_worker._on_credit_drop_message_task = _never_finishes

    warmup = _credit(0, CreditPhase.WARMUP)
    profiling = _credit(0, CreditPhase.PROFILING)
    untouched = _credit(1, CreditPhase.PROFILING)
    for credit in (warmup, profiling, untouched):
        mock_worker._schedule_credit_drop_task(credit)

    warmup_task = mock_worker.credit_tasks[_credit_task_key(warmup)]
    profiling_task = mock_worker.credit_tasks[_credit_task_key(profiling)]
    untouched_task = mock_worker.credit_tasks[_credit_task_key(untouched)]

    await mock_worker._on_cancel_credits_message(CancelCredits(credit_ids={0}))

    assert warmup_task.cancelling() or warmup_task.cancelled()
    assert profiling_task.cancelling() or profiling_task.cancelled()
    assert not (untouched_task.cancelling() or untouched_task.cancelled())
    untouched_task.cancel()


@pytest.mark.asyncio
async def test_on_cancel_credits_message_unknown_id_is_a_noop(mock_worker) -> None:
    await mock_worker._on_cancel_credits_message(CancelCredits(credit_ids={999}))

    assert mock_worker.credit_tasks == {}
