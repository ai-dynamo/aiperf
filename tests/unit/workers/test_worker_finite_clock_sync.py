# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for distributed Kubernetes worker clock synchronization in finite replay."""

import asyncio
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.messages import BaseServiceErrorMessage
from aiperf.credit.messages import TimePong, WorkerDispatchable
from aiperf.credit.structs import Credit, CreditContext
from aiperf.workers.clock_offset_tracker import ClockOffsetTracker
from aiperf.workers.worker import Worker


@pytest.mark.asyncio
async def test_finite_preflight_handle_pong_feeds_clock_observation() -> None:
    """When finite_preflight is enabled, pong router timestamps calibrate the tracker."""
    tracker = ClockOffsetTracker(finite_preflight=True, min_samples=1)
    tracker._pending_pong_sequence = 1
    tracker._pending_pong_future = asyncio.get_running_loop().create_future()

    pong = TimePong(
        sequence=1,
        sent_at_ns=1_000_000,
        router_sent_wall_ns=1_000_000_000,
    )
    with patch.object(ClockOffsetTracker, "_now_ns", return_value=1_000_100_000):
        tracker.handle_pong(pong)

    assert len(tracker._window) == 1
    assert tracker.sample_count == 1
    assert tracker.offset_ns == 100_000
    assert tracker._pending_pong_future.result() == pong
    assert tracker.is_currently_calibrated is True


@pytest.mark.asyncio
async def test_worker_ready_requires_finite_clock_calibration_convergence() -> None:
    """Under Kubernetes, finite replay blocks WorkerDispatchable until clock converges."""
    worker = SimpleNamespace(
        service_id="worker-1",
        _finite_replay_enabled=True,
        _tracks_clock_offset=True,
        _worker_ready_event=asyncio.Event(),
        _worker_ready_lock=asyncio.Lock(),
        _await_return_channel_ready=AsyncMock(),
        _measure_baseline_rtt=AsyncMock(),
        clock_offset_tracker=MagicMock(
            is_currently_calibrated=False,
            baseline_rtt_ns=None,
        ),
        credit_dealer_client=AsyncMock(),
        _publish_startup_state=AsyncMock(),
        _dataset_state_retry_task=None,
    )

    with pytest.raises(
        RuntimeError, match="Finite replay clock calibration did not converge"
    ):
        await Worker._mark_worker_ready_locked(worker)

    worker.credit_dealer_client.send.assert_not_called()
    assert not worker._worker_ready_event.is_set()

    # When calibrated, WorkerDispatchable is emitted
    worker.clock_offset_tracker.is_currently_calibrated = True
    worker.clock_offset_tracker.baseline_rtt_ns = 500_000

    await Worker._mark_worker_ready_locked(worker)

    worker.credit_dealer_client.send.assert_called_once_with(
        WorkerDispatchable(worker_id="worker-1")
    )
    assert worker._worker_ready_event.is_set()


@pytest.mark.asyncio
async def test_clock_remeasure_task_fails_worker_when_calibration_lost_in_finite_replay() -> (
    None
):
    """Periodic remeasurement task transitions worker to FAILED when calibration is lost."""

    async def mock_fail(error: Exception) -> None:
        raise asyncio.CancelledError(str(error))

    tracker = MagicMock(
        baseline_measurement_count=1,
        is_currently_calibrated=False,
        baseline_rtt_ns=500_000,
        estimated_one_way_ns=250_000,
    )
    worker = SimpleNamespace(
        service_id="worker-1",
        _tracks_clock_offset=True,
        _finite_replay_enabled=True,
        clock_offset_tracker=tracker,
        _measure_baseline_rtt=AsyncMock(),
        publish=AsyncMock(),
        error=MagicMock(),
        debug=MagicMock(),
        _fail=mock_fail,
    )
    worker._handle_finite_clock_calibration_failure = MethodType(
        Worker._handle_finite_clock_calibration_failure, worker
    )

    with pytest.raises(asyncio.CancelledError):
        await Worker._clock_remeasure_task(worker)

    assert tracker.baseline_rtt_ns is None
    assert tracker.estimated_one_way_ns is None
    worker.publish.assert_awaited_once()
    published = worker.publish.await_args.args[0]
    assert isinstance(published, BaseServiceErrorMessage)
    assert published.service_id == "worker-1"


def test_uncalibrated_worker_rejects_finite_start_and_eof_callbacks() -> None:
    """Callbacks raise immediately if the Kubernetes worker clock loses calibration."""
    worker = SimpleNamespace(
        service_id="worker-1",
        _tracks_clock_offset=True,
        clock_offset_tracker=MagicMock(
            is_currently_calibrated=False,
            baseline_rtt_ns=None,
            correction_ns=None,
            offset_range_ns=None,
        ),
        credit_return_push_client=SimpleNamespace(send=AsyncMock()),
        _transport_start_tasks={},
    )
    worker._snapshot_finite_clock_state = MethodType(
        Worker._snapshot_finite_clock_state, worker
    )
    credit = Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="root",
        x_correlation_id="corr-1",
        turn_index=0,
        num_turns=1,
        issued_at_ns=1,
        finite_replay=True,
    )
    context = CreditContext(credit=credit, drop_perf_ns=1)

    on_start = Worker._make_transport_start_callback(worker, context)
    on_eof = Worker._make_transport_eof_callback(worker, context)

    assert on_start is not None and on_eof is not None

    with pytest.raises(
        RuntimeError, match="Finite replay worker clock is not calibrated"
    ):
        on_start(1_000_000)

    with pytest.raises(
        RuntimeError, match="Finite replay worker clock lost calibration"
    ):
        on_eof(2_000_000)


@pytest.mark.asyncio
async def test_failed_transport_start_send_populates_context_error() -> None:
    """If TransportDispatched send fails, credit return attaches the failure reason."""

    async def raise_failure() -> None:
        raise RuntimeError("ZMQ push send failed")

    failed_task = asyncio.create_task(raise_failure())

    worker = SimpleNamespace(
        service_id="worker-1",
        credit_return_push_client=SimpleNamespace(send=AsyncMock()),
        _transport_start_tasks={(CreditPhase.PROFILING, None, 1): failed_task},
    )
    credit = Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="root",
        x_correlation_id="corr-1",
        turn_index=0,
        num_turns=1,
        issued_at_ns=1,
        finite_replay=True,
    )
    context = CreditContext(credit=credit, drop_perf_ns=1)
    context.transport_start_sent = True

    await Worker._send_ordered_credit_return(worker, context)

    assert context.error is not None
    assert "Finite transport-start notification failed" in context.error
    returned = worker.credit_return_push_client.send.await_args.args[0]
    assert returned.error == context.error
    assert returned.transport_eof_wall_ns is None
    assert context.returned
    assert not worker._transport_start_tasks
