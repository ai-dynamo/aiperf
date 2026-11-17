# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for distributed Kubernetes worker clock synchronization in finite replay."""

import asyncio
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase, LifecycleState
from aiperf.common.environment import Environment
from aiperf.common.messages import BaseServiceErrorMessage
from aiperf.common.mixins.aiperf_lifecycle_mixin import AIPerfLifecycleMixin
from aiperf.credit.messages import TimePing, TimePong, WorkerDispatchable
from aiperf.credit.structs import Credit, CreditContext
from aiperf.workers.clock_offset_tracker import ClockOffsetTracker
from aiperf.workers.worker import Worker


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("missed_pongs", "expected_ready"),
    [
        param(6, True, id="late-router-calibrates-within-budget"),
        param(26, False, id="insufficient-samples-block-readiness"),
    ],
)  # fmt: skip
async def test_finite_startup_retries_until_calibrated(
    missed_pongs: int, expected_ready: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Environment.WORKER, "CLOCK_PROBE_BUDGET", 30.0)
    monkeypatch.setattr(Environment.WORKER, "CLOCK_PROBE_TIMEOUT", 1.0)
    monkeypatch.setattr(Environment.WORKER, "CLOCK_PROBE_COUNT", 5)
    monkeypatch.setattr(Environment.WORKER, "CLOCK_OFFSET_MIN_SAMPLES", 5)
    tracker = ClockOffsetTracker(observe_router_pongs=True, min_samples=5)
    attempts = 0
    dispatchable: list[WorkerDispatchable] = []

    async def send(message: TimePing | WorkerDispatchable) -> None:
        nonlocal attempts
        if isinstance(message, WorkerDispatchable):
            dispatchable.append(message)
            return
        attempts += 1
        if attempts <= missed_pongs:
            tracker._pending_pong_future.set_exception(TimeoutError())
            return
        tracker.handle_pong(
            TimePong(
                sequence=message.sequence,
                sent_at_ns=message.sent_at_ns,
                router_sent_wall_ns=tracker._now_ns(),
            )
        )

    worker = SimpleNamespace(
        service_id="worker-1",
        _finite_replay_enabled=True,
        _tracks_clock_offset=True,
        _worker_ready_event=asyncio.Event(),
        _clock_probe_lock=asyncio.Lock(),
        _await_return_channel_ready=AsyncMock(),
        clock_offset_tracker=tracker,
        credit_dealer_client=SimpleNamespace(send=send),
        _publish_startup_state=AsyncMock(),
        _dataset_state_retry_task=None,
        warning=MagicMock(),
    )
    worker._measure_baseline_rtt = MethodType(Worker._measure_baseline_rtt, worker)

    if expected_ready:
        await Worker._mark_worker_ready_locked(worker)
        assert dispatchable == [WorkerDispatchable(worker_id="worker-1")]
        assert attempts == 11
    else:
        with pytest.raises(RuntimeError, match="did not converge before readiness"):
            await Worker._mark_worker_ready_locked(worker)
        assert not dispatchable
        assert attempts == 30
    assert worker._worker_ready_event.is_set() is expected_ready
    assert tracker.is_currently_calibrated is expected_ready


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("finite", "budget", "probe_count", "expected_attempts"),
    [
        param(True, 30.0, 5, 30, id="finite-preserves-budget"),
        param(True, 1.0, 5, 10, id="finite-minimum-samples"),
        param(True, 1.0, 12, 12, id="finite-requested-probes"),
        param(False, 30.0, 5, 30, id="steady-state-budget"),
        param(False, 1.0, 5, 1, id="steady-state-no-finite-minimum"),
    ],
)  # fmt: skip
async def test_measure_baseline_rtt_retains_retry_limits(
    finite: bool,
    budget: float,
    probe_count: int,
    expected_attempts: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Environment.WORKER, "CLOCK_PROBE_BUDGET", budget)
    monkeypatch.setattr(Environment.WORKER, "CLOCK_PROBE_TIMEOUT", 1.0)
    monkeypatch.setattr(Environment.WORKER, "CLOCK_OFFSET_MIN_SAMPLES", 5)
    tracker = SimpleNamespace(measure_baseline_rtt=AsyncMock())
    worker = SimpleNamespace(
        _finite_replay_enabled=finite,
        _clock_probe_lock=asyncio.Lock(),
        clock_offset_tracker=tracker,
        credit_dealer_client=SimpleNamespace(send=AsyncMock()),
    )

    await Worker._measure_baseline_rtt(worker, probe_count=probe_count)

    tracker.measure_baseline_rtt.assert_awaited_once_with(
        send_ping=worker.credit_dealer_client.send,
        probe_count=probe_count,
        timeout=1.0,
        max_attempts=expected_attempts,
    )


@pytest.mark.asyncio
async def test_clock_failure_in_tracked_background_task_finishes_failed() -> None:
    component = AIPerfLifecycleMixin(id="worker-1")
    component.service_id = "worker-1"
    component._finite_replay_enabled = True
    component._tracks_clock_offset = True
    component.clock_offset_tracker = ClockOffsetTracker(observe_router_pongs=True)
    component.clock_offset_tracker.baseline_rtt_ns = 500_000
    component.clock_offset_tracker.estimated_one_way_ns = 250_000
    component._measure_baseline_rtt = AsyncMock()
    component.publish = AsyncMock()
    component._handle_finite_clock_calibration_failure = MethodType(
        Worker._handle_finite_clock_calibration_failure, component
    )
    await component.initialize_and_start()
    other_task = component.execute_async(asyncio.Event().wait())

    async def remeasure() -> None:
        await Worker._clock_remeasure_task(component)

    task = component.execute_async(
        component._background_task_loop(remeasure, immediate=True)
    )
    await asyncio.wait_for(task, timeout=1.0)
    await asyncio.gather(other_task, return_exceptions=True)

    assert component.state == LifecycleState.FAILED
    assert component.stopped_event.is_set()
    assert (
        component.failure_shutdown_timeout
        == Environment.SERVICE.FAILURE_SHUTDOWN_TIMEOUT
    )
    assert other_task.cancelled()
    assert component.clock_offset_tracker.baseline_rtt_ns is None
    assert component.clock_offset_tracker.estimated_one_way_ns is None
    component.publish.assert_awaited_once()
    assert isinstance(component.publish.await_args.args[0], BaseServiceErrorMessage)


@pytest.mark.asyncio
async def test_observe_router_pongs_handle_pong_feeds_clock_observation() -> None:
    """When observe_router_pongs is enabled, pong router timestamps calibrate the tracker."""
    tracker = ClockOffsetTracker(observe_router_pongs=True, min_samples=1)
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


@pytest.mark.asyncio
async def test_clock_remeasure_probe_error_routes_to_finite_failure() -> None:
    probe_error = OSError("ping send failed")
    worker = SimpleNamespace(
        _tracks_clock_offset=True,
        _finite_replay_enabled=True,
        clock_offset_tracker=MagicMock(baseline_measurement_count=1),
        _measure_baseline_rtt=AsyncMock(side_effect=probe_error),
        _handle_finite_clock_calibration_failure=AsyncMock(),
    )

    await Worker._clock_remeasure_task(worker)

    worker._handle_finite_clock_calibration_failure.assert_awaited_once_with(
        probe_error
    )


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
@pytest.mark.parametrize(
    ("original_error", "expected_error"),
    [
        param("DatasetLoaderError: original diagnostic", "DatasetLoaderError: original diagnostic", id="preserves-original"),
        param(None, "Finite transport-start notification was not queued", id="missing-start-fallback"),
    ],
)  # fmt: skip
async def test_missing_transport_start_preserves_error_or_sets_fallback(
    original_error: str | None, expected_error: str
) -> None:
    worker = SimpleNamespace(
        service_id="worker-1",
        credit_return_push_client=SimpleNamespace(send=AsyncMock()),
        _transport_start_tasks={},
    )
    context = CreditContext(
        credit=Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="root",
            x_correlation_id="corr-1",
            turn_index=0,
            num_turns=1,
            issued_at_ns=1,
            finite_replay=True,
        ),
        drop_perf_ns=1,
        error=original_error,
    )

    await Worker._send_ordered_credit_return(worker, context)

    returned = worker.credit_return_push_client.send.await_args.args[0]
    assert context.error == expected_error
    assert returned.error == expected_error
    assert context.returned


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "original_error",
    [
        param(None, id="send-error-fallback"),
        param("DatasetLoaderError: original diagnostic", id="preserves-original-on-send-error"),
    ],
)  # fmt: skip
async def test_failed_transport_start_send_populates_context_error(
    original_error: str | None,
) -> None:
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
    context = CreditContext(credit=credit, drop_perf_ns=1, error=original_error)
    context.transport_start_sent = True

    await Worker._send_ordered_credit_return(worker, context)

    assert context.error is not None
    if original_error is None:
        assert "Finite transport-start notification failed" in context.error
    else:
        assert context.error == original_error
    returned = worker.credit_return_push_client.send.await_args.args[0]
    assert returned.error == context.error
    assert returned.transport_eof_wall_ns is None
    assert context.returned
    assert not worker._transport_start_tasks
