# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The consumer half of clock-offset correction.

Task 11 shipped the estimator; without a reader the measurement never reaches
a record and cross-pod timestamps stay skewed. Every ``RequestRecord`` leaving
a worker carries the offset that was current when it was emitted, so downstream
can map worker wall-clock into the controller's frame.

Sign convention (get this wrong and correction doubles the error instead of
removing it): a sample is ``received - issued``, so a worker clock running
*ahead* of the controller yields a *positive* offset, and ``correct_timestamp``
SUBTRACTS -- ``controller_time = worker_time - offset``.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import param

from aiperf.common.models import RequestRecord
from aiperf.workers.clock_offset_tracker import ClockOffsetTracker


def test_offset_sign_round_trips_a_known_skew() -> None:
    """Recover the controller timestamp from a worker timestamp exactly.

    Worker clock is 5 ms ahead; transit adds a further 1 ms of positive bias to
    each raw sample, so the min-filtered estimate is the smallest sample (6 ms).
    A sign flip returns 2_012_000_000 -- double the 6 ms error rather than
    removing it -- not 2_000_000_000.
    """
    tracker = ClockOffsetTracker()
    skew_ns = 5_000_000

    # Controller issues at T; worker receives at T + skew + transit.
    for transit_ns in (4_000_000, 1_000_000, 9_000_000):
        issued = 1_000_000_000
        tracker.observe(
            issued_at_ns=issued, received_at_ns=issued + skew_ns + transit_ns
        )

    # min() over the window picks the least-delayed sample: skew + 1 ms.
    assert tracker.offset_ns == skew_ns + 1_000_000

    worker_now_ns = 2_000_000_000 + skew_ns + 1_000_000
    assert tracker.correct_timestamp(worker_now_ns) == 2_000_000_000


def test_offset_is_none_before_any_sample() -> None:
    tracker = ClockOffsetTracker()
    assert tracker.offset_ns is None
    assert tracker.correct_timestamp(4_242) == 4_242


def test_request_record_carries_clock_offset() -> None:
    assert RequestRecord().clock_offset_ns is None
    assert RequestRecord(clock_offset_ns=-7).clock_offset_ns == -7


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "group_managed, expected_sent",
    [
        param(True, ["WorkerConnected"], id="kubernetes-defers-dispatchable"),
        param(
            False,
            ["WorkerConnected", "WorkerDispatchable"],
            id="local-dispatchable-immediately",
        ),
    ],
)  # fmt: skip
async def test_worker_announces_connectivity_without_awaiting_the_rtt_probe(
    group_managed: bool, expected_sent: list[str]
) -> None:
    """Connectivity is announced without awaiting the RTT probe.

    Regression test for a real-cluster failure: the probe used to run *before*
    the worker announced itself so its pings would not queue behind credits. On
    a multi-pod cluster the credit ROUTER is not echoing yet at that moment, so
    every probe timed out, the worker blocked for the whole budget, and service
    registration timed out -- a diagnostic taking down startup. The probe is now
    dispatched behind the announcement, so a router that never echoes costs a
    measurement and nothing else.

    The probe is Kubernetes-only and group-managed mode returns early, so the
    dispatch must sit above that branch or it never runs where it matters.
    """
    from aiperf.workers.worker import Worker

    sent: list[object] = []
    worker = MagicMock(spec=Worker)
    # The probe is enabled exactly when the worker runs under Kubernetes.
    worker._tracks_clock_offset = group_managed
    probe_coros: list[object] = []
    worker._measure_baseline_rtt = MagicMock(side_effect=lambda: "probe-coro")
    worker.execute_async = MagicMock(side_effect=probe_coros.append)
    worker.credit_dealer_client = AsyncMock()
    worker.credit_dealer_client.send = AsyncMock(
        side_effect=lambda msg: sent.append(type(msg).__name__)
    )
    worker._publish_startup_state = AsyncMock()
    worker.service_id = "worker-7f2a"
    worker._is_group_managed_mode = MagicMock(return_value=group_managed)
    worker.pod_lifecycle_dealer_client = None
    worker._ensure_group_dataset_state_retry = MagicMock()
    worker._complete_group_startup_flow = AsyncMock()
    worker._mark_worker_ready = AsyncMock(
        side_effect=lambda: sent.append("WorkerDispatchable")
    )

    await Worker._send_worker_ready_message(worker)

    assert sent == expected_sent
    # Dispatched, never awaited on this path -- in both modes.
    assert probe_coros == (["probe-coro"] if group_managed else [])


@pytest.mark.asyncio
async def test_kubernetes_worker_is_not_dispatchable_until_dataset_ready() -> None:
    """A group-managed worker must not join the routing pool on connect alone.

    This is the invariant whose loss let the router hand 1182 credits to a pod
    that had missed the dataset broadcast; every one failed without reaching the
    inference server, and the records barrier then hung waiting for records that
    could never arrive.
    """
    from aiperf.workers.worker import Worker

    sent: list[object] = []
    worker = MagicMock(spec=Worker)
    worker._tracks_clock_offset = False
    worker.credit_dealer_client = AsyncMock()
    worker.credit_dealer_client.send = AsyncMock(
        side_effect=lambda msg: sent.append(type(msg).__name__)
    )
    worker._publish_startup_state = AsyncMock()
    worker.service_id = "worker-7f2a"
    worker._is_group_managed_mode = MagicMock(return_value=True)
    worker.pod_lifecycle_dealer_client = None
    worker._ensure_group_dataset_state_retry = MagicMock()
    worker._complete_group_startup_flow = AsyncMock()

    await Worker._send_worker_ready_message(worker)

    assert "WorkerDispatchable" not in sent
    # The poll is what makes a missed one-shot broadcast recoverable.
    worker._ensure_group_dataset_state_retry.assert_called_once()


@pytest.mark.asyncio
async def test_probe_budget_bounds_a_router_that_never_echoes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A silent router must not stall worker readiness past registration."""
    import asyncio

    from aiperf.common.environment import Environment
    from aiperf.workers.worker import Worker

    monkeypatch.setattr(Environment.WORKER, "CLOCK_PROBE_BUDGET", 0.01)

    worker = MagicMock(spec=Worker)
    worker.clock_offset_tracker = MagicMock()

    # Stands in for the serial probe loop when no TimePong ever arrives: the
    # real call would run probe_count timeouts back to back.
    async def _never_echoes(**_: object) -> None:
        await asyncio.Event().wait()

    worker.clock_offset_tracker.measure_baseline_rtt = AsyncMock(
        side_effect=_never_echoes
    )
    worker.credit_dealer_client = AsyncMock()
    worker.warning = MagicMock()

    # Unbounded, this never returns; the budget is what makes it complete.
    await Worker._measure_baseline_rtt(worker)

    worker.warning.assert_called_once()
    assert "budget" in worker.warning.call_args[0][0]


@pytest.mark.asyncio
async def test_kubernetes_worker_defers_record_correction_until_clock_calibrates() -> (
    None
):
    """One-way credit samples must not shift Kubernetes records before calibration."""
    from aiperf.workers.worker import Worker

    worker = MagicMock(spec=Worker)
    worker.clock_offset_tracker = ClockOffsetTracker()
    worker._tracks_clock_offset = True
    for _ in range(4):
        worker.clock_offset_tracker.observe(issued_at_ns=1_000, received_at_ns=3_500)
    worker.task_stats = MagicMock()
    worker.execute_async = MagicMock()
    worker.inference_results_push_client = AsyncMock()
    worker.service_id = "worker-7f2a"

    record = RequestRecord()
    await Worker._send_inference_result_message(worker, record)

    assert record.clock_offset_ns is None


@pytest.mark.asyncio
async def test_kubernetes_worker_stamps_offset_after_clock_calibrates() -> None:
    """The fifth credit sample enables controller-frame record correction."""
    from aiperf.workers.worker import Worker

    worker = MagicMock(spec=Worker)
    worker.clock_offset_tracker = ClockOffsetTracker()
    worker._tracks_clock_offset = True
    for _ in range(5):
        worker.clock_offset_tracker.observe(issued_at_ns=1_000, received_at_ns=3_500)
    worker.task_stats = MagicMock()
    worker.execute_async = MagicMock()
    worker.inference_results_push_client = AsyncMock()
    worker.service_id = "worker-7f2a"

    record = RequestRecord()
    await Worker._send_inference_result_message(worker, record)

    assert record.clock_offset_ns == 2_500
