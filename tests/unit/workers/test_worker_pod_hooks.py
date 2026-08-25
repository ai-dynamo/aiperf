# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pod-awareness hooks on the Worker: startup-state reporting and clock offset.

These hooks are inert in single-node multiprocess mode (the messages are
published on the same bus every other worker message uses) and load-bearing in
Kubernetes mode, where the controller needs per-worker startup visibility and
worker clocks are not the controller's clock.
"""

from unittest.mock import AsyncMock, patch

import pytest

from aiperf.common.enums import CreditPhase, WorkerStartupState
from aiperf.common.messages import WorkerStartupStateMessage
from aiperf.credit.messages import TimePong
from aiperf.credit.structs import Credit
from aiperf.plugin.enums import ServiceRunType
from aiperf.workers.clock_offset_tracker import ClockOffsetTracker
from aiperf.workers.worker import Worker
from tests.harness.fake_tokenizer import FakeTokenizer


@pytest.fixture
async def mock_worker(
    benchmark_run,
    fake_tokenizer: FakeTokenizer,
    skip_service_registration,
):
    """A fully initialized and started Worker (no SystemController needed)."""
    worker = Worker(run=benchmark_run, service_id="mock-service-id")
    await worker.initialize()
    await worker.start()
    yield worker
    await worker.stop()


def _credit(issued_at_ns: int) -> Credit:
    return Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="conv-1",
        x_correlation_id="x-1",
        turn_index=0,
        num_turns=1,
        issued_at_ns=issued_at_ns,
    )


class TestStartupStateReporting:
    @pytest.mark.asyncio
    async def test_publishes_each_distinct_transition(
        self, mock_worker: Worker
    ) -> None:
        published: list[WorkerStartupStateMessage] = []
        mock_worker.publish = AsyncMock(side_effect=lambda m: published.append(m))

        await mock_worker._publish_startup_state(WorkerStartupState.STARTING)
        await mock_worker._publish_startup_state(WorkerStartupState.READY)

        assert [m.startup_state for m in published] == [
            WorkerStartupState.STARTING,
            WorkerStartupState.READY,
        ]
        assert all(m.service_id == mock_worker.service_id for m in published)

    @pytest.mark.asyncio
    async def test_repeated_state_is_not_republished(self, mock_worker: Worker) -> None:
        mock_worker.publish = AsyncMock()
        await mock_worker._publish_startup_state(WorkerStartupState.SHUTTING_DOWN)
        await mock_worker._publish_startup_state(WorkerStartupState.SHUTTING_DOWN)
        assert mock_worker.publish.await_count == 1

    @pytest.mark.asyncio
    async def test_startup_reaches_ready(self, mock_worker: Worker) -> None:
        """The started fixture worker has already run its @on_start hook."""
        assert mock_worker._startup_state == WorkerStartupState.READY

    @pytest.mark.asyncio
    async def test_shutdown_reports_shutting_down(self, mock_worker: Worker) -> None:
        mock_worker.publish = AsyncMock()
        await mock_worker._send_worker_shutdown_message()
        states = [
            call.args[0].startup_state
            for call in mock_worker.publish.await_args_list
            if isinstance(call.args[0], WorkerStartupStateMessage)
        ]
        assert WorkerStartupState.SHUTTING_DOWN in states


class TestClockOffsetHooks:
    @pytest.mark.asyncio
    async def test_worker_owns_a_clock_offset_tracker(
        self, mock_worker: Worker
    ) -> None:
        assert isinstance(mock_worker.clock_offset_tracker, ClockOffsetTracker)
        assert mock_worker.clock_offset_tracker.offset_ns is None

    @pytest.mark.asyncio
    async def test_credit_receipt_observes_the_offset_in_kubernetes_mode(
        self, mock_worker: Worker
    ) -> None:
        mock_worker._tracks_clock_offset = True
        mock_worker._schedule_credit_drop_task(_credit(issued_at_ns=1))
        assert mock_worker.clock_offset_tracker.sample_count == 1
        assert mock_worker.clock_offset_tracker.offset_ns is not None

    @pytest.mark.asyncio
    async def test_credit_receipt_skips_the_offset_in_local_mode(
        self, mock_worker: Worker
    ) -> None:
        """Nothing reads the offset outside Kubernetes, so the credit hot path
        must not pay for the sample."""
        assert mock_worker._tracks_clock_offset is False
        mock_worker._schedule_credit_drop_task(_credit(issued_at_ns=1))
        assert mock_worker.clock_offset_tracker.sample_count == 0
        assert mock_worker.clock_offset_tracker.offset_ns is None

    def test_offset_tracking_follows_the_service_run_type(self, benchmark_run) -> None:
        """The gate is derived from the run type, not hardcoded off."""
        benchmark_run.cfg.runtime.service_run_type = ServiceRunType.KUBERNETES
        worker = Worker(run=benchmark_run, service_id="k8s-worker")
        assert worker._tracks_clock_offset is True

    @pytest.mark.asyncio
    async def test_time_pong_is_routed_to_the_tracker(
        self, mock_worker: Worker
    ) -> None:
        pong = TimePong(sequence=0, sent_at_ns=123)
        with patch.object(ClockOffsetTracker, "handle_pong") as handle_pong:
            await mock_worker._on_credit_message(pong)
        handle_pong.assert_called_once_with(pong)

    @pytest.mark.asyncio
    async def test_unknown_credit_message_still_warns(
        self, mock_worker: Worker, caplog: pytest.LogCaptureFixture
    ) -> None:
        await mock_worker._on_credit_message(object())  # type: ignore[arg-type]
        assert "Unknown credit message type" in caplog.text
