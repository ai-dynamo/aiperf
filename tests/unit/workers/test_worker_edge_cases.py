# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-case tests for `aiperf.workers.worker.Worker`.

Complements `tests/unit/workers/test_worker.py` (happy-path coverage) by
exercising:

- Cancellation paths (`asyncio.CancelledError`, `CancelCredits`,
  cancel-before-start in the done callback).
- Per-credit error/exception handlers (`_on_credit_drop_message_task` finally
  block always returns the credit; `_process_credit` records `ErrorDetails`).
- Credit-channel dispatch (`_on_credit_message`) for `TimePong`,
  `InFlightReconciliation`, and unknown messages.
- Group-managed pod peer commands (`SHUTDOWN`, `ABORT`, unknown,
  `PROFILE_CONFIGURE`).
- Shutdown-message error suppression (`zmq.ZMQError`, timeout, connection).
- `_query_pod_dataset_state` timeouts and wrong-typed responses.
- Conversation-fallback `ErrorMessage` path (sends an error record + raises).
- `_publish_startup_state` duplicate suppression.
- `_send_inference_result_message` task-stats counters.
- `_make_first_token_callback` disabled path.

These tests construct the `Worker` directly without starting the lifecycle —
just like `test_worker.py` — and patch ZMQ-facing async clients with
`AsyncMock` so message-bus traffic stays in-process.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import zmq
from pytest import param

from aiperf.common.enums import CommandType, CreditPhase, WorkerStartupState
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.messages import ErrorMessage, WorkerStartupStateMessage
from aiperf.common.messages.dataset_messages import ConversationResponseMessage
from aiperf.common.models import (
    Conversation,
    ErrorDetails,
    ProcessHealth,
    RequestRecord,
)
from aiperf.common.pod_lifecycle_structs import (
    GroupDatasetStateSnapshot,
    GroupPeerCommand,
    GroupPeerCommandAck,
    GroupWorkerStartupState,
)
from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.credit.messages import (
    CancelCredits,
    CreditReturn,
    FirstToken,
    InFlightReconciliation,
    InFlightReport,
    TimePong,
)
from aiperf.credit.structs import Credit, CreditContext
from aiperf.plugin.enums import ServiceRunType
from aiperf.workers.worker import Worker
from tests.harness.fake_tokenizer import FakeTokenizer

_STUB_PROCESS_HEALTH = ProcessHealth(
    create_time=0.0, uptime=1.0, cpu_usage=0.0, memory_usage=0
)


# ============================================================
# Fixtures
# ============================================================


def _make_run(cfg: AIPerfConfig) -> BenchmarkRun:
    return BenchmarkRun(
        benchmark_id="test",
        cfg=cfg.benchmark,
        artifact_dir=Path("/tmp/test"),
    )


@pytest.fixture
def mock_worker(
    config: AIPerfConfig,
    fake_tokenizer: FakeTokenizer,
    skip_service_registration,
    mock_psutil_process,
    monkeypatch: pytest.MonkeyPatch,
) -> Worker:
    """Construct a non-group-managed Worker without starting its lifecycle.

    The shared ``config`` fixture defaults to ``MULTIPROCESSING`` which
    ``RuntimeConfig.uses_worker_group_manager`` treats as group-managed. Force
    the in-process bypass so ``_publish_startup_state`` and friends route
    through ``self.publish`` instead of the pod-lifecycle dealer.
    """
    monkeypatch.setenv("AIPERF_FAKE_IN_PROCESS_MODE", "1")
    worker = Worker(run=_make_run(config), service_id="edge-worker")
    worker._measure_baseline_rtt = AsyncMock()
    worker.get_process_health = Mock(return_value=_STUB_PROCESS_HEALTH)
    worker.get_pss_memory = Mock(return_value=None)
    worker.credit_return_push_client.send = AsyncMock()
    worker.credit_dealer_client.send = AsyncMock()
    return worker


@pytest.fixture
def k8s_worker(
    config: AIPerfConfig,
    fake_tokenizer: FakeTokenizer,
    skip_service_registration,
    mock_psutil_process,
) -> Worker:
    """Construct a Kubernetes-mode Worker for group-managed peer-command tests."""
    config.benchmark.runtime.service_run_type = ServiceRunType.KUBERNETES
    worker = Worker(run=_make_run(config), service_id="k8s-edge-worker")
    worker._pod_index = "0"
    worker._measure_baseline_rtt = AsyncMock()
    worker.get_process_health = Mock(return_value=_STUB_PROCESS_HEALTH)
    worker.get_pss_memory = Mock(return_value=None)
    worker.credit_return_push_client.send = AsyncMock()
    worker.credit_dealer_client.send = AsyncMock()
    if worker.pod_lifecycle_dealer_client is not None:
        worker.pod_lifecycle_dealer_client.send = AsyncMock()
    return worker


@pytest.fixture
def credit_factory():
    """Build a Credit + CreditContext with overridable defaults."""

    def _make(
        credit_id: int = 1,
        turn_index: int = 0,
        num_turns: int = 1,
        x_correlation_id: str = "corr-1",
    ) -> CreditContext:
        return CreditContext(
            credit=Credit(
                id=credit_id,
                phase="profiling",
                conversation_id="conv-1",
                x_correlation_id=x_correlation_id,
                turn_index=turn_index,
                num_turns=num_turns,
                issued_at_ns=1_000_000,
            ),
            drop_perf_ns=2_000_000,
        )

    return _make


# ============================================================
# Credit task lifecycle: cancellation + finally-always-returns
# ============================================================


@pytest.mark.asyncio
class TestWorkerCancellation:
    """Verify cancellation paths on the credit-processing task."""

    async def test_credit_task_cancelled_during_processing_marks_cancelled_and_returns(
        self, mock_worker: Worker, credit_factory
    ) -> None:
        """CancelledError mid-`_process_credit` flips `cancelled` and still sends CreditReturn."""
        ctx = credit_factory()

        async def raise_cancel(_ctx: CreditContext) -> None:
            raise asyncio.CancelledError

        mock_worker._process_credit = raise_cancel  # type: ignore[assignment]

        await mock_worker._on_credit_drop_message_task(ctx)

        assert ctx.cancelled is True
        assert ctx.returned is True
        sent = mock_worker.credit_return_push_client.send.await_args.args[0]
        assert isinstance(sent, CreditReturn)
        assert sent.cancelled is True
        assert sent.error is None

    async def test_credit_task_exception_records_error_in_credit_return(
        self, mock_worker: Worker, credit_factory
    ) -> None:
        """A non-cancel exception in `_process_credit` is logged but credit is still returned."""
        ctx = credit_factory()
        ctx.error = "boom"  # pretend `_process_credit` populated the error
        mock_worker._process_credit = AsyncMock(side_effect=RuntimeError("boom"))

        await mock_worker._on_credit_drop_message_task(ctx)

        sent = mock_worker.credit_return_push_client.send.await_args.args[0]
        assert isinstance(sent, CreditReturn)
        assert sent.error == "boom"
        assert ctx.returned is True

    async def test_credit_task_uninitialized_inference_client_returns_credit(
        self, mock_worker: Worker, credit_factory
    ) -> None:
        """`NotInitializedError` is caught at the per-credit boundary; finally still sends a return."""
        ctx = credit_factory()
        mock_worker.inference_client = None  # type: ignore[assignment]

        # `_process_credit` is reached via the `if not self.inference_client` guard, so
        # we don't actually call it; the NotInitializedError is raised in-line.
        # The except Exception branch absorbs it.
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                Worker,
                "_process_credit",
                AsyncMock(side_effect=NotInitializedError("nope")),
            )
            await mock_worker._on_credit_drop_message_task(ctx)

        sent = mock_worker.credit_return_push_client.send.await_args.args[0]
        assert isinstance(sent, CreditReturn)
        assert ctx.returned is True

    async def test_process_credit_cancelled_evicts_session(
        self, mock_worker: Worker, credit_factory
    ) -> None:
        """Cancellation during `_process_credit` should evict the session even on a non-final turn."""
        ctx = credit_factory(turn_index=0, num_turns=3)
        evict_calls: list[str] = []
        mock_worker.session_manager.evict = lambda x_corr: evict_calls.append(x_corr)  # type: ignore[assignment]

        async def cancel_immediately(*_args: Any, **_kwargs: Any) -> Any:
            raise asyncio.CancelledError

        mock_worker._get_or_create_session = cancel_immediately  # type: ignore[assignment]

        with pytest.raises(asyncio.CancelledError):
            await mock_worker._process_credit(ctx)

        assert ctx.cancelled is True
        assert evict_calls == ["corr-1"]

    async def test_process_credit_exception_captures_error_details(
        self, mock_worker: Worker, credit_factory
    ) -> None:
        """A real exception is captured into `credit_context.error` as ErrorDetails."""
        ctx = credit_factory(turn_index=0, num_turns=1)

        async def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("dataset offline")

        mock_worker._get_or_create_session = boom  # type: ignore[assignment]
        mock_worker.session_manager.evict = Mock()  # type: ignore[assignment]

        await mock_worker._process_credit(ctx)

        assert isinstance(ctx.error, ErrorDetails)
        # Final turn → session evicted
        mock_worker.session_manager.evict.assert_called_once_with("corr-1")

    async def test_process_credit_non_final_non_cancel_does_not_evict(
        self, mock_worker: Worker, credit_factory
    ) -> None:
        """Non-final turn that completes normally must not evict the cached session."""
        ctx = credit_factory(turn_index=0, num_turns=3)
        mock_worker._get_or_create_session = AsyncMock(
            return_value=Mock(
                advance_turn=Mock(),
                conversation=Mock(system_message=None, user_context_message=None),
                turn_index=0,
                turn_list=[],
                url_index=None,
                x_correlation_id="corr-1",
                should_store_response=Mock(return_value=False),
            )
        )
        mock_worker._dispatch_turn = AsyncMock()
        mock_worker.session_manager.evict = Mock()  # type: ignore[assignment]

        await mock_worker._process_credit(ctx)

        mock_worker.session_manager.evict.assert_not_called()

    async def test_on_cancel_credits_cancels_known_inflight_tasks(
        self, mock_worker: Worker
    ) -> None:
        """`_on_cancel_credits_message` cancels exactly the tasks named, ignores unknown ids."""
        loop = asyncio.get_running_loop()
        cancelled_a = asyncio.Event()
        cancelled_b = asyncio.Event()

        async def waiter(evt: asyncio.Event) -> None:
            try:
                await (
                    loop.create_future()
                )  # never completes; not asyncio.sleep (auto-fixture nukes it)
            except asyncio.CancelledError:
                evt.set()
                raise

        task_a = asyncio.create_task(waiter(cancelled_a))
        task_b = asyncio.create_task(waiter(cancelled_b))
        await asyncio.sleep(0)  # let waiters start awaiting the future
        mock_worker.credit_tasks[(CreditPhase.PROFILING, 1)] = task_a
        mock_worker.credit_tasks[(CreditPhase.PROFILING, 2)] = task_b

        await mock_worker._on_cancel_credits_message(CancelCredits(credit_ids={1, 99}))

        with pytest.raises(asyncio.CancelledError):
            await task_a
        assert cancelled_a.is_set()
        assert not cancelled_b.is_set()
        # Cleanup task B.
        task_b.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task_b


# ============================================================
# Credit done-callback: cancelled-before-start path
# ============================================================


@pytest.mark.asyncio
class TestCreditDoneCallback:
    """Verify the `_on_credit_drop_message_task_done` synchronous callback."""

    async def test_done_callback_returns_credit_when_task_cancelled_before_start(
        self, mock_worker: Worker, credit_factory
    ) -> None:
        """Tasks cancelled before the finally block must trigger a credit return in the done callback."""
        ctx = credit_factory(credit_id=42)
        ctx.returned = False  # finally block never ran

        async def never_runs() -> None:  # pragma: no cover - cancelled before start
            raise AssertionError("should not execute")

        task = asyncio.create_task(never_runs())
        mock_worker.credit_tasks[(CreditPhase.PROFILING, 42)] = task
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        mock_worker._on_credit_drop_message_task_done(task, ctx)

        # Done callback schedules an async send — let it run.
        await asyncio.sleep(0)
        assert (CreditPhase.PROFILING, 42) not in mock_worker.credit_tasks
        assert ctx.returned is True
        assert ctx.cancelled is True
        sent = mock_worker.credit_return_push_client.send.await_args.args[0]
        assert isinstance(sent, CreditReturn)
        assert sent.cancelled is True

    async def test_done_callback_skips_send_when_already_returned(
        self, mock_worker: Worker, credit_factory
    ) -> None:
        """If finally block already sent the return, the done callback must not send again."""
        ctx = credit_factory(credit_id=7)
        ctx.returned = True

        async def noop() -> None:
            return None

        task = asyncio.create_task(noop())
        mock_worker.credit_tasks[(CreditPhase.PROFILING, 7)] = task
        await task

        mock_worker._on_credit_drop_message_task_done(task, ctx)

        # No additional send; tracking dict still cleaned up.
        await asyncio.sleep(0)
        assert (CreditPhase.PROFILING, 7) not in mock_worker.credit_tasks
        assert mock_worker.credit_return_push_client.send.await_count == 0


# ============================================================
# Credit channel message dispatch
# ============================================================


@pytest.mark.asyncio
class TestOnCreditMessageDispatch:
    """`_on_credit_message` should route every wrapped struct correctly."""

    async def test_credit_message_routes_credit_to_scheduler(
        self, mock_worker: Worker, credit_factory
    ) -> None:
        scheduled: list[Credit] = []
        mock_worker._schedule_credit_drop_task = lambda c: scheduled.append(c)  # type: ignore[assignment]

        ctx = credit_factory(credit_id=11)
        await mock_worker._on_credit_message(ctx.credit)

        assert scheduled == [ctx.credit]

    async def test_credit_message_routes_cancel_to_handler(
        self, mock_worker: Worker
    ) -> None:
        called = AsyncMock()
        mock_worker._on_cancel_credits_message = called  # type: ignore[assignment]

        msg = CancelCredits(credit_ids={1})
        await mock_worker._on_credit_message(msg)

        called.assert_awaited_once_with(msg)

    async def test_credit_message_routes_timepong_to_clock_tracker(
        self, mock_worker: Worker
    ) -> None:
        fake_tracker = MagicMock()
        mock_worker.clock_offset_tracker = fake_tracker
        msg = TimePong(sequence=1, sent_at_ns=12345)
        await mock_worker._on_credit_message(msg)
        fake_tracker.handle_pong.assert_called_once_with(msg)

    async def test_credit_message_routes_reconciliation(
        self, mock_worker: Worker
    ) -> None:
        """`InFlightReconciliation` triggers an `InFlightReport` covering live credit ids."""
        sentinel_task = asyncio.create_task(asyncio.sleep(0))
        mock_worker.credit_tasks[(CreditPhase.PROFILING, 5)] = sentinel_task
        mock_worker.credit_tasks[(CreditPhase.WARMUP, 6)] = sentinel_task

        await mock_worker._on_credit_message(
            InFlightReconciliation(credit_ids=frozenset({5, 6}))
        )

        sent = mock_worker.credit_dealer_client.send.await_args.args[0]
        assert isinstance(sent, InFlightReport)
        assert sent.credit_ids == frozenset({5, 6})
        await sentinel_task

    async def test_credit_message_unknown_struct_warns_only(
        self, mock_worker: Worker
    ) -> None:
        """Unknown credit-channel messages must not crash the worker."""
        mock_worker.warning = Mock()
        await mock_worker._on_credit_message(object())  # type: ignore[arg-type]
        mock_worker.warning.assert_called_once()


# ============================================================
# Conversation fallback error path
# ============================================================


@pytest.mark.asyncio
class TestConversationFallbackErrors:
    """`_request_conversation_from_dataset_manager` must surface ErrorMessage payloads."""

    async def test_fallback_error_response_sends_error_record_and_raises(
        self, mock_worker: Worker, credit_factory
    ) -> None:
        """ErrorMessage from DatasetManager triggers an inference-result push and a ValueError."""
        ctx = credit_factory()
        err = ErrorDetails(message="dataset gone", code=500)
        err_msg = ErrorMessage(error=err)
        mock_worker.conversation_request_client.request = AsyncMock(
            return_value=err_msg
        )
        mock_worker._send_inference_result_message = AsyncMock()

        with pytest.raises(
            ValueError, match="Failed to retrieve conversation response"
        ):
            await mock_worker._request_conversation_from_dataset_manager("conv-1", ctx)

        mock_worker._send_inference_result_message.assert_awaited_once()
        record = mock_worker._send_inference_result_message.await_args.args[0]
        assert isinstance(record, RequestRecord)
        assert record.error == err

    async def test_fallback_success_returns_conversation(
        self, mock_worker: Worker, credit_factory
    ) -> None:
        ctx = credit_factory()
        conv = Conversation(session_id="conv-1", turns=[])
        mock_worker.conversation_request_client.request = AsyncMock(
            return_value=ConversationResponseMessage(
                service_id="dataset_manager", conversation=conv
            )
        )

        result = await mock_worker._request_conversation_from_dataset_manager(
            "conv-1", ctx
        )

        assert result is conv


# ============================================================
# Group-managed pod-peer commands
# ============================================================


@pytest.mark.asyncio
class TestPodPeerCommands:
    """`_handle_pod_peer_command` dispatches lifecycle commands and acks."""

    async def test_shutdown_command_calls_stop_and_acks(
        self, k8s_worker: Worker
    ) -> None:
        k8s_worker.stop = AsyncMock()
        cmd = GroupPeerCommand(
            cid="c-1",
            service_id=k8s_worker.service_id,
            command=str(CommandType.SHUTDOWN),
        )

        await k8s_worker._handle_pod_peer_command(cmd)

        k8s_worker.stop.assert_awaited_once()
        ack = k8s_worker.pod_lifecycle_dealer_client.send.await_args.args[0]
        assert isinstance(ack, GroupPeerCommandAck)
        assert ack.cid == "c-1"

    async def test_profile_configure_command_invokes_configure_and_acks(
        self, k8s_worker: Worker
    ) -> None:
        k8s_worker._configure_for_profiling = AsyncMock()
        cmd = GroupPeerCommand(
            cid="c-2",
            service_id=k8s_worker.service_id,
            command=str(CommandType.PROFILE_CONFIGURE),
        )

        await k8s_worker._handle_pod_peer_command(cmd)

        k8s_worker._configure_for_profiling.assert_awaited_once()
        ack = k8s_worker.pod_lifecycle_dealer_client.send.await_args.args[0]
        assert isinstance(ack, GroupPeerCommandAck)

    async def test_unknown_command_warns_and_does_not_ack(
        self, k8s_worker: Worker
    ) -> None:
        """Unknown peer commands are logged but do NOT ack (post-ack short-circuited)."""
        k8s_worker.warning = Mock()
        cmd = GroupPeerCommand(
            cid="c-3", service_id=k8s_worker.service_id, command="not-a-command"
        )

        await k8s_worker._handle_pod_peer_command(cmd)

        k8s_worker.warning.assert_called_once()
        # Source: handler returns early without sending ack on unknown commands.
        k8s_worker.pod_lifecycle_dealer_client.send.assert_not_called()

    async def test_abort_command_force_exits(
        self, k8s_worker: Worker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ABORT must call `os._exit(1)` after best-effort ack — never `self.stop()`."""
        exit_calls: list[int] = []

        def fake_exit(code: int) -> None:
            exit_calls.append(code)
            raise SystemExit(
                code
            )  # simulate process termination so caller can't recover

        monkeypatch.setattr("aiperf.workers.worker.os._exit", fake_exit)
        k8s_worker.stop = AsyncMock()

        cmd = GroupPeerCommand(
            cid="c-4", service_id=k8s_worker.service_id, command=str(CommandType.ABORT)
        )

        with pytest.raises(SystemExit):
            await k8s_worker._handle_pod_peer_command(cmd)

        assert exit_calls == [1]
        k8s_worker.stop.assert_not_awaited()
        # Best-effort ack went out before the exit.
        ack = k8s_worker.pod_lifecycle_dealer_client.send.await_args.args[0]
        assert isinstance(ack, GroupPeerCommandAck)

    async def test_abort_command_swallows_ack_send_failure_before_exit(
        self, k8s_worker: Worker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the ack send raises, ABORT must still hit `os._exit(1)` — best-effort ack."""
        exit_calls: list[int] = []

        def fake_exit(code: int) -> None:
            exit_calls.append(code)
            raise SystemExit(code)

        monkeypatch.setattr("aiperf.workers.worker.os._exit", fake_exit)
        k8s_worker.pod_lifecycle_dealer_client.send = AsyncMock(
            side_effect=zmq.ZMQError("disconnected")
        )

        cmd = GroupPeerCommand(
            cid="c-5", service_id=k8s_worker.service_id, command=str(CommandType.ABORT)
        )

        with pytest.raises(SystemExit):
            await k8s_worker._handle_pod_peer_command(cmd)

        assert exit_calls == [1]

    async def test_handle_pod_peer_command_noop_when_lifecycle_client_missing(
        self, mock_worker: Worker
    ) -> None:
        """Local-mode worker has no pod-lifecycle client; handler must early-return safely."""
        mock_worker.pod_lifecycle_dealer_client = None
        mock_worker.stop = AsyncMock()

        cmd = GroupPeerCommand(
            cid="c-6",
            service_id=mock_worker.service_id,
            command=str(CommandType.SHUTDOWN),
        )

        await mock_worker._handle_pod_peer_command(cmd)

        mock_worker.stop.assert_not_awaited()


# ============================================================
# Shutdown error suppression
# ============================================================


@pytest.mark.asyncio
class TestShutdownMessageErrorSuppression:
    """`_send_worker_shutdown_message` swallows ZMQ/timeout/connection errors."""

    @pytest.mark.parametrize(
        "exc",
        [
            param(zmq.ZMQError("disconnected"), id="zmq-error"),
            param(TimeoutError(), id="timeout"),
            param(ConnectionError("refused"), id="connection-error"),
        ],
    )  # fmt: skip
    async def test_send_failure_is_suppressed(
        self, mock_worker: Worker, exc: Exception
    ) -> None:
        mock_worker.credit_dealer_client.send = AsyncMock(side_effect=exc)
        mock_worker.warning = Mock()

        # Must not raise.
        await mock_worker._send_worker_shutdown_message()

        mock_worker.warning.assert_called_once()

    async def test_cancelled_error_propagates(self, mock_worker: Worker) -> None:
        """CancelledError MUST re-raise so caller sees the cancellation."""
        mock_worker.credit_dealer_client.send = AsyncMock(
            side_effect=asyncio.CancelledError
        )

        with pytest.raises(asyncio.CancelledError):
            await mock_worker._send_worker_shutdown_message()

    async def test_unrelated_exception_propagates(self, mock_worker: Worker) -> None:
        """A non-shutdown-class exception (RuntimeError) must NOT be silently swallowed."""
        mock_worker.credit_dealer_client.send = AsyncMock(
            side_effect=RuntimeError("boom")
        )

        with pytest.raises(RuntimeError, match="boom"):
            await mock_worker._send_worker_shutdown_message()


# ============================================================
# Pod-local dataset state query
# ============================================================


@pytest.mark.asyncio
class TestQueryPodDatasetState:
    """`_query_pod_dataset_state` resilience tests."""

    async def test_returns_none_when_no_lifecycle_client(
        self, mock_worker: Worker
    ) -> None:
        mock_worker.pod_lifecycle_dealer_client = None
        assert await mock_worker._query_pod_dataset_state() is None

    async def test_returns_none_on_timeout(self, k8s_worker: Worker) -> None:
        k8s_worker.pod_lifecycle_dealer_client.request = AsyncMock(
            side_effect=TimeoutError
        )
        assert await k8s_worker._query_pod_dataset_state() is None

    async def test_returns_none_on_wrong_response_type(
        self, k8s_worker: Worker
    ) -> None:
        k8s_worker.pod_lifecycle_dealer_client.request = AsyncMock(
            return_value=object()
        )
        assert await k8s_worker._query_pod_dataset_state() is None

    async def test_caches_snapshot_on_success(self, k8s_worker: Worker) -> None:
        snap = GroupDatasetStateSnapshot(
            rid="r-1",
            service_id="pod-manager",
            benchmark_generation="g",
            dataset_generation="d",
            ready=True,
        )
        k8s_worker.pod_lifecycle_dealer_client.request = AsyncMock(return_value=snap)

        result = await k8s_worker._query_pod_dataset_state()

        assert result is snap
        assert k8s_worker._latest_pod_dataset_state is snap


# ============================================================
# Startup-state publication idempotency
# ============================================================


@pytest.mark.asyncio
class TestPublishStartupState:
    """`_publish_startup_state` should suppress duplicate transitions."""

    async def test_duplicate_state_is_suppressed(self, mock_worker: Worker) -> None:
        mock_worker.publish = AsyncMock()

        await mock_worker._publish_startup_state(WorkerStartupState.STARTING)
        await mock_worker._publish_startup_state(WorkerStartupState.STARTING)

        assert mock_worker.publish.await_count == 1
        msg = mock_worker.publish.await_args.args[0]
        assert isinstance(msg, WorkerStartupStateMessage)
        assert msg.startup_state == WorkerStartupState.STARTING

    async def test_state_transition_publishes_again(self, mock_worker: Worker) -> None:
        mock_worker.publish = AsyncMock()

        await mock_worker._publish_startup_state(WorkerStartupState.STARTING)
        await mock_worker._publish_startup_state(WorkerStartupState.READY)

        assert mock_worker.publish.await_count == 2

    async def test_group_managed_routes_through_pod_lifecycle(
        self, k8s_worker: Worker
    ) -> None:
        await k8s_worker._publish_startup_state(WorkerStartupState.READY)
        sent = k8s_worker.pod_lifecycle_dealer_client.send.await_args.args[0]
        assert isinstance(sent, GroupWorkerStartupState)
        assert sent.startup_state == str(WorkerStartupState.READY)


# ============================================================
# task_stats counters via _send_inference_result_message
# ============================================================


@pytest.mark.asyncio
class TestSendInferenceResultStats:
    """`_send_inference_result_message` updates task counters and pushes serialized bytes."""

    async def test_invalid_record_increments_failed(
        self,
        mock_worker: Worker,
        sample_request_record: RequestRecord,
    ) -> None:
        """A record with an attached error increments `failed` and pushes serialized bytes."""
        sample_request_record.error = ErrorDetails(message="bad", code=500)
        assert sample_request_record.valid is False

        mock_worker._serialize_inference_wire = Mock(return_value=b"serialized")
        mock_worker.inference_results_push_client.push_raw = AsyncMock()

        await mock_worker._send_inference_result_message(sample_request_record)
        await asyncio.sleep(0)

        assert mock_worker.task_stats.completed == 0
        assert mock_worker.task_stats.failed == 1
        mock_worker.inference_results_push_client.push_raw.assert_awaited_once_with(
            b"serialized"
        )

    async def test_record_pushes_through_thread_pool_serialization(
        self,
        mock_worker: Worker,
        sample_request_record: RequestRecord,
    ) -> None:
        """Serialization runs via `asyncio.to_thread` and bytes are pushed unchanged."""
        sample_request_record.error = ErrorDetails(message="bad", code=500)
        mock_worker._serialize_inference_wire = Mock(return_value=b"payload-A")
        mock_worker.inference_results_push_client.push_raw = AsyncMock()

        await mock_worker._send_inference_result_message(sample_request_record)
        await asyncio.sleep(0)

        mock_worker._serialize_inference_wire.assert_called_once_with(
            sample_request_record
        )
        mock_worker.inference_results_push_client.push_raw.assert_awaited_once_with(
            b"payload-A"
        )


# ============================================================
# FirstToken callback construction + invocation
# ============================================================


@pytest.mark.asyncio
class TestFirstTokenCallbackConstruction:
    """`_make_first_token_callback` is enabled only when prefill_concurrency is set."""

    async def test_callback_is_none_when_prefill_disabled(
        self, mock_worker: Worker, credit_factory
    ) -> None:
        ctx = credit_factory()
        # Default fixture config has no phase with prefill_concurrency set.
        assert mock_worker._prefill_concurrency_enabled is False
        cb = mock_worker._make_first_token_callback(ctx)
        assert cb is None

    async def test_callback_sends_first_token_and_marks_credit_context(
        self, mock_worker: Worker, credit_factory
    ) -> None:
        """When enabled, callback emits FirstToken and flips `first_token_sent` on hit."""
        mock_worker._prefill_concurrency_enabled = True

        # Endpoint returns "meaningful" content.
        mock_endpoint = Mock()
        mock_endpoint.parse_response = Mock(
            return_value=Mock(data=Mock())  # truthy `.data`
        )
        mock_worker.inference_client.endpoint = mock_endpoint

        ctx = credit_factory(credit_id=99)
        cb = mock_worker._make_first_token_callback(ctx)
        assert cb is not None

        result = await cb(123_456, MagicMock())

        assert result is True
        assert ctx.first_token_sent is True
        sent = mock_worker.credit_return_push_client.send.await_args.args[0]
        assert isinstance(sent, FirstToken)
        assert sent.credit_id == 99
        assert sent.ttft_ns == 123_456

    async def test_callback_returns_false_when_no_meaningful_content(
        self, mock_worker: Worker, credit_factory
    ) -> None:
        """Empty / usage-only chunks must not emit FirstToken or flip the flag."""
        mock_worker._prefill_concurrency_enabled = True
        mock_endpoint = Mock()
        mock_endpoint.parse_response = Mock(return_value=None)
        mock_worker.inference_client.endpoint = mock_endpoint

        ctx = credit_factory()
        cb = mock_worker._make_first_token_callback(ctx)
        assert cb is not None

        result = await cb(0, MagicMock())

        assert result is False
        assert ctx.first_token_sent is False
        mock_worker.credit_return_push_client.send.assert_not_called()


# ============================================================
# Schedule + tracking dict
# ============================================================


@pytest.mark.asyncio
class TestScheduleCreditDropTask:
    """`_schedule_credit_drop_task` registers the task in `credit_tasks` and fires the done callback."""

    async def test_schedule_inserts_and_callback_clears(
        self, mock_worker: Worker, credit_factory
    ) -> None:
        ctx = credit_factory(credit_id=77)

        # Replace inner work with a noop so the task completes immediately.
        async def fast_path(_ctx: CreditContext) -> None:
            _ctx.returned = True

        mock_worker._on_credit_drop_message_task = fast_path  # type: ignore[assignment]

        mock_worker._schedule_credit_drop_task(ctx.credit)

        assert (CreditPhase.PROFILING, 77) in mock_worker.credit_tasks
        task = mock_worker.credit_tasks[(CreditPhase.PROFILING, 77)]
        await task
        # Done callback runs synchronously after await; pop happens inside it.
        await asyncio.sleep(0)
        assert (CreditPhase.PROFILING, 77) not in mock_worker.credit_tasks
