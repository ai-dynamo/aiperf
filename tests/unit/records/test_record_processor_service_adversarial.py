# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial coverage tests for `RecordProcessor`.

These tests exercise the branches the existing happy-path suite leaves
uncovered:

1. `_uses_controller_control_channel` — group-mode flips the answer.
2. `_on_pod_lifecycle_message` — every branch:
    a. GroupTokenizerReady early-return
    b. Non-GroupPeerCommand ignored
    c. Dealer-client absent → ignored
    d. PROFILE_CONFIGURE delegates to `_configure_for_profiling`
    e. SHUTDOWN delegates to `self.stop()`
    f. ABORT logs error, sends best-effort ack, calls `os._exit(1)`
    g. Unknown command logs warning and skips ack
3. `_register_with_worker_group_manager` — delegates to retry helper, skip
   when no dealer client.
4. `_flush_pending_metric_records_task` — delegates to `_flush_pending_metric_records`.
5. `_profile_configure_command` — delegates to `_configure_for_profiling`.
6. `_profile_complete_command` — stops every child, in order.
7. `get_tokenizer` — delegates to parser.
8. `_on_tokenizer_ready` — success vs failure (failure os._exit's).
9. `_merge_metric_results` — exception filter + dict merge order.
10. `_flush_pending_metric_records` — single-record path produces a single
    wire message (not a batch); empty buffer → noop.
11. `_enqueue_metric_record` — flush triggered when batch fills.
12. `_free_record_data` — RAW export-level keeps responses; default frees.
13. `_process_record` — None values filtered out of returned list.
14. `main()` — bootstrap dispatch.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.common.enums import CommandType, ExportLevel
from aiperf.common.metric_records_wire import (
    MetricRecordMetadata,
    MetricRecordsBatchWireMessage,
    MetricRecordsData,
    MetricRecordsWireMessage,
)
from aiperf.common.pod_lifecycle_structs import (
    GroupManagerToPeerMessage,
    GroupPeerCommand,
    GroupPeerCommandAck,
    GroupTokenizerReady,
)
from aiperf.records.record_processor_service import RecordProcessor, main

# ============================================================================
# Canonical metadata helper
# ============================================================================

_VALID_METADATA: dict[str, Any] = {
    "request_num": 0,
    "session_num": 0,
    "conversation_id": "conv-1",
    "turn_index": 0,
    "request_start_ns": 100,
    "request_end_ns": 200,
    "worker_id": "worker-1",
    "record_processor_id": "rp-1",
    "benchmark_phase": "profiling",
}


def _meta_with(**overrides: Any) -> MetricRecordMetadata:
    return MetricRecordMetadata(**{**_VALID_METADATA, **overrides})


def _record_data(**overrides: Any) -> MetricRecordsData:
    metrics = overrides.pop("metrics", {"request_latency": 1.0})
    return MetricRecordsData(metadata=_meta_with(**overrides), metrics=metrics)


def _make_processor(
    *,
    bind_methods: list[str] | None = None,
    has_dealer: bool = False,
    export_level: ExportLevel = ExportLevel.SUMMARY,
    ingest_batch_size: int = 100,
    children: list[Any] | None = None,
) -> MagicMock:
    """Build a `RecordProcessor` mock with `bind_methods` bound to the real fn."""
    proc = MagicMock(spec=RecordProcessor)
    proc.service_id = "rp-1"
    proc.service_type = "record_processor"
    proc._pending_metric_records = []
    proc._ingest_batch_size = ingest_batch_size
    proc._tokenizer_bundles = {}
    proc._tokenizer_ready = MagicMock()
    proc.run = MagicMock()
    proc.run.cfg.runtime.uses_worker_group_manager = has_dealer
    proc.run.cfg.output.export_level = export_level
    proc._children = children or []
    proc.records_processors = []
    proc.pod_lifecycle_dealer_client = AsyncMock() if has_dealer else None
    proc.records_push_client = MagicMock()
    proc.records_push_client.push = AsyncMock()
    proc.inference_result_parser = MagicMock()
    proc.inference_result_parser.configure = AsyncMock()
    proc.inference_result_parser.get_tokenizer = AsyncMock()
    proc._configure_for_profiling = AsyncMock()
    proc._flush_pending_metric_records = AsyncMock()
    proc.stop = AsyncMock()
    for level in ("trace", "debug", "info", "warning", "error", "exception"):
        setattr(proc, level, MagicMock())

    method_map = {
        "_uses_controller_control_channel": RecordProcessor._uses_controller_control_channel,
        "_on_pod_lifecycle_message": RecordProcessor._on_pod_lifecycle_message,
        "_register_with_worker_group_manager": RecordProcessor._register_with_worker_group_manager,
        "_flush_pending_metric_records_task": RecordProcessor._flush_pending_metric_records_task,
        "_configure_for_profiling": RecordProcessor._configure_for_profiling,
        "_profile_configure_command": RecordProcessor._profile_configure_command,
        "_profile_complete_command": RecordProcessor._profile_complete_command,
        "get_tokenizer": RecordProcessor.get_tokenizer,
        "_on_tokenizer_ready": RecordProcessor._on_tokenizer_ready,
        "_merge_metric_results": RecordProcessor._merge_metric_results,
        "_flush_pending_metric_records": RecordProcessor._flush_pending_metric_records,
        "_enqueue_metric_record": RecordProcessor._enqueue_metric_record,
        "_free_record_data": RecordProcessor._free_record_data,
        "_process_record": RecordProcessor._process_record,
        "_on_inference_results": RecordProcessor._on_inference_results,
    }
    for name in bind_methods or []:
        fn = method_map[name]
        wrapped = getattr(fn, "__wrapped__", fn)
        setattr(proc, name, wrapped.__get__(proc))
    return proc


# ============================================================================
# 1) `_uses_controller_control_channel`
# ============================================================================


class TestUsesControllerControlChannel:
    def test_group_mode_returns_false(self) -> None:
        proc = _make_processor(
            bind_methods=["_uses_controller_control_channel"],
            has_dealer=True,
        )
        assert proc._uses_controller_control_channel() is False

    def test_non_group_mode_returns_true(self) -> None:
        proc = _make_processor(
            bind_methods=["_uses_controller_control_channel"],
            has_dealer=False,
        )
        assert proc._uses_controller_control_channel() is True


# ============================================================================
# 2) `_on_pod_lifecycle_message` — every branch
# ============================================================================


class TestOnPodLifecycleMessage:
    """The lifecycle dispatcher is a state machine. Every branch is locked."""

    @pytest.mark.asyncio
    async def test_tokenizer_ready_message_routes_to_tokenizer_handler(
        self,
    ) -> None:
        proc = _make_processor(
            bind_methods=["_on_pod_lifecycle_message"], has_dealer=True
        )
        proc._on_tokenizer_ready = AsyncMock()
        msg = GroupTokenizerReady(
            service_id="wgm-1",
            bundles={"test-model": "/path/to/bundle"},
            success=True,
        )

        await proc._on_pod_lifecycle_message(msg)

        proc._on_tokenizer_ready.assert_awaited_once_with(msg)
        # Tokenizer-ready early-return → no ack-send (the tokenizer handler
        # owns lifecycle).
        proc.pod_lifecycle_dealer_client.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_command_message_ignored(self) -> None:
        """Anything that isn't a `GroupPeerCommand` (or tokenizer ready)
        falls through with no side effect."""
        proc = _make_processor(
            bind_methods=["_on_pod_lifecycle_message"], has_dealer=True
        )
        # Use a real GroupManagerToPeerMessage subtype that isn't either.
        # GroupPeerShutdown is sent in the OTHER direction, but isinstance()
        # is what gates this — pass a dummy that's neither.
        weird = MagicMock(spec=GroupManagerToPeerMessage)

        await proc._on_pod_lifecycle_message(weird)

        proc.stop.assert_not_awaited()
        proc._configure_for_profiling.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_dealer_client_short_circuits(self) -> None:
        """If `pod_lifecycle_dealer_client` is None, skip processing — even
        for valid commands. Important for non-K8s lifecycle paths."""
        proc = _make_processor(
            bind_methods=["_on_pod_lifecycle_message"], has_dealer=False
        )
        msg = GroupPeerCommand(
            cid="c1", service_id="rp-1", command=str(CommandType.PROFILE_CONFIGURE)
        )

        await proc._on_pod_lifecycle_message(msg)

        proc._configure_for_profiling.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_profile_configure_command_delegates_and_acks(self) -> None:
        proc = _make_processor(
            bind_methods=["_on_pod_lifecycle_message"], has_dealer=True
        )
        msg = GroupPeerCommand(
            cid="c1", service_id="rp-1", command=str(CommandType.PROFILE_CONFIGURE)
        )

        await proc._on_pod_lifecycle_message(msg)

        proc._configure_for_profiling.assert_awaited_once()
        # Ack sent.
        sent = proc.pod_lifecycle_dealer_client.send.await_args.args[0]
        assert isinstance(sent, GroupPeerCommandAck)
        assert sent.cid == "c1"

    @pytest.mark.asyncio
    async def test_abort_command_force_exits_with_code_1(self) -> None:
        """ABORT triggers `os._exit(1)` — locks the gotcha referenced in
        `feedback_prefer_os_exit_for_hard_kills.md` and the source comment
        at lines 162-171.
        """
        proc = _make_processor(
            bind_methods=["_on_pod_lifecycle_message"], has_dealer=True
        )
        msg = GroupPeerCommand(
            cid="c1", service_id="rp-1", command=str(CommandType.ABORT)
        )

        # Real os._exit() never returns; simulate with SystemExit so the
        # post-exit ack block is unreachable and our send-call-count assertion
        # matches production behavior.
        with (
            patch(
                "aiperf.records.record_processor_service.os._exit",
                side_effect=SystemExit(1),
            ) as mock_exit,
            pytest.raises(SystemExit),
        ):
            await proc._on_pod_lifecycle_message(msg)

        # Best-effort ack attempted (the one inside the ABORT branch).
        proc.pod_lifecycle_dealer_client.send.assert_awaited_once()
        # Force-exited.
        mock_exit.assert_called_once_with(1)
        # Error logged with explicit "force-exiting" wording.
        assert any("force-exiting" in str(c.args[0]) for c in proc.error.call_args_list)

    @pytest.mark.asyncio
    async def test_abort_swallows_ack_send_failure_then_exits(self) -> None:
        """ABORT-path ack failure must not block the os._exit — the WGM is
        already gone; we just want the kubelet restart."""
        proc = _make_processor(
            bind_methods=["_on_pod_lifecycle_message"], has_dealer=True
        )
        proc.pod_lifecycle_dealer_client.send = AsyncMock(
            side_effect=ConnectionError("peer gone")
        )
        msg = GroupPeerCommand(
            cid="c1", service_id="rp-1", command=str(CommandType.ABORT)
        )

        with (
            patch(
                "aiperf.records.record_processor_service.os._exit",
                side_effect=SystemExit(1),
            ) as mock_exit,
            pytest.raises(SystemExit),
        ):
            await proc._on_pod_lifecycle_message(msg)

        # Still exited, despite the ack failure.
        mock_exit.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_unknown_command_warns_and_skips_ack(self) -> None:
        """Unknown command words must not be silently acked — defensive
        log so misrouting is visible. The source uses `return` after the
        warning, so the trailing ack-send block is bypassed.
        """
        proc = _make_processor(
            bind_methods=["_on_pod_lifecycle_message"], has_dealer=True
        )
        msg = GroupPeerCommand(cid="c1", service_id="rp-1", command="nonsense-command")

        await proc._on_pod_lifecycle_message(msg)

        proc.warning.assert_called_once()
        assert "nonsense-command" in proc.warning.call_args.args[0]
        # No ack sent because of the early return.
        proc.pod_lifecycle_dealer_client.send.assert_not_awaited()


# ============================================================================
# 3) `_register_with_worker_group_manager`
# ============================================================================


class TestRegisterWithWorkerGroupManager:
    @pytest.mark.asyncio
    async def test_no_dealer_client_skips_registration(self) -> None:
        proc = _make_processor(
            bind_methods=["_register_with_worker_group_manager"], has_dealer=False
        )

        # Should not raise (no dealer to talk to).
        await proc._register_with_worker_group_manager()

    @pytest.mark.asyncio
    async def test_with_dealer_client_calls_hello_with_retry(self) -> None:
        proc = _make_processor(
            bind_methods=["_register_with_worker_group_manager"], has_dealer=True
        )
        proc._pod_index = "0"
        with patch(
            "aiperf.records.record_processor_service._send_group_peer_hello_with_retry",
            new=AsyncMock(),
        ) as helper:
            await proc._register_with_worker_group_manager()

        helper.assert_awaited_once()
        kwargs = helper.await_args.kwargs
        assert kwargs["service_id"] == "rp-1"
        assert kwargs["pod_index"] == "0"


# ============================================================================
# 4) Background tasks + simple delegators
# ============================================================================


class TestSimpleDelegators:
    @pytest.mark.asyncio
    async def test_flush_pending_metric_records_task_delegates(self) -> None:
        proc = _make_processor(bind_methods=["_flush_pending_metric_records_task"])

        await proc._flush_pending_metric_records_task()

        proc._flush_pending_metric_records.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_configure_for_profiling_delegates_to_parser(self) -> None:
        proc = _make_processor(bind_methods=["_configure_for_profiling"])

        await proc._configure_for_profiling()

        proc.inference_result_parser.configure.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_profile_configure_command_delegates(self) -> None:
        proc = _make_processor(bind_methods=["_profile_configure_command"])
        cmd = MagicMock()

        await proc._profile_configure_command(cmd)

        proc._configure_for_profiling.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_profile_complete_stops_every_child(self) -> None:
        """PROFILE_COMPLETE must call .stop() on every child processor so
        their buffers (e.g. RawRecordWriterProcessor) flush to disk before
        the aggregator reads them."""
        child_a = MagicMock()
        child_a.stop = AsyncMock()
        child_b = MagicMock()
        child_b.stop = AsyncMock()
        proc = _make_processor(
            bind_methods=["_profile_complete_command"],
            children=[child_a, child_b],
        )
        cmd = MagicMock()

        await proc._profile_complete_command(cmd)

        child_a.stop.assert_awaited_once()
        child_b.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_tokenizer_delegates_to_parser(self) -> None:
        proc = _make_processor(bind_methods=["get_tokenizer"])
        tok = MagicMock()
        proc.inference_result_parser.get_tokenizer = AsyncMock(return_value=tok)

        result = await proc.get_tokenizer("test-model")

        proc.inference_result_parser.get_tokenizer.assert_awaited_once_with(
            "test-model"
        )
        assert result is tok


# ============================================================================
# 5) `_on_tokenizer_ready`
# ============================================================================


class TestOnTokenizerReady:
    @pytest.mark.asyncio
    async def test_failure_force_exits_with_code_1(self) -> None:
        """Tokenizer download failure → kubelet must restart this pod.

        Locks the same `os._exit(1)` contract as ABORT — see
        `feedback_prefer_os_exit_for_hard_kills.md`."""
        proc = _make_processor(bind_methods=["_on_tokenizer_ready"])
        msg = GroupTokenizerReady(
            service_id="wgm-1",
            bundles={},
            success=False,
            error_message="HF rate limited",
        )

        with (
            patch(
                "aiperf.records.record_processor_service.os._exit",
                side_effect=SystemExit(1),
            ) as mock_exit,
            pytest.raises(SystemExit),
        ):
            await proc._on_tokenizer_ready(msg)

        # Logged the upstream error message.
        assert any(
            "HF rate limited" in str(c.args[0]) for c in proc.error.call_args_list
        )
        mock_exit.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_success_updates_bundles_and_signals_event(self) -> None:
        proc = _make_processor(bind_methods=["_on_tokenizer_ready"])
        msg = GroupTokenizerReady(
            service_id="wgm-1",
            bundles={"test-model": "/snapshots/test", "phi": "/snapshots/phi"},
            success=True,
        )

        await proc._on_tokenizer_ready(msg)

        assert proc._tokenizer_bundles == {
            "test-model": "/snapshots/test",
            "phi": "/snapshots/phi",
        }
        proc._tokenizer_ready.set.assert_called_once()


# ============================================================================
# 6) `_merge_metric_results` — exception filter + dict merge ordering
# ============================================================================


class TestMergeMetricResults:
    def test_exceptions_filtered_remaining_dicts_merged(self) -> None:
        proc = _make_processor(bind_methods=["_merge_metric_results"])
        good_a = {"request_latency": 12.0}
        bad = RuntimeError("processor crashed")
        good_b = {"ttft": 5.0}

        merged = proc._merge_metric_results([good_a, bad, good_b])

        assert merged == {"request_latency": 12.0, "ttft": 5.0}
        # Error logged.
        assert any(
            "processor crashed" in str(c.args[0]) for c in proc.error.call_args_list
        )

    def test_later_keys_overwrite_earlier_in_merge(self) -> None:
        """Adversarial: same key in two dicts — last writer wins per .update() semantics."""
        proc = _make_processor(bind_methods=["_merge_metric_results"])
        merged = proc._merge_metric_results([{"x": 1.0}, {"x": 2.0, "y": 3.0}])
        assert merged == {"x": 2.0, "y": 3.0}

    def test_empty_input_returns_empty_dict(self) -> None:
        proc = _make_processor(bind_methods=["_merge_metric_results"])
        assert proc._merge_metric_results([]) == {}

    def test_all_exceptions_returns_empty_dict_and_logs_each(self) -> None:
        proc = _make_processor(bind_methods=["_merge_metric_results"])
        merged = proc._merge_metric_results(
            [ValueError("v"), TypeError("t"), RuntimeError("r")]
        )
        assert merged == {}
        # Each exception logged.
        assert proc.error.call_count == 3


# ============================================================================
# 7) `_flush_pending_metric_records` — single vs batch
# ============================================================================


class TestFlushPendingMetricRecords:
    @pytest.mark.asyncio
    async def test_empty_buffer_short_circuits(self) -> None:
        proc = _make_processor(bind_methods=["_flush_pending_metric_records"])
        proc._pending_metric_records = []

        # Real method, not the AsyncMock from _make_processor.
        await proc._flush_pending_metric_records()

        proc.records_push_client.push.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_single_record_uses_single_wire_message_not_batch(self) -> None:
        """One pending record → MetricRecordsWireMessage (not batch).

        Locks the size-class optimization and rules out the silent
        regression where every flush would always batch.
        """
        proc = _make_processor(bind_methods=["_flush_pending_metric_records"])
        proc._pending_metric_records = [_record_data()]

        await proc._flush_pending_metric_records()

        pushed = proc.records_push_client.push.await_args.args[0]
        assert isinstance(pushed, MetricRecordsWireMessage)
        assert not isinstance(pushed, MetricRecordsBatchWireMessage)
        assert proc._pending_metric_records == []

    @pytest.mark.asyncio
    async def test_multiple_records_use_batch_wire_message(self) -> None:
        proc = _make_processor(bind_methods=["_flush_pending_metric_records"])
        proc._pending_metric_records = [_record_data(), _record_data()]

        await proc._flush_pending_metric_records()

        pushed = proc.records_push_client.push.await_args.args[0]
        assert isinstance(pushed, MetricRecordsBatchWireMessage)
        assert len(pushed.records) == 2
        assert proc._pending_metric_records == []

    @pytest.mark.asyncio
    async def test_buffer_swap_happens_before_push_protects_against_concurrent_append(
        self,
    ) -> None:
        """The flush method captures the buffer reference and resets to []
        BEFORE awaiting the push. Adversarial: a record appended during
        the await must NOT be lost — it must appear in the next flush."""
        proc = _make_processor(bind_methods=["_flush_pending_metric_records"])
        proc._pending_metric_records = [_record_data()]
        appended_during_push = _record_data(request_num=99)
        push_started = False

        async def _push_observe(_msg: Any) -> None:
            nonlocal push_started
            push_started = True
            # During push, simulate a concurrent append.
            proc._pending_metric_records.append(appended_during_push)

        proc.records_push_client.push.side_effect = _push_observe

        await proc._flush_pending_metric_records()

        assert push_started
        # The record appended during push survived the swap.
        assert proc._pending_metric_records == [appended_during_push]


# ============================================================================
# 8) `_enqueue_metric_record` — auto-flush trigger
# ============================================================================


class TestEnqueueMetricRecord:
    @pytest.mark.asyncio
    async def test_enqueue_below_batch_size_does_not_flush(self) -> None:
        proc = _make_processor(
            bind_methods=["_enqueue_metric_record"], ingest_batch_size=10
        )

        await proc._enqueue_metric_record(
            metadata=_meta_with(),
            metrics={"x": 1.0},
            trace_data=None,
            error=None,
        )

        assert len(proc._pending_metric_records) == 1
        proc._flush_pending_metric_records.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_enqueue_at_batch_size_triggers_flush(self) -> None:
        proc = _make_processor(
            bind_methods=["_enqueue_metric_record"], ingest_batch_size=2
        )
        # Pre-fill to size-1.
        proc._pending_metric_records = [_record_data()]

        await proc._enqueue_metric_record(
            metadata=_meta_with(),
            metrics={"x": 1.0},
            trace_data=None,
            error=None,
        )

        proc._flush_pending_metric_records.assert_awaited_once()


# ============================================================================
# 9) `_free_record_data` — RAW export keeps responses; default frees them
# ============================================================================


class TestFreeRecordData:
    """Adversarial verification of the GC-aid behavior at lines 425-450."""

    def test_summary_export_frees_responses(self) -> None:
        proc = _make_processor(
            bind_methods=["_free_record_data"], export_level=ExportLevel.SUMMARY
        )
        record = MagicMock()
        record.responses = [MagicMock(), MagicMock()]
        record.turns = [MagicMock()]
        original_trace = MagicMock(name="trace")
        record.trace_data = original_trace
        record.request_headers = {"X": "Y"}
        record.error = None
        record.request_info = MagicMock()
        record.request_info.turns = [MagicMock()]
        record.request_info.system_message = "sys"
        record.request_info.user_context_message = "user"
        parsed = MagicMock()
        parsed.responses = [MagicMock()]

        trace_data, _error = proc._free_record_data(record, parsed)

        # The trace_data tuple element is the captured original; record.trace_data
        # was freed (set to None) AFTER capture.
        assert trace_data is original_trace
        # SUMMARY export → responses freed.
        assert record.responses is None
        assert record.turns is None
        assert record.trace_data is None
        assert record.request_headers is None
        assert record.request_info.turns is None
        assert record.request_info.system_message is None
        assert record.request_info.user_context_message is None
        # Parsed responses freed unconditionally.
        assert parsed.responses is None

    def test_raw_export_keeps_responses_for_writer(self) -> None:
        """RAW export means the raw writer still needs the responses; don't free."""
        proc = _make_processor(
            bind_methods=["_free_record_data"], export_level=ExportLevel.RAW
        )
        record = MagicMock()
        responses = [MagicMock()]
        record.responses = responses
        record.turns = [MagicMock()]
        record.trace_data = MagicMock()
        record.request_headers = {"X": "Y"}
        record.error = None
        record.request_info = MagicMock()
        parsed = MagicMock()

        proc._free_record_data(record, parsed)

        # Responses preserved despite the rest being freed.
        assert record.responses is responses
        assert record.turns is None

    def test_returns_trace_and_error_tuple(self) -> None:
        proc = _make_processor(bind_methods=["_free_record_data"])
        record = MagicMock()
        record.responses = []
        record.turns = []
        trace = MagicMock(name="trace")
        err = MagicMock(name="err")
        record.trace_data = trace
        record.error = err
        record.request_headers = {}
        record.request_info = MagicMock()
        parsed = MagicMock()

        trace_out, error_out = proc._free_record_data(record, parsed)

        assert trace_out is trace
        assert error_out is err

    def test_no_request_info_does_not_crash(self) -> None:
        """Adversarial: `record.request_info is None` should not raise."""
        proc = _make_processor(bind_methods=["_free_record_data"])
        record = MagicMock()
        record.responses = []
        record.turns = []
        record.trace_data = None
        record.error = None
        record.request_headers = None
        record.request_info = None
        parsed = MagicMock()

        # Should not raise.
        proc._free_record_data(record, parsed)


# ============================================================================
# 10) `_process_record` — None values filtered out
# ============================================================================


class TestProcessRecord:
    @pytest.mark.asyncio
    async def test_none_results_filtered_from_returned_list(self) -> None:
        """A processor returning None (no metrics for this record) must NOT
        end up in the merge list — it would crash `.update()`. Locks the
        `result is not None` filter at line 463."""
        proc = _make_processor(bind_methods=["_process_record"])
        proc_a = MagicMock()
        proc_a.process_record = AsyncMock(return_value={"x": 1.0})
        proc_b = MagicMock()
        proc_b.process_record = AsyncMock(return_value=None)
        proc_c = MagicMock()
        proc_c.process_record = AsyncMock(return_value={"y": 2.0})
        proc.records_processors = [proc_a, proc_b, proc_c]

        results = await proc._process_record(MagicMock(), _meta_with())

        # None elided; both real dicts present.
        assert {"x": 1.0} in results
        assert {"y": 2.0} in results
        assert None not in results

    @pytest.mark.asyncio
    async def test_processor_exceptions_returned_not_raised(self) -> None:
        """asyncio.gather with `return_exceptions=True` means processor
        crashes come back as BaseException instances; merging filters them."""
        proc = _make_processor(bind_methods=["_process_record"])
        crashing = MagicMock()
        crashing.process_record = AsyncMock(side_effect=ValueError("bad input"))
        proc.records_processors = [crashing]

        results = await proc._process_record(MagicMock(), _meta_with())

        # The exception came through (not raised by gather).
        assert any(isinstance(r, ValueError) for r in results)


# ============================================================================
# 11) Error-message contract assertions
# ============================================================================


class TestErrorMessageContracts:
    """Validate that error logs name the offending peer/cause, per the
    project's semantic-ceiling rule.
    """

    @pytest.mark.asyncio
    async def test_abort_log_names_service_id_and_force_exit(self) -> None:
        proc = _make_processor(
            bind_methods=["_on_pod_lifecycle_message"], has_dealer=True
        )
        proc.service_id = "rp-7"
        msg = GroupPeerCommand(
            cid="c1", service_id="rp-7", command=str(CommandType.ABORT)
        )

        with (
            patch(
                "aiperf.records.record_processor_service.os._exit",
                side_effect=SystemExit(1),
            ),
            pytest.raises(SystemExit),
        ):
            await proc._on_pod_lifecycle_message(msg)

        err_text = " ".join(str(c.args[0]) for c in proc.error.call_args_list)
        assert "rp-7" in err_text
        assert "force-exiting" in err_text

    @pytest.mark.asyncio
    async def test_tokenizer_failure_log_names_specific_error(self) -> None:
        proc = _make_processor(bind_methods=["_on_tokenizer_ready"])
        msg = GroupTokenizerReady(
            service_id="wgm-1",
            bundles={},
            success=False,
            error_message="403 Forbidden from huggingface.co",
        )

        with (
            patch(
                "aiperf.records.record_processor_service.os._exit",
                side_effect=SystemExit(1),
            ),
            pytest.raises(SystemExit),
        ):
            await proc._on_tokenizer_ready(msg)

        err_text = " ".join(str(c.args[0]) for c in proc.error.call_args_list)
        assert "403 Forbidden" in err_text


# ============================================================================
# 12) `main()`
# ============================================================================


class TestMainEntryPoint:
    def test_main_dispatches_record_processor_service_type(self) -> None:
        with patch("aiperf.common.bootstrap.bootstrap_and_run_service") as bootstrap:
            main()

        bootstrap.assert_called_once()
        from aiperf.plugin.enums import ServiceType

        assert bootstrap.call_args.args[0] == ServiceType.RECORD_PROCESSOR


# ============================================================================
# 12b) `__init__` — plugin loading branch matrix
# ============================================================================


class TestRecordProcessorInit:
    """Lock the `__init__` plugin-loader branches.

    Covers:
    - tokenizer-ready pre-set for non-K8s runs
    - tokenizer-ready left unset for K8s runs (until WGM advertises)
    - dealer client created only when uses_worker_group_manager=True
    - PostProcessorDisabled → silent skip (debug log)
    - other exception → re-raised after exception() log
    - happy multi-plugin path with attach_child_lifecycle
    """

    @staticmethod
    def _make_run(
        *,
        service_run_type: str = "process",
        uses_worker_group_manager: bool = False,
    ) -> MagicMock:
        run = MagicMock()
        run.cfg.benchmark.runtime.service_run_type = service_run_type
        run.cfg.benchmark.runtime.uses_worker_group_manager = uses_worker_group_manager
        return run

    @staticmethod
    def _patch_super(captured: dict[str, Any]) -> Any:
        def _capture(self: Any, **kwargs: Any) -> None:
            captured.update(kwargs)
            # Mimic the BaseComponentService attributes that the real super
            # sets up. We poke them directly because we're skipping the
            # full lifecycle wiring.
            self.run = kwargs["run"]
            self.service_id = kwargs.get("service_id", "rp-init")
            self.comms = MagicMock()
            self.attach_child_lifecycle = MagicMock()
            self.debug = MagicMock()
            self.exception = MagicMock()

        return patch(
            "aiperf.common.mixins.PullClientMixin.__init__",
            new=_capture,
        )

    def _build_processor(
        self,
        run: MagicMock,
        plugin_entries: list[Any],
        plugin_get_class: Any,
    ) -> tuple[RecordProcessor, dict[str, Any]]:
        captured: dict[str, Any] = {}
        with (
            self._patch_super(captured),
            patch(
                "aiperf.records.record_processor_service.InferenceResultParser",
                return_value=MagicMock(),
            ),
            patch(
                "aiperf.records.record_processor_service.plugins.iter_entries",
                return_value=iter(plugin_entries),
            ),
            patch(
                "aiperf.records.record_processor_service.plugins.get_class",
                side_effect=plugin_get_class,
            ),
        ):
            proc = RecordProcessor(run=run, service_id="rp-init")
        return proc, captured

    def test_non_k8s_pre_sets_tokenizer_ready_event(self) -> None:
        run = self._make_run(service_run_type="process")
        proc, _ = self._build_processor(run, [], lambda *_: None)

        assert proc._tokenizer_ready.is_set()
        # No dealer in non-group mode.
        assert proc.pod_lifecycle_dealer_client is None

    def test_kubernetes_leaves_tokenizer_ready_unset_until_wgm_signals(
        self,
    ) -> None:
        from aiperf.plugin.enums import ServiceRunType

        run = self._make_run(service_run_type=ServiceRunType.KUBERNETES)
        proc, _ = self._build_processor(run, [], lambda *_: None)

        assert not proc._tokenizer_ready.is_set()

    def test_group_mode_creates_dealer_client(self) -> None:
        run = self._make_run(uses_worker_group_manager=True)
        proc, _ = self._build_processor(run, [], lambda *_: None)

        assert proc.pod_lifecycle_dealer_client is not None
        # Dealer wired its receiver to the lifecycle handler.
        proc.pod_lifecycle_dealer_client.register_receiver.assert_called_once()

    def test_disabled_plugin_silently_skipped(self) -> None:
        run = self._make_run()
        entries = [SimpleNamespace(name="disabled-one")]

        def _get_class(_cat: Any, _name: str) -> Any:
            from aiperf.common.exceptions import PostProcessorDisabled

            def _disabled(**_: Any) -> Any:
                raise PostProcessorDisabled("disabled")

            return _disabled

        proc, _ = self._build_processor(run, entries, _get_class)

        assert proc.records_processors == []

    def test_plugin_failure_propagates_with_exception_log(self) -> None:
        run = self._make_run()
        entries = [SimpleNamespace(name="boom")]

        def _get_class(_cat: Any, _name: str) -> Any:
            def _explode(**_: Any) -> Any:
                raise RuntimeError("plugin construct error")

            return _explode

        captured: dict[str, Any] = {}
        with (
            self._patch_super(captured),
            patch(
                "aiperf.records.record_processor_service.InferenceResultParser",
                return_value=MagicMock(),
            ),
            patch(
                "aiperf.records.record_processor_service.plugins.iter_entries",
                return_value=iter(entries),
            ),
            patch(
                "aiperf.records.record_processor_service.plugins.get_class",
                side_effect=_get_class,
            ),
            pytest.raises(RuntimeError, match="plugin construct error"),
        ):
            RecordProcessor(run=run, service_id="rp-init")

    def test_successful_plugin_attached_to_lifecycle(self) -> None:
        run = self._make_run()
        entries = [SimpleNamespace(name="good")]
        instance = MagicMock()

        def _get_class(_cat: Any, _name: str) -> Any:
            return lambda **_: instance

        proc, _ = self._build_processor(run, entries, _get_class)

        assert proc.records_processors == [instance]
        proc.attach_child_lifecycle.assert_called_with(instance)


# ============================================================================
# 13) Parametrized command-routing — branch matrix
# ============================================================================


class TestPodLifecycleCommandMatrix:
    """Locks the dispatch table for the lifecycle handler."""

    @pytest.mark.parametrize(
        "command_str,expected_method,expects_ack",
        [
            param(
                str(CommandType.PROFILE_CONFIGURE),
                "_configure_for_profiling",
                True,
                id="profile-configure",
            ),
            param(
                str(CommandType.SHUTDOWN),
                "stop",
                True,
                id="shutdown",
            ),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_command_routes_to_method(
        self,
        command_str: str,
        expected_method: str,
        expects_ack: bool,
    ) -> None:
        proc = _make_processor(
            bind_methods=["_on_pod_lifecycle_message"], has_dealer=True
        )
        msg = GroupPeerCommand(cid="cid-x", service_id="rp-1", command=command_str)

        await proc._on_pod_lifecycle_message(msg)

        getattr(proc, expected_method).assert_awaited_once()
        if expects_ack:
            sent = proc.pod_lifecycle_dealer_client.send.await_args.args[0]
            assert isinstance(sent, GroupPeerCommandAck)
            assert sent.cid == "cid-x"


# ============================================================================
# 14) Real-config sanity check (anti-MagicMock-drift insurance)
# ============================================================================


class TestRecordProcessorRealConfigInteraction:
    """Builds a real `BenchmarkConfig` and exercises `_free_record_data`
    against it. Locks the contract that the export-level branch reads from
    `run.cfg.output.export_level` — not some stale path that MagicMock
    auto-creates."""

    @pytest.mark.parametrize(
        "raw_flag,responses_kept",
        [
            param(False, False, id="summary-frees"),
            param(True, True, id="raw-keeps"),
        ],
    )  # fmt: skip
    def test_real_config_drives_free_record_data(
        self, raw_flag: bool, responses_kept: bool
    ) -> None:
        from aiperf.config import BenchmarkConfig

        config = BenchmarkConfig(
            models=["test-model"],
            endpoint={"type": "chat", "urls": ["http://localhost:8000/v1/test"]},
            datasets=[
                {
                    "name": "default",
                    "type": "synthetic",
                    "entries": 1,
                    "prompts": {"isl": 8, "osl": 8},
                }
            ],
            phases=[
                {
                    "name": "default",
                    "type": "concurrency",
                    "requests": 1,
                    "concurrency": 1,
                }
            ],
            artifacts={"raw": raw_flag},
        )
        # Sanity: derived export_level matches expectation.
        assert (config.output.export_level == ExportLevel.RAW) == raw_flag

        proc = MagicMock(spec=RecordProcessor)
        proc.run = MagicMock()
        proc.run.cfg = config
        proc._free_record_data = RecordProcessor._free_record_data.__get__(proc)

        record = MagicMock()
        responses = [MagicMock()]
        record.responses = responses
        record.turns = []
        record.trace_data = None
        record.error = None
        record.request_headers = None
        record.request_info = None
        parsed = MagicMock()

        proc._free_record_data(record, parsed)

        if responses_kept:
            assert record.responses is responses
        else:
            assert record.responses is None


# Ref import to keep `contextlib` listed (used by source's ABORT path; left
# unused here but kept for the absence-of-cm assertion if it ever comes
# back).
_ = contextlib
