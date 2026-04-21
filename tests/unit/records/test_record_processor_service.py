# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.inference_wire import build_inference_results_wire_message
from aiperf.common.metric_records_wire import (
    MetricRecordMetadata,
    MetricRecordsBatchWireMessage,
    MetricRecordsData,
)
from aiperf.common.utils import compute_time_ns
from aiperf.records.record_processor_service import RecordProcessor


class TestRecordProcessorCreateMetricRecordMetadata:
    """Test the RecordProcessor._create_metric_record_metadata method."""

    @pytest.fixture
    def mock_record_processor(self, service_config, user_config):
        """Create a mock RecordProcessor instance for testing."""
        instance = MagicMock(spec=RecordProcessor)
        instance.service_id = "test-processor-id"
        instance.info = MagicMock()
        return instance

    def test_create_metadata_without_end_and_no_responses(
        self, mock_record_processor, sample_request_record
    ):
        """Test creating metadata when RequestRecord has no end_perf_ns and no responses."""
        sample_request_record.end_perf_ns = None
        sample_request_record.responses = []
        sample_request_record.request_info.credit_num = 1
        sample_request_record.request_info.credit_phase = "profiling"
        sample_request_record.recv_start_perf_ns = (
            sample_request_record.start_perf_ns + 10_000
        )

        worker_id = "worker-1"

        metadata = RecordProcessor._create_metric_record_metadata(
            mock_record_processor, sample_request_record, worker_id
        )

        # When no end_perf_ns and no responses, should use start_perf_ns as fallback
        expected_end_ns = sample_request_record.timestamp_ns
        assert metadata.request_start_ns == sample_request_record.timestamp_ns
        assert metadata.request_end_ns == expected_end_ns
        assert metadata.worker_id == worker_id
        assert metadata.record_processor_id == "test-processor-id"

    def test_create_metadata_last_response_perf_ns_takes_precedence(
        self, mock_record_processor, sample_request_record
    ):
        """Test that last_response_perf_ns takes precedence over end_perf_ns."""
        last_response_perf_ns = sample_request_record.start_perf_ns + 150_000
        sample_request_record.end_perf_ns = (
            sample_request_record.start_perf_ns + 200_000
        )
        sample_request_record.request_info.credit_num = 2

        worker_id = "worker-2"

        metadata = RecordProcessor._create_metric_record_metadata(
            mock_record_processor,
            sample_request_record,
            worker_id,
            last_response_perf_ns=last_response_perf_ns,
        )

        # Should use last_response_perf_ns (not end_perf_ns)
        expected_end_ns = compute_time_ns(
            sample_request_record.timestamp_ns,
            sample_request_record.start_perf_ns,
            last_response_perf_ns,
        )
        assert metadata.request_end_ns == expected_end_ns
        assert metadata.worker_id == worker_id

    def test_create_metadata_with_cancellation(
        self, mock_record_processor, sample_request_record
    ):
        """Test creating metadata for a cancelled request."""
        cancellation_perf_ns = sample_request_record.start_perf_ns + 75_000
        sample_request_record.end_perf_ns = (
            sample_request_record.start_perf_ns + 100_000
        )
        sample_request_record.cancellation_perf_ns = cancellation_perf_ns
        sample_request_record.request_info.credit_num = 3

        worker_id = "worker-3"

        metadata = RecordProcessor._create_metric_record_metadata(
            mock_record_processor, sample_request_record, worker_id
        )

        expected_cancellation_time = compute_time_ns(
            sample_request_record.timestamp_ns,
            sample_request_record.start_perf_ns,
            cancellation_perf_ns,
        )
        assert metadata.was_cancelled is True
        assert metadata.cancellation_time_ns == expected_cancellation_time
        assert metadata.worker_id == worker_id

    def test_create_metadata_populates_request_num_from_credit_num(
        self, mock_record_processor, sample_request_record
    ):
        """request_num should be set from credit_num."""
        sample_request_record.request_info.credit_num = 7
        sample_request_record.request_info.session_num = 3
        sample_request_record.end_perf_ns = (
            sample_request_record.start_perf_ns + 100_000
        )

        metadata = RecordProcessor._create_metric_record_metadata(
            mock_record_processor, sample_request_record, "worker-1"
        )

        assert metadata.request_num == 7

    def test_create_metadata_populates_session_num_from_request_info(
        self, mock_record_processor, sample_request_record
    ):
        """session_num should come from request_info.session_num when present."""
        sample_request_record.request_info.credit_num = 10
        sample_request_record.request_info.session_num = 5
        sample_request_record.end_perf_ns = (
            sample_request_record.start_perf_ns + 100_000
        )

        metadata = RecordProcessor._create_metric_record_metadata(
            mock_record_processor, sample_request_record, "worker-1"
        )

        assert metadata.session_num == 5

    def test_create_metadata_session_num_falls_back_to_credit_num(
        self, mock_record_processor, sample_request_record
    ):
        """session_num should fall back to credit_num when request_info.session_num is None."""
        sample_request_record.request_info.credit_num = 8
        sample_request_record.request_info.session_num = None
        sample_request_record.end_perf_ns = (
            sample_request_record.start_perf_ns + 100_000
        )

        metadata = RecordProcessor._create_metric_record_metadata(
            mock_record_processor, sample_request_record, "worker-1"
        )

        assert metadata.session_num == 8

    @pytest.mark.parametrize(
        "field_name,field_value,expected_metadata_field",
        [
            ("recv_start_perf_ns", None, "request_ack_ns"),
        ],
    )
    def test_create_metadata_with_optional_fields_none(
        self,
        mock_record_processor,
        sample_request_record,
        field_name: str,
        field_value,
        expected_metadata_field: str,
    ):
        """Test creating metadata when optional fields are None."""
        setattr(sample_request_record, field_name, field_value)
        sample_request_record.request_info.credit_num = 4

        worker_id = "worker-4"

        metadata = RecordProcessor._create_metric_record_metadata(
            mock_record_processor, sample_request_record, worker_id
        )

        assert getattr(metadata, expected_metadata_field) is None
        assert metadata.worker_id == worker_id


class TestRecordProcessorWireMessages:
    @pytest.mark.asyncio
    async def test_on_inference_results_accepts_wire_message(
        self,
        sample_request_record,
        sample_parsed_record,
    ) -> None:
        """RecordProcessor should consume the msgspec wire envelope and emit MetricRecordsWireMessage."""
        processor = MagicMock(spec=RecordProcessor)
        processor.run = MagicMock()
        processor.run.cfg = sample_request_record.request_info.config.model_copy(
            deep=True
        )
        processor.run.cfg.output.records = []
        processor.service_id = "record-processor-1"
        processor.inference_result_parser = MagicMock()
        processor.inference_result_parser.parse_request_record = AsyncMock(
            return_value=sample_parsed_record
        )
        processor._process_record = AsyncMock(return_value=[{"request_latency": 12.5}])
        processor._merge_metric_results = MagicMock(
            return_value={"request_latency": 12.5}
        )
        processor._enqueue_metric_record = AsyncMock()
        processor._free_record_data = MagicMock(return_value=(None, None))
        metadata = MetricRecordMetadata(
            request_num=1,
            session_num=1,
            conversation_id="conversation-1",
            turn_index=0,
            request_start_ns=100,
            request_end_ns=200,
            worker_id="worker-7",
            record_processor_id="record-processor-1",
            benchmark_phase="profiling",
        )
        processor._create_metric_record_metadata = MagicMock(return_value=metadata)
        processor.records_push_client = MagicMock()
        processor.records_push_client.push = AsyncMock()

        record = copy.deepcopy(sample_request_record)
        record.responses = []
        record.turns = record.request_info.turns
        wire_message = build_inference_results_wire_message(
            service_id="worker-7",
            record=record,
        )

        await RecordProcessor._on_inference_results(processor, wire_message)

        processor.inference_result_parser.parse_request_record.assert_awaited_once()
        processor._create_metric_record_metadata.assert_called_once()
        metadata_args = processor._create_metric_record_metadata.call_args[0]
        assert metadata_args[1] == "worker-7"

        processor._enqueue_metric_record.assert_awaited_once_with(
            metadata=metadata,
            metrics={"request_latency": 12.5},
            trace_data=None,
            error=None,
        )

    @pytest.mark.asyncio
    async def test_flush_pending_metric_records_batches_multiple_records(self) -> None:
        processor = MagicMock(spec=RecordProcessor)
        processor.service_id = "record-processor-1"
        processor.records_push_client = MagicMock()
        processor.records_push_client.push = AsyncMock()
        processor._pending_metric_records = [
            MetricRecordsData(
                metadata=MetricRecordMetadata(
                    request_num=1,
                    session_num=1,
                    conversation_id="conversation-1",
                    turn_index=0,
                    request_start_ns=100,
                    request_end_ns=200,
                    worker_id="worker-1",
                    record_processor_id="record-processor-1",
                    benchmark_phase="profiling",
                ),
                metrics={"request_latency": 12.5},
            ),
            MetricRecordsData(
                metadata=MetricRecordMetadata(
                    request_num=2,
                    session_num=2,
                    conversation_id="conversation-2",
                    turn_index=0,
                    request_start_ns=200,
                    request_end_ns=300,
                    worker_id="worker-2",
                    record_processor_id="record-processor-1",
                    benchmark_phase="profiling",
                ),
                metrics={"request_latency": 9.5},
            ),
        ]

        await RecordProcessor._flush_pending_metric_records(processor)

        pushed_message = processor.records_push_client.push.call_args.args[0]
        assert isinstance(pushed_message, MetricRecordsBatchWireMessage)
        assert len(pushed_message.records) == 2
        assert processor._pending_metric_records == []


class TestRecordProcessorShutdownResilience:
    """RecordProcessor must tolerate a disconnected WorkerGroupManager during teardown.

    In k8s JobSet teardown, the WorkerGroupManager DEALER peer can be torn down
    before this record-processor finishes its @on_stop hook. When that race hits,
    the ZMQ send raises. Before the fix, the exception propagated out of the
    on_stop hook, tripped the service lifecycle's _fail path, and caused the
    container to exit non-zero — which tripped the Job's backoffLimit and
    cascaded the whole JobSet to Failed. The fix swallows the send error on
    shutdown paths only, matching worker.py's existing pattern.
    """

    @pytest.fixture
    def processor(self):
        instance = MagicMock(spec=RecordProcessor)
        instance.service_id = "record-processor-1"
        instance.service_type = "record_processor"
        instance.pod_lifecycle_dealer_client = AsyncMock()
        instance._flush_pending_metric_records = AsyncMock()
        instance.stop = AsyncMock()
        instance.warning = MagicMock()
        instance.debug = MagicMock()
        return instance

    @pytest.mark.asyncio
    async def test_on_stop_swallows_dealer_send_failure(self, processor):
        """If the group manager DEALER peer has already exited, send raises —
        the on_stop hook must log and return, never propagate."""
        processor.pod_lifecycle_dealer_client.send = AsyncMock(
            side_effect=ConnectionError("peer gone")
        )

        await RecordProcessor._notify_worker_group_manager_shutdown(processor)

        processor._flush_pending_metric_records.assert_awaited_once()
        processor.pod_lifecycle_dealer_client.send.assert_awaited_once()
        processor.warning.assert_called_once()
        assert "peer already disconnected" in processor.warning.call_args.args[0]

    @pytest.mark.asyncio
    async def test_on_stop_sends_group_peer_shutdown_when_peer_up(self, processor):
        """Happy path: dealer up → GroupPeerShutdown is sent, no warning."""
        await RecordProcessor._notify_worker_group_manager_shutdown(processor)

        sent = processor.pod_lifecycle_dealer_client.send.await_args.args[0]
        from aiperf.common.pod_lifecycle_structs import GroupPeerShutdown

        assert isinstance(sent, GroupPeerShutdown)
        assert sent.service_id == "record-processor-1"
        processor.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_stop_skips_send_when_dealer_client_absent(self, processor):
        """No dealer client (non-k8s mode) → flush only, no send, no error."""
        processor.pod_lifecycle_dealer_client = None

        await RecordProcessor._notify_worker_group_manager_shutdown(processor)

        processor._flush_pending_metric_records.assert_awaited_once()
        processor.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_pod_lifecycle_ack_swallows_send_failure(self, processor):
        """The SHUTDOWN command's ack send races the same peer teardown —
        a dealer failure here must not propagate either, otherwise the
        awaited stop() completes but the hook itself raises."""
        from aiperf.common.enums import CommandType
        from aiperf.common.pod_lifecycle_structs import GroupPeerCommand

        processor.pod_lifecycle_dealer_client.send = AsyncMock(
            side_effect=ConnectionError("peer gone")
        )
        message = GroupPeerCommand(
            cid="cmd-1",
            service_id="record-processor-1",
            command=str(CommandType.SHUTDOWN),
        )

        await RecordProcessor._on_pod_lifecycle_message(processor, message)

        processor.stop.assert_awaited_once()
        processor.pod_lifecycle_dealer_client.send.assert_awaited_once()
        # Debug log, not a warning — ack failures during SHUTDOWN are expected.
        processor.debug.assert_called_once()

    @pytest.mark.asyncio
    async def test_pod_lifecycle_ack_sent_when_peer_up(self, processor):
        """Happy path: SHUTDOWN command → stop() awaited, ack sent."""
        from aiperf.common.enums import CommandType
        from aiperf.common.pod_lifecycle_structs import (
            GroupPeerCommand,
            GroupPeerCommandAck,
        )

        message = GroupPeerCommand(
            cid="cmd-2",
            service_id="record-processor-1",
            command=str(CommandType.SHUTDOWN),
        )

        await RecordProcessor._on_pod_lifecycle_message(processor, message)

        processor.stop.assert_awaited_once()
        sent = processor.pod_lifecycle_dealer_client.send.await_args.args[0]
        assert isinstance(sent, GroupPeerCommandAck)
        assert sent.cid == "cmd-2"
        processor.debug.assert_not_called()
