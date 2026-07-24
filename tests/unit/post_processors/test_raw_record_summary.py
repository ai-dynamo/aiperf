# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import orjson
import pytest

from aiperf.common.messages import MetricRecordsData
from aiperf.common.models import (
    ParsedResponseRecord,
    RawRecordSummaryInfo,
    SSEMessage,
)
from aiperf.post_processors.metric_record_processor import MetricRecordProcessor
from aiperf.post_processors.raw_record_writer_processor import RawRecordAggregator
from aiperf.post_processors.record_observer_context import RecordObserverContext
from aiperf.records.raw_record_summary import build_raw_record_summary
from tests.unit.post_processors.conftest import (
    create_exporter_config,
    create_metric_metadata,
    raw_record_processor,
)

START_NS = 1_000_000_000


def _with_nvext_packets(record: ParsedResponseRecord) -> ParsedResponseRecord:
    first_packet = {
        "id": "cmpl-123",
        "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
        "nvext": {
            "timing": {
                "request_received_ms": 101.0,
                "prefill_wait_time_ms": 2.5,
                "prefill_time_ms": 18.0,
                "ttft_ms": 42.0,
                "total_time_ms": 55.0,
                "kv_hit_rate": 0.875,
                "router_queue_depth": 3,
            }
        },
    }
    final_packet = {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "nvext": {"worker_id": "decode-worker-1"},
    }
    record.request.start_perf_ns = START_NS
    record.request.status = 200
    record.request.responses = [
        SSEMessage.parse(
            f"data: {orjson.dumps(first_packet).decode()}",
            START_NS + 10_000_000,
        ),
        SSEMessage.parse(
            f"data: {orjson.dumps(final_packet).decode()}",
            START_NS + 28_000_000,
        ),
        SSEMessage.parse("data: [DONE]", START_NS + 30_000_000),
    ]
    return record


class TestRawRecordSummaryExtraction:
    def test_extracts_compact_nvext_and_chunk_timing(
        self, sample_parsed_record_with_raw_responses: ParsedResponseRecord
    ):
        record = _with_nvext_packets(sample_parsed_record_with_raw_responses)

        summary = build_raw_record_summary(record)

        assert summary.request_id == "cmpl-123"
        assert summary.status == 200
        assert summary.data_chunk_count == 2
        assert summary.finish_reason == "stop"
        assert summary.first_chunk_ms == 10.0
        assert summary.last_chunk_ms == 28.0
        assert summary.stream_decode_ms == 18.0
        assert summary.nvext is not None
        assert summary.nvext.worker_id == "decode-worker-1"
        assert summary.nvext.timing == {
            "request_received_ms": 101.0,
            "prefill_wait_time_ms": 2.5,
            "prefill_time_ms": 18.0,
            "ttft_ms": 42.0,
            "total_time_ms": 55.0,
            "kv_hit_rate": 0.875,
            "router_queue_depth": 3,
        }

    def test_handles_packets_without_nvext(
        self, sample_parsed_record_with_raw_responses: ParsedResponseRecord
    ):
        summary = build_raw_record_summary(sample_parsed_record_with_raw_responses)
        assert summary.data_chunk_count == 2
        assert summary.nvext is None


class TestRawSummaryRecordsPipeline:
    @pytest.mark.asyncio
    async def test_metric_producer_attaches_summary_only_for_raw_export(
        self,
        run_raw,
        sample_parsed_record_with_raw_responses: ParsedResponseRecord,
    ):
        record = _with_nvext_packets(sample_parsed_record_with_raw_responses)
        processor = object.__new__(MetricRecordProcessor)
        processor.run = run_raw
        processor.valid_parse_funcs = []
        processor.error_parse_funcs = []

        result = await MetricRecordProcessor.process_record(
            processor,
            record,
            create_metric_metadata(),
        )

        assert result.raw_summary is not None
        assert result.raw_summary.request_id == "cmpl-123"

    @pytest.mark.asyncio
    async def test_raw_observer_writes_joinable_summary_sidecar(
        self,
        cfg_raw,
        run_raw,
        sample_parsed_record_with_raw_responses: ParsedResponseRecord,
    ):
        record = _with_nvext_packets(sample_parsed_record_with_raw_responses)
        metadata = create_metric_metadata(
            session_num=7,
            conversation_id="conv-summary",
            x_request_id="req-summary",
        )
        metric_data = MetricRecordsData(
            metadata=metadata,
            metrics={},
            raw_summary=build_raw_record_summary(record),
        )

        async with raw_record_processor("processor-summary", run_raw) as processor:
            await processor.observe(
                RecordObserverContext(
                    record=record,
                    metadata=metadata,
                    produced={"metric_records": [metric_data]},
                )
            )

        fragment = RawRecordSummaryInfo.model_validate_json(
            processor.summary_output_file.read_text().splitlines()[0]
        )
        assert fragment.metadata.session_num == 7
        assert fragment.metadata.x_request_id == "req-summary"
        assert fragment.request_id == "cmpl-123"

        aggregator = RawRecordAggregator(
            exporter_config=create_exporter_config(cfg_raw)
        )
        await aggregator.export()
        sidecar = RawRecordSummaryInfo.model_validate_json(
            aggregator.summary_output_file.read_text().splitlines()[0]
        )
        assert sidecar == fragment

    @pytest.mark.asyncio
    async def test_raw_observer_writes_summary_without_metric_producer_output(
        self,
        run_raw,
        sample_parsed_record_with_raw_responses: ParsedResponseRecord,
    ):
        record = _with_nvext_packets(sample_parsed_record_with_raw_responses)
        metadata = create_metric_metadata()

        async with raw_record_processor("processor-no-metrics", run_raw) as processor:
            await processor.observe(
                RecordObserverContext(
                    record=record,
                    metadata=metadata,
                    produced={},
                )
            )

        fragment = RawRecordSummaryInfo.model_validate_json(
            processor.summary_output_file.read_text().splitlines()[0]
        )
        assert fragment.request_id == "cmpl-123"
