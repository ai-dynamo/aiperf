# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.channel_codecs import RECORDS_CODEC
from aiperf.common.message_codecs import (
    MsgspecStructCodec,
    codec_cache_key,
    get_message_codec,
)
from aiperf.common.metric_records_wire import (
    MetricRecordMetadata,
    MetricRecordsBatchWireMessage,
    MetricRecordsData,
    MetricRecordsWireMessage,
    build_metric_records_batch_wire_message,
    build_metric_records_wire_message,
)


class TestMessageCodecs:
    def test_records_codec_round_trips_metric_records_wire_message(self) -> None:
        """MessagePack records codec should round-trip the msgspec wire payload."""
        message = build_metric_records_wire_message(
            service_id="record-processor-1",
            metadata=MetricRecordMetadata(
                request_num=7,
                session_num=3,
                conversation_id="conversation-1",
                turn_index=1,
                request_start_ns=1_000,
                request_end_ns=2_000,
                worker_id="worker-1",
                record_processor_id="record-processor-1",
                benchmark_phase="profiling",
            ),
            metrics={"request_latency": 12.5, "output_sequence_length": 8},
            trace_data=None,
            error=None,
        )

        encoded = RECORDS_CODEC.encode(message)
        decoded = RECORDS_CODEC.decode(encoded)

        assert isinstance(decoded, MetricRecordsWireMessage)
        assert decoded.service_id == "record-processor-1"
        assert decoded.metadata.worker_id == "worker-1"
        assert decoded.metrics == {
            "request_latency": 12.5,
            "output_sequence_length": 8,
        }

    def test_records_codec_round_trips_metric_records_batch_wire_message(self) -> None:
        """MessagePack records codec should round-trip the batched msgspec wire payload."""
        message = build_metric_records_batch_wire_message(
            service_id="record-processor-1",
            records=[
                MetricRecordsData(
                    metadata=MetricRecordMetadata(
                        request_num=7,
                        session_num=3,
                        conversation_id="conversation-1",
                        turn_index=1,
                        request_start_ns=1_000,
                        request_end_ns=2_000,
                        worker_id="worker-1",
                        record_processor_id="record-processor-1",
                        benchmark_phase="profiling",
                    ),
                    metrics={"request_latency": 12.5},
                ),
                MetricRecordsData(
                    metadata=MetricRecordMetadata(
                        request_num=8,
                        session_num=4,
                        conversation_id="conversation-2",
                        turn_index=1,
                        request_start_ns=2_000,
                        request_end_ns=3_000,
                        worker_id="worker-2",
                        record_processor_id="record-processor-1",
                        benchmark_phase="profiling",
                    ),
                    metrics={"request_latency": 9.5},
                ),
            ],
        )

        encoded = RECORDS_CODEC.encode(message)
        decoded = RECORDS_CODEC.decode(encoded)

        assert isinstance(decoded, MetricRecordsBatchWireMessage)
        assert decoded.service_id == "record-processor-1"
        assert len(decoded.records) == 2
        assert decoded.records[0].metrics == {"request_latency": 12.5}
        assert decoded.records[1].metrics == {"request_latency": 9.5}

    def test_codec_cache_key_uses_msgspec_default_and_custom_cache_keys(self) -> None:
        """Codec cache keys should stay stable for client cache partitioning."""
        assert codec_cache_key(None) == "msgspec-message"
        assert codec_cache_key(RECORDS_CODEC) == "records-msgpack"

    def test_get_message_codec_returns_msgspec_struct_codec(self) -> None:
        """get_message_codec() must return an MsgspecStructCodec instance."""
        codec = get_message_codec()
        assert isinstance(codec, MsgspecStructCodec)
        assert codec.cache_key == "msgspec-message"

    def test_get_message_codec_is_singleton(self) -> None:
        """Repeated calls to get_message_codec() return the same object."""
        assert get_message_codec() is get_message_codec()
