# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.channel_codecs import RECORDS_CODEC
from aiperf.common.message_codecs import JSON_MESSAGE_CODEC, codec_cache_key
from aiperf.common.messages import MetricRecordsMessage
from aiperf.common.models.record_models import MetricRecordMetadata


class TestMessageCodecs:
    def test_records_codec_round_trips_metric_records_message(self) -> None:
        """MessagePack records codec should rehydrate the routed Pydantic subclass."""
        message = MetricRecordsMessage(
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
            results=[{"request_latency": 12.5, "output_sequence_length": 8}],
        )

        encoded = RECORDS_CODEC.encode(message)
        decoded = RECORDS_CODEC.decode(encoded)

        assert isinstance(decoded, MetricRecordsMessage)
        assert decoded.service_id == "record-processor-1"
        assert decoded.metadata.worker_id == "worker-1"
        assert decoded.results == [
            {"request_latency": 12.5, "output_sequence_length": 8}
        ]

    def test_codec_cache_key_uses_json_default_and_custom_cache_keys(self) -> None:
        """Codec cache keys should stay stable for client cache partitioning."""
        assert codec_cache_key(None) == JSON_MESSAGE_CODEC.cache_key
        assert codec_cache_key(RECORDS_CODEC) == "records-msgpack"
