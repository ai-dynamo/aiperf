# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import textwrap

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

    def test_codec_built_before_credit_messages_import_round_trips(self) -> None:
        """Codec built BEFORE aiperf.credit.messages is imported must still
        decode credit-phase messages.

        The credit-phase message module lives outside the eagerly-imported
        common/messages tree, so without the explicit import inside
        ``_build_message_codec`` the tagged-union snapshot would silently
        exclude the five credit-phase message types whenever the codec
        builder won the import race. Needs a fresh interpreter — this test
        process has long since imported everything.
        """
        script = textwrap.dedent(
            """
            import sys

            from aiperf.common.message_codecs import get_message_codec

            assert "aiperf.credit.messages" not in sys.modules, (
                "precondition broken: credit messages already imported before "
                "the codec was built; this test no longer exercises the race"
            )
            codec = get_message_codec()

            from aiperf.common.enums import CreditPhase
            from aiperf.common.models import CreditPhaseStats
            from aiperf.credit.messages import CreditPhaseProgressMessage

            message = CreditPhaseProgressMessage(
                service_id="timing-manager",
                stats=CreditPhaseStats(phase=CreditPhase.PROFILING, requests_sent=7),
            )
            decoded = codec.decode(codec.encode(message))
            assert isinstance(decoded, CreditPhaseProgressMessage), type(decoded)
            assert decoded.stats.requests_sent == 7
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
