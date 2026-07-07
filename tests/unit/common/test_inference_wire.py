# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import orjson

from aiperf.common.inference_wire import (
    build_inference_results_wire_message,
    decode_inference_results_wire_message,
    encode_inference_results_wire_message,
    wire_message_to_request_record,
)
from aiperf.common.models import (
    AioHttpTraceData,
    BaseTraceData,
    BinaryResponse,
    Image,
    RequestRecord,
    SSEMessage,
    Text,
    TextResponse,
    Turn,
)


def _full_aiohttp_trace() -> AioHttpTraceData:
    """AioHttpTraceData carrying both the timing subset and the heavy raw fields.

    Timing values are chosen so every ``http_req_*`` metric resolves to a
    positive, distinct number after a wire round-trip (see the metric-level
    regression in ``tests/unit/metrics/test_http_trace_metrics.py``).
    """
    return AioHttpTraceData(
        trace_type="aiohttp",
        reference_perf_ns=1000,
        reference_time_ns=1_000_000_000_000,
        # Timing subset (must always survive the wire).
        connection_pool_wait_start_perf_ns=1000,
        connection_pool_wait_end_perf_ns=1050,
        tcp_connect_start_perf_ns=1050,
        tcp_connect_end_perf_ns=1120,
        dns_lookup_start_perf_ns=1050,
        dns_lookup_end_perf_ns=1080,
        request_send_start_perf_ns=1120,
        request_send_end_perf_ns=1150,
        response_receive_start_perf_ns=1200,
        response_receive_end_perf_ns=1400,
        request_chunks_count=2,
        request_bytes_total=64,
        response_chunks_count=3,
        response_bytes_total=128,
        # Heavy raw fields (gated behind include_raw_trace_fields).
        request_chunks=[(1120, 32), (1150, 32)],
        response_chunks=[(1200, 40), (1300, 40), (1400, 48)],
        request_headers={"Content-Type": "application/json"},
        response_headers={"Server": "mock"},
        local_ip="127.0.0.1",
        local_port=5555,
        remote_ip="127.0.0.1",
        remote_port=8765,
    )


class TestInferenceWire:
    def test_round_trips_trimmed_wire_message(
        self,
        sample_request_info,
    ) -> None:
        """The alternate msgspec wire model should rehydrate into the current record shape."""
        request_info = copy.deepcopy(sample_request_info)
        turn = Turn(
            texts=[Text(contents=["hello", " world"])],
            images=[Image(contents=["img-0", "img-1"])],
            role="user",
            model="test-model",
            max_tokens=42,
        )
        request_info.turns = [turn]
        request_info.turn_index = 3
        request_info.credit_num = 7
        request_info.session_num = 11
        request_info.credit_phase = "profiling"
        request_info.credit_issued_ns = 1_000
        request_info.credit_received_ns = 1_025
        request_info.system_message = "system"
        request_info.user_context_message = "context"

        sse_payload = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "choices": [{"delta": {"content": "hello"}}],
        }
        raw_payload = {
            "messages": [{"role": "user", "content": "hello world"}],
            "model": "test-model",
        }
        record = RequestRecord(
            request_info=request_info,
            request_headers={"Content-Type": "application/json"},
            model_name="test-model",
            timestamp_ns=10,
            start_perf_ns=11,
            end_perf_ns=25,
            recv_start_perf_ns=15,
            status=200,
            responses=[
                SSEMessage.parse(
                    f"data: {orjson.dumps(sse_payload).decode()}\n\n",
                    perf_ns=16,
                ),
                TextResponse(
                    perf_ns=17,
                    text='{"result":"ok"}',
                    content_type="application/json",
                ),
                BinaryResponse(
                    perf_ns=18,
                    raw_bytes=b"\x00\x01\x02",
                    content_type="application/octet-stream",
                ),
            ],
            credit_drop_latency=4,
            clock_offset_ns=9,
            turns=[turn.copy_with_stripped_media()],
        )

        wire_message = build_inference_results_wire_message(
            service_id="worker-1",
            record=record,
            raw_payload=raw_payload,
            include_request_headers=True,
            include_status=True,
            include_raw_trace_fields=False,
        )
        encoded = encode_inference_results_wire_message(wire_message)
        decoded = decode_inference_results_wire_message(encoded)
        service_id, rebuilt = wire_message_to_request_record(
            message=decoded,
        )

        assert service_id == "worker-1"
        assert decoded.record.metadata.credit_num == 7
        assert decoded.record.metadata.requested_max_tokens == 42
        assert decoded.record.prompt is not None
        assert len(decoded.record.prompt.turns) == 1
        assert decoded.record.prompt.turns[0].image_count == 2

        assert rebuilt.model_name == "test-model"
        assert rebuilt.request_info.credit_num == 7
        assert rebuilt.request_info.session_num == 11
        assert rebuilt.request_info.turn_index == 3
        assert rebuilt.request_info.credit_issued_ns == 1_000
        assert rebuilt.request_info.credit_received_ns == 1_025
        assert rebuilt.request_info.system_message == "system"
        assert rebuilt.request_info.user_context_message == "context"
        assert rebuilt.turns[-1].max_tokens == 42
        assert rebuilt.request_headers == {"Content-Type": "application/json"}
        assert rebuilt.status == 200
        assert rebuilt.credit_drop_latency == 4
        assert rebuilt.clock_offset_ns == 9
        assert rebuilt.raw_payload == raw_payload
        assert len(rebuilt.turns[0].texts) == 1
        assert rebuilt.turns[0].texts[0].contents == ["hello", " world"]
        assert len(rebuilt.turns[0].images) == 1
        assert len(rebuilt.turns[0].images[0].contents) == 2
        assert len(rebuilt.responses) == 3
        assert isinstance(rebuilt.responses[0], SSEMessage)
        assert rebuilt.responses[0].get_json()["id"] == "chatcmpl-test"
        assert isinstance(rebuilt.responses[1], TextResponse)
        assert rebuilt.responses[1].text == '{"result":"ok"}'
        assert isinstance(rebuilt.responses[2], BinaryResponse)
        assert rebuilt.responses[2].raw_bytes == b"\x00\x01\x02"

    def test_falls_back_to_request_info_turns_when_record_turns_missing(
        self,
        sample_request_info,
    ) -> None:
        """The alternate wire model should still project prompt data from request_info."""
        request_info = copy.deepcopy(sample_request_info)
        request_info.turns = [
            Turn(
                texts=[Text(contents=["fallback prompt"])],
                images=[Image(contents=["img-0"])],
                max_tokens=19,
            )
        ]

        record = RequestRecord(
            request_info=request_info,
            model_name="test-model",
            timestamp_ns=10,
            start_perf_ns=11,
            responses=[],
            turns=[],
        )

        wire_message = build_inference_results_wire_message(
            service_id="worker-1",
            record=record,
        )

        assert wire_message.record.prompt is not None
        assert len(wire_message.record.prompt.turns) == 1
        assert wire_message.record.prompt.turns[0].texts[0].contents == (
            "fallback prompt",
        )
        assert wire_message.record.prompt.turns[0].image_count == 1
        assert wire_message.record.metadata.requested_max_tokens == 19

    def test_round_trips_trace_data_when_enabled(
        self,
        sample_request_info,
    ) -> None:
        """Trace payload should survive the msgspec wire path when explicitly included."""
        request_info = copy.deepcopy(sample_request_info)
        record = RequestRecord(
            request_info=request_info,
            model_name="test-model",
            timestamp_ns=10,
            start_perf_ns=11,
            responses=[],
            turns=request_info.turns,
            trace_data=BaseTraceData(
                trace_type="httpcore",
                request_send_start_perf_ns=111,
                request_send_end_perf_ns=222,
                response_status_code=200,
            ),
        )

        wire_message = build_inference_results_wire_message(
            service_id="worker-1",
            record=record,
            include_raw_trace_fields=True,
        )
        rebuilt_service_id, rebuilt_record = wire_message_to_request_record(
            message=decode_inference_results_wire_message(
                encode_inference_results_wire_message(wire_message)
            ),
        )

        assert rebuilt_service_id == "worker-1"
        assert wire_message.record.trace_data is not None
        assert wire_message.record.trace_data.trace_type == "httpcore"
        assert rebuilt_record.trace_data is not None
        assert rebuilt_record.trace_data.trace_type == "httpcore"
        assert rebuilt_record.trace_data.request_send_start_perf_ns == 111
        assert rebuilt_record.trace_data.response_status_code == 200

    def test_trace_timing_subset_always_rides_the_wire(
        self,
        sample_request_info,
    ) -> None:
        """Regression: the timing subset survives even with raw trace export OFF.

        This is the behavioral-parity contract behind ``--export-http-trace``:
        the flag gates only the heavy raw fields (per-chunk lists, headers,
        socket info). The perf_ns timing subset, chunk counts, and byte totals
        must always round-trip so the aggregate ``http_req_*`` metrics compute
        on a default run.
        """
        request_info = copy.deepcopy(sample_request_info)
        record = RequestRecord(
            request_info=request_info,
            model_name="test-model",
            timestamp_ns=10,
            start_perf_ns=11,
            responses=[],
            turns=request_info.turns,
            trace_data=_full_aiohttp_trace(),
        )

        wire_message = build_inference_results_wire_message(
            service_id="worker-1",
            record=record,
            include_raw_trace_fields=False,
        )
        _, rebuilt_record = wire_message_to_request_record(
            message=decode_inference_results_wire_message(
                encode_inference_results_wire_message(wire_message)
            ),
        )

        trace = rebuilt_record.trace_data
        assert isinstance(trace, AioHttpTraceData)
        # Timing subset kept.
        assert trace.connection_pool_wait_start_perf_ns == 1000
        assert trace.connection_pool_wait_end_perf_ns == 1050
        assert trace.tcp_connect_start_perf_ns == 1050
        assert trace.tcp_connect_end_perf_ns == 1120
        assert trace.dns_lookup_start_perf_ns == 1050
        assert trace.dns_lookup_end_perf_ns == 1080
        assert trace.request_send_start_perf_ns == 1120
        assert trace.request_send_end_perf_ns == 1150
        assert trace.response_receive_start_perf_ns == 1200
        assert trace.response_receive_end_perf_ns == 1400
        assert trace.request_chunks_count == 2
        assert trace.request_bytes_total == 64
        assert trace.response_chunks_count == 3
        assert trace.response_bytes_total == 128
        # Heavy raw fields dropped.
        assert trace.request_chunks == []
        assert trace.response_chunks == []
        assert trace.request_headers is None
        assert trace.response_headers is None
        assert trace.local_ip is None
        assert trace.local_port is None
        assert trace.remote_ip is None
        assert trace.remote_port is None
        # The original record must not be mutated by the projection.
        assert record.trace_data.request_headers == {"Content-Type": "application/json"}
        assert record.trace_data.request_chunks == [(1120, 32), (1150, 32)]

    def test_raw_trace_fields_ride_the_wire_when_enabled(
        self,
        sample_request_info,
    ) -> None:
        """With raw trace export ON, the heavy per-chunk/header/socket fields survive."""
        request_info = copy.deepcopy(sample_request_info)
        record = RequestRecord(
            request_info=request_info,
            model_name="test-model",
            timestamp_ns=10,
            start_perf_ns=11,
            responses=[],
            turns=request_info.turns,
            trace_data=_full_aiohttp_trace(),
        )

        wire_message = build_inference_results_wire_message(
            service_id="worker-1",
            record=record,
            include_raw_trace_fields=True,
        )
        _, rebuilt_record = wire_message_to_request_record(
            message=decode_inference_results_wire_message(
                encode_inference_results_wire_message(wire_message)
            ),
        )

        trace = rebuilt_record.trace_data
        assert isinstance(trace, AioHttpTraceData)
        assert trace.request_chunks == [(1120, 32), (1150, 32)]
        assert trace.response_chunks == [(1200, 40), (1300, 40), (1400, 48)]
        assert trace.request_headers == {"Content-Type": "application/json"}
        assert trace.response_headers == {"Server": "mock"}
        assert trace.local_ip == "127.0.0.1"
        assert trace.local_port == 5555
        assert trace.remote_ip == "127.0.0.1"
        assert trace.remote_port == 8765

    def test_omits_raw_export_fields_when_not_requested(
        self,
        sample_request_info,
    ) -> None:
        """Raw-export-only baggage should stay off the wire unless explicitly requested.

        The trace timing subset always survives (see
        ``test_trace_timing_subset_always_rides_the_wire``); this test asserts the
        *other* raw-export baggage (request headers, status, raw payload) and the
        heavy raw trace fields stay off unless requested.
        """
        request_info = copy.deepcopy(sample_request_info)
        record = RequestRecord(
            request_info=request_info,
            request_headers={"Authorization": "Bearer secret"},
            model_name="test-model",
            timestamp_ns=10,
            start_perf_ns=11,
            status=202,
            responses=[],
            turns=request_info.turns,
            raw_payload={"messages": [{"role": "user", "content": "hidden"}]},
            trace_data=BaseTraceData(
                trace_type="httpcore",
                request_send_start_perf_ns=111,
                request_send_end_perf_ns=222,
                request_headers={"Authorization": "Bearer secret"},
            ),
        )

        wire_message = build_inference_results_wire_message(
            service_id="worker-1",
            record=record,
            include_request_headers=False,
            include_status=False,
            include_raw_trace_fields=False,
        )
        _, rebuilt_record = wire_message_to_request_record(
            message=wire_message,
        )

        assert wire_message.record.request_headers is None
        assert wire_message.record.status is None
        assert wire_message.record.raw_payload is None
        # Timing subset kept, but the raw trace headers stripped.
        assert wire_message.record.trace_data is not None
        assert wire_message.record.trace_data.request_send_start_perf_ns == 111
        assert wire_message.record.trace_data.request_headers is None
        assert rebuilt_record.request_headers is None
        assert rebuilt_record.status is None
        assert rebuilt_record.trace_data is not None
        assert rebuilt_record.trace_data.request_headers is None
        assert (
            not hasattr(rebuilt_record, "raw_payload")
            or rebuilt_record.raw_payload is None
        )
