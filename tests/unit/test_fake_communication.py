# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from aiperf.common.channel_codecs import RAW_INFERENCE_CODEC, RECORDS_CODEC
from aiperf.common.enums import CommAddress, MessageType
from aiperf.common.inference_wire import (
    InferenceResultsWireMessage,
    build_inference_results_wire_message,
    encode_inference_results_wire_message,
)
from aiperf.common.metric_records_wire import (
    MetricRecordMetadata,
    MetricRecordsWireMessage,
    build_metric_records_wire_message,
)
from tests.harness.fake_communication import FakeCommunication, FakeCommunicationBus


class TestFakeCommunication:
    @pytest.mark.asyncio
    async def test_streaming_router_routes_by_address_and_identity(self) -> None:
        """Streaming dealer identities can repeat across addresses without collisions."""
        bus = FakeCommunicationBus()
        FakeCommunication.set_shared_bus(bus)
        comm = FakeCommunication()

        credit_router = comm.create_streaming_router_client("fake://credit", bind=True)
        return_router = comm.create_streaming_router_client(
            "fake://credit_return", bind=True
        )
        credit_dealer = comm.create_streaming_dealer_client(
            "fake://credit",
            identity="worker-1",
        )
        return_dealer = comm.create_streaming_dealer_client(
            "fake://credit_return",
            identity="worker-1",
        )

        received_credit: list[dict[str, str]] = []
        received_return: list[dict[str, str]] = []

        async def on_credit(message: dict[str, str]) -> None:
            received_credit.append(message)

        async def on_return(message: dict[str, str]) -> None:
            received_return.append(message)

        credit_dealer.register_receiver(on_credit)
        return_dealer.register_receiver(on_return)

        await credit_router.send_to("worker-1", {"channel": "credit"})
        await return_router.send_to("worker-1", {"channel": "return"})

        assert received_credit == [{"channel": "credit"}]
        assert received_return == [{"channel": "return"}]

    def test_client_cache_partitions_by_codec(self) -> None:
        """Fake communication should mirror the real transport cache partitioning."""
        FakeCommunication.set_shared_bus(FakeCommunicationBus())
        comm = FakeCommunication()

        json_client = comm.create_push_client(CommAddress.RECORDS)
        msgpack_client = comm.create_push_client(
            CommAddress.RECORDS,
            codec=RECORDS_CODEC,
        )

        assert json_client is not msgpack_client

    @pytest.mark.asyncio
    async def test_push_raw_decodes_raw_inference_msgspec_payload(
        self,
        sample_request_record,
    ) -> None:
        """push_raw should decode the configured raw-inference codec before dispatching."""
        bus = FakeCommunicationBus()
        FakeCommunication.set_shared_bus(bus)
        comm = FakeCommunication()

        push_client = comm.create_push_client(
            CommAddress.RAW_INFERENCE_PROXY_FRONTEND,
            codec=RAW_INFERENCE_CODEC,
        )
        pull_client = comm.create_pull_client(
            CommAddress.RAW_INFERENCE_PROXY_BACKEND,
            codec=RAW_INFERENCE_CODEC,
        )

        received: list[InferenceResultsWireMessage] = []

        async def callback(message: InferenceResultsWireMessage) -> None:
            received.append(message)

        pull_client.register_pull_callback(MessageType.INFERENCE_RESULTS, callback)

        record = copy.deepcopy(sample_request_record)
        record.responses = []
        record.turns = record.request_info.turns
        data = encode_inference_results_wire_message(
            build_inference_results_wire_message(
                service_id="worker-1",
                record=record,
            )
        )

        await push_client.push_raw(data)

        assert len(received) == 1
        assert received[0].service_id == "worker-1"
        assert received[0].record.metadata.credit_num == record.request_info.credit_num
        assert any(
            isinstance(payload.payload, InferenceResultsWireMessage)
            for payload in bus.sent_payloads
        )

    @pytest.mark.asyncio
    async def test_push_raw_decodes_records_msgpack_payload(self) -> None:
        """Fake push_raw should also honor msgpack codecs on the records channel."""
        bus = FakeCommunicationBus()
        FakeCommunication.set_shared_bus(bus)
        comm = FakeCommunication()

        push_client = comm.create_push_client(
            CommAddress.RECORDS,
            codec=RECORDS_CODEC,
        )
        pull_client = comm.create_pull_client(
            CommAddress.RECORDS,
            codec=RECORDS_CODEC,
        )

        received: list[MetricRecordsWireMessage] = []

        async def callback(message: MetricRecordsWireMessage) -> None:
            received.append(message)

        pull_client.register_pull_callback(MessageType.METRIC_RECORDS, callback)

        message = build_metric_records_wire_message(
            service_id="record-processor-1",
            metadata=MetricRecordMetadata(
                request_num=1,
                session_num=1,
                conversation_id="conversation-1",
                turn_index=0,
                request_start_ns=100,
                request_end_ns=200,
                worker_id="worker-1",
                record_processor_id="rp-1",
                benchmark_phase="profiling",
            ),
            metrics={"request_latency": 3.14},
            trace_data=None,
            error=None,
        )

        await push_client.push_raw(RECORDS_CODEC.encode(message))

        assert len(received) == 1
        assert received[0].service_id == "record-processor-1"
        assert received[0].metrics == {"request_latency": 3.14}
