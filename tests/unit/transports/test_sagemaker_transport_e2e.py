# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end proof that AioHttpTransport's SageMaker eventstream path works
over a real network connection, not just in isolated unit mocks.

Spins up a real ``aiohttp.web`` server on loopback (via aiohttp's own
``TestServer``/``TestClient`` test utilities) that responds with genuine
``application/vnd.amazon.eventstream`` binary framing, then drives the
actual ``AioHttpTransport.send_request()`` -- real sockets, real
Content-Type-based ``is_eventstream`` detection (``aiohttp_client.py:181``),
real ``EventStreamReader`` decoding -- and asserts the resulting
``RequestRecord.responses`` contains real ``AwsEventStreamMessage``
instances with the expected content. Nothing in this test module is mocked.
"""

from __future__ import annotations

import struct
import zlib

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from aiperf.common.enums import CreditPhase
from aiperf.common.models import AwsEventStreamMessage
from aiperf.common.models.record_models import RequestInfo
from aiperf.transports.aiohttp_transport import AioHttpTransport
from tests.unit.transports.conftest import create_model_endpoint_info


def _encode_header(name: str, value: str) -> bytes:
    name_b = name.encode("utf-8")
    value_b = value.encode("utf-8")
    return (
        struct.pack(">B", len(name_b))
        + name_b
        + struct.pack(">B", 7)  # header value type 7 == string
        + struct.pack(">H", len(value_b))
        + value_b
    )


def _encode_frame(payload: bytes, *, event_type: str = "PayloadPart") -> bytes:
    """Hand-roll one real AWS eventstream binary frame."""
    headers = (
        _encode_header(":message-type", "event")
        + _encode_header(":event-type", event_type)
        + _encode_header(":content-type", "application/json")
    )
    headers_len = len(headers)
    total_len = 4 + 4 + 4 + headers_len + len(payload) + 4
    prelude = struct.pack(">II", total_len, headers_len)
    prelude_crc = struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF)
    message_no_crc = prelude + prelude_crc + headers + payload
    message_crc = struct.pack(">I", zlib.crc32(message_no_crc) & 0xFFFFFFFF)
    return message_no_crc + message_crc


async def _sagemaker_invoke_handler(request: web.Request) -> web.StreamResponse:
    """Real SageMaker-shaped streaming handler: two PayloadPart chunks over
    application/vnd.amazon.eventstream, matching InvokeEndpointWithResponseStream."""
    response = web.StreamResponse(
        status=200,
        headers={"Content-Type": "application/vnd.amazon.eventstream"},
    )
    await response.prepare(request)
    await response.write(
        _encode_frame(b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n')
    )
    await response.write(
        _encode_frame(b'data: {"choices":[{"delta":{"content":" world"}}]}\n')
    )
    await response.write_eof()
    return response


def _request_info(model_endpoint) -> RequestInfo:
    return RequestInfo(
        model_endpoint=model_endpoint,
        turns=[],
        endpoint_headers={},
        endpoint_params={},
        turn_index=0,
        credit_num=1,
        credit_phase=CreditPhase.PROFILING,
        x_request_id="test-request-id",
        x_correlation_id="test-correlation-id",
        conversation_id="test-conversation-id",
        is_final_turn=True,
    )


class TestSageMakerTransportEndToEnd:
    """Drive the real AioHttpTransport over a real loopback socket against a
    real SageMaker-shaped eventstream response -- no mocks anywhere in this
    class, unlike every other transport test which mocks post_request or the
    stream reader directly."""

    @pytest.mark.asyncio
    async def test_streaming_sagemaker_response_decodes_to_stream_messages(
        self,
    ) -> None:
        app = web.Application()
        app.router.add_post(
            "/endpoints/my-endpoint/invocations-response-stream",
            _sagemaker_invoke_handler,
        )
        server = TestServer(app)
        async with TestClient(server) as client:
            await client.start_server()
            base_url = str(client.make_url(""))

            model_endpoint = create_model_endpoint_info(
                base_url=base_url,
                custom_endpoint="/endpoints/my-endpoint/invocations-response-stream",
                streaming=True,
                aws_service="sagemaker",
            )
            transport = AioHttpTransport(model_endpoint=model_endpoint)
            await transport.initialize()
            try:
                record = await transport.send_request(
                    _request_info(model_endpoint),
                    {"messages": [{"role": "user", "content": "hi"}]},
                )
            finally:
                await transport.stop()

        assert record.error is None
        assert len(record.responses) == 2
        for response in record.responses:
            assert isinstance(response, AwsEventStreamMessage)

        assert record.responses[0].get_json() == {
            "choices": [{"delta": {"content": "Hello"}}]
        }
        assert record.responses[1].get_json() == {
            "choices": [{"delta": {"content": " world"}}]
        }

    @pytest.mark.asyncio
    async def test_first_token_callback_fires_with_stream_message(self) -> None:
        app = web.Application()
        app.router.add_post(
            "/endpoints/my-endpoint/invocations-response-stream",
            _sagemaker_invoke_handler,
        )
        server = TestServer(app)
        async with TestClient(server) as client:
            await client.start_server()
            base_url = str(client.make_url(""))

            model_endpoint = create_model_endpoint_info(
                base_url=base_url,
                custom_endpoint="/endpoints/my-endpoint/invocations-response-stream",
                streaming=True,
                aws_service="sagemaker",
            )
            transport = AioHttpTransport(model_endpoint=model_endpoint)
            await transport.initialize()

            observed: list[AwsEventStreamMessage] = []

            async def first_token_callback(ttft_ns: int, message) -> bool:
                observed.append(message)
                return True

            try:
                record = await transport.send_request(
                    _request_info(model_endpoint),
                    {"messages": [{"role": "user", "content": "hi"}]},
                    first_token_callback=first_token_callback,
                )
            finally:
                await transport.stop()

        assert record.error is None
        assert len(observed) == 1
        assert isinstance(observed[0], AwsEventStreamMessage)
        assert observed[0].get_json() == {"choices": [{"delta": {"content": "Hello"}}]}
