# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end proof that the SageMaker transport works over a real connection.

Spins up a real ``aiohttp.web`` server on loopback that responds with genuine
``application/vnd.amazon.eventstream`` binary framing, then drives the actual
``SageMakerTransport.send_request()`` -- real sockets, real content-type-based
reader selection, real botocore frame decoding, real URL derivation. Nothing in
this module is mocked, unlike the other transport tests which stub
``post_request`` or the stream reader.

The frame encoder is deliberately imported from the mock server rather than
re-implemented here: one hand-rolled copy of the wire format is enough, and two
that could drift is worse than none.
"""

from __future__ import annotations

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from aiperf_mock_server.eventstream import (
    EVENTSTREAM_CONTENT_TYPE,
    encode_payload_part,
)

# botocore ships only in the optional aiperf[aws] extra, and CI installs
# "--extra test --no-dev" on Windows-on-ARM. Skip the module rather than
# failing collection there; these tests decode real eventstream frames.
pytest.importorskip("botocore")

from aiperf.common.enums import CreditPhase
from aiperf.common.models import AwsEventStreamMessage, TextResponse
from aiperf.common.models.record_models import RequestInfo
from aiperf.transports.aws.sagemaker import SageMakerTransport
from tests.unit.transports.conftest import create_model_endpoint_info

_ENDPOINT_NAME = "my-endpoint"
_STREAM_PATH = f"/endpoints/{_ENDPOINT_NAME}/invocations-response-stream"
_UNARY_PATH = f"/endpoints/{_ENDPOINT_NAME}/invocations"


async def _stream_handler(request: web.Request) -> web.StreamResponse:
    """Two PayloadPart chunks, matching InvokeEndpointWithResponseStream."""
    response = web.StreamResponse(
        status=200, headers={"Content-Type": EVENTSTREAM_CONTENT_TYPE}
    )
    await response.prepare(request)
    await response.write(
        encode_payload_part(b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n')
    )
    await response.write(
        encode_payload_part(b'data: {"choices":[{"delta":{"content":" world"}}]}\n')
    )
    await response.write_eof()
    return response


async def _unary_handler(request: web.Request) -> web.Response:
    """Plain JSON body, matching non-streaming InvokeEndpoint."""
    return web.json_response(
        {
            "id": "chatcmpl-1",
            # `object` is what the chat parser dispatches on; a real vLLM
            # container behind SageMaker returns it unchanged.
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "Hello world"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2},
        }
    )


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


async def _drive(path: str, *, streaming: bool, handler, **kwargs):
    """Run one real request against a loopback server.

    Returns ``(record, model_endpoint)``: the transport does not populate
    ``record.request_info``, and the non-streaming test needs the endpoint to
    build a parser.
    """
    app = web.Application()
    app.router.add_post(path, handler)
    async with TestClient(TestServer(app)) as client:
        await client.start_server()
        model_endpoint = create_model_endpoint_info(
            base_url=str(client.make_url("")),
            # Deliberately left unset so the transport derives the path from
            # the endpoint name -- that derivation is part of what is under test.
            custom_endpoint=None,
            streaming=streaming,
            sagemaker_endpoint_name=_ENDPOINT_NAME,
        )
        transport = SageMakerTransport(model_endpoint=model_endpoint)
        await transport.initialize()
        try:
            record = await transport.send_request(
                _request_info(model_endpoint),
                {"messages": [{"role": "user", "content": "hi"}]},
                **kwargs,
            )
            return record, model_endpoint
        finally:
            await transport.stop()


class TestSageMakerStreamingEndToEnd:
    @pytest.mark.asyncio
    async def test_eventstream_frames_decode_to_stream_messages(self) -> None:
        record, _ = await _drive(_STREAM_PATH, streaming=True, handler=_stream_handler)

        assert record.error is None
        assert len(record.responses) == 2
        assert all(isinstance(r, AwsEventStreamMessage) for r in record.responses)
        assert record.responses[0].get_json() == {
            "choices": [{"delta": {"content": "Hello"}}]
        }
        assert record.responses[1].get_json() == {
            "choices": [{"delta": {"content": " world"}}]
        }

    @pytest.mark.asyncio
    async def test_first_token_callback_fires_with_a_stream_message(self) -> None:
        observed: list[AwsEventStreamMessage] = []

        async def first_token_callback(ttft_ns: int, message) -> bool:
            observed.append(message)
            return True

        record, _ = await _drive(
            _STREAM_PATH,
            streaming=True,
            handler=_stream_handler,
            first_token_callback=first_token_callback,
        )

        assert record.error is None
        assert len(observed) == 1
        assert isinstance(observed[0], AwsEventStreamMessage)
        assert observed[0].get_json() == {"choices": [{"delta": {"content": "Hello"}}]}


class TestSageMakerNonStreamingEndToEnd:
    """The unary operation is a separate SageMaker API on a different path, so
    streaming coverage alone would leave half the integration untested."""

    @pytest.mark.asyncio
    async def test_json_response_decodes_and_parses_through_the_chat_endpoint(
        self,
    ) -> None:
        record, model_endpoint = await _drive(
            _UNARY_PATH, streaming=False, handler=_unary_handler
        )

        assert record.error is None
        assert len(record.responses) == 1
        assert isinstance(record.responses[0], TextResponse)

        # The whole point of the structural-protocol design: the stock chat
        # parser reads a SageMaker response with no SageMaker awareness at all.
        from aiperf.endpoints.openai_chat import ChatEndpoint

        endpoint = ChatEndpoint(model_endpoint=model_endpoint)
        parsed = endpoint.parse_response(record.responses[0])

        assert parsed is not None
        assert parsed.data is not None
        assert parsed.usage == {"prompt_tokens": 2, "completion_tokens": 2}
