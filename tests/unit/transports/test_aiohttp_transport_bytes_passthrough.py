# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-serialized payload passthrough in AioHttpTransport.send_request."""

from unittest.mock import AsyncMock

import orjson
import pytest
from pytest import param

from aiperf.common.models.record_models import RequestRecord
from aiperf.transports.aiohttp_transport import AioHttpTransport
from tests.unit.transports.test_aiohttp_transport import create_request_info


class TestBytesPayloadPassthrough:
    """A pre-serialized bytes payload must reach aiohttp ``data=`` verbatim."""

    # Routing a bytes payload through orjson.dumps double-encodes it into a JSON
    # string ("..."), so the bytes path must short-circuit.

    @pytest.fixture
    def transport(self, model_endpoint_non_streaming) -> AioHttpTransport:
        """An initialized transport whose HTTP post is stubbed out."""
        return AioHttpTransport(model_endpoint=model_endpoint_non_streaming)

    async def _sent_body(
        self, transport: AioHttpTransport, model_endpoint_non_streaming, payload
    ):
        """Send ``payload`` and return the body handed to the aiohttp client."""
        await transport.initialize()
        transport.aiohttp_client.post_request = AsyncMock(return_value=RequestRecord())
        request_info = create_request_info(model_endpoint_non_streaming)
        await transport.send_request(request_info, payload)
        return transport.aiohttp_client.post_request.call_args[0][1]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload",
        [
            param(
                b'{"model":"m","messages":[{"role":"user","content":"hi"}]}',
                id="bytes",
            ),
            param(bytearray(b'{"k":"v"}'), id="bytearray"),
        ],
    )  # fmt: skip
    async def test_preserialized_payload_sent_verbatim(
        self, transport: AioHttpTransport, model_endpoint_non_streaming, payload
    ) -> None:
        """A bytes-like payload is forwarded as the same object, never re-encoded."""
        body = await self._sent_body(transport, model_endpoint_non_streaming, payload)
        assert body is payload

    @pytest.mark.asyncio
    async def test_dict_payload_still_orjson_dumped(
        self, transport: AioHttpTransport, model_endpoint_non_streaming
    ) -> None:
        """A dict payload keeps its existing orjson.dumps behavior (unchanged)."""
        payload = {"model": "m", "temperature": 0.7}
        body = await self._sent_body(transport, model_endpoint_non_streaming, payload)
        assert body == orjson.dumps(payload)
