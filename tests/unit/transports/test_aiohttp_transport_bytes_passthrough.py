# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import orjson
import pytest

from aiperf.common.models.record_models import RequestRecord
from aiperf.transports.aiohttp_transport import AioHttpTransport
from tests.unit.transports.test_aiohttp_transport import create_request_info


class TestBytesPayloadPassthrough:
    """A pre-serialized bytes payload must reach aiohttp ``data=`` verbatim.

    The transport double-encodes a bytes payload into a JSON string (``"..."``)
    if it routes through ``orjson.dumps``; the bytes path must short-circuit.
    """

    @pytest.fixture
    def transport(self, model_endpoint_non_streaming):
        return AioHttpTransport(model_endpoint=model_endpoint_non_streaming)

    async def _setup(self, transport):
        await transport.initialize()
        transport.aiohttp_client.post_request = AsyncMock(return_value=RequestRecord())

    @pytest.mark.asyncio
    async def test_bytes_payload_sent_verbatim(
        self, transport, model_endpoint_non_streaming
    ):
        """A bytes payload is forwarded byte-identical, not orjson.dumps'd."""
        await self._setup(transport)

        request_info = create_request_info(model_endpoint_non_streaming)
        payload = b'{"model":"m","messages":[{"role":"user","content":"hi"}]}'

        await transport.send_request(request_info, payload)

        body = transport.aiohttp_client.post_request.call_args[0][1]
        # Identity: the exact bytes object is forwarded, never re-encoded.
        assert body is payload

    @pytest.mark.asyncio
    async def test_bytearray_payload_sent_verbatim(
        self, transport, model_endpoint_non_streaming
    ):
        """A bytearray payload is also short-circuited (no orjson.dumps)."""
        await self._setup(transport)

        request_info = create_request_info(model_endpoint_non_streaming)
        payload = bytearray(b'{"k":"v"}')

        await transport.send_request(request_info, payload)

        body = transport.aiohttp_client.post_request.call_args[0][1]
        # Identity: the exact bytearray object is forwarded, never re-encoded.
        assert body is payload

    @pytest.mark.asyncio
    async def test_dict_payload_still_orjson_dumped(
        self, transport, model_endpoint_non_streaming
    ):
        """A dict payload keeps its existing orjson.dumps behavior (unchanged)."""
        await self._setup(transport)

        request_info = create_request_info(model_endpoint_non_streaming)
        payload = {"model": "m", "temperature": 0.7}

        await transport.send_request(request_info, payload)

        body = transport.aiohttp_client.post_request.call_args[0][1]
        assert body == orjson.dumps(payload)
