# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for GenericGrpcClient."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import grpc.aio
import pytest

from aiperf.transports.grpc.grpc_client import (
    GenericGrpcClient,
    GrpcStreamCall,
    GrpcUnaryResult,
    _identity,
)
from aiperf.transports.grpc.grpc_defaults import DEFAULT_CHANNEL_OPTIONS


class TestIdentity:
    """Tests for the identity passthrough function."""

    def test_identity_returns_same_bytes(self) -> None:
        """Identity function should return input unchanged."""
        data = b"hello"
        assert _identity(data) is data


class TestGrpcUnaryResult:
    """Tests for GrpcUnaryResult dataclass."""

    def test_frozen(self) -> None:
        """GrpcUnaryResult should be immutable."""
        result = GrpcUnaryResult(data=b"resp", trailing_metadata=())
        with pytest.raises(AttributeError):
            result.data = b"other"

    def test_fields(self) -> None:
        """GrpcUnaryResult stores data and trailing_metadata."""
        meta = (("key", "val"),)
        result = GrpcUnaryResult(data=b"resp", trailing_metadata=meta)
        assert result.data == b"resp"
        assert result.trailing_metadata == meta


class TestGrpcStreamCall:
    """Tests for GrpcStreamCall wrapper."""

    @pytest.mark.asyncio
    async def test_aiter_yields_chunks(self) -> None:
        """Async iteration should yield chunks from the underlying call."""
        chunks = [b"chunk1", b"chunk2", b"chunk3"]

        class FakeCall:
            async def __aiter__(self):
                for c in chunks:
                    yield c

        stream = GrpcStreamCall(FakeCall())
        result = []
        async for chunk in stream:
            result.append(chunk)

        assert result == chunks

    @pytest.mark.asyncio
    async def test_initial_metadata(self) -> None:
        """initial_metadata should delegate to the underlying call."""
        mock_call = AsyncMock()
        expected = grpc.aio.Metadata()
        mock_call.initial_metadata = AsyncMock(return_value=expected)

        stream = GrpcStreamCall(mock_call)
        result = await stream.initial_metadata()

        assert result is expected

    @pytest.mark.asyncio
    async def test_trailing_metadata(self) -> None:
        """trailing_metadata should delegate to the underlying call."""
        mock_call = AsyncMock()
        expected = grpc.aio.Metadata()
        mock_call.trailing_metadata = AsyncMock(return_value=expected)

        stream = GrpcStreamCall(mock_call)
        result = await stream.trailing_metadata()

        assert result is expected

    def test_cancel(self) -> None:
        """cancel should delegate to the underlying call."""
        mock_call = MagicMock()
        mock_call.cancel.return_value = True

        stream = GrpcStreamCall(mock_call)
        assert stream.cancel() is True
        mock_call.cancel.assert_called_once()


class TestGenericGrpcClientInit:
    """Tests for GenericGrpcClient initialization."""

    @patch("aiperf.transports.grpc.grpc_client.grpc.aio.insecure_channel")
    def test_insecure_channel_created(self, mock_insecure: MagicMock) -> None:
        """Insecure client should create an insecure channel."""
        mock_insecure.return_value = MagicMock()
        client = GenericGrpcClient(target="localhost:8001", secure=False)

        mock_insecure.assert_called_once_with(
            "localhost:8001", options=DEFAULT_CHANNEL_OPTIONS
        )
        assert client._target == "localhost:8001"

    @patch("aiperf.transports.grpc.grpc_client.grpc.aio.secure_channel")
    @patch("aiperf.transports.grpc.grpc_client.grpc.ssl_channel_credentials")
    def test_secure_channel_created_default_creds(
        self, mock_ssl_creds: MagicMock, mock_secure: MagicMock
    ) -> None:
        """Secure client should create TLS channel with default credentials."""
        mock_creds = MagicMock()
        mock_ssl_creds.return_value = mock_creds
        mock_secure.return_value = MagicMock()

        GenericGrpcClient(target="secure-host:443", secure=True)

        mock_ssl_creds.assert_called_once()
        mock_secure.assert_called_once_with(
            "secure-host:443", mock_creds, options=DEFAULT_CHANNEL_OPTIONS
        )

    @patch("aiperf.transports.grpc.grpc_client.grpc.aio.secure_channel")
    def test_secure_channel_custom_credentials(self, mock_secure: MagicMock) -> None:
        """Secure client should use provided SSL credentials."""
        custom_creds = MagicMock(spec=grpc.ChannelCredentials)
        mock_secure.return_value = MagicMock()

        GenericGrpcClient(target="host:443", secure=True, ssl_credentials=custom_creds)

        mock_secure.assert_called_once_with(
            "host:443", custom_creds, options=DEFAULT_CHANNEL_OPTIONS
        )

    @patch("aiperf.transports.grpc.grpc_client.grpc.aio.insecure_channel")
    def test_custom_channel_options(self, mock_insecure: MagicMock) -> None:
        """Custom channel options should override defaults."""
        custom_opts = [("grpc.max_receive_message_length", 1024)]
        mock_insecure.return_value = MagicMock()

        GenericGrpcClient(target="host:8001", secure=False, channel_options=custom_opts)

        mock_insecure.assert_called_once_with("host:8001", options=custom_opts)

    @patch("aiperf.transports.grpc.grpc_client.grpc.aio.insecure_channel")
    def test_timeout_stored(self, mock_insecure: MagicMock) -> None:
        """Client should store the default timeout."""
        mock_insecure.return_value = MagicMock()
        client = GenericGrpcClient(target="host:8001", timeout=30.0)

        assert client._timeout == 30.0


class TestGenericGrpcClientClose:
    """Tests for GenericGrpcClient.close()."""

    @pytest.mark.asyncio
    async def test_close_closes_channel(self) -> None:
        """close() should close the underlying channel."""
        mock_channel = AsyncMock()
        with patch(
            "aiperf.transports.grpc.grpc_client.grpc.aio.insecure_channel",
            return_value=mock_channel,
        ):
            client = GenericGrpcClient(target="host:8001")
            await client.close()

        mock_channel.close.assert_awaited_once()


class TestGenericGrpcClientWaitForReady:
    """Tests for wait_for_ready() state machine."""

    @pytest.mark.asyncio
    async def test_already_ready_returns_immediately(self) -> None:
        """If channel is already READY, should return immediately."""
        mock_channel = MagicMock()
        mock_channel.get_state.return_value = grpc.ChannelConnectivity.READY
        with patch(
            "aiperf.transports.grpc.grpc_client.grpc.aio.insecure_channel",
            return_value=mock_channel,
        ):
            client = GenericGrpcClient(target="host:8001")
            await client.wait_for_ready(timeout=1.0)

        mock_channel.get_state.assert_called_once_with(try_to_connect=True)

    @pytest.mark.asyncio
    async def test_idle_transitions_to_ready(self) -> None:
        """Should wait for state change from IDLE to READY."""
        mock_channel = AsyncMock()
        states = iter(
            [
                grpc.ChannelConnectivity.IDLE,
                grpc.ChannelConnectivity.CONNECTING,
                grpc.ChannelConnectivity.READY,
            ]
        )
        # First call with try_to_connect=True, then without
        call_count = 0

        def get_state(try_to_connect: bool = False) -> grpc.ChannelConnectivity:
            nonlocal call_count
            call_count += 1
            return next(states)

        mock_channel.get_state = get_state
        mock_channel.wait_for_state_change = AsyncMock()

        with patch(
            "aiperf.transports.grpc.grpc_client.grpc.aio.insecure_channel",
            return_value=mock_channel,
        ):
            client = GenericGrpcClient(target="host:8001")
            await client.wait_for_ready(timeout=5.0)

        assert mock_channel.wait_for_state_change.await_count == 2

    @pytest.mark.asyncio
    async def test_shutdown_raises_connection_error(self) -> None:
        """SHUTDOWN state should raise ConnectionError."""
        mock_channel = AsyncMock()
        states = iter(
            [
                grpc.ChannelConnectivity.IDLE,
                grpc.ChannelConnectivity.SHUTDOWN,
            ]
        )

        def get_state(try_to_connect: bool = False) -> grpc.ChannelConnectivity:
            return next(states)

        mock_channel.get_state = get_state
        mock_channel.wait_for_state_change = AsyncMock()

        with patch(
            "aiperf.transports.grpc.grpc_client.grpc.aio.insecure_channel",
            return_value=mock_channel,
        ):
            client = GenericGrpcClient(target="host:8001")
            with pytest.raises(ConnectionError, match="shutdown"):
                await client.wait_for_ready(timeout=5.0)

    @pytest.mark.asyncio
    async def test_timeout_raises_timeout_error(self) -> None:
        """Should raise TimeoutError when channel doesn't become ready."""
        mock_channel = MagicMock()
        mock_channel.get_state = MagicMock(
            return_value=grpc.ChannelConnectivity.CONNECTING
        )

        # wait_for_state_change returns a future that never resolves
        async def never_resolve(_state):
            await asyncio.get_event_loop().create_future()

        mock_channel.wait_for_state_change = never_resolve

        with patch(
            "aiperf.transports.grpc.grpc_client.grpc.aio.insecure_channel",
            return_value=mock_channel,
        ):
            client = GenericGrpcClient(target="host:8001")
            with pytest.raises(asyncio.TimeoutError):
                await client.wait_for_ready(timeout=0.01)


class FakeGrpcCall:
    """Fake gRPC call object that is awaitable and has trailing_metadata."""

    def __init__(self, response: bytes = b"response", trailing: tuple = ()) -> None:
        self._response = response
        self._trailing = trailing

    def __await__(self):
        return self._do().__await__()

    async def _do(self):
        return self._response

    async def trailing_metadata(self):
        return self._trailing


class TestGenericGrpcClientUnary:
    """Tests for unary() method."""

    @pytest.mark.asyncio
    async def test_unary_uses_identity_serializer(self) -> None:
        """unary() should register identity serializer/deserializer."""
        mock_channel = MagicMock()
        mock_callable = MagicMock()
        mock_callable.return_value = FakeGrpcCall()
        mock_channel.unary_unary.return_value = mock_callable

        with patch(
            "aiperf.transports.grpc.grpc_client.grpc.aio.insecure_channel",
            return_value=mock_channel,
        ):
            client = GenericGrpcClient(target="host:8001")
            await client.unary("/svc/Method", b"request")

        mock_channel.unary_unary.assert_called_once_with(
            "/svc/Method",
            request_serializer=_identity,
            response_deserializer=_identity,
        )

    @pytest.mark.asyncio
    async def test_unary_passes_metadata(self) -> None:
        """unary() should pass metadata to the call."""
        mock_channel = MagicMock()
        mock_callable = MagicMock()
        mock_callable.return_value = FakeGrpcCall()
        mock_channel.unary_unary.return_value = mock_callable

        metadata = [("key", "val")]
        with patch(
            "aiperf.transports.grpc.grpc_client.grpc.aio.insecure_channel",
            return_value=mock_channel,
        ):
            client = GenericGrpcClient(target="host:8001", timeout=10.0)
            await client.unary("/svc/Method", b"req", metadata=metadata)

        mock_callable.assert_called_once_with(b"req", metadata=metadata, timeout=10.0)

    @pytest.mark.asyncio
    async def test_unary_explicit_timeout_overrides_default(self) -> None:
        """Explicit timeout should override the client default."""
        mock_channel = MagicMock()
        mock_callable = MagicMock()
        mock_callable.return_value = FakeGrpcCall()
        mock_channel.unary_unary.return_value = mock_callable

        with patch(
            "aiperf.transports.grpc.grpc_client.grpc.aio.insecure_channel",
            return_value=mock_channel,
        ):
            client = GenericGrpcClient(target="host:8001", timeout=30.0)
            await client.unary("/svc/Method", b"req", timeout=5.0)

        mock_callable.assert_called_once_with(b"req", metadata=None, timeout=5.0)

    @pytest.mark.asyncio
    async def test_unary_zero_timeout_not_treated_as_none(self) -> None:
        """timeout=0 should be passed as 0, not fall back to default."""
        mock_channel = MagicMock()
        mock_callable = MagicMock()
        mock_callable.return_value = FakeGrpcCall()
        mock_channel.unary_unary.return_value = mock_callable

        with patch(
            "aiperf.transports.grpc.grpc_client.grpc.aio.insecure_channel",
            return_value=mock_channel,
        ):
            client = GenericGrpcClient(target="host:8001", timeout=30.0)
            await client.unary("/svc/Method", b"req", timeout=0)

        mock_callable.assert_called_once_with(b"req", metadata=None, timeout=0)

    @pytest.mark.asyncio
    async def test_unary_returns_grpc_unary_result(self) -> None:
        """unary() should return GrpcUnaryResult with data and metadata."""
        mock_channel = MagicMock()
        mock_callable = MagicMock()
        trailing = (("x-server", "gpu-0"),)
        mock_callable.return_value = FakeGrpcCall(
            response=b"the-response", trailing=trailing
        )
        mock_channel.unary_unary.return_value = mock_callable

        with patch(
            "aiperf.transports.grpc.grpc_client.grpc.aio.insecure_channel",
            return_value=mock_channel,
        ):
            client = GenericGrpcClient(target="host:8001")
            result = await client.unary("/svc/Method", b"req")

        assert isinstance(result, GrpcUnaryResult)
        assert result.data == b"the-response"
        assert result.trailing_metadata == trailing


class TestGenericGrpcClientServerStream:
    """Tests for server_stream() method."""

    def test_server_stream_returns_grpc_stream_call(self) -> None:
        """server_stream() should return a GrpcStreamCall wrapper."""
        mock_channel = MagicMock()
        mock_callable = MagicMock()
        mock_call = MagicMock()
        mock_callable.return_value = mock_call
        mock_channel.unary_stream.return_value = mock_callable

        with patch(
            "aiperf.transports.grpc.grpc_client.grpc.aio.insecure_channel",
            return_value=mock_channel,
        ):
            client = GenericGrpcClient(target="host:8001")
            result = client.server_stream("/svc/Stream", b"req")

        assert isinstance(result, GrpcStreamCall)

    def test_server_stream_zero_timeout_not_treated_as_none(self) -> None:
        """timeout=0 should be passed as 0, not fall back to default."""
        mock_channel = MagicMock()
        mock_callable = MagicMock()
        mock_call = MagicMock()
        mock_callable.return_value = mock_call
        mock_channel.unary_stream.return_value = mock_callable

        with patch(
            "aiperf.transports.grpc.grpc_client.grpc.aio.insecure_channel",
            return_value=mock_channel,
        ):
            client = GenericGrpcClient(target="host:8001", timeout=30.0)
            client.server_stream("/svc/Stream", b"req", timeout=0)

        mock_callable.assert_called_once_with(b"req", metadata=None, timeout=0)
