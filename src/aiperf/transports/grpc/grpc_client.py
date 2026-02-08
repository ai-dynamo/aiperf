# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic proto-free async gRPC client."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator
from typing import Any

import grpc
import grpc.aio

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.transports.grpc.grpc_defaults import DEFAULT_CHANNEL_OPTIONS


def _identity(x: bytes) -> bytes:
    """Identity passthrough for gRPC serializer/deserializer."""
    return x


@dataclasses.dataclass(frozen=True, slots=True)
class GrpcUnaryResult:
    """Result from a unary gRPC call with response data and metadata."""

    data: bytes
    trailing_metadata: tuple[tuple[str, str | bytes], ...]


class GrpcStreamCall:
    """Wrapper around grpc.aio.UnaryStreamCall exposing metadata access.

    Provides async iteration over response chunks plus access to
    initial/trailing metadata for trace data capture.
    """

    __slots__ = ("_call",)

    def __init__(self, call: grpc.aio.UnaryStreamCall) -> None:
        self._call = call

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Yield raw response bytes from the server stream."""
        async for chunk in self._call:
            yield chunk

    async def initial_metadata(self) -> grpc.aio.Metadata:
        """Await and return the server's initial metadata (response headers)."""
        return await self._call.initial_metadata()

    async def trailing_metadata(self) -> grpc.aio.Metadata:
        """Await and return the server's trailing metadata (after stream ends)."""
        return await self._call.trailing_metadata()

    def cancel(self) -> bool:
        """Cancel the underlying RPC call."""
        return self._call.cancel()


class GenericGrpcClient(AIPerfLoggerMixin):
    """Proto-free async gRPC client operating on raw bytes.

    Uses identity serializer/deserializer so callers handle
    serialization externally via pluggable callables.
    """

    def __init__(
        self,
        target: str,
        *,
        secure: bool = False,
        timeout: float | None = None,
        channel_options: list[tuple[str, Any]] | None = None,
        ssl_credentials: grpc.ChannelCredentials | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the gRPC client.

        Args:
            target: gRPC target string (host:port).
            secure: Whether to use TLS.
            timeout: Default timeout in seconds for RPCs.
            channel_options: gRPC channel options. Defaults to DEFAULT_CHANNEL_OPTIONS.
            ssl_credentials: TLS credentials for secure channels.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(**kwargs)
        self._target = target
        self._timeout = timeout
        options = channel_options or DEFAULT_CHANNEL_OPTIONS

        if secure:
            credentials = ssl_credentials or grpc.ssl_channel_credentials()
            self._channel: grpc.aio.Channel = grpc.aio.secure_channel(
                target, credentials, options=options
            )
        else:
            self._channel = grpc.aio.insecure_channel(target, options=options)

    async def close(self) -> None:
        """Close the gRPC channel."""
        if self._channel:
            await self._channel.close()

    async def wait_for_ready(self, timeout: float | None = None) -> None:
        """Wait for the gRPC channel to reach READY state.

        Separates connection establishment time from request processing time,
        enabling accurate cancellation timing (cancel_after_ns starts after
        the channel is ready, not during connection setup).

        Args:
            timeout: Maximum seconds to wait. None means wait indefinitely.

        Raises:
            asyncio.TimeoutError: Channel didn't become ready within timeout.
            ConnectionError: Channel entered SHUTDOWN state.
        """
        state = self._channel.get_state(try_to_connect=True)
        if state == grpc.ChannelConnectivity.READY:
            return

        async def _wait() -> None:
            nonlocal state
            while state != grpc.ChannelConnectivity.READY:
                if state == grpc.ChannelConnectivity.SHUTDOWN:
                    raise ConnectionError("gRPC channel is shutdown")
                await self._channel.wait_for_state_change(state)
                state = self._channel.get_state()

        await asyncio.wait_for(_wait(), timeout=timeout)

    async def unary(
        self,
        method: str,
        request_data: bytes,
        *,
        metadata: list[tuple[str, str]] | None = None,
        timeout: float | None = None,
    ) -> GrpcUnaryResult:
        """Send a unary RPC. Returns response bytes and trailing metadata.

        Args:
            method: Fully-qualified gRPC method path (e.g. "/service/Method").
            request_data: Serialized request bytes.
            metadata: Optional gRPC metadata (key-value pairs).
            timeout: RPC timeout in seconds. Falls back to client default.

        Returns:
            GrpcUnaryResult with response data and trailing metadata.
        """
        callable_ = self._channel.unary_unary(
            method,
            request_serializer=_identity,
            response_deserializer=_identity,
        )
        call = callable_(
            request_data, metadata=metadata, timeout=timeout or self._timeout
        )
        response_bytes = await call
        trailing = await call.trailing_metadata()
        return GrpcUnaryResult(
            data=response_bytes,
            trailing_metadata=tuple(trailing),
        )

    def server_stream(
        self,
        method: str,
        request_data: bytes,
        *,
        metadata: list[tuple[str, str]] | None = None,
        timeout: float | None = None,
    ) -> GrpcStreamCall:
        """Create a server-streaming RPC call.

        Returns a GrpcStreamCall wrapper that supports async iteration over
        response chunks and access to initial/trailing metadata.

        Args:
            method: Fully-qualified gRPC method path (e.g. "/service/Method").
            request_data: Serialized request bytes.
            metadata: Optional gRPC metadata (key-value pairs).
            timeout: RPC timeout in seconds. Falls back to client default.

        Returns:
            GrpcStreamCall wrapper for async iteration and metadata access.
        """
        callable_ = self._channel.unary_stream(
            method,
            request_serializer=_identity,
            response_deserializer=_identity,
        )
        call = callable_(
            request_data, metadata=metadata, timeout=timeout or self._timeout
        )
        return GrpcStreamCall(call)
