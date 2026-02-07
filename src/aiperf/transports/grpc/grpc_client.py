# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic proto-free async gRPC client."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import grpc
import grpc.aio

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.transports.grpc.grpc_defaults import DEFAULT_CHANNEL_OPTIONS


def _identity(x: bytes) -> bytes:
    """Identity passthrough for gRPC serializer/deserializer."""
    return x


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

    async def unary(
        self,
        method: str,
        request_data: bytes,
        *,
        metadata: list[tuple[str, str]] | None = None,
        timeout: float | None = None,
    ) -> bytes:
        """Send a unary RPC. Returns raw response bytes.

        Args:
            method: Fully-qualified gRPC method path (e.g. "/service/Method").
            request_data: Serialized request bytes.
            metadata: Optional gRPC metadata (key-value pairs).
            timeout: RPC timeout in seconds. Falls back to client default.

        Returns:
            Serialized response bytes.
        """
        callable_ = self._channel.unary_unary(
            method,
            request_serializer=_identity,
            response_deserializer=_identity,
        )
        return await callable_(
            request_data, metadata=metadata, timeout=timeout or self._timeout
        )

    async def server_stream(
        self,
        method: str,
        request_data: bytes,
        *,
        metadata: list[tuple[str, str]] | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[bytes]:
        """Send a server-streaming RPC. Yields raw response bytes.

        Args:
            method: Fully-qualified gRPC method path (e.g. "/service/Method").
            request_data: Serialized request bytes.
            metadata: Optional gRPC metadata (key-value pairs).
            timeout: RPC timeout in seconds. Falls back to client default.

        Yields:
            Serialized response bytes for each stream chunk.
        """
        callable_ = self._channel.unary_stream(
            method,
            request_serializer=_identity,
            response_deserializer=_identity,
        )
        call = callable_(
            request_data, metadata=metadata, timeout=timeout or self._timeout
        )
        async for chunk in call:
            yield chunk
