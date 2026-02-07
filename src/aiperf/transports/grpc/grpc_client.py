# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-level async gRPC client for KServe V2 inference protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import grpc
import grpc.aio

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.transports.grpc.grpc_defaults import DEFAULT_CHANNEL_OPTIONS
from aiperf.transports.grpc.proto import grpc_predict_v2_pb2 as pb2
from aiperf.transports.grpc.proto import grpc_predict_v2_pb2_grpc as pb2_grpc


class GrpcClient(AIPerfLoggerMixin):
    """Low-level async gRPC client for KServe V2 inference protocol.

    Manages a single gRPC channel with HTTP/2 multiplexing.
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

        self._stub = pb2_grpc.GRPCInferenceServiceStub(self._channel)

    async def close(self) -> None:
        """Close the gRPC channel."""
        if self._channel:
            await self._channel.close()

    async def model_infer(
        self,
        request: pb2.ModelInferRequest,
        *,
        metadata: list[tuple[str, str]] | None = None,
        timeout: float | None = None,
    ) -> pb2.ModelInferResponse:
        """Unary ModelInfer RPC.

        Args:
            request: ModelInferRequest protobuf.
            metadata: Optional gRPC metadata (key-value pairs).
            timeout: RPC timeout in seconds. Falls back to client default.

        Returns:
            ModelInferResponse protobuf.
        """
        return await self._stub.ModelInfer(
            request,
            metadata=metadata,
            timeout=timeout or self._timeout,
        )

    async def model_stream_infer(
        self,
        request: pb2.ModelInferRequest,
        *,
        metadata: list[tuple[str, str]] | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[pb2.ModelStreamInferResponse]:
        """Server-side streaming ModelStreamInfer RPC.

        Args:
            request: ModelInferRequest protobuf.
            metadata: Optional gRPC metadata (key-value pairs).
            timeout: RPC timeout in seconds. Falls back to client default.

        Yields:
            ModelStreamInferResponse protobuf messages.
        """
        call = self._stub.ModelStreamInfer(
            request,
            metadata=metadata,
            timeout=timeout or self._timeout,
        )
        async for response in call:
            yield response

    async def model_ready(
        self,
        model_name: str,
        *,
        model_version: str = "",
        timeout: float | None = None,
    ) -> bool:
        """Check model readiness (health check).

        Args:
            model_name: Name of the model to check.
            model_version: Version of the model to check.
            timeout: RPC timeout in seconds.

        Returns:
            True if the model is ready.
        """
        request = pb2.ModelReadyRequest(name=model_name, version=model_version)
        response = await self._stub.ModelReady(
            request, timeout=timeout or self._timeout
        )
        return response.ready
