# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import orjson

from aiperf.common.enums import TransportType
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.factories import TransportFactory
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.models import ErrorDetails, RequestInfo, RequestRecord
from aiperf.transports.aiohttp_client import AioHttpClient
from aiperf.transports.base_transports import BaseTransport, TransportMetadata


@TransportFactory.register(TransportType.HTTP)
class AioHttpTransport(BaseTransport):
    """HTTP/1.1 transport implementation using aiohttp.

    Provides high-performance async HTTP client with:
    - Connection pooling and TCP optimization
    - SSE (Server-Sent Events) streaming support
    - Automatic error handling and timing
    - Custom TCP connector configuration
    """

    def __init__(
        self, tcp_kwargs: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """Initialize HTTP transport with optional TCP configuration.

        Args:
            tcp_kwargs: TCP connector configuration (socket options, timeouts, etc.)
            **kwargs: Additional arguments passed to parent classes
        """
        super().__init__(**kwargs)
        self.tcp_kwargs = tcp_kwargs
        self.aiohttp_client = None

    @on_init
    async def _init_aiohttp_client(self) -> None:
        """Initialize the AioHttpClient."""
        self.aiohttp_client = AioHttpClient(
            timeout=self.model_endpoint.endpoint.timeout, tcp_kwargs=self.tcp_kwargs
        )

    @on_stop
    async def _close_aiohttp_client(self) -> None:
        """Cleanup hook to close aiohttp session on stop."""
        if self.aiohttp_client:
            await self.aiohttp_client.close()
            self.aiohttp_client = None

    @classmethod
    def metadata(cls) -> TransportMetadata:
        """Return HTTP transport metadata."""
        return TransportMetadata(
            transport_type=TransportType.HTTP,
            url_schemes=["http", "https"],
        )

    def get_transport_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """Build HTTP-specific headers based on streaming mode.

        Args:
            request_info: Request context with endpoint configuration

        Returns:
            HTTP headers (Content-Type and Accept)
        """
        accept = (
            "text/event-stream"
            if request_info.model_endpoint.endpoint.streaming
            else "application/json"
        )
        return {"Content-Type": "application/json", "Accept": accept}

    def get_url(self, request_info: RequestInfo) -> str:
        """Build HTTP URL from base_url and endpoint path.

        Constructs the full URL by combining the base URL with the endpoint path
        from metadata or custom endpoint. Adds http:// scheme if missing.

        Args:
            request_info: Request context with model endpoint info

        Returns:
            Complete HTTP URL with scheme and endpoint path
        """
        endpoint_info = request_info.model_endpoint.endpoint

        # Start with base URL
        base_url = endpoint_info.base_url.rstrip("/")

        # Determine the endpoint path
        if endpoint_info.custom_endpoint:
            # Use custom endpoint path if provided
            path = endpoint_info.custom_endpoint.lstrip("/")
            url = f"{base_url}/{path}"
        else:
            # Get endpoint path from endpoint metadata
            from aiperf.common.factories import EndpointFactory

            endpoint_metadata = EndpointFactory.get_metadata(endpoint_info.type)
            endpoint_path = endpoint_metadata.endpoint_path
            if (
                self.model_endpoint.endpoint.streaming
                and endpoint_metadata.streaming_path is not None
            ):
                endpoint_path = endpoint_metadata.streaming_path
            if not endpoint_path:
                # No endpoint path, just use base URL
                url = base_url

            else:
                path = endpoint_path.lstrip("/")
                # Handle /v1 base URL with v1/ path prefix to avoid duplication
                if base_url.endswith("/v1") and path.startswith("v1/"):
                    path = path.removeprefix("v1/")
                url = f"{base_url}/{path}"
        return url if url.startswith("http") else f"http://{url}"

    async def send_request(
        self, request_info: RequestInfo, payload: dict[str, Any]
    ) -> RequestRecord:
        """Send HTTP POST request with JSON payload.

        Args:
            request_info: Request context and metadata
            payload: JSON-serializable request payload

        Returns:
            Request record with responses, timing, and any errors
        """
        if self.aiohttp_client is None:
            raise NotInitializedError(
                "AioHttpTransport not initialized. Call initialize() before send_request()."
            )

        start_perf_ns = time.perf_counter_ns()
        headers = None
        try:
            url = self.build_url(request_info)
            headers = self.build_headers(request_info)

            # Serialize with orjson for performance
            json_str = orjson.dumps(payload).decode("utf-8")
            record = await self.aiohttp_client.post_request(url, json_str, headers)
            record.request_headers = headers
        except Exception as e:
            # Capture all exceptions with timing and error details
            record = RequestRecord(
                request_headers=headers or request_info.endpoint_headers,
                start_perf_ns=start_perf_ns,
                end_perf_ns=time.perf_counter_ns(),
                error=ErrorDetails.from_exception(e),
            )
            self.exception(f"HTTP request failed: {e!r}")

        return record
