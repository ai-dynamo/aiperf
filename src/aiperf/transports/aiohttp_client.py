# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import socket
import time
from typing import Any

import aiohttp

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.exceptions import SSEResponseError
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import (
    ErrorDetails,
    RequestRecord,
    TextResponse,
)
from aiperf.transports.http_defaults import AioHttpDefaults, SocketDefaults
from aiperf.transports.sse_utils import AsyncSSEStreamReader


class AioHttpClient(AIPerfLoggerMixin):
    """A high-performance HTTP client for communicating with HTTP based REST APIs using aiohttp.

    This class is optimized for maximum performance and accurate timing measurements,
    making it ideal for benchmarking scenarios.
    """

    def __init__(
        self,
        timeout: float | None = None,
        tcp_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the AioHttpClient."""
        super().__init__(**kwargs)
        self.tcp_connector = create_tcp_connector(**tcp_kwargs or {})
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def close(self) -> None:
        """Close the client."""
        if self.tcp_connector:
            await self.tcp_connector.close()
            self.tcp_connector = None

    async def _request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        data: str | None = None,
        **kwargs: Any,
    ) -> RequestRecord:
        """Generic request method that handles common logic for all HTTP methods.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: The URL to send the request to
            headers: Request headers
            data: Request payload (for POST, PUT, etc.)
            **kwargs: Additional arguments to pass to the request

        Returns:
            RequestRecord with the response data
        """
        self.debug(lambda: f"Sending {method} request to {url}")

        record: RequestRecord = RequestRecord(
            start_perf_ns=time.perf_counter_ns(),
        )

        try:
            # Make raw HTTP request with precise timing using aiohttp
            async with aiohttp.ClientSession(
                connector=self.tcp_connector,
                timeout=self.timeout,
                headers=headers,
                skip_auto_headers=[
                    *list(headers.keys()),
                    "User-Agent",
                    "Accept-Encoding",
                ],
                connector_owner=False,
            ) as session:
                record.start_perf_ns = time.perf_counter_ns()
                async with session.request(
                    method, url, data=data, headers=headers, **kwargs
                ) as response:
                    record.status = response.status
                    # Check for HTTP errors
                    if response.status != 200:
                        error_text = await response.text()
                        record.error = ErrorDetails(
                            code=response.status,
                            type=response.reason,
                            message=error_text,
                        )
                        return record

                    record.recv_start_perf_ns = time.perf_counter_ns()

                    if (
                        method == "POST"
                        and response.content_type == "text/event-stream"
                    ):
                        # Parse SSE stream with optimal performance
                        async for message in AsyncSSEStreamReader(response.content):
                            AsyncSSEStreamReader.inspect_message_for_error(message)
                            record.responses.append(message)
                    else:
                        raw_response = await response.text()
                        record.end_perf_ns = time.perf_counter_ns()
                        record.responses.append(
                            TextResponse(
                                perf_ns=record.end_perf_ns,
                                content_type=response.content_type,
                                text=raw_response,
                            )
                        )
                    record.end_perf_ns = time.perf_counter_ns()
                    self.debug(
                        lambda: f"{method} request to {url} completed in {(record.end_perf_ns - record.start_perf_ns) / NANOS_PER_SECOND} seconds"
                    )
        except SSEResponseError as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Error in SSE response: {e!r}")
            record.error = ErrorDetails.from_exception(e)
        except Exception as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Error in aiohttp request: {e!r}")
            record.error = ErrorDetails.from_exception(e)

        return record

    async def post_request(
        self,
        url: str,
        payload: str,
        headers: dict[str, str],
        **kwargs: Any,
    ) -> RequestRecord:
        """Send a streaming or non-streaming POST request to the specified URL with the given payload and headers.

        If the response is an SSE stream, the response will be parsed into a list of SSE messages.
        Otherwise, the response will be parsed into a TextResponse object.
        """
        return await self._request("POST", url, headers, data=payload, **kwargs)

    async def get_request(
        self, url: str, headers: dict[str, str], **kwargs: Any
    ) -> RequestRecord:
        """Send a GET request to the specified URL with the given headers.

        The response will be parsed into a TextResponse object.
        """
        return await self._request("GET", url, headers, **kwargs)


def create_tcp_connector(**kwargs) -> aiohttp.TCPConnector:
    """Create a new connector with the given configuration."""

    def socket_factory(addr_info):
        """Custom socket factory optimized for SSE streaming performance."""
        family, sock_type, proto, _, _ = addr_info
        sock = socket.socket(family=family, type=sock_type, proto=proto)
        SocketDefaults.apply_to_socket(sock)
        return sock

    default_kwargs: dict[str, Any] = AioHttpDefaults.get_default_kwargs()
    default_kwargs["socket_factory"] = socket_factory
    default_kwargs.update(kwargs)

    return aiohttp.TCPConnector(
        **default_kwargs,
    )
