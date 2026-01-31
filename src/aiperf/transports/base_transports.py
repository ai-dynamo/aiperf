# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata as importlib_metadata
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import (
    RequestInfo,
    RequestRecord,
    SSEMessage,
)
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.protocols import AIPerfLifecycleProtocol
from aiperf.common.types import RequestInputT
from aiperf.plugin.schema.schemas import TransportMetadata

FirstTokenCallback = Callable[[int, SSEMessage], Awaitable[bool]]
"""
Type alias for a callback that is called with the ttft_ns and the first SSE message:

Args:
    ttft_ns: duration from request start
    message: the first SSE message

Returns:
    True if this is meaningful content (stop looking for first token), False otherwise

This callback is used to determine if the first token has been received and can be released.
It is used to release prefill concurrency.
"""


@runtime_checkable
class TransportProtocol(AIPerfLifecycleProtocol, Protocol):
    """Protocol for a transport that sends requests to an inference server."""

    def __init__(self, **kwargs) -> None: ...

    @classmethod
    def metadata(cls) -> TransportMetadata: ...

    def get_transport_headers(self, request_info: RequestInfo) -> dict[str, str]: ...

    def build_headers(self, request_info: RequestInfo) -> dict[str, str]: ...

    def build_url(self, request_info: RequestInfo) -> str: ...

    def get_url(self, request_info: RequestInfo) -> str: ...

    async def send_request(
        self, request_info: RequestInfo, payload: RequestInputT
    ) -> RequestRecord: ...


class BaseTransport(AIPerfLifecycleMixin, ABC):
    """Base class for all transport protocol implementations.

    Transports handle the protocol layer (HTTP, gRPC, etc.).
    """

    def __init__(self, model_endpoint: ModelEndpointInfo, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_endpoint: ModelEndpointInfo = model_endpoint
        self.user_agent: str = f"aiperf/{importlib_metadata.version('aiperf')}"
        self.base_headers: dict[str, str] = {
            "User-Agent": self.user_agent,
        }

    @classmethod
    @abstractmethod
    def metadata(cls) -> TransportMetadata:
        """Return transport metadata for discovery and registration.

        Returns:
            Metadata describing transport type and supported URL schemes
        """
        ...

    def get_transport_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """Get protocol-specific headers (e.g., Content-Type, Accept).

        Override in subclasses to add transport-specific headers.

        Args:
            request_info: Request context

        Returns:
            Dictionary of transport-specific HTTP headers
        """
        return {}

    def build_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """Compose final headers from universal, endpoint, and transport sources.

        Merges headers in priority order:
        1. Universal headers (User-Agent, correlation IDs)
        2. Endpoint-specific headers (auth, custom)
        3. Transport-specific headers (Content-Type, Accept)

        Args:
            request_info: Request context with endpoint headers

        Returns:
            Complete header dictionary for request
        """
        headers: dict[str, str] = self.base_headers.copy()

        if request_info.x_request_id:
            headers["X-Request-ID"] = request_info.x_request_id
        if request_info.x_correlation_id:
            headers["X-Correlation-ID"] = request_info.x_correlation_id

        headers.update(request_info.endpoint_headers)
        headers.update(self.get_transport_headers(request_info))

        return headers

    def build_url(self, request_info: RequestInfo) -> str:
        """Build complete URL with query parameters from the request context.

        Preserves existing query params from base URL and merges with
        endpoint-specific params (endpoint params take precedence).

        Args:
            request_info: Request context with endpoint params

        Returns:
            Complete URL with merged query parameters
        """
        base_url = self.get_url(request_info)
        parsed = urlparse(base_url)

        # Parse existing query params from URL
        existing_params = parse_qs(parsed.query, keep_blank_values=True)
        # Flatten from lists to single values (take first)
        params = {k: v[0] if v else "" for k, v in existing_params.items()}

        # Merge endpoint params (these override existing params)
        if request_info.endpoint_params:
            params.update(request_info.endpoint_params)

        # Rebuild URL with merged params
        if params:
            return urlunparse(parsed._replace(query=urlencode(params)))

        return base_url

    @abstractmethod
    def get_url(self, request_info: RequestInfo) -> str:
        """Build base URL without query parameters.

        Args:
            request_info: Request context with model endpoint info

        Returns:
            Base URL for the request
        """
        ...

    @abstractmethod
    async def send_request(
        self,
        request_info: RequestInfo,
        payload: RequestInputT,
        *,
        first_token_callback: FirstTokenCallback | None = None,
    ) -> RequestRecord:
        """Execute request via this transport protocol.

        Args:
            request_info: Request context and metadata
            payload: Request payload (format depends on transport)
            first_token_callback: Optional callback fired on first SSE message with ttft_ns

        Returns:
            Record containing responses, timing, and any errors
        """
        ...
