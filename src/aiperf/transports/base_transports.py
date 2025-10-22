# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import RequestInfo, RequestRecord, TransportMetadata
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.types import RequestInputT


class BaseTransport(AIPerfLifecycleMixin, ABC):
    """Base class for all transport protocol implementations.

    Transports handle the protocol layer (HTTP, gRPC, etc.).
    """

    def __init__(self, model_endpoint: ModelEndpointInfo, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_endpoint = model_endpoint

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
        headers: dict[str, str] = {"User-Agent": "aiperf/1.0"}

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
        self, request_info: RequestInfo, payload: RequestInputT
    ) -> RequestRecord:
        """Execute request via this transport protocol.

        Args:
            request_info: Request context and metadata
            payload: Request payload (format depends on transport)

        Returns:
            Record containing responses, timing, and any errors
        """
        ...
