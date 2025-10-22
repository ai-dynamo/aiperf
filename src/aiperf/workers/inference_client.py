# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiperf.common.factories import EndpointFactory, TransportFactory
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import ModelEndpointInfo, RequestInfo, RequestRecord


class InferenceClient(AIPerfLifecycleMixin):
    """Inference client for the worker."""

    def __init__(self, model_endpoint: ModelEndpointInfo, **kwargs):
        super().__init__(model_endpoint=model_endpoint, **kwargs)
        self.model_endpoint = model_endpoint

        # Detect and set transport type if not explicitly set
        if not model_endpoint.transport:
            model_endpoint.transport = TransportFactory.detect_from_url(
                model_endpoint.endpoint.base_url
            )
            if not model_endpoint.transport:
                raise ValueError(
                    f"No transport found for URL: {model_endpoint.endpoint.base_url}"
                )

        # Create endpoint and transport instances
        self.endpoint = EndpointFactory.create_instance(
            self.model_endpoint.endpoint.type,
            model_endpoint=self.model_endpoint,
        )
        self.transport = TransportFactory.create_instance(
            self.model_endpoint.transport,
            model_endpoint=self.model_endpoint,
        )
        self.attach_child_lifecycle(self.transport)

    async def send_request(self, request_info: RequestInfo) -> RequestRecord:
        """Send request via transport.

        Handles the complete request lifecycle:
        1. Populates endpoint headers and params on request_info
        2. Formats the payload using the endpoint
        3. Sends the request via the transport

        Args:
            request_info: The request information.

        Returns:
            RequestRecord containing the response data and metadata.
        """
        request_info.endpoint_headers = self.endpoint.get_endpoint_headers(request_info)
        request_info.endpoint_params = self.endpoint.get_endpoint_params(request_info)
        formatted_payload = self.endpoint.format_payload(request_info)
        return await self.transport.send_request(
            request_info, payload=formatted_payload
        )
