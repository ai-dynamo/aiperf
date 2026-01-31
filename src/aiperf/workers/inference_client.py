# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import (
    ErrorDetails,
    ModelEndpointInfo,
    RequestInfo,
    RequestRecord,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType

if TYPE_CHECKING:
    from aiperf.transports.base_transports import FirstTokenCallback


def detect_transport_from_url(url: str) -> str:
    """Detect transport type from URL scheme.

    Looks up registered transports and matches their url_schemes metadata
    against the URL's scheme.

    Args:
        url: URL to detect transport for.

    Returns:
        Transport plugin name (e.g., 'http').

    Raises:
        ValueError: If no transport supports the URL scheme.
    """
    parsed = urlparse(url)
    # urlparse mishandles URLs without schemes (e.g., 'localhost:8765')
    if parsed.scheme and not parsed.netloc:
        parsed = urlparse(f"http://{url}")
    scheme = parsed.scheme.lower() if parsed.scheme else "http"

    for entry in plugins.list_entries(PluginType.TRANSPORT):
        if scheme in entry.metadata.get("url_schemes", []):
            return entry.name

    raise ValueError(f"No transport found for URL scheme '{scheme}' in: {url}")


class InferenceClient(AIPerfLifecycleMixin):
    """Inference client for the worker."""

    def __init__(self, model_endpoint: ModelEndpointInfo, service_id: str, **kwargs):
        super().__init__(model_endpoint=model_endpoint, service_id=service_id, **kwargs)
        self.model_endpoint = model_endpoint
        self.service_id = service_id

        # Detect and set transport type if not explicitly set
        if not model_endpoint.transport:
            model_endpoint.transport = detect_transport_from_url(
                model_endpoint.endpoint.base_url,
            )

        # Create endpoint and transport instances
        EndpointClass = plugins.get_class(
            PluginType.ENDPOINT, self.model_endpoint.endpoint.type
        )
        self.endpoint = EndpointClass(model_endpoint=self.model_endpoint)
        TransportClass = plugins.get_class(
            PluginType.TRANSPORT, str(self.model_endpoint.transport)
        )
        self.transport = TransportClass(model_endpoint=self.model_endpoint)
        self.attach_child_lifecycle(self.transport)

    async def _send_request_to_transport(
        self,
        request_info: RequestInfo,
        first_token_callback: FirstTokenCallback | None = None,
    ) -> RequestRecord:
        """Send request via transport.

        Handles the complete request lifecycle:
        1. Populates endpoint headers and params on request_info
        2. Formats the payload using the endpoint
        3. Sends the request via the transport

        Note: Cancellation is handled by the transport layer, which ensures the
        request is always sent before being cancelled (simulating real client behavior).

        Args:
            request_info: The request information (includes cancel_after_ns).
            first_token_callback: Optional callback fired on first SSE message with ttft_ns

        Returns:
            RequestRecord containing the response data and metadata.
        """
        request_info.endpoint_headers = self.endpoint.get_endpoint_headers(request_info)
        request_info.endpoint_params = self.endpoint.get_endpoint_params(request_info)
        formatted_payload = self.endpoint.format_payload(request_info)
        return await self.transport.send_request(
            request_info,
            payload=formatted_payload,
            first_token_callback=first_token_callback,
        )

    async def _send_request_internal(
        self,
        request_info: RequestInfo,
        first_token_callback: FirstTokenCallback | None = None,
    ) -> RequestRecord:
        """Send request to transport and handle exceptions.

        Cancellation is now handled at the transport layer, which ensures the
        request is always sent before being cancelled.
        """
        pre_send_perf_ns, pre_send_timestamp_ns = None, None
        try:
            # Save the current perf_ns before sending the request so it can be used to calculate
            # the start_perf_ns of the request in case of an exception.
            pre_send_perf_ns, pre_send_timestamp_ns = (
                time.perf_counter_ns(),
                time.time_ns(),
            )

            # Transport handles cancellation internally (cancel_after_ns is in request_info)
            result = await self._send_request_to_transport(
                request_info=request_info, first_token_callback=first_token_callback
            )

            if self.is_debug_enabled:
                self.debug(
                    f"pre_send_perf_ns to start_perf_ns latency: {result.start_perf_ns - pre_send_perf_ns} ns"
                )
            return result
        except Exception as e:
            self.error(
                f"Error calling inference server API at {self.model_endpoint.endpoint.base_url}: {e!r}"
            )
            return RequestRecord(
                request_info=request_info,
                timestamp_ns=pre_send_timestamp_ns or time.time_ns(),
                # Try and use the pre_send_perf_ns if it is available, otherwise use the current time.
                start_perf_ns=pre_send_perf_ns or time.perf_counter_ns(),
                end_perf_ns=time.perf_counter_ns(),
                error=ErrorDetails.from_exception(e),
            )

    async def send_request(
        self,
        request_info: RequestInfo,
        first_token_callback: FirstTokenCallback | None = None,
    ) -> RequestRecord:
        """Send a request to the inference API. Will return an error record if the call fails.

        Args:
            request_info: The request information.
            first_token_callback: Optional callback fired on first SSE message with ttft_ns

        Returns:
            RequestRecord containing the response data and metadata.
        """
        if self.is_trace_enabled:
            self.trace(
                f"Calling inference API for turn: {request_info.turns[request_info.turn_index]}"
            )
        record = await self._send_request_internal(request_info, first_token_callback)
        return self._enrich_request_record(record=record, request_info=request_info)

    def _enrich_request_record(
        self,
        *,
        record: RequestRecord,
        request_info: RequestInfo,
    ) -> RequestRecord:
        """Enrich a RequestRecord with the original request info."""
        record.model_name = (
            request_info.turns[request_info.turn_index].model
            or self.model_endpoint.primary_model_name
        )
        record.request_info = request_info

        # Copy turns with stripped multimodal data to avoid mutating original session
        # and reduce memory usage (placeholders instead of large image/audio/video data)
        record.turns = [turn.copy_with_stripped_media() for turn in request_info.turns]

        # If this is the first turn, calculate the credit drop latency
        if request_info.turn_index == 0 and request_info.drop_perf_ns is not None:
            record.credit_drop_latency = (
                record.start_perf_ns - request_info.drop_perf_ns
            )

        # Preserve headers set by transport; only use endpoint headers if not set
        if record.request_headers is None:
            record.request_headers = request_info.endpoint_headers
        return record
