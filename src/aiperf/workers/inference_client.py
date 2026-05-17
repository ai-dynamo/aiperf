# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import orjson

from aiperf.common.enums import RequestContentType
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import (
    ErrorDetails,
    MetricInputs,
    ModelEndpointInfo,
    RequestInfo,
    RequestRecord,
)
from aiperf.common.redact import redact_headers
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, TransportType

Payload = dict[str, Any] | bytes

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
            model_endpoint.transport = TransportType(
                detect_transport_from_url(model_endpoint.endpoint.base_url)
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
        self._prepare_payload_for_transport: Callable[[Payload, RequestInfo], Payload]
        if (
            model_endpoint.endpoint.request_content_type
            == RequestContentType.MULTIPART_FORM_DATA
        ):
            self._prepare_payload_for_transport = self._prepare_multipart_payload
        else:
            self._prepare_payload_for_transport = self._prepare_json_payload
        self.attach_child_lifecycle(self.transport)

    def _prepare_json_payload(
        self, payload: Payload, request_info: RequestInfo
    ) -> Payload:
        if isinstance(payload, dict):
            payload = orjson.dumps(payload)
        request_info.payload_bytes = payload
        return payload

    def _prepare_multipart_payload(
        self, payload: Payload, request_info: RequestInfo
    ) -> Payload:
        if isinstance(payload, bytes):
            request_info.payload_bytes = payload
        else:
            try:
                request_info.payload_bytes = orjson.dumps(payload)
            except TypeError:
                request_info.payload_bytes = None
        return payload

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

        # Resolution order:
        # 1. request_info.payload_bytes already set by the PAYLOAD_BYTES mmap fast path.
        # 2. The current turn carries a raw_payload dict from a verbatim-payload loader.
        # 3. Build via endpoint.format_payload for structured datasets.
        if request_info.payload_bytes is not None:
            payload: dict[str, Any] | bytes = request_info.payload_bytes
        else:
            current_turn = request_info.turns[-1] if request_info.turns else None
            if current_turn is not None and current_turn.raw_payload is not None:
                payload = current_turn.raw_payload
            else:
                payload = self.endpoint.format_payload(request_info)

        payload = self._prepare_payload_for_transport(payload, request_info)

        return await self.transport.send_request(
            request_info,
            payload=payload,
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
        if not request_info.turns and not request_info.payload_bytes:
            raise ValueError(
                f"RequestInfo has no turns and no payload_bytes "
                f"(credit_num={request_info.credit_num}, "
                f"conversation_id={request_info.conversation_id})"
            )

        if self.is_trace_enabled and request_info.turns:
            self.trace(f"Calling inference API for turn: {request_info.turns[-1]}")
        record = await self._send_request_internal(request_info, first_token_callback)
        # Redact sensitive headers on the request_info now that the transport has
        # consumed them.  This prevents raw credentials from flowing back through
        # ZMQ messages (which are TRACE-logged as serialised JSON / repr).
        request_info.endpoint_headers = (
            redact_headers(request_info.endpoint_headers) or {}
        )
        return self._finalize_request_record(record=record, request_info=request_info)

    @staticmethod
    def _enrich_request_record(
        record: RequestRecord, request_info: RequestInfo
    ) -> RequestRecord:
        """Attach a ``MetricInputs`` to the record before the ZMQ hop to the
        record processor.

        The full ``RequestInfo`` carries transport-only extras (model_endpoint,
        endpoint_headers, endpoint_params, drop_perf_ns, cancel_after_ns, the
        worker-side ``turns`` list, ...) that the record-processor pipeline
        never reads; ``MetricInputs`` is the flat wire schema carrying only
        routing identity, DAG fields, and optional inline payload bytes.
        When ``from_mmap`` is True the records pipeline resolves bytes via
        its own mmap client, so wire bytes are dropped.
        """
        record.metric_inputs = MetricInputs(
            credit_num=request_info.credit_num,
            credit_phase=request_info.credit_phase,
            conversation_id=request_info.conversation_id,
            turn_index=request_info.turn_index,
            x_request_id=request_info.x_request_id,
            x_correlation_id=request_info.x_correlation_id,
            credit_issued_ns=request_info.credit_issued_ns,
            agent_depth=request_info.agent_depth,
            parent_correlation_id=request_info.parent_correlation_id,
            # ``MetricInputs.payload_bytes`` is plain ``bytes | None``; under
            # the msgpack records-pipeline wire it rides as a length-prefixed
            # bin span (no base64 inflation, binary-transparent). When the
            # payload was fetched from mmap, the records process resolves it
            # via its own client -- wire bytes are dropped.
            payload_bytes=None
            if request_info.from_mmap
            else request_info.payload_bytes,
        )
        return record

    def _finalize_request_record(
        self,
        *,
        record: RequestRecord,
        request_info: RequestInfo,
    ) -> RequestRecord:
        """Enrich a RequestRecord with the original request info."""
        record.model_name = (
            (request_info.turns[-1].model or self.model_endpoint.primary_model_name)
            if request_info.turns
            else self.model_endpoint.primary_model_name
        )
        self._enrich_request_record(record, request_info)

        # If this is the first turn, calculate the credit drop latency
        if request_info.turn_index == 0 and request_info.drop_perf_ns is not None:
            record.credit_drop_latency = (
                record.start_perf_ns - request_info.drop_perf_ns
            )

        # Always redact at this boundary to guarantee no raw headers leak downstream,
        # even if a transport pre-populates record.request_headers.
        source_headers = (
            record.request_headers
            if record.request_headers is not None
            else request_info.endpoint_headers
        )
        record.request_headers = redact_headers(source_headers)
        return record
