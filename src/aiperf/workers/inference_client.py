# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import orjson

from aiperf.common.enums import RequestContentType
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import (
    ErrorDetails,
    ModelEndpointInfo,
    RecordContext,
    RequestInfo,
    RequestRecord,
)
from aiperf.common.redact import redact_headers
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, TransportType

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

    def __init__(
        self,
        model_endpoint: ModelEndpointInfo,
        service_id: str,
        *,
        strip_record_payload_bytes: bool = False,
        **kwargs,
    ):
        super().__init__(model_endpoint=model_endpoint, service_id=service_id, **kwargs)
        self.model_endpoint = model_endpoint
        self.service_id = service_id
        # When True, omit canonical request payload bytes from the slim
        # RecordContext after dispatch (memory optimization for large prompts).
        # Resolved by the worker via record payload-retention auto-detection.
        self.strip_record_payload_bytes = strip_record_payload_bytes

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
        if request_info.payload_bytes is not None:
            # PAYLOAD_BYTES fast path: the mmap dataset ships pre-encoded wire
            # bytes; send them verbatim without decoding or re-encoding.
            formatted_payload: dict[str, Any] | bytes = request_info.payload_bytes
        else:
            raw_payload = request_info.turns[-1].raw_payload
            formatted_payload = (
                raw_payload
                if raw_payload is not None
                else self.endpoint.format_payload(request_info)
            )
        # Canonicalise to bytes and stash on request_info. Two wins: (1) the
        # transport skips its own orjson.dumps on the dict path, (2) the
        # record processor can read the exact wire payload for raw-export.
        # Multipart endpoints (image_edit / image_generation / video) are the
        # exception: the transport must receive the structured dict so it can
        # build the FormData. Pre-dumped bytes would be sent verbatim as JSON
        # and the server's form parser would reject them (422, prompt=null).
        wire_payload: dict[str, Any] | bytes = formatted_payload
        if isinstance(formatted_payload, dict):
            encoded = orjson.dumps(formatted_payload)
            request_info.payload_bytes = encoded
            is_multipart = (
                self.model_endpoint.endpoint.request_content_type
                == RequestContentType.MULTIPART_FORM_DATA
            )
            wire_payload = formatted_payload if is_multipart else encoded
        else:
            request_info.payload_bytes = formatted_payload
        return await self.transport.send_request(
            request_info,
            payload=wire_payload,
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
        if not request_info.turns and not request_info.payload_bytes:
            raise ValueError(
                f"RequestInfo has no turns (credit_num={request_info.credit_num}, "
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
        """Attach a slim ``RecordContext`` (downcast from ``RequestInfo``) to
        the record before the ZMQ hop to the record processor.

        The full ``RequestInfo`` carries transport-only extras
        (``model_endpoint``, ``turns``, ``system_message``,
        ``user_context_message``, ``endpoint_headers``, ``endpoint_params``,
        ``drop_perf_ns``, ``cancel_after_ns``, ...) that the record-processor
        pipeline never reads; downcasting saves ~500-900 bytes per record at
        high throughput. The full ``turns`` list never travels — live records
        drive off the canonical ``payload_bytes``.
        """
        ctx_field_names = set(RecordContext.model_fields.keys())
        ri_dump = request_info.model_dump(include=ctx_field_names)
        record.request_info = RecordContext.model_validate(ri_dump)
        return record

    def _finalize_request_record(
        self,
        *,
        record: RequestRecord,
        request_info: RequestInfo,
    ) -> RequestRecord:
        """Enrich a RequestRecord with the original request info."""
        last_turn = request_info.turns[-1] if request_info.turns else None
        record.model_name = (
            last_turn.model if last_turn else None
        ) or self.model_endpoint.primary_model_name
        # Hoist per-turn scalars onto the RecordContext so the record
        # processor's metrics (requested_osl, audio_duration,
        # replay_send_schedule_offset) can read them without walking turns:
        # max_tokens / scheduled_send_ms from the dispatch turn,
        # audio_duration_seconds from the first turn (ASR requests are
        # single-turn; mirrors the pre-hoist turns[0] read). Guarded for the
        # payload-bytes fast path, which dispatches with an empty turns list.
        first_turn = request_info.turns[0] if request_info.turns else None
        request_info.max_tokens = last_turn.max_tokens if last_turn else None
        request_info.audio_duration_seconds = (
            first_turn.audio_duration_seconds if first_turn else None
        )
        request_info.scheduled_send_ms = (
            float(last_turn.timestamp)
            if last_turn is not None and last_turn.timestamp is not None
            else None
        )
        self._enrich_request_record(record, request_info)

        # When stripping is enabled (large-prompt memory optimization,
        # resolved by the worker's payload-retention auto-detection), drop
        # the canonical request payload bytes from the slim record context
        # after dispatch.
        if self.strip_record_payload_bytes and record.request_info is not None:
            record.request_info.payload_bytes = None

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
