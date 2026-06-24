# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import orjson

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
from aiperf.plugin.enums import PluginType
from aiperf.workers.dynamo_session_control import (
    build_session_control,
    merge_session_control,
)

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

        # Legacy Dynamo session_control only: session_ids this worker has already
        # sent an 'open' for. 'open' is not idempotent and must be sent exactly
        # once on the first request the worker makes for a session -- which under
        # agentic replay is the WARMUP turn (k_i), not turn_index 0. The
        # StickyCreditRouter pins every turn of a session (warmup + profiling) to
        # one worker, so this per-process set sees them all. Entries are dropped
        # on 'close' to bound the set to in-flight sessions.
        self._dynamo_opened_sessions: set[str] = set()

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
        if request_info.payload_bytes is not None:
            # PAYLOAD_BYTES fast path: bytes were validated at dataset-load time
            # by the mmap loader / DatasetManager. Defensive guard against any
            # invalid bytes that bypass upstream validation — round-trip
            # through orjson.loads so a malformed payload turns into an error
            # RequestRecord rather than reaching the wire. Body-mutating features
            # (cache-bust, Dynamo session_control) are refused against this
            # verbatim-bytes path at dataset load, so nothing is injected here.
            try:
                orjson.loads(request_info.payload_bytes)
            except (orjson.JSONDecodeError, ValueError, TypeError) as e:
                raise ValueError(
                    f"invalid JSON in pre-serialised payload_bytes: {e}"
                ) from e
            formatted_payload = request_info.payload_bytes
        else:
            current_turn = request_info.turns[-1] if request_info.turns else None
            if current_turn and current_turn.raw_payload is not None:
                formatted_payload = current_turn.raw_payload
            else:
                formatted_payload = self.endpoint.format_payload(request_info)
            # Dynamo conversation-aware routing (opt-in): overlay
            # nvext.session_control onto the structured request body. Done here,
            # after the endpoint built the dict, so it is endpoint-agnostic and
            # never mutates a cached Turn. The verbatim PAYLOAD_BYTES path is
            # excluded by the dataset-load guard, so it is not handled here.
            endpoint = self.model_endpoint.endpoint
            if endpoint.use_dynamo_conv_aware_routing:
                session_id = request_info.x_correlation_id
                legacy = endpoint.use_legacy_dynamo_session_control
                session_control = build_session_control(
                    session_id=session_id,
                    is_final_turn=request_info.is_final_turn,
                    timeout_seconds=endpoint.dynamo_session_timeout_seconds,
                    legacy=legacy,
                    already_opened=session_id in self._dynamo_opened_sessions,
                )
                # Track the open/close lifecycle so legacy 'open' is sent exactly
                # once per session (modern 'bind' is stateless and ignores this).
                if legacy:
                    if session_control.get("action") == "open":
                        self._dynamo_opened_sessions.add(session_id)
                    elif request_info.is_final_turn:
                        self._dynamo_opened_sessions.discard(session_id)
                formatted_payload = merge_session_control(
                    formatted_payload, session_control
                )
        # Canonicalise to bytes and stash on request_info. Two wins: (1) the
        # transport skips its own orjson.dumps on the dict path, (2) the
        # record processor can drop request_info.turns before the ZMQ hop
        # and still replay the exact wire payload for raw-export.
        if isinstance(formatted_payload, dict):
            formatted_payload = orjson.dumps(formatted_payload)
        request_info.payload_bytes = formatted_payload
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
        return self._enrich_request_record(record=record, request_info=request_info)

    def _enrich_request_record(
        self,
        *,
        record: RequestRecord,
        request_info: RequestInfo,
    ) -> RequestRecord:
        """Enrich a RequestRecord with a slim RecordContext.

        Down-casts the full ``RequestInfo`` (which carries the
        ``ModelEndpointInfo``, transport headers / URL params, and
        pre-send-only timing fields) into a pure ``RecordContext`` before
        attaching it to the record. Only the slim context crosses the ZMQ
        hop to the record processor.

        The tokeniser and the raw-record exporter both read
        ``request_info.payload_bytes`` unless ``strip_record_payload_bytes``
        is set (see ``AIPERF_RECORD_STRIP_PAYLOAD_BYTES``); ``osl_mismatch`` reads
        ``max_tokens``; image/audio/video metrics derive their counts from
        the endpoint's single-pass ``extract_payload_inputs`` at
        parse-time. ``turns`` is never populated on the attached context
        — live records travel turn-less and consumers drive off
        ``payload_bytes``.
        """
        turn_model = request_info.turns[-1].model if request_info.turns else None
        record.model_name = turn_model or self.model_endpoint.primary_model_name

        max_tokens = request_info.turns[-1].max_tokens if request_info.turns else None
        audio_duration_seconds = (
            request_info.turns[-1].audio_duration_seconds
            if request_info.turns
            else None
        )

        payload_bytes = (
            None if self.strip_record_payload_bytes else request_info.payload_bytes
        )

        record.request_info = RecordContext(
            credit_num=request_info.credit_num,
            credit_phase=request_info.credit_phase,
            conversation_id=request_info.conversation_id,
            turn_index=request_info.turn_index,
            x_request_id=request_info.x_request_id,
            x_correlation_id=request_info.x_correlation_id,
            credit_issued_ns=request_info.credit_issued_ns,
            agent_depth=request_info.agent_depth,
            parent_correlation_id=request_info.parent_correlation_id,
            root_correlation_id=request_info.root_correlation_id,
            payload_bytes=payload_bytes,
            max_tokens=max_tokens,
            audio_duration_seconds=audio_duration_seconds,
            cache_bust_marker=request_info.cache_bust_marker,
            cache_bust_target=request_info.cache_bust_target,
            # system_message / user_context_message stay on RequestInfo —
            # format_payload inlined them into payload_bytes before dispatch,
            # so the record processor (which reads only payload_bytes) does
            # not need them on the wire.
        )

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
