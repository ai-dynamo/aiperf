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
from aiperf.plugin.enums import PluginType, TransportType
from aiperf.transports.base_transports import effective_streaming
from aiperf.workers.session_routing import RoutingContext, SessionRoutingBase

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

        # Session-routing plugin (selected via --session-routing): one instance
        # per worker, invoked at the request-serialization chokepoint to stamp
        # per-session identity (headers and/or body). None when routing is off.
        self._routing: SessionRoutingBase | None = None
        self._routing_mode: str | None = None
        self._warned_bytes_routing = False
        endpoint_info = model_endpoint.endpoint
        if endpoint_info.session_routing is not None:
            routing_cls = plugins.get_class(
                PluginType.SESSION_ROUTING, endpoint_info.session_routing
            )
            self._routing = routing_cls(
                routing_cls.Options(**endpoint_info.session_routing_opts)
            )
            self._routing_mode = endpoint_info.session_routing

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

    @property
    def session_routing_active(self) -> bool:
        """True when a --session-routing plugin owns per-session identity."""
        return self._routing is not None

    def notify_session_end(self, x_correlation_id: str) -> None:
        """Post-session pass-through to the routing plugin (idempotent hook).

        Called by the worker terminal-eviction paths on ANY terminal outcome of
        a session. On this codebase those are: a successful final turn, a
        cancellation, and a cancel-before-start (the done-callback path whose
        finally block never runs). Idempotency is the plugin's responsibility --
        this hook does not dedupe. No-op when session routing is unset.

        A plugin exception is logged (naming the plugin and session) and
        swallowed: this cleanup hook must never break the worker's core
        session-eviction lifecycle.
        """
        if self._routing is None:
            return
        try:
            self._routing.on_session_end(x_correlation_id)
        except Exception as e:
            self.warning(
                f"session-routing plugin {self._routing_mode!r} on_session_end "
                f"failed for session {x_correlation_id!r}; continuing eviction: {e!r}"
            )

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
        # Session-routing chokepoint: build the per-request routing context once
        # and let the plugin stamp its headers now (merged onto the endpoint
        # headers). The same context feeds the structured body transform below.
        routing_ctx: RoutingContext | None = None
        if self._routing is not None:
            routing_ctx = RoutingContext(
                x_correlation_id=request_info.x_correlation_id,
                parent_correlation_id=request_info.parent_correlation_id,
                root_correlation_id=request_info.root_correlation_id,
                is_final_turn=request_info.is_final_turn,
                is_parent_final=request_info.is_parent_final,
                is_tree_final=request_info.is_tree_final,
            )
            # Attribute a plugin fault to the routing plugin (not the server):
            # this raise is caught by _send_request_internal and becomes an error
            # record whose message names the plugin instead of the endpoint.
            try:
                routing_headers = self._routing.headers(routing_ctx)
            except Exception as e:
                raise RuntimeError(
                    f"session-routing plugin {self._routing_mode!r} failed in headers(): {e!r}"
                ) from e
            request_info.endpoint_headers.update(routing_headers)

        raw_payload_bytes = request_info.turns[-1].raw_payload_bytes
        raw_payload = request_info.turns[-1].raw_payload
        if raw_payload_bytes is not None:
            # Pre-serialized body (weka graph-IR bytes path): the bytes ARE valid
            # JSON, so send and record them verbatim. orjson.dumps(<bytes>) would
            # corrupt payload_bytes into a JSON string and break ISL/raw-export.
            # Body-based routing transforms cannot apply to a verbatim-bytes
            # payload (the header stamp above still does) -- warn ONCE so a
            # body-mutating mode never silently loses its bind/close writes.
            if (
                routing_ctx is not None
                and self._routing.mutates_body
                and not self._warned_bytes_routing
            ):
                self._warned_bytes_routing = True
                self.warning(
                    f"session-routing mode {self._routing_mode!r} mutates the "
                    "request BODY, but this workload sends pre-serialized "
                    "verbatim bytes (graph-IR replay); the body transform is "
                    "skipped for those requests (headers still apply). Use a "
                    "header-based mode (e.g. dynamo_headers) for byte-exact "
                    "graph replay."
                )
            payload = raw_payload_bytes
            request_info.payload_bytes = raw_payload_bytes
        else:
            payload = (
                raw_payload
                if raw_payload is not None
                else self.endpoint.format_payload(request_info)
            )
            # Body-based session routing (e.g. Dynamo nvext.session_control):
            # overlay onto the structured body, endpoint-agnostic, after the
            # payload dict is in hand. transform_body returns a copy, so this
            # never mutates a cached Turn.raw_payload dict (the copy-on-write
            # contract is load-bearing here).
            if routing_ctx is not None and isinstance(payload, dict):
                try:
                    payload = self._routing.transform_body(payload, routing_ctx)
                except Exception as e:
                    raise RuntimeError(
                        f"session-routing plugin {self._routing_mode!r} failed "
                        f"in transform_body(): {e!r}"
                    ) from e
            request_info.payload_bytes = orjson.dumps(payload)
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
        if not request_info.turns:
            raise ValueError(
                f"RequestInfo has no turns (credit_num={request_info.credit_num}, "
                f"conversation_id={request_info.conversation_id})"
            )
        if self.is_trace_enabled:
            self.trace(f"Calling inference API for turn: {request_info.turns[-1]}")
        record = await self._send_request_internal(request_info, first_token_callback)
        # Stamp the per-request effective wire mode as ground truth before the
        # downcast in _finalize_request_record. This holds even for error
        # records: a mid-stream failure was still a streamed send.
        record.streamed = effective_streaming(request_info)
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
        (``model_endpoint``, ``turns``, ``endpoint_headers``,
        ``endpoint_params``, ``drop_perf_ns``, ``cancel_after_ns``, ...) that
        the record-processor pipeline never reads; downcasting saves
        ~500-900 bytes per record at high throughput.
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
        record.model_name = (
            request_info.turns[-1].model or self.model_endpoint.primary_model_name
        )
        # Hoist per-turn scalars onto the RecordContext so the record
        # processor's metrics (requested_osl, audio_duration) can read them
        # without walking turns: max_tokens from the dispatch turn,
        # audio_duration_seconds from the first turn (ASR requests are
        # single-turn; mirrors the pre-hoist turns[0] read).
        request_info.max_tokens = request_info.turns[-1].max_tokens
        request_info.audio_duration_seconds = request_info.turns[
            0
        ].audio_duration_seconds
        self._enrich_request_record(record, request_info)

        # When stripping is enabled (large-prompt memory optimization,
        # resolved by the worker's payload-retention auto-detection), drop
        # the canonical request payload bytes from the slim record context
        # after dispatch.
        if self.strip_record_payload_bytes and record.request_info is not None:
            record.request_info.payload_bytes = None

        # Copy turns with stripped multimodal data to avoid mutating original session
        # and reduce memory usage (placeholders instead of large image/audio/video data)
        record.turns = [turn.copy_with_stripped_media() for turn in request_info.turns]

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
