# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gRPC transport for KServe V2 Open Inference Protocol."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any
from urllib.parse import urlparse

import grpc
import orjson

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import ConnectionReuseStrategy
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.models import ErrorDetails, RequestInfo, RequestRecord, TextResponse
from aiperf.plugin.schema.schemas import TransportMetadata
from aiperf.transports.base_transports import BaseTransport, FirstTokenCallback
from aiperf.transports.grpc.grpc_client import GrpcClient
from aiperf.transports.grpc.payload_converter import (
    dict_to_model_infer_request,
    model_infer_response_to_dict,
)
from aiperf.transports.grpc.status_mapping import grpc_status_to_http
from aiperf.transports.grpc.trace_data import GrpcTraceData


class GrpcTransport(BaseTransport):
    """gRPC transport for KServe V2 Open Inference Protocol.

    Uses grpc.aio for async gRPC with HTTP/2 multiplexing.
    Supports insecure (grpc://) and TLS (grpcs://) channels.
    Supports unary (ModelInfer) and streaming (ModelStreamInfer) RPCs.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._grpc_client: GrpcClient | None = None

    @classmethod
    def metadata(cls) -> TransportMetadata:
        """Return gRPC transport metadata."""
        return TransportMetadata(
            transport_type="grpc",
            url_schemes=["grpc", "grpcs"],
        )

    @on_init
    async def _init_grpc_client(self) -> None:
        """Parse target from URL, detect secure/insecure, create GrpcClient."""
        base_url = self.model_endpoint.endpoint.base_url
        parsed = urlparse(base_url)
        scheme = parsed.scheme.lower()
        secure = scheme == "grpcs"

        # Extract host:port from URL
        target = parsed.netloc or parsed.path
        if not target:
            raise ValueError(f"Cannot parse gRPC target from URL: {base_url}")

        # Warn about connection reuse strategies that don't apply to gRPC
        reuse_strategy = self.model_endpoint.endpoint.connection_reuse_strategy
        if reuse_strategy in (
            ConnectionReuseStrategy.NEVER,
            ConnectionReuseStrategy.STICKY_USER_SESSIONS,
        ):
            self.warning(
                f"Connection reuse strategy '{reuse_strategy}' is not applicable to gRPC "
                f"(HTTP/2 multiplexing handles connection reuse). Using POOLED behavior."
            )

        timeout = self.model_endpoint.endpoint.timeout
        self._grpc_client = GrpcClient(
            target=target,
            secure=secure,
            timeout=timeout,
        )
        self.info(
            f"gRPC client initialized: target={target}, secure={secure}, timeout={timeout}"
        )

    @on_stop
    async def _close_grpc_client(self) -> None:
        """Close gRPC client and channel."""
        if self._grpc_client:
            client = self._grpc_client
            self._grpc_client = None
            await client.close()

    def get_transport_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """gRPC uses metadata, not HTTP headers. Returns empty dict."""
        return {}

    def get_url(self, request_info: RequestInfo) -> str:
        """Return the gRPC target (host:port) for the request.

        Strips the grpc:// or grpcs:// scheme, returning just the target.
        Supports multi-URL via url_index for load balancing.

        Args:
            request_info: Request context with model endpoint info.

        Returns:
            gRPC target string (host:port).
        """
        endpoint_info = request_info.model_endpoint.endpoint
        base_url = endpoint_info.get_url(request_info.url_index)
        parsed = urlparse(base_url)
        return parsed.netloc or parsed.path or base_url

    def _build_grpc_metadata(
        self, request_info: RequestInfo
    ) -> list[tuple[str, str]] | None:
        """Convert build_headers() dict to gRPC metadata tuples.

        gRPC metadata keys must be lowercase ASCII. HTTP-style headers are
        converted to lowercase for compatibility.

        Args:
            request_info: Request context.

        Returns:
            List of (key, value) tuples or None if empty.
        """
        headers = self.build_headers(request_info)
        if not headers:
            return None
        # gRPC metadata keys must be lowercase
        metadata = [(k.lower(), v) for k, v in headers.items()]
        return metadata or None

    async def send_request(
        self,
        request_info: RequestInfo,
        payload: dict[str, Any],
        *,
        first_token_callback: FirstTokenCallback | None = None,
    ) -> RequestRecord:
        """Send a gRPC request to the inference server.

        Converts the endpoint's dict payload to a ModelInferRequest protobuf,
        sends via gRPC (unary or streaming), and converts responses back to
        TextResponse objects with JSON-serialized dicts.

        Args:
            request_info: Request context and metadata.
            payload: V2 JSON-format dict from the endpoint's format_payload().
            first_token_callback: Optional callback fired on first streaming response.

        Returns:
            RequestRecord with responses, timing, and any errors.
        """
        if self._grpc_client is None:
            raise NotInitializedError(
                "GrpcTransport not initialized. Call initialize() before send_request()."
            )

        trace_data = GrpcTraceData()
        start_perf_ns = time.perf_counter_ns()
        record = RequestRecord(
            start_perf_ns=start_perf_ns,
            trace_data=trace_data,
        )

        try:
            # Build protobuf request
            model_name = request_info.model_endpoint.primary_model_name
            grpc_request = dict_to_model_infer_request(
                payload,
                model_name=model_name,
                request_id=request_info.x_request_id or "",
            )

            # Build gRPC metadata from headers
            grpc_metadata = self._build_grpc_metadata(request_info)

            # Record request send timing
            trace_data.request_send_start_perf_ns = time.perf_counter_ns()
            request_size = grpc_request.ByteSize()
            trace_data.request_chunks.append(
                (trace_data.request_send_start_perf_ns, request_size)
            )

            is_streaming = self.model_endpoint.endpoint.streaming

            if is_streaming:
                await self._send_streaming_request(
                    record=record,
                    trace_data=trace_data,
                    grpc_request=grpc_request,
                    grpc_metadata=grpc_metadata,
                    first_token_callback=first_token_callback,
                    cancel_after_ns=request_info.cancel_after_ns,
                )
            else:
                await self._send_unary_request(
                    record=record,
                    trace_data=trace_data,
                    grpc_request=grpc_request,
                    grpc_metadata=grpc_metadata,
                    cancel_after_ns=request_info.cancel_after_ns,
                )

        except asyncio.CancelledError:
            record.end_perf_ns = time.perf_counter_ns()
            record.cancellation_perf_ns = record.end_perf_ns
            record.error = ErrorDetails(
                type="RequestCancellationError",
                message="Request cancelled by external signal",
                code=499,
            )
            self.debug("gRPC request cancelled by external signal")
            raise
        except grpc.aio.AioRpcError as e:
            record.end_perf_ns = time.perf_counter_ns()
            status_code = e.code().value[0]
            http_status = grpc_status_to_http(status_code)
            trace_data.grpc_status_code = status_code
            trace_data.grpc_status_message = e.details() or str(e.code())
            trace_data.error_timestamp_perf_ns = record.end_perf_ns
            record.status = http_status
            record.error = ErrorDetails(
                type=f"gRPC:{e.code().name}",
                message=e.details() or str(e),
                code=http_status,
            )
            self.error(f"gRPC error: {e.code().name} - {e.details()}")
        except Exception as e:
            record.end_perf_ns = time.perf_counter_ns()
            trace_data.error_timestamp_perf_ns = record.end_perf_ns
            record.error = ErrorDetails.from_exception(e)
            self.error(f"gRPC request failed: {e!r}")

        return record

    async def _send_unary_request(
        self,
        *,
        record: RequestRecord,
        trace_data: GrpcTraceData,
        grpc_request: Any,
        grpc_metadata: list[tuple[str, str]] | None,
        cancel_after_ns: int | None,
    ) -> None:
        """Send a unary ModelInfer RPC, optionally with cancellation timeout.

        Args:
            record: RequestRecord to populate.
            trace_data: Trace data to populate.
            grpc_request: ModelInferRequest protobuf.
            grpc_metadata: gRPC metadata tuples.
            cancel_after_ns: Cancel after this many ns, or None for no cancellation.
        """
        assert self._grpc_client is not None  # noqa: S101

        if cancel_after_ns is None:
            response = await self._grpc_client.model_infer(
                grpc_request, metadata=grpc_metadata
            )
        else:
            timeout_s = cancel_after_ns / NANOS_PER_SECOND
            task = asyncio.create_task(
                self._grpc_client.model_infer(grpc_request, metadata=grpc_metadata)
            )
            try:
                response = await asyncio.wait_for(task, timeout=timeout_s)
            except asyncio.TimeoutError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                end_ns = time.perf_counter_ns()
                record.end_perf_ns = end_ns
                record.cancellation_perf_ns = end_ns
                record.error = ErrorDetails(
                    type="RequestCancellationError",
                    message=f"Request cancelled {timeout_s:.3f}s after being sent",
                    code=499,
                )
                self.debug(f"gRPC request cancelled {timeout_s:.3f}s after being sent")
                return

        perf_ns = time.perf_counter_ns()
        trace_data.response_receive_start_perf_ns = perf_ns
        trace_data.response_receive_end_perf_ns = perf_ns
        response_size = response.ByteSize()
        trace_data.response_chunks.append((perf_ns, response_size))
        trace_data.grpc_status_code = 0  # OK
        record.status = 200

        response_dict = model_infer_response_to_dict(response)
        json_str = orjson.dumps(response_dict).decode("utf-8")
        record.responses.append(
            TextResponse(
                perf_ns=perf_ns,
                text=json_str,
                content_type="application/json",
            )
        )
        record.end_perf_ns = perf_ns

    async def _send_streaming_request(
        self,
        *,
        record: RequestRecord,
        trace_data: GrpcTraceData,
        grpc_request: Any,
        grpc_metadata: list[tuple[str, str]] | None,
        first_token_callback: FirstTokenCallback | None,
        cancel_after_ns: int | None,
    ) -> None:
        """Send a streaming ModelStreamInfer RPC, collecting responses.

        Args:
            record: RequestRecord to populate.
            trace_data: Trace data to populate.
            grpc_request: ModelInferRequest protobuf.
            grpc_metadata: gRPC metadata tuples.
            first_token_callback: Optional callback on first non-error response.
            cancel_after_ns: Cancel after this many ns, or None for no cancellation.
        """
        assert self._grpc_client is not None  # noqa: S101

        first_token_acquired = False

        async def _consume_stream() -> None:
            nonlocal first_token_acquired
            assert self._grpc_client is not None  # noqa: S101

            async for stream_response in self._grpc_client.model_stream_infer(
                grpc_request, metadata=grpc_metadata
            ):
                if stream_response.error_message:
                    error_perf_ns = time.perf_counter_ns()
                    trace_data.error_timestamp_perf_ns = error_perf_ns
                    record.end_perf_ns = error_perf_ns
                    record.error = ErrorDetails(
                        type="gRPC:STREAM_ERROR",
                        message=stream_response.error_message,
                        code=500,
                    )
                    break

                perf_ns = time.perf_counter_ns()
                response_size = stream_response.infer_response.ByteSize()
                trace_data.response_chunks.append((perf_ns, response_size))

                if trace_data.response_receive_start_perf_ns is None:
                    trace_data.response_receive_start_perf_ns = perf_ns

                response_dict = model_infer_response_to_dict(
                    stream_response.infer_response
                )
                json_str = orjson.dumps(response_dict).decode("utf-8")
                text_response = TextResponse(
                    perf_ns=perf_ns,
                    text=json_str,
                    content_type="application/json",
                )
                record.responses.append(text_response)

                if first_token_callback and not first_token_acquired:
                    ttft_ns = perf_ns - record.start_perf_ns
                    first_token_acquired = await first_token_callback(
                        ttft_ns, text_response
                    )

            # Stream completed successfully (or with inline error)
            end_ns = time.perf_counter_ns()
            trace_data.response_receive_end_perf_ns = end_ns
            if record.end_perf_ns is None:
                record.end_perf_ns = end_ns
            if record.error is None:
                trace_data.grpc_status_code = 0  # OK
                record.status = 200

        if cancel_after_ns is None:
            await _consume_stream()
        else:
            timeout_s = cancel_after_ns / NANOS_PER_SECOND
            task = asyncio.create_task(_consume_stream())
            try:
                await asyncio.wait_for(task, timeout=timeout_s)
            except asyncio.TimeoutError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                end_ns = time.perf_counter_ns()
                record.end_perf_ns = end_ns
                record.cancellation_perf_ns = end_ns
                record.error = ErrorDetails(
                    type="RequestCancellationError",
                    message=f"Request cancelled {timeout_s:.3f}s after being sent",
                    code=499,
                )
                self.debug(
                    f"gRPC streaming request cancelled {timeout_s:.3f}s after being sent"
                )
