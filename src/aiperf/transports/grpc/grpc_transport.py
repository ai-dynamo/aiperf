# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic gRPC transport with pluggable serializers.

Protocol-agnostic: all proto knowledge is loaded dynamically from the
endpoint's ``grpc`` metadata in plugins.yaml. No proto imports here.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import time
from typing import Any, Protocol, runtime_checkable
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
from aiperf.transports.grpc.grpc_client import GenericGrpcClient
from aiperf.transports.grpc.status_mapping import grpc_status_to_http
from aiperf.transports.grpc.stream_chunk import StreamChunk
from aiperf.transports.grpc.trace_data import GrpcTraceData


@runtime_checkable
class GrpcSerializerProtocol(Protocol):
    """Interface that gRPC serializer classes must implement."""

    def serialize_request(
        self, payload: dict[str, Any], model_name: str, request_id: str = ""
    ) -> bytes: ...

    def deserialize_response(self, data: bytes) -> tuple[dict[str, Any], int]: ...

    def deserialize_stream_response(self, data: bytes) -> StreamChunk: ...


class GrpcTransport(BaseTransport):
    """Generic gRPC transport with pluggable serializers.

    Uses grpc.aio for async gRPC with HTTP/2 multiplexing.
    Supports insecure (grpc://) and TLS (grpcs://) channels.
    Supports unary and streaming RPCs for any proto schema.

    The serializer class and gRPC method paths are loaded dynamically from
    the endpoint's ``grpc`` metadata block in plugins.yaml. This keeps the
    transport fully proto-agnostic.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._grpc_client: GenericGrpcClient | None = None
        self._serializer: GrpcSerializerProtocol | None = None
        self._unary_method: str | None = None
        self._stream_method: str | None = None

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
        self._grpc_client = GenericGrpcClient(
            target=target,
            secure=secure,
            timeout=timeout,
        )
        self.info(
            f"gRPC client initialized: target={target}, secure={secure}, timeout={timeout}"
        )

    @on_init
    async def _init_serializer(self) -> None:
        """Load gRPC serializer from endpoint metadata in plugins.yaml."""
        from aiperf.plugin import plugins

        endpoint_type = str(self.model_endpoint.endpoint.type)
        metadata = plugins.get_endpoint_metadata(endpoint_type)

        if metadata.grpc is None:
            raise ValueError(
                f"Endpoint '{endpoint_type}' does not have gRPC configuration in metadata. "
                f"Add a 'grpc' block with serializer, method, and stream_method to plugins.yaml."
            )

        # Load serializer class from module:Class path
        module_path, _, class_name = metadata.grpc.serializer.rpartition(":")
        if not module_path or not class_name:
            raise ValueError(
                f"Invalid serializer format: {metadata.grpc.serializer!r}. "
                f"Expected 'module.path:ClassName'."
            )

        module = importlib.import_module(module_path)
        serializer_cls = getattr(module, class_name)
        self._serializer = serializer_cls()
        self._unary_method = metadata.grpc.method
        self._stream_method = metadata.grpc.stream_method

        self.debug(f"gRPC serializer loaded: {metadata.grpc.serializer}")

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
        if self._grpc_client is None or self._serializer is None:
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
            # Serialize request to bytes
            model_name = request_info.model_endpoint.primary_model_name
            request_bytes = self._serializer.serialize_request(
                payload,
                model_name=model_name,
                request_id=request_info.x_request_id or "",
            )

            # Build gRPC metadata from headers
            grpc_metadata = self._build_grpc_metadata(request_info)

            # Record request send timing
            trace_data.request_send_start_perf_ns = time.perf_counter_ns()
            request_size = len(request_bytes)
            trace_data.request_chunks.append(
                (trace_data.request_send_start_perf_ns, request_size)
            )

            is_streaming = self.model_endpoint.endpoint.streaming

            if is_streaming:
                await self._send_streaming_request(
                    record=record,
                    trace_data=trace_data,
                    request_bytes=request_bytes,
                    grpc_metadata=grpc_metadata,
                    first_token_callback=first_token_callback,
                    cancel_after_ns=request_info.cancel_after_ns,
                )
            else:
                await self._send_unary_request(
                    record=record,
                    trace_data=trace_data,
                    request_bytes=request_bytes,
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
        request_bytes: bytes,
        grpc_metadata: list[tuple[str, str]] | None,
        cancel_after_ns: int | None,
    ) -> None:
        """Send a unary ModelInfer RPC, optionally with cancellation timeout.

        Args:
            record: RequestRecord to populate.
            trace_data: Trace data to populate.
            request_bytes: Serialized request bytes.
            grpc_metadata: gRPC metadata tuples.
            cancel_after_ns: Cancel after this many ns, or None for no cancellation.
        """
        assert self._grpc_client is not None  # noqa: S101
        assert self._serializer is not None  # noqa: S101
        assert self._unary_method is not None  # noqa: S101

        if cancel_after_ns is None:
            response_bytes = await self._grpc_client.unary(
                self._unary_method, request_bytes, metadata=grpc_metadata
            )
        else:
            timeout_s = cancel_after_ns / NANOS_PER_SECOND
            task = asyncio.create_task(
                self._grpc_client.unary(
                    self._unary_method, request_bytes, metadata=grpc_metadata
                )
            )
            try:
                response_bytes = await asyncio.wait_for(task, timeout=timeout_s)
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
        response_dict, response_size = self._serializer.deserialize_response(
            response_bytes
        )
        trace_data.response_chunks.append((perf_ns, response_size))
        trace_data.grpc_status_code = 0  # OK
        record.status = 200

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
        request_bytes: bytes,
        grpc_metadata: list[tuple[str, str]] | None,
        first_token_callback: FirstTokenCallback | None,
        cancel_after_ns: int | None,
    ) -> None:
        """Send a streaming ModelStreamInfer RPC, collecting responses.

        Args:
            record: RequestRecord to populate.
            trace_data: Trace data to populate.
            request_bytes: Serialized request bytes.
            grpc_metadata: gRPC metadata tuples.
            first_token_callback: Optional callback on first non-error response.
            cancel_after_ns: Cancel after this many ns, or None for no cancellation.
        """
        assert self._grpc_client is not None  # noqa: S101
        assert self._serializer is not None  # noqa: S101
        assert self._stream_method is not None  # noqa: S101

        first_token_acquired = False

        async def _consume_stream() -> None:
            nonlocal first_token_acquired
            assert self._grpc_client is not None  # noqa: S101

            async for chunk_bytes in self._grpc_client.server_stream(
                self._stream_method, request_bytes, metadata=grpc_metadata
            ):
                chunk = self._serializer.deserialize_stream_response(chunk_bytes)

                if chunk.error_message:
                    error_perf_ns = time.perf_counter_ns()
                    trace_data.error_timestamp_perf_ns = error_perf_ns
                    record.end_perf_ns = error_perf_ns
                    record.error = ErrorDetails(
                        type="gRPC:STREAM_ERROR",
                        message=chunk.error_message,
                        code=500,
                    )
                    break

                perf_ns = time.perf_counter_ns()
                trace_data.response_chunks.append((perf_ns, chunk.response_size))

                if trace_data.response_receive_start_perf_ns is None:
                    trace_data.response_receive_start_perf_ns = perf_ns

                json_str = orjson.dumps(chunk.response_dict).decode("utf-8")
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
