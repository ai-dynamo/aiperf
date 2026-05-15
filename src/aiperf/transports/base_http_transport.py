# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base HTTP transport providing shared URL construction, header logic, and video polling."""

from __future__ import annotations

import asyncio
import time
from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable

import aiohttp
import orjson

from aiperf.common.enums import RequestContentType, VideoJobStatus
from aiperf.common.environment import Environment
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.models import (
    BinaryResponse,
    ErrorDetails,
    RequestInfo,
    RequestRecord,
    TextResponse,
)
from aiperf.plugin import plugins
from aiperf.transports.base_transports import BaseTransport


@runtime_checkable
class HTTPClientProtocol(Protocol):
    """Protocol defining the minimal HTTP client interface used by BaseHTTPTransport.

    Both AioHttpClient and HttpCoreClient satisfy this protocol, enabling
    video polling and other shared logic to work with either transport.
    """

    async def post_request(
        self, url: str, payload: bytes, headers: dict[str, str], **kwargs: Any
    ) -> RequestRecord: ...

    async def get_request(
        self, url: str, headers: dict[str, str], **kwargs: Any
    ) -> RequestRecord: ...

    async def close(self) -> None: ...


def _append_path_deduped(base_url: str, path: str) -> str:
    """Append *path* to *base_url*, deduplicating when the URL already contains the path.

    Handles common user mistakes like providing ``http://server:8000/v1/chat/completions``
    as the base URL when the endpoint path is ``v1/chat/completions``.
    """
    if base_url.endswith(f"/{path}"):
        return base_url
    # Handle /v1 base URL with v1/ path prefix to avoid /v1/v1/...
    if base_url.endswith("/v1") and path.startswith("v1/"):
        path = path.removeprefix("v1/")
    return f"{base_url}/{path}"


class BaseHTTPTransport(BaseTransport):
    """Base class for HTTP-based transports (HTTP/1.1 and HTTP/2).

    Provides shared URL construction, header building, streaming path resolution,
    and video polling logic. Subclasses implement client lifecycle and send_request().
    """

    @property
    @abstractmethod
    def http_client(self) -> HTTPClientProtocol | None:
        """Return the underlying HTTP client instance, or None if not initialized."""
        ...

    def get_transport_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """Build HTTP-specific headers based on streaming mode.

        When request_content_type is multipart/form-data, Content-Type is
        omitted so aiohttp can auto-set it with the correct boundary.
        """
        accept = (
            "text/event-stream"
            if self.run.cfg.endpoint.streaming
            else "application/json"
        )
        headers: dict[str, str] = {"Accept": accept}
        content_type = self.run.cfg.endpoint.request_content_type
        if content_type != RequestContentType.MULTIPART_FORM_DATA:
            headers["Content-Type"] = (
                content_type or RequestContentType.APPLICATION_JSON
            )
        return headers

    @staticmethod
    def _build_form_data(payload: dict[str, Any]) -> aiohttp.FormData:
        """Build multipart form data from a payload dict."""
        form_data = aiohttp.FormData()
        for key, value in payload.items():
            if value is not None:
                str_value = (
                    str(value).lower() if isinstance(value, bool) else str(value)
                )
                form_data.add_field(key, str_value)
        return form_data

    def get_url(self, request_info: RequestInfo) -> str:
        """Build HTTP URL from base_url and endpoint path.

        Constructs the full URL by combining the base URL with the endpoint path
        from metadata or custom endpoint. Adds http:// scheme if missing.

        When multiple URLs are configured, uses request_info.url_index to select
        the appropriate URL for load balancing.

        Args:
            request_info: Request context with model endpoint info

        Returns:
            Complete HTTP URL with scheme and endpoint path
        """
        endpoint_info = self.run.cfg.endpoint

        # Start with base URL - use url_index for multi-URL load balancing
        url_index = request_info.url_index if request_info.url_index is not None else 0
        base_url = endpoint_info.urls[url_index % len(endpoint_info.urls)].rstrip("/")

        # Determine the endpoint path
        if endpoint_info.path:
            # Use custom endpoint path if provided
            path = endpoint_info.path.lstrip("/")
        else:
            # Get endpoint path from endpoint metadata
            endpoint_metadata = plugins.get_endpoint_metadata(endpoint_info.type)
            path = endpoint_metadata.endpoint_path
            if (
                self.run.cfg.endpoint.streaming
                and endpoint_metadata.streaming_path is not None
            ):
                path = endpoint_metadata.streaming_path

        if not path:
            url = base_url
        else:
            path = path.lstrip("/")
            url = _append_path_deduped(base_url, path)
        # Add scheme if missing for proper parsing
        return url if url.startswith("http") else f"http://{url}"

    # -- Video polling methods (shared by aiohttp and httpcore transports) --

    def _parse_video_response(
        self,
        record: RequestRecord,
        context: str,
    ) -> tuple[dict[str, Any], TextResponse] | ErrorDetails:
        """Parse JSON response from a video API request record.

        Args:
            record: The request record to parse
            context: Description for error messages (e.g., "submit", "poll")

        Returns:
            Tuple of (parsed_json, text_response) on success, or ErrorDetails on failure
        """
        if record.error:
            return record.error
        if not record.responses:
            return ErrorDetails(
                type="VideoGenerationError",
                message=f"No response from video {context}",
                code=500,
            )
        response = record.responses[0]
        if not isinstance(response, TextResponse):
            return ErrorDetails(
                type="VideoGenerationError",
                message=f"Unexpected response type from video {context}",
                code=500,
            )
        try:
            return orjson.loads(response.text), response
        except orjson.JSONDecodeError:
            snippet = response.text[:200] if response.text else "<empty>"
            return ErrorDetails(
                type="VideoGenerationError",
                message=f"Invalid JSON in video {context} response (status {record.status}): {snippet}",
                code=500,
            )

    async def _submit_video_job(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        *,
        use_form_data: bool = False,
    ) -> tuple[str, TextResponse] | ErrorDetails:
        """Submit video generation job via POST /v1/videos.

        Returns (job_id, response) on success, ErrorDetails on failure.
        """
        if self.http_client is None:
            raise NotInitializedError("HTTP client not initialized")
        body: bytes | aiohttp.FormData = (
            self._build_form_data(payload) if use_form_data else orjson.dumps(payload)
        )
        record = await self.http_client.post_request(url, body, headers)
        result = self._parse_video_response(record, "submit")
        if isinstance(result, ErrorDetails):
            return result

        job_data, response = result
        job_id = job_data.get("id")
        if not job_id:
            return ErrorDetails(
                type="VideoGenerationError",
                message=f"No job ID returned: {job_data}",
                code=500,
            )
        latency_ms = (record.end_perf_ns - record.start_perf_ns) / 1e6
        self.info(f"Video job {job_id} submitted ({latency_ms:.0f}ms)")
        return job_id, response

    async def _poll_video_job(
        self,
        job_id: str,
        poll_url: str,
        headers: dict[str, str],
        *,
        timeout: float,
        poll_interval: float,
    ) -> tuple[dict[str, Any], float] | ErrorDetails:
        """Poll video job until completed/failed. Returns (data, elapsed) or error."""
        if self.http_client is None:
            raise NotInitializedError("HTTP client not initialized")
        self.info(f"Polling video job {job_id}")
        poll_start = time.perf_counter_ns()

        while (time.perf_counter_ns() - poll_start) / 1e9 < timeout:
            record = await self.http_client.get_request(poll_url, headers)
            result = self._parse_video_response(record, "poll")
            if isinstance(result, ErrorDetails):
                return result

            data, _ = result
            status = data.get("status", "")

            if status == VideoJobStatus.COMPLETED:
                elapsed = (time.perf_counter_ns() - poll_start) / 1e9
                self.info(f"Video job {job_id} completed in {elapsed:.1f}s")
                return data, elapsed

            if status == VideoJobStatus.FAILED:
                error_info = data.get("error", {})
                msg = (
                    error_info.get("message", "Unknown error")
                    if isinstance(error_info, dict)
                    else str(error_info)
                )
                self.error(f"Video job {job_id} failed: {msg}")
                return ErrorDetails(
                    type="VideoGenerationError",
                    message=f"Video generation failed: {msg}",
                    code=500,
                )

            await asyncio.sleep(poll_interval)

        self.error(f"Video job {job_id} timed out after {timeout}s")
        return ErrorDetails(
            type="TimeoutError",
            message=f"Video generation timed out after {timeout}s",
            code=504,
        )

    async def _download_video_content(
        self,
        job_id: str,
        content_url: str,
        headers: dict[str, str],
    ) -> bytes | ErrorDetails:
        """Download video content via GET /v1/videos/{id}/content.

        Returns video bytes on success, ErrorDetails on failure.
        Used when --download-video-content is enabled.
        """
        if self.http_client is None:
            raise NotInitializedError("HTTP client not initialized")
        try:
            record = await self.http_client.get_request(content_url, headers)
            if record.error:
                return ErrorDetails(
                    type="VideoDownloadError",
                    message=f"Failed to download video {job_id}: {record.error}",
                    code=record.status or 500,
                )
            if record.responses and isinstance(record.responses[0], BinaryResponse):
                self.info(
                    f"Video {job_id} downloaded ({len(record.responses[0].raw_bytes)} bytes)"
                )
                return record.responses[0].raw_bytes
            return ErrorDetails(
                type="VideoDownloadError",
                message=f"No content returned for video {job_id}",
                code=500,
            )
        except Exception as e:  # noqa: BLE001 - per-request; convert to ErrorDetails and return
            return ErrorDetails(
                type="VideoDownloadError",
                message=f"Failed to download video {job_id}: {e!r}",
                code=500,
            )

    async def _run_video_pipeline(
        self,
        request_info: RequestInfo,
        payload: dict[str, Any],
        headers: dict[str, str],
        responses: list[TextResponse | BinaryResponse],
    ) -> ErrorDetails | None:
        """Submit, poll, and optionally download video content.

        Appends produced responses to *responses* in place. Returns ErrorDetails on
        failure, or None on success.
        """
        submit_url = self.build_url(request_info)
        use_form_data = (
            self.run.cfg.endpoint.request_content_type
            == RequestContentType.MULTIPART_FORM_DATA
        )

        result = await self._submit_video_job(
            submit_url, payload, headers, use_form_data=use_form_data
        )
        if isinstance(result, ErrorDetails):
            return result
        job_id, submit_response = result
        responses.append(submit_response)

        poll_url = f"{submit_url.rstrip('/')}/{job_id}"
        poll_result = await self._poll_video_job(
            job_id,
            poll_url,
            headers,
            timeout=self.run.cfg.endpoint.timeout,
            poll_interval=Environment.HTTP.VIDEO_POLL_INTERVAL,
        )
        if isinstance(poll_result, ErrorDetails):
            return poll_result

        data, _ = poll_result
        responses.append(
            TextResponse(
                perf_ns=time.perf_counter_ns(),
                content_type="application/json",
                text=orjson.dumps(data).decode(),
            )
        )

        if self.run.cfg.endpoint.download_video_content:
            content_url = data.get("url") or f"{poll_url}/content"
            download_result = await self._download_video_content(
                job_id, content_url, headers
            )
            if isinstance(download_result, ErrorDetails):
                return download_result

        return None

    async def _send_video_request_with_polling(
        self,
        request_info: RequestInfo,
        payload: dict[str, Any],
    ) -> RequestRecord:
        """Send video generation request and poll until complete."""
        if self.http_client is None:
            raise NotInitializedError("HTTP client not initialized")

        start_ns = time.perf_counter_ns()
        headers = self.build_headers(request_info)
        responses: list[TextResponse | BinaryResponse] = []

        def make_record(
            error: ErrorDetails | None = None, status: int | None = None
        ) -> RequestRecord:
            return RequestRecord(
                request_info=request_info,
                request_headers=headers,
                start_perf_ns=start_ns,
                end_perf_ns=time.perf_counter_ns(),
                responses=responses,
                error=error,
                status=status,
            )

        try:
            error = await self._run_video_pipeline(
                request_info, payload, headers, responses
            )
            if error is not None:
                return make_record(error=error)
            return make_record(status=200)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - per-request; convert to ErrorDetails and return
            self.exception(f"Video generation failed: {e!r}")
            return make_record(error=ErrorDetails.from_exception(e))
