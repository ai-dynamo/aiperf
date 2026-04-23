# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Progress client for fetching job status from controller pods.

This module provides an async HTTP client that queries the controller pod's
health/progress API to retrieve real-time job execution status and download
benchmark result files.
"""

import asyncio
import dataclasses
import logging
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.parse import quote

import aiohttp
from pydantic import ValidationError

from aiperf.common.enums import CreditPhase, WorkerStartupState
from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats
from aiperf.common.models import WorkerStats
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.operator.k8s_helpers import retry_with_backoff
from aiperf.operator.progress_download import (
    make_decompressor,
    save_decompressed,
    save_transcoded_zstd,
    save_zstd_passthrough,
)
from aiperf.operator.progress_models import (
    BACKOFF_MULTIPLIER,
    INITIAL_BACKOFF_SEC,
    MAX_RETRIES,
    RETRYABLE_STATUS_CODES,
    ControllerAggregateWorkerStatus,
    JobProgress,
)
from aiperf.transports.aiohttp_client import create_tcp_connector

__all__ = [
    "BACKOFF_MULTIPLIER",
    "INITIAL_BACKOFF_SEC",
    "MAX_RETRIES",
    "RETRYABLE_STATUS_CODES",
    "ControllerAggregateWorkerStatus",
    "JobProgress",
    "ProgressClient",
]

logger = logging.getLogger(__name__)


class ProgressClient:
    """Async HTTP client for fetching job progress from controller pods.

    This client connects to the controller pod's HTTP API to retrieve
    real-time progress information during job execution. Includes retry
    logic with exponential backoff for transient failures.

    All public methods require the client to be entered as an async context
    manager; calling them outside of ``async with`` raises ``RuntimeError``.

    controller_host format:
        Every public method takes a ``controller_host`` string. In-cluster
        this is the fully qualified headless-service DNS name produced by
        :func:`aiperf.kubernetes.jobset.controller_dns_name`, of the form
        ``<jobset>-controller-0-0.<jobset>.<namespace>.svc.cluster.local``
        (the leaf ``controller-0-0`` is the single controller replica's
        JobSet pod DNS). When port-forwarding from outside the cluster,
        ``localhost`` or any reachable IP/hostname is also accepted.

    Example:
        >>> async with ProgressClient() as client:
        ...     progress = await client.get_progress(
        ...         "run-1234-controller-0-0.run-1234.aiperf.svc.cluster.local"
        ...     )
        ...     if stats := progress.profiling_stats:
        ...         print(f"Progress: {stats.requests_completed}/{stats.total_expected_requests}")
    """

    __slots__ = ("_port", "_session", "_max_retries", "_initial_backoff")

    PROGRESS_ENDPOINT = "/api/progress"
    WORKERS_ENDPOINT = "/api/workers"
    TIMEOUT_SECONDS = 10.0  # Increased for slow networks

    def __init__(
        self,
        port: int | None = None,
        max_retries: int = MAX_RETRIES,
        initial_backoff: float = INITIAL_BACKOFF_SEC,
    ) -> None:
        """Initialize the progress client.

        Args:
            port: The HTTP port on the controller pod. Defaults to
                  K8sEnvironment.PORTS.API_SERVICE (where progress endpoint is served).
            max_retries: Maximum number of retry attempts for transient failures.
            initial_backoff: Initial backoff duration in seconds.
        """
        self._port = port or K8sEnvironment.PORTS.API_SERVICE
        self._session: aiohttp.ClientSession | None = None
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff

    async def __aenter__(self) -> "ProgressClient":
        """Enter async context and create HTTP session."""
        connector = create_tcp_connector()
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.TIMEOUT_SECONDS),
            connector=connector,
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context and close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _request_with_retry(self, url: str) -> dict[str, Any] | None:
        """Make an HTTP request with exponential backoff retry on transient failures.

        Args:
            url: The URL to request.

        Returns:
            JSON response dict on success, None on persistent failure.

        Raises:
            aiohttp.ClientError: On non-retryable errors.
        """
        if self._session is None:
            raise RuntimeError(
                "ProgressClient._request_with_retry() called outside async context; "
                "wrap in 'async with ProgressClient(...) as pc:'"
            )

        async def _do_request() -> dict[str, Any]:
            assert self._session is not None  # noqa: S101
            async with self._session.get(url) as response:
                if response.status in RETRYABLE_STATUS_CODES:
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=response.status,
                        message=f"Retryable status {response.status}",
                    )
                response.raise_for_status()
                return await response.json()

        try:
            return await retry_with_backoff(
                _do_request,
                max_retries=self._max_retries,
                initial_delay=self._initial_backoff,
                backoff_multiplier=BACKOFF_MULTIPLIER,
                description=f"GET {url}",
            )
        except aiohttp.ClientResponseError as e:
            if e.status in RETRYABLE_STATUS_CODES:
                logger.warning(
                    f"Request to {url} failed after {self._max_retries + 1} "
                    f"attempts with status {e.status}"
                )
                return None
            raise

    async def get_progress(self, controller_host: str) -> JobProgress:
        """Fetch progress from the controller pod with retry logic.

        Args:
            controller_host: see class docstring.

        Returns:
            JobProgress with current execution status (per-phase
            :class:`CombinedPhaseStats` plus :class:`ControllerAggregateWorkerStatus`).
            On connection failure, returns an empty ``JobProgress`` with
            ``connection_error`` populated rather than raising.

        Raises:
            RuntimeError: If called outside ``async with ProgressClient() as c:``.
            aiohttp.ClientError: Non-transport errors from non-retryable HTTP
                responses (transport errors are caught and returned as
                ``connection_error`` on the result).

        Example:
            >>> async with ProgressClient() as c:
            ...     progress = await c.get_progress(
            ...         "run-1234-controller-0-0.run-1234.aiperf.svc.cluster.local"
            ...     )
            ...     if progress.connection_error:
            ...         print("controller not reachable yet")
            ...     elif progress.is_complete:
            ...         print("profiling done, safe to download results")
        """
        url = f"http://{controller_host}:{self._port}{self.PROGRESS_ENDPOINT}"

        try:
            data = await self._request_with_retry(url)
            if data is None:
                return JobProgress(
                    connection_error=f"Failed after {self._max_retries + 1} retries to {url}"
                )
            return self._parse_progress_response(data)
        except aiohttp.ClientError as e:
            # Return empty progress with detailed connection error for debugging.
            # Include URL to help diagnose DNS resolution vs connection issues.
            # Common cases: controller pod not ready, network issues, DNS not yet available.
            error_type = type(e).__name__
            error_msg = (
                f"{error_type} connecting to {controller_host}:{self._port} - {e}. "
                f"Check if controller pod is running and DNS is resolvable."
            )
            return JobProgress(connection_error=error_msg)

    def _parse_progress_response(self, data: dict[str, Any]) -> JobProgress:
        """Parse the progress API response into JobProgress.

        Args:
            data: Raw JSON response from the progress API.

        Returns:
            JobProgress with parsed phase stats.
        """
        phases: dict[CreditPhase, CombinedPhaseStats] = {}

        # Filter to declared dataclass fields so older/newer peers adding computed
        # or auxiliary keys don't break parsing (dataclass constructor rejects
        # unknown kwargs).
        valid_fields = {f.name for f in dataclasses.fields(CombinedPhaseStats)}
        for phase_name, phase_data in data.get("phases", {}).items():
            try:
                phase = CreditPhase(phase_name)
                filtered = {k: v for k, v in phase_data.items() if k in valid_fields}
                phases[phase] = CombinedPhaseStats(**filtered)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping malformed phase '{phase_name}': {e}")
                continue

        workers_data = data.get("workers", {})
        try:
            workers = ControllerAggregateWorkerStatus(**workers_data)
        except ValidationError as e:
            logger.warning(f"Falling back to default aggregate worker status: {e}")
            workers = ControllerAggregateWorkerStatus()

        return JobProgress(
            phases=phases,
            workers=workers,
            error=data.get("error"),
        )

    async def get_worker_startup_states(
        self, controller_host: str
    ) -> dict[str, WorkerStartupState] | None:
        """Fetch current worker startup states from the controller pod.

        Args:
            controller_host: see class docstring.

        Returns:
            Mapping of ``worker_id`` -> :class:`WorkerStartupState` for every
            worker whose startup state has been reported, or ``None`` when the
            endpoint is temporarily unreachable or returned a non-200
            retryable response.

        Raises:
            RuntimeError: If called outside ``async with ProgressClient() as c:``
                (propagated from :meth:`_request_with_retry`).
            aiohttp.ClientError: Non-transport errors from non-retryable HTTP
                responses (transport errors are logged and ``None`` is returned).

        Example:
            >>> async with ProgressClient() as c:
            ...     states = await c.get_worker_startup_states(
            ...         "run-1234-controller-0-0.run-1234.aiperf.svc.cluster.local"
            ...     )
            ...     ready = sum(1 for s in (states or {}).values() if s.is_ready)
        """
        url = f"http://{controller_host}:{self._port}{self.WORKERS_ENDPOINT}"

        try:
            data = await self._request_with_retry(url)
            if data is None:
                return None

            states: dict[str, WorkerStartupState] = {}
            for worker_id, worker_data in data.get("workers", {}).items():
                try:
                    worker = WorkerStats(**worker_data)
                except (TypeError, ValueError) as e:
                    logger.warning(
                        f"Skipping malformed worker payload for {worker_id}: {e}"
                    )
                    continue
                if worker.startup_state is not None:
                    states[worker_id] = worker.startup_state
            return states
        except aiohttp.ClientError as e:
            logger.warning(f"Failed to fetch worker startup states from {url}: {e}")
            return None

    async def check_health(self, controller_host: str) -> bool:
        """Check if the controller pod is healthy.

        Probes the ``/health`` endpoint served on the API service port.
        Does not retry — used for fast liveness polling.

        Args:
            controller_host: see class docstring.

        Returns:
            ``True`` if the controller responds with HTTP 200; ``False`` on
            any non-200 response or transport failure (errors are swallowed
            so callers can poll cheaply).

        Raises:
            RuntimeError: If called outside ``async with ProgressClient() as c:``.

        Example:
            >>> async with ProgressClient() as c:
            ...     while not await c.check_health(host):
            ...         await asyncio.sleep(1.0)
        """
        if self._session is None:
            raise RuntimeError(
                "ProgressClient.check_health() called outside async context; "
                "wrap in 'async with ProgressClient(...) as pc:'"
            )

        # API service exposes /health endpoint on the API_SERVICE port
        url = f"http://{controller_host}:{self._port}/health"

        try:
            async with self._session.get(url) as response:
                return response.status == 200
        except aiohttp.ClientError:
            return False

    async def get_metrics(self, controller_host: str) -> dict[str, Any] | None:
        """Fetch AIPerf benchmark metrics from the controller pod with retry logic.

        Queries ``/api/metrics`` which returns a JSON-serialized
        :class:`aiperf.api.routers.metrics.MetricsResponse`. The dict has
        these top-level keys:

        - ``aiperf_version`` (str): AIPerf version.
        - ``benchmark_id`` (str | None): Benchmark identifier.
        - ``model`` (str | None): Comma-separated model names.
        - ``endpoint_type`` (str | None): e.g. ``"chat"``.
        - ``streaming`` (bool | None): Streaming flag.
        - ``concurrency`` (int | None): Concurrency setting.
        - ``request_rate`` (float | None): Request rate setting.
        - ``metrics`` (dict[str, Any]): Real-time metric values keyed by tag
          (e.g. ``request_latency``, ``output_token_throughput``).

        Non-200 responses are returned as ``None`` rather than raising, so
        this method is safe to poll during controller startup.

        Args:
            controller_host: see class docstring.

        Returns:
            The metrics dict described above on HTTP 200, else ``None``.

        Raises:
            RuntimeError: If called outside ``async with ProgressClient() as c:``
                (propagated from :meth:`_request_with_retry`).

        Example:
            >>> async with ProgressClient() as c:
            ...     if (m := await c.get_metrics(host)) is not None:
            ...         print(m["metrics"].get("request_latency"))
        """
        url = f"http://{controller_host}:{self._port}/api/metrics"

        try:
            metrics = await self._request_with_retry(url)
            if metrics:
                logger.debug(f"Fetched metrics from {controller_host}")
            return metrics
        except aiohttp.ClientError as e:
            logger.warning(f"Failed to fetch metrics from {url}: {e}")
            return None

    async def get_server_metrics(self, controller_host: str) -> dict[str, Any] | None:
        """Fetch real-time inference-server metrics from the controller pod.

        Queries ``/api/server-metrics``, which mirrors the latest
        :class:`aiperf.common.messages.RealtimeServerMetricsMessage`
        received over the message bus. The dict typically contains:

        - ``endpoint_summaries`` (dict[str, dict]): Per-endpoint
          :class:`ServerMetricsEndpointSummary` payloads keyed by endpoint
          URL (queue depth, cache usage, latency, throughput).
        - ``message`` (str, optional): Present only when no server metrics
          have arrived yet (e.g. ``"No server metrics available yet"``), in
          which case ``endpoint_summaries`` is an empty dict.

        Non-200 responses are returned as ``None`` rather than raising.

        Args:
            controller_host: see class docstring.

        Returns:
            The server-metrics dict described above on HTTP 200, else ``None``.

        Raises:
            RuntimeError: If called outside ``async with ProgressClient() as c:``
                (propagated from :meth:`_request_with_retry`).

        Example:
            >>> async with ProgressClient() as c:
            ...     sm = await c.get_server_metrics(host)
            ...     for ep, summary in (sm or {}).get("endpoint_summaries", {}).items():
            ...         print(ep, summary)
        """
        url = f"http://{controller_host}:{self._port}/api/server-metrics"

        try:
            metrics = await self._request_with_retry(url)
            if metrics:
                logger.debug(f"Fetched server metrics from {controller_host}")
            return metrics
        except aiohttp.ClientError as e:
            logger.warning(f"Failed to fetch server metrics from {url}: {e}")
            return None

    async def send_shutdown(self, controller_host: str) -> bool:
        """Send shutdown signal to the controller pod's API service.

        POSTs to ``/api/shutdown``. The controller acknowledges with a
        2xx/3xx response and begins graceful shutdown asynchronously; this
        method does NOT wait for the pod to terminate.

        Args:
            controller_host: see class docstring.

        Returns:
            ``True`` if the controller returned HTTP < 400 (shutdown
            accepted), ``False`` on 4xx/5xx or transport failure.

        Raises:
            RuntimeError: If called outside ``async with ProgressClient() as c:``.

        Example:
            >>> async with ProgressClient() as c:
            ...     if not await c.send_shutdown(host):
            ...         log.warning("controller did not accept shutdown; falling back to pod delete")
        """
        if self._session is None:
            raise RuntimeError(
                "ProgressClient.send_shutdown() called outside async context; "
                "wrap in 'async with ProgressClient(...) as pc:'"
            )

        url = f"http://{controller_host}:{self._port}/api/shutdown"

        try:
            async with self._session.post(url) as response:
                logger.info(
                    f"Shutdown signal sent to {controller_host}: {response.status}"
                )
                return response.status < 400
        except aiohttp.ClientError as e:
            logger.warning(f"Failed to send shutdown to {url}: {e}")
            return False

    async def get_results_list(
        self, controller_host: str
    ) -> list[dict[str, Any]] | None:
        """Fetch list of available result files from the controller pod.

        Queries ``/api/results/list``; the ``files`` array of the
        :class:`ResultsListResponse` is returned directly.

        Args:
            controller_host: see class docstring.

        Returns:
            List of file-info dicts, each with at minimum ``name`` (str,
            relative path under the controller's results directory) and
            ``size`` (int, uncompressed byte size). Returns ``None`` on
            failure or when ``files`` is absent.

        Raises:
            RuntimeError: If called outside ``async with ProgressClient() as c:``
                (propagated from :meth:`_request_with_retry`).

        Example:
            >>> async with ProgressClient() as c:
            ...     for f in await c.get_results_list(host) or []:
            ...         print(f["name"], f["size"])
        """
        url = f"http://{controller_host}:{self._port}/api/results/list"

        try:
            data = await self._request_with_retry(url)
            if data:
                return data.get("files", [])
            return None
        except aiohttp.ClientError as e:
            logger.warning(f"Failed to fetch results list from {url}: {e}")
            return None

    def _build_result_file_url(self, controller_host: str, filename: str) -> str | None:
        """Validate filename and build the results-file URL.

        Returns None if the filename is unsafe (empty, ``.``, ``..`` segments).
        """
        # Allow nested subpaths (e.g. checkpoints/...) but reject any segment
        # that would escape the results dir on the server side.
        parts = [p for p in filename.replace("\\", "/").split("/") if p]
        if any(p in ("", ".", "..") for p in parts) or not parts:
            logger.warning(f"Refusing unsafe filename: {filename!r}")
            return None
        safe_path = "/".join(parts)
        return (
            f"http://{controller_host}:{self._port}"
            f"/api/results/files/{quote(safe_path, safe='/')}"
        )

    async def download_result_file(
        self,
        controller_host: str,
        filename: str,
        dest_path: Path,
    ) -> bool:
        """Download a result file from the controller pod with zstd compression.

        Requests ``Accept-Encoding: zstd, gzip, identity`` and streams the
        response body to disk. When
        :attr:`OperatorEnvironment.RESULTS.COMPRESS_ON_DISK` is enabled,
        zstd-encoded responses are persisted verbatim as ``<name>.zst``
        (zero-copy); otherwise the payload is decompressed to ``dest_path``.

        Path-traversal segments (``""``, ``"."``, ``".."``) are rejected
        client-side in addition to the server-side check.

        Args:
            controller_host: see class docstring.
            filename: Relative path under the results directory (e.g.
                ``"metrics.json"`` or ``"checkpoints/run0.parquet"``).
            dest_path: Destination file path. Parent dirs are created; the
                final name may be overridden by the server's ``X-Filename``
                header.

        Returns:
            ``True`` on successful download, ``False`` on HTTP 404, unsafe
            filename, or any transport error (partial files are removed).

        Raises:
            RuntimeError: If called outside ``async with ProgressClient() as c:``.

        Example:
            >>> async with ProgressClient() as c:
            ...     ok = await c.download_result_file(
            ...         host, "metrics.json", Path("/tmp/metrics.json")
            ...     )
        """
        if self._session is None:
            raise RuntimeError(
                "ProgressClient.download_result_file() called outside async context; "
                "wrap in 'async with ProgressClient(...) as pc:'"
            )

        url = self._build_result_file_url(controller_host, filename)
        if url is None:
            return False

        try:
            return await self._stream_result_file(url, filename, dest_path)
        except aiohttp.ClientError as e:
            logger.warning(f"Failed to download {filename}: {e}")
            # Remove partial file so retries don't consume corrupted data.
            # With COMPRESS_ON_DISK the writers in progress_download.save_*
            # land on ``dest_path.name + ".zst"``, not on ``dest_path``
            # itself, so clean both to avoid leaking a half-written .zst.
            dest_path.unlink(missing_ok=True)
            (dest_path.parent / (dest_path.name + ".zst")).unlink(missing_ok=True)
            return False

    async def _stream_result_file(
        self, url: str, filename: str, dest_path: Path
    ) -> bool:
        """Open a zstd-aware download session and stream the response to disk.

        Returns ``True`` on success, ``False`` on HTTP 404.
        """
        assert self._session is not None  # noqa: S101
        headers = {"Accept-Encoding": "zstd, gzip, identity"}
        # Disable auto_decompress: we handle zstd/gzip decompression manually
        # in _download_response(). aiohttp doesn't support zstd natively and
        # would reject the response with a 400 error.
        async with (
            aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300.0),
                connector=self._session.connector,
                connector_owner=False,
                auto_decompress=False,
            ) as dl_session,
            dl_session.get(url, headers=headers) as response,
        ):
            if response.status == 404:
                logger.debug(f"Result file not found: {filename}")
                return False
            response.raise_for_status()
            content_encoding = response.headers.get("Content-Encoding", "identity")
            dest_path = self._resolve_dest_path(response, dest_path)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            await self._download_response(response, dest_path, content_encoding)
            logger.info(f"Downloaded {filename} -> {dest_path}")
            return True

    @staticmethod
    def _resolve_dest_path(response: aiohttp.ClientResponse, dest_path: Path) -> Path:
        """Apply the server's ``X-Filename`` override to ``dest_path`` if present."""
        x_filename = response.headers.get("X-Filename")
        if not x_filename:
            return dest_path
        safe_name = Path(x_filename).name
        if not safe_name:
            return dest_path
        return dest_path.parent / safe_name

    async def _download_response(
        self,
        response: aiohttp.ClientResponse,
        dest_path: Path,
        content_encoding: str,
    ) -> None:
        """Download response to file, with optional on-disk zstd compression.

        When COMPRESS_ON_DISK is enabled:
        - zstd-encoded responses are saved directly as .zst (no decompression)
        - gzip/identity responses are decompressed then re-compressed as .zst
        When disabled, behaves as before (decompress to raw files).
        """
        from aiperf.operator.environment import OperatorEnvironment

        compress_on_disk = OperatorEnvironment.RESULTS.COMPRESS_ON_DISK

        if compress_on_disk and content_encoding == "zstd":
            # Save zstd bytes directly — zero processing cost
            await save_zstd_passthrough(response, dest_path)
            return

        decompressor = make_decompressor(content_encoding)
        if compress_on_disk:
            await save_transcoded_zstd(response, dest_path, decompressor)
        else:
            await save_decompressed(response, dest_path, decompressor)

    async def download_all_results(
        self,
        controller_host: str,
        dest_dir: Path,
        max_concurrent: int = 5,
    ) -> list[str]:
        """Download all available result files from the controller pod.

        Discovers files via :meth:`get_results_list`, then downloads them
        concurrently via :meth:`download_result_file` using a semaphore to
        limit parallelism. Exceptions from individual downloads are logged
        but do not abort the batch.

        Args:
            controller_host: see class docstring.
            dest_dir: Destination directory (created if missing). Each file's
                relative path from the server is preserved beneath it.
            max_concurrent: Maximum concurrent downloads. Defaults to 5.

        Returns:
            List of successfully downloaded filenames (server-side relative
            paths). Empty if the controller reports no files or is
            unreachable.

        Raises:
            RuntimeError: If called outside ``async with ProgressClient() as c:``
                (propagated from :meth:`download_result_file`).

        Example:
            >>> async with ProgressClient() as c:
            ...     got = await c.download_all_results(host, Path("./artifacts"))
            ...     print(f"downloaded {len(got)} files")
        """
        available = await self.get_results_list(controller_host)
        if not available:
            return []

        dest_dir.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _download_one(file_info: dict[str, Any]) -> str | None:
            filename = file_info["name"]
            dest_path = dest_dir / filename
            async with semaphore:
                if await self.download_result_file(
                    controller_host, filename, dest_path
                ):
                    return filename
            return None

        results = await asyncio.gather(
            *[_download_one(f) for f in available], return_exceptions=True
        )
        failed = [r for r in results if isinstance(r, BaseException)]
        if failed:
            logger.warning(f"{len(failed)}/{len(available)} file downloads failed")
        return [r for r in results if isinstance(r, str)]
