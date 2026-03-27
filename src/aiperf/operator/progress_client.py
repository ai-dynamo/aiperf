# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Progress client for fetching job status from controller pods.

This module provides an async HTTP client that queries the controller pod's
health/progress API to retrieve real-time job execution status and download
benchmark result files.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

import aiofiles
import aiohttp
import zstandard as zstd
from pydantic import Field

from aiperf.common.enums import CreditPhase, WorkerStartupState
from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats
from aiperf.common.models import AIPerfBaseModel, WorkerStats
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.operator.k8s_helpers import retry_with_backoff
from aiperf.transports.aiohttp_client import create_tcp_connector

logger = logging.getLogger(__name__)

# Retry configuration for transient failures
MAX_RETRIES = 3
INITIAL_BACKOFF_SEC = 0.5
BACKOFF_MULTIPLIER = 2.0
# HTTP status codes that are retryable (transient failures)
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


class JobProgress(AIPerfBaseModel):
    """Aggregated progress across all benchmark phases.

    This model wraps phase-specific progress stats (CombinedPhaseStats) for
    each benchmark phase (warmup, profiling), providing a complete view of
    job execution status.

    Attributes:
        phases: Progress stats for each phase (warmup, profiling).
        error: Error message if the job failed.
        connection_error: Connection error message if API request failed.
    """

    phases: dict[CreditPhase, CombinedPhaseStats] = Field(
        default_factory=dict,
        description="Progress stats for each benchmark phase",
    )
    error: str | None = Field(
        default=None,
        description="Error message if job failed",
    )
    connection_error: str | None = Field(
        default=None,
        description="Connection error if progress API was unreachable",
    )

    @property
    def current_phase(self) -> CreditPhase | None:
        """Get the most recently started phase."""
        if not self.phases:
            return None
        return max(
            self.phases.items(),
            key=lambda x: x[1].start_ns or 0,
        )[0]

    @property
    def is_complete(self) -> bool:
        """Check if the profiling phase has fully completed (requests sent AND records processed)."""
        profiling = self.phases.get("profiling")
        if profiling is None:
            return False
        if not profiling.is_requests_complete:
            return False
        # Wait for records to finish processing too — the controller won't
        # export results until all records are received.
        return profiling.is_records_complete

    @property
    def profiling_stats(self) -> CombinedPhaseStats | None:
        """Get the profiling phase stats (primary benchmark phase)."""
        return self.phases.get("profiling")

    @property
    def warmup_stats(self) -> CombinedPhaseStats | None:
        """Get the warmup phase stats."""
        return self.phases.get("warmup")


class ProgressClient:
    """Async HTTP client for fetching job progress from controller pods.

    This client connects to the controller pod's HTTP API to retrieve
    real-time progress information during job execution. Includes retry
    logic with exponential backoff for transient failures.

    Example:
        >>> async with ProgressClient() as client:
        ...     progress = await client.get_progress("controller-0-0.ns.svc.cluster.local")
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

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
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
            raise RuntimeError("ProgressClient must be used as async context manager")

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
            controller_host: The DNS name or IP of the controller pod.

        Returns:
            JobProgress with current execution status. Returns empty JobProgress
            with connection_error set on connection failure.
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

        for phase_name, phase_data in data.get("phases", {}).items():
            try:
                phase = CreditPhase(phase_name)
                phases[phase] = CombinedPhaseStats(**phase_data)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping malformed phase '{phase_name}': {e}")
                continue

        return JobProgress(
            phases=phases,
            error=data.get("error"),
        )

    async def get_worker_startup_states(
        self, controller_host: str
    ) -> dict[str, WorkerStartupState] | None:
        """Fetch current worker startup states from the controller pod.

        Args:
            controller_host: The DNS name or IP of the controller pod.

        Returns:
            Mapping of worker_id -> startup state, or None when the endpoint is
            temporarily unreachable.
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

        Args:
            controller_host: The DNS name or IP of the controller pod.

        Returns:
            True if the controller responds to health checks.
        """
        if self._session is None:
            raise RuntimeError("ProgressClient must be used as async context manager")

        # API service exposes /health endpoint on the API_SERVICE port
        url = f"http://{controller_host}:{self._port}/health"

        try:
            async with self._session.get(url) as response:
                return response.status == 200
        except aiohttp.ClientError:
            return False

    async def get_metrics(self, controller_host: str) -> dict[str, Any] | None:
        """Fetch metrics from the controller pod with retry logic.

        Args:
            controller_host: The DNS name or IP of the controller pod.

        Returns:
            Metrics dict from /api/metrics endpoint, or None on failure.
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
        """Fetch server metrics from the controller pod with retry logic.

        Args:
            controller_host: The DNS name or IP of the controller pod.

        Returns:
            Server metrics dict from /api/server-metrics endpoint, or None on failure.
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

        Args:
            controller_host: The DNS name or IP of the controller pod.

        Returns:
            True if shutdown was accepted, False otherwise.
        """
        if self._session is None:
            raise RuntimeError("ProgressClient must be used as async context manager")

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

        Args:
            controller_host: The DNS name or IP of the controller pod.

        Returns:
            List of file info dicts with 'name' and 'size' keys, or None on failure.
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

    async def download_result_file(
        self,
        controller_host: str,
        filename: str,
        dest_path: Path,
    ) -> bool:
        """Download a result file from the controller pod with zstd compression.

        Args:
            controller_host: The DNS name or IP of the controller pod.
            filename: The result filename (e.g., 'metrics.json').
            dest_path: Destination path to save the file.

        Returns:
            True if download succeeded, False otherwise.
        """
        if self._session is None:
            raise RuntimeError("ProgressClient must be used as async context manager")

        url = f"http://{controller_host}:{self._port}/api/results/files/{filename}"
        headers = {"Accept-Encoding": "zstd, gzip, identity"}

        try:
            timeout = aiohttp.ClientTimeout(total=300.0)
            # Disable auto_decompress: we handle zstd/gzip decompression manually
            # in _download_response(). aiohttp doesn't support zstd natively and
            # would reject the response with a 400 error.
            connector = self._session.connector
            async with (
                aiohttp.ClientSession(
                    timeout=timeout,
                    connector=connector,
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
                x_filename = response.headers.get("X-Filename")
                if x_filename:
                    safe_name = Path(x_filename).name
                    if safe_name:
                        dest_path = dest_path.parent / safe_name

                dest_path.parent.mkdir(parents=True, exist_ok=True)
                await self._download_response(response, dest_path, content_encoding)

                logger.info(f"Downloaded {filename} -> {dest_path}")
                return True

        except aiohttp.ClientError as e:
            logger.warning(f"Failed to download {filename}: {e}")
            # Remove partial file so retries don't consume corrupted data
            if dest_path.exists():
                dest_path.unlink(missing_ok=True)
            return False

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
        import zlib

        from aiperf.operator.environment import OperatorEnvironment

        compress_on_disk = OperatorEnvironment.RESULTS.COMPRESS_ON_DISK

        if compress_on_disk and content_encoding == "zstd":
            # Save zstd bytes directly — zero processing cost
            zst_path = dest_path.parent / (dest_path.name + ".zst")
            async with aiofiles.open(zst_path, "wb") as f:
                async for chunk in response.content.iter_chunked(64 * 1024):
                    if chunk:
                        await f.write(chunk)
            return

        # Decompress the response
        if content_encoding == "zstd":
            dctx = zstd.ZstdDecompressor()
            decompressor = dctx.decompressobj()
        elif content_encoding == "gzip":
            decompressor = zlib.decompressobj(wbits=31)
        else:
            decompressor = None

        if compress_on_disk:
            # Decompress from wire encoding, re-compress as zstd for storage
            import zstandard

            zst_path = dest_path.parent / (dest_path.name + ".zst")
            cctx = zstandard.ZstdCompressor(level=3)
            compressor = cctx.compressobj()

            async with aiofiles.open(zst_path, "wb") as f:
                async for chunk in response.content.iter_chunked(64 * 1024):
                    if decompressor is not None:
                        chunk = decompressor.decompress(chunk)
                    if chunk:
                        compressed = compressor.compress(chunk)
                        if compressed:
                            await f.write(compressed)
                if decompressor is not None:
                    remaining = decompressor.flush()
                    if remaining:
                        compressed = compressor.compress(remaining)
                        if compressed:
                            await f.write(compressed)
                final = compressor.flush()
                if final:
                    await f.write(final)
        else:
            # Original behavior: decompress and save raw
            async with aiofiles.open(dest_path, "wb") as f:
                async for chunk in response.content.iter_chunked(64 * 1024):
                    if decompressor is not None:
                        chunk = decompressor.decompress(chunk)
                    if chunk:
                        await f.write(chunk)
                if decompressor is not None:
                    remaining = decompressor.flush()
                    if remaining:
                        await f.write(remaining)

    async def download_all_results(
        self,
        controller_host: str,
        dest_dir: Path,
        max_concurrent: int = 5,
    ) -> list[str]:
        """Download all available result files from the controller pod.

        Discovers files via /api/results/list, then downloads them concurrently
        using a semaphore to limit parallelism.

        Args:
            controller_host: The DNS name or IP of the controller pod.
            dest_dir: Destination directory to save files.
            max_concurrent: Maximum concurrent downloads.

        Returns:
            List of successfully downloaded filenames.
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
