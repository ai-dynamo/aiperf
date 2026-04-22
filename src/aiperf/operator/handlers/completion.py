# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Completion handling and result fetching for AIPerfJob."""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import kopf
import kr8s

from aiperf.kubernetes.client import get_api
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.jobset import controller_dns_name
from aiperf.kubernetes.kr8s_resources import AsyncJobSet
from aiperf.kubernetes.results_sidecar import CHECKPOINTS_DIR_NAME
from aiperf.operator import events
from aiperf.operator.client_cache import (
    get_or_create_progress_client,
    is_cancellation_requested,
    job_key,
)
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.job_index import index_job_completed
from aiperf.operator.k8s_helpers import retry_with_backoff
from aiperf.operator.models import FetchResult, MetricsSummary
from aiperf.operator.progress_client import ProgressClient
from aiperf.operator.status import ConditionType, Phase, StatusBuilder, parse_timestamp

logger = logging.getLogger(__name__)


async def handle_completion(
    body: dict[str, Any],
    namespace: str,
    jobset_name: str,
    job_id: str,
    status: dict[str, Any],
    sb: StatusBuilder,
    result: FetchResult | None = None,
) -> None:
    """Handle job completion: fetch results and update status."""
    # Short-circuit if on_delete has signaled cancellation. The CR is
    # about to disappear; skipping fetch/JobSet-delete/status patches
    # keeps the delete from blocking on retry backoff.
    if is_cancellation_requested(job_key(namespace, job_id)):
        logger.info(
            f"Cancellation requested for {namespace}/{job_id}, "
            "skipping completion handling"
        )
        return

    # Backfill conditions for fast-completing jobs that skipped RUNNING phase
    total_workers = status.get("workers", {}).get("total", 1)
    if not sb.conditions.is_condition_true(ConditionType.WORKERS_READY):
        sb.conditions.set_true(
            ConditionType.WORKERS_READY,
            "CompletedBeforeMonitor",
            f"Job completed before workers ({total_workers}) were observed ready",
        )
    if not sb.conditions.is_condition_true(ConditionType.BENCHMARK_RUNNING):
        sb.conditions.set_true(
            ConditionType.BENCHMARK_RUNNING,
            "CompletedBeforeMonitor",
            "Job completed before running state was observed",
        )

    sb.set_completion_time()

    # Calculate duration
    start_time = status.get("startTime")
    duration_sec = None
    if start_time:
        try:
            start_dt = parse_timestamp(start_time)
            duration_sec = (datetime.now(timezone.utc) - start_dt).total_seconds()
        except (ValueError, TypeError):
            pass

    # Fetch results with retry
    host = controller_dns_name(jobset_name, namespace)
    if result is None:
        result = await fetch_results_with_retry(host, namespace, job_id)

    has_metrics = bool(result.metrics and result.metrics.get("metrics"))
    key_result_files = {"profile_export_aiperf.json", "profile_export_aiperf.csv"}
    has_files = bool(key_result_files & set(result.downloaded or []))
    # A partial fetch can set has_files=True but still populate result.error
    # (e.g. checkpoints saved but key export files missing). Treat error as
    # authoritative so a false-success Completed phase never overwrites the
    # real failure signal.
    has_error = bool(result.error)
    success = has_files and not has_error

    logger.info(
        f"Results for {job_id}: has_metrics={has_metrics}, has_files={has_files}, "
        f"metrics_keys={list(result.metrics.keys()) if result.metrics else []}"
    )

    if has_metrics:
        sb.set_results(result.metrics)

        summary = MetricsSummary.from_metrics(result.metrics)
        summary_dict = summary.to_status_dict()
        if summary_dict:
            sb.set_summary(summary_dict)
    elif has_files:
        # API metrics empty/unavailable but files downloaded.
        # Parse metrics from the JSON export file and store in CR.
        file_metrics = _parse_metrics_from_files(result.downloaded, namespace, job_id)
        if file_metrics:
            sb.set_results(file_metrics)
            logger.info(f"Parsed metrics from result files for {job_id}")

    if has_files:
        dest_dir = OperatorEnvironment.RESULTS.DIR / namespace / job_id
        sb.set_results_path(str(dest_dir))
        events.results_stored(body, str(dest_dir), len(result.downloaded))
        logger.info(f"Downloaded {len(result.downloaded)} result files to {dest_dir}")

    # Set condition based on what was actually retrieved.
    # Result files are the authoritative source - /api/metrics is a convenience
    # that duplicates what's derivable from the files. Files alone = full success,
    # but only if FetchResult.error is empty: a partial fetch can set has_files
    # while still reporting an error for missing key artifacts.
    if success:
        reason = "ResultsStored"
        if has_metrics:
            msg = f"Metrics and {len(result.downloaded)} result files stored"
        else:
            msg = f"{len(result.downloaded)} result files stored"
            logger.info(
                f"Metrics fetch skipped/failed for {jobset_name} - "
                f"result files are sufficient"
            )
        sb.set_phase(Phase.COMPLETED)
        sb.conditions.set_true(ConditionType.RESULTS_AVAILABLE, reason, msg)
    else:
        sb.set_phase(Phase.FAILED)
        failure_reason = "ResultsFetchFailed"
        failure_msg = (
            result.error
            if has_error
            else "Failed to fetch complete result files from controller"
        )
        sb.conditions.set_false(
            ConditionType.RESULTS_AVAILABLE,
            failure_reason,
            failure_msg,
        )
        if has_files and has_error:
            logger.warning(
                f"Partial results for {jobset_name}: key files present but "
                f"fetch reported error: {result.error}"
            )
        elif has_metrics:
            logger.warning(
                f"Metrics were fetched for {jobset_name}, but complete result files were not available"
            )
        else:
            logger.warning(f"No result files downloaded for {jobset_name}")
        events.results_failed(body, failure_msg)

    sb.finalize()
    if success:
        events.completed(body, job_id, duration_sec)

    # Update job index with completion data. Results are already persisted
    # to disk, so a failure here only affects discoverability via the
    # index/history API - don't retry the whole completion handler, but
    # set a status condition and event so operators can see the gap.
    try:
        await index_job_completed(
            namespace=namespace,
            job_id=job_id,
            phase="Completed" if success else "Failed",
            metrics=result.metrics,
            downloaded_files=result.downloaded,
        )
    except Exception as e:
        logger.exception(f"Failed to update job index for {job_id}")
        sb.conditions.set_false(
            ConditionType.INDEX_UPDATED,
            "IndexUpdateFailed",
            f"Index write failed: {e}",
        )
        sb.finalize()
        kopf.event(
            body,
            type="Warning",
            reason="IndexUpdateFailed",
            message=f"Job index update failed (results still on disk): {e}",
        )

    # Delete the JobSet to free cluster resources after complete result files are stored.
    # Keep pods alive for retry on the next monitor tick if fetch failed or only
    # partial/non-authoritative artifacts were available.
    # Skip the delete entirely if cancellation was requested — K8s GC via
    # ownerReferences will reap the JobSet once the CR is gone.
    if success and not is_cancellation_requested(job_key(namespace, job_id)):
        try:
            api = await get_api()
            js = await AsyncJobSet.get(jobset_name, namespace=namespace, api=api)
            await js.delete()
            logger.info(f"Deleted JobSet {jobset_name} after results stored")
        except kr8s.NotFoundError:
            pass
        except kr8s.ServerError as e:
            logger.warning(
                f"Failed to delete JobSet {jobset_name} after completion: {e}"
            )


# Progress-aware retry: give up after this many CONSECUTIVE attempts with
# no new bytes/files arriving in the results dir. Tuning rationale below.
_NO_PROGRESS_STAGNATION_LIMIT = 5


async def _fetch_with_progress_aware_retry(
    fetch_once: Callable[[], Awaitable[FetchResult]],
    *,
    dest_dir: Path,
    job_id: str,
    initial_delay: float,
    description: str,
    is_cancelled: Callable[[], bool] | None = None,
    max_delay: float = 30.0,
    backoff_multiplier: float = 2.0,
    stagnation_limit: int = _NO_PROGRESS_STAGNATION_LIMIT,
) -> FetchResult:
    """Retry ``fetch_once`` until it returns, or until no progress has been
    made for ``stagnation_limit`` consecutive attempts.

    Unlike a plain count-based retry (``retry_with_backoff``), this tolerates
    controllers that take minutes to finalise their export. Records-manager
    is single-core bound on pure-Python aggregation; at high concurrency its
    final ``summarize()`` can run longer than any fixed retry window. As long
    as the partial checkpoint on disk is still growing or new files are
    landing, we know the controller is still working and should keep waiting.
    Only if nothing changes across ``stagnation_limit`` consecutive attempts
    (wall-clock ~60s at default cap) do we give up.

    Progress signal: total bytes of all files directly under ``dest_dir``
    and its subtree. A file that grows counts as progress; a file that
    shrinks or disappears does not.

    ``is_cancelled``, if supplied, is polled between attempts and at each
    sleep wakeup; returning True aborts the loop promptly. This avoids
    blocking a CR deletion behind a full stagnation window.
    """

    def _snapshot_bytes() -> int:
        total = 0
        if not dest_dir.exists():
            return 0
        for entry in dest_dir.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    # File vanished between rglob and stat — skip silently.
                    continue
        return total

    delay = initial_delay
    attempt = 0
    no_progress_streak = 0
    last_bytes = _snapshot_bytes()
    pending_exc: BaseException | None = None
    while True:
        if is_cancelled is not None and is_cancelled():
            return await fetch_once()
        attempt += 1
        try:
            return await fetch_once()
        except _IncompleteResultsError as e:
            pending_exc = e
        except Exception as e:
            # Transient errors (network, IO) count as no-progress attempts
            # too; if they persist past stagnation_limit we bubble them up.
            pending_exc = e
            logger.debug(
                "%s attempt %d raised %s; treating as no-progress",
                description,
                attempt,
                type(e).__name__,
            )

        if is_cancelled is not None and is_cancelled():
            return await fetch_once()

        bytes_now = _snapshot_bytes()
        if bytes_now > last_bytes:
            # Controller is still producing output — reset stagnation.
            no_progress_streak = 0
            logger.debug(
                "%s attempt %d progressing: %d -> %d bytes on disk; retrying in %.1fs",
                description,
                attempt,
                last_bytes,
                bytes_now,
                delay,
            )
        else:
            no_progress_streak += 1
            logger.debug(
                "%s attempt %d no new bytes (%d) — stagnation %d/%d",
                description,
                attempt,
                bytes_now,
                no_progress_streak,
                stagnation_limit,
            )
            if no_progress_streak >= stagnation_limit:
                logger.warning(
                    "%s stalled at %d bytes for %d consecutive attempts — giving up",
                    description,
                    bytes_now,
                    no_progress_streak,
                )
                assert pending_exc is not None
                raise pending_exc
        last_bytes = bytes_now
        jittered_delay = delay * random.uniform(0.8, 1.2)
        await asyncio.sleep(jittered_delay)
        delay = min(delay * backoff_multiplier, max_delay)


async def fetch_results_with_retry(
    controller_host: str,
    namespace: str,
    job_id: str,
    max_retries: int = OperatorEnvironment.RESULTS.MAX_RETRIES,
    retry_delay: float = OperatorEnvironment.RESULTS.RETRY_DELAY,
    dest_dir: Path | None = None,
) -> FetchResult:
    """Fetch results from controller pod with retry logic.

    Uses the cached ProgressClient for the job. Falls back to creating
    a temporary client if no cached one exists (e.g. after restart).
    Args:
        controller_host: Controller pod DNS name.
        namespace: Kubernetes namespace (used for results directory scoping).
        job_id: Job identifier for results directory.
        max_retries: Maximum retry attempts.
        retry_delay: Delay between retries (with exponential backoff).
        dest_dir: Explicit destination directory for results. When None,
            defaults to ``RESULTS.DIR / namespace / job_id``.

    Returns:
        FetchResult with metrics dict and list of downloaded files.
    """
    for label, value in [("namespace", namespace), ("job_id", job_id)]:
        if not value or value in (".", ".."):
            logger.error(f"Invalid {label} for results storage: {value!r}")
            return FetchResult(
                metrics=None, downloaded=[], error=f"Invalid {label}: {value!r}"
            )
        try:
            safe = (OperatorEnvironment.RESULTS.DIR / value).resolve()
            safe.relative_to(OperatorEnvironment.RESULTS.DIR.resolve())
        except (ValueError, OSError):
            logger.error(f"Path traversal detected in {label}: {value!r}")
            return FetchResult(
                metrics=None, downloaded=[], error=f"Path traversal in {label}"
            )

    key = job_key(namespace, job_id)
    client = await get_or_create_progress_client(key)

    if dest_dir is None:
        dest_dir = OperatorEnvironment.RESULTS.DIR / namespace / job_id

    # Mutable state shared across retry attempts so partial progress
    # (e.g. metrics fetched but files not yet) survives retries.
    # Use None (not yet attempted) vs [] (attempted, no files) to avoid
    # treating a valid empty download list as "not yet fetched".
    state: dict[str, Any] = {"metrics": None, "downloaded": None, "checkpoints": None}

    # Key result files that indicate a complete export. If downloads
    # succeed but none of these are present, export is still in progress
    # and we should retry to capture the full set.
    _KEY_FILES = {"profile_export_aiperf.json", "profile_export_aiperf.csv"}
    sidecar_port = K8sEnvironment.PORTS.RESULTS_SIDECAR

    def _split_downloaded(
        paths: list[str] | None,
    ) -> tuple[list[str], list[str]]:
        final_files: list[str] = []
        checkpoint_files: list[str] = []
        for path in paths or []:
            if path.startswith(f"{CHECKPOINTS_DIR_NAME}/"):
                checkpoint_files.append(path)
            else:
                final_files.append(path)
        return final_files, checkpoint_files

    def _merge_downloaded(
        current: list[str] | None, new: list[str] | None
    ) -> list[str] | None:
        if not new:
            return current
        if not current:
            return list(new)
        return sorted(set(current) | set(new))

    async def _fetch_once() -> FetchResult:
        # Short-circuit on cancellation so the outer retry_with_backoff
        # stops immediately (returning this FetchResult is success from
        # the retry wrapper's perspective, so no further attempts fire).
        if is_cancellation_requested(key):
            return FetchResult(
                metrics=state["metrics"],
                downloaded=state["downloaded"] or [],
                checkpoints=state["checkpoints"] or [],
                error="Cancelled by CR deletion",
            )

        if state["metrics"] is None:
            state["metrics"] = await client.get_metrics(controller_host)

        if OperatorEnvironment.RESULTS.DIR.exists():
            downloaded = await client.download_all_results(controller_host, dest_dir)
            if downloaded:
                final_files, checkpoint_files = _split_downloaded(downloaded)
                state["downloaded"] = _merge_downloaded(
                    state["downloaded"], final_files
                )
                state["checkpoints"] = _merge_downloaded(
                    state["checkpoints"], checkpoint_files
                )

            has_key_file = bool(_KEY_FILES & set(state["downloaded"] or []))
            if not has_key_file and sidecar_port != K8sEnvironment.PORTS.API_SERVICE:
                async with ProgressClient(port=sidecar_port) as sidecar_client:
                    sidecar_downloaded = await sidecar_client.download_all_results(
                        controller_host, dest_dir
                    )
                if sidecar_downloaded:
                    final_files, checkpoint_files = _split_downloaded(
                        sidecar_downloaded
                    )
                    state["downloaded"] = _merge_downloaded(
                        state["downloaded"], final_files
                    )
                    state["checkpoints"] = _merge_downloaded(
                        state["checkpoints"], checkpoint_files
                    )

        if state["metrics"] is not None and state["downloaded"] is not None:
            has_key_file = bool(_KEY_FILES & set(state["downloaded"]))
            if has_key_file:
                return FetchResult(
                    metrics=state["metrics"],
                    downloaded=state["downloaded"],
                    checkpoints=state["checkpoints"] or [],
                )
            logger.info(
                f"Downloaded {len(state['downloaded'])} files but missing key "
                f"export files, retrying..."
            )
        raise _IncompleteResultsError(
            state["metrics"],
            state["downloaded"] or [],
            state["checkpoints"] or [],
        )

    try:
        return await _fetch_with_progress_aware_retry(
            _fetch_once,
            dest_dir=dest_dir,
            job_id=job_id,
            initial_delay=retry_delay,
            description=f"results fetch for {job_id}",
            is_cancelled=lambda: is_cancellation_requested(key),
            stagnation_limit=max(max_retries, 1),
        )
    except _IncompleteResultsError as e:
        return e.to_fetch_result(job_id)
    except Exception as e:
        logger.warning(f"Results fetch failed for {job_id}: {e}")
        return FetchResult(
            metrics=state["metrics"],
            downloaded=state["downloaded"] or [],
            checkpoints=state["checkpoints"] or [],
            error=f"Failed to fetch results: {e}",
        )


class _IncompleteResultsError(Exception):
    """Raised when results are partially fetched (metrics or files missing)."""

    def __init__(
        self,
        metrics: dict[str, Any] | None,
        downloaded: list[str],
        checkpoints: list[str],
    ) -> None:
        self.metrics = metrics
        self.downloaded = downloaded
        self.checkpoints = checkpoints
        super().__init__("Incomplete results")

    def to_fetch_result(self, job_id: str) -> FetchResult:
        """Convert to a FetchResult with appropriate error message."""
        error = ""
        if not self.metrics and not self.downloaded:
            error = "Failed to fetch results"
            logger.warning(f"No metrics or files retrieved for {job_id}")
        elif not self.metrics:
            error = "Failed to fetch metrics (files downloaded)"
            logger.warning(
                f"Metrics fetch failed for {job_id}, "
                f"files downloaded: {len(self.downloaded)}"
            )
        elif not self.downloaded:
            error = "Failed to download result files (metrics fetched)"
            logger.warning(f"File download failed for {job_id}, metrics retrieved")

        return FetchResult(
            metrics=self.metrics,
            downloaded=self.downloaded,
            checkpoints=self.checkpoints,
            error=error,
        )


def _parse_metrics_from_files(
    downloaded: list[str],
    namespace: str,
    job_id: str,
) -> dict[str, Any] | None:
    """Parse metrics from downloaded result files.

    Looks for profile_export_aiperf.json (or .json.zst) which contains the
    full benchmark results in a format compatible with the CR status.
    """
    import orjson

    dest_dir = OperatorEnvironment.RESULTS.DIR / namespace / job_id

    try:
        candidate_paths: list[Path] = []
        for name in downloaded:
            candidate = dest_dir / name
            candidate_paths.append(candidate)
        candidate_paths.extend(
            [
                dest_dir / "profile_export_aiperf.json.zst",
                dest_dir / "profile_export_aiperf.json",
            ]
        )
        candidate_paths.sort(key=lambda p: 0 if p.suffix == ".zst" else 1)

        seen: set[Path] = set()
        for path in candidate_paths:
            if path in seen or not path.exists():
                continue
            seen.add(path)

            if path.suffix == ".zst":
                import io

                import zstandard

                raw = (
                    zstandard.ZstdDecompressor()
                    .stream_reader(io.BytesIO(path.read_bytes()))
                    .read()
                )
                data = orjson.loads(raw)
            else:
                data = orjson.loads(path.read_bytes())

            if not isinstance(data, dict):
                continue
            # Newer exports wrap metrics under "metrics"; older ones put them
            # at the top level. Accept either shape and return a dict that
            # has a populated "metrics" key so downstream readers always see
            # the same structure in CR status.
            if isinstance(data.get("metrics"), dict) and data["metrics"]:
                return data
            if data.get("request_throughput"):
                return {
                    "metrics": data,
                    **{k: v for k, v in data.items() if k != "metrics"},
                }
    except Exception as e:
        logger.warning(f"Failed to parse metrics from {dest_dir}: {e}")
    return None
