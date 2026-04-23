# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Completion handling and result fetching for AIPerfJob."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import kopf
import orjson
import zstandard
from kubernetes_asyncio import client
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.client import k8s_client
from aiperf.kubernetes.cr_refs import JOBSET_GROUP, JOBSET_PLURAL, JOBSET_VERSION
from aiperf.kubernetes.jobset import controller_dns_name
from aiperf.operator import events
from aiperf.operator.client_cache import (
    get_or_create_progress_client,
    is_cancellation_requested,
    job_key,
)
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.handlers._completion_fetch import (
    _NO_PROGRESS_STAGNATION_LIMIT,  # re-exported for tests
    _fetch_with_progress_aware_retry,  # re-exported for tests
    _IncompleteResultsError,  # re-exported for tests/monitor
    fetch_results_with_retry,
)
from aiperf.operator.job_index import index_job_completed
from aiperf.operator.models import ControllerFetchResult, MetricsSummary
from aiperf.operator.progress_client import ProgressClient  # re-exported for tests
from aiperf.operator.status import ConditionType, Phase, StatusBuilder, parse_timestamp

__all__ = [
    "ProgressClient",
    "_IncompleteResultsError",
    "_NO_PROGRESS_STAGNATION_LIMIT",
    "_fetch_with_progress_aware_retry",
    "_parse_metrics_from_files",
    "fetch_results_with_retry",
    "get_or_create_progress_client",
    "handle_completion",
]

logger = logging.getLogger(__name__)

_KEY_RESULT_FILES = frozenset(
    {"profile_export_aiperf.json", "profile_export_aiperf.csv"}
)


async def handle_completion(
    body: dict[str, Any],
    namespace: str,
    jobset_name: str,
    job_id: str,
    *,
    status: dict[str, Any],
    sb: StatusBuilder,
    result: ControllerFetchResult | None = None,
) -> None:
    """Finalize a completed AIPerfJob: fetch results, patch status, update index.

    Precondition: caller MUST hold the completion claim via
    ``try_claim_completion``; without it this double-fetches and double-patches.

    Side effects: fetches results (unless ``result`` is supplied), writes
    phase/results/summary/resultsPath + conditions on ``sb``, updates the
    job index (degrading to a condition + event on failure), emits
    ResultsStored/ResultsFailed/Completed kopf events, and deletes the
    backing JobSet on success. Short-circuits with no side effects if
    ``on_delete`` has already requested cancellation for this job. The
    ``result`` kwarg lets the salvage path skip the HTTP round-trip.
    """
    # Short-circuit if on_delete has signaled cancellation. The CR is
    # about to disappear; skipping fetch/JobSet-delete/status patches
    # keeps the delete from blocking on retry backoff.
    if is_cancellation_requested(job_key(namespace, job_id)):
        logger.info(
            f"Cancellation requested for {namespace}/{job_id}, "
            "skipping completion handling"
        )
        return

    _backfill_pre_completion_conditions(status, sb)
    sb.set_completion_time()
    duration_sec = _compute_duration_seconds(status)

    if result is None:
        host = controller_dns_name(jobset_name, namespace)
        result = await fetch_results_with_retry(host, namespace, job_id)

    flags = _compute_result_flags(result, job_id)

    _record_results_on_status(
        body=body,
        namespace=namespace,
        job_id=job_id,
        result=result,
        sb=sb,
        has_metrics=flags.has_metrics,
        has_files=flags.has_files,
    )
    _set_results_phase_and_condition(
        body=body,
        jobset_name=jobset_name,
        result=result,
        sb=sb,
        has_metrics=flags.has_metrics,
        has_files=flags.has_files,
        has_error=flags.has_error,
        success=flags.success,
    )

    sb.finalize()
    if flags.success:
        events.completed(body, job_id, duration_sec)

    await _update_job_index_safe(
        body=body,
        namespace=namespace,
        job_id=job_id,
        result=result,
        sb=sb,
        success=flags.success,
    )

    await _maybe_delete_jobset_after_success(namespace, jobset_name, job_id, flags)


async def _maybe_delete_jobset_after_success(
    namespace: str, jobset_name: str, job_id: str, flags: _ResultFlags
) -> None:
    """Delete the backing JobSet to free cluster resources once results are stored.

    Keep pods alive for retry on the next monitor tick if fetch failed or only
    partial/non-authoritative artifacts were available. Skip the delete on
    cancellation — K8s GC via ownerReferences will reap the JobSet.
    """
    if flags.success and not is_cancellation_requested(job_key(namespace, job_id)):
        await _delete_backing_jobset(namespace, jobset_name)


@dataclass(frozen=True, slots=True)
class _ResultFlags:
    """Derived booleans describing a ``ControllerFetchResult``."""

    has_metrics: bool
    has_files: bool
    has_error: bool
    success: bool


def _compute_result_flags(result: ControllerFetchResult, job_id: str) -> _ResultFlags:
    """Derive has_metrics/has_files/has_error/success flags and log a summary.

    A partial fetch can set has_files=True but still populate result.error
    (e.g. checkpoints saved but key export files missing). Treat error as
    authoritative so a false-success Completed phase never overwrites the
    real failure signal.
    """
    has_metrics = bool(result.metrics and result.metrics.get("metrics"))
    has_files = bool(_KEY_RESULT_FILES & set(result.downloaded or []))
    has_error = bool(result.error)
    success = has_files and not has_error

    logger.info(
        f"Results for {job_id}: has_metrics={has_metrics}, has_files={has_files}, "
        f"metrics_keys={list(result.metrics.keys()) if result.metrics else []}"
    )
    return _ResultFlags(
        has_metrics=has_metrics,
        has_files=has_files,
        has_error=has_error,
        success=success,
    )


def _backfill_pre_completion_conditions(
    status: dict[str, Any], sb: StatusBuilder
) -> None:
    """Backfill conditions for fast-completing jobs that skipped RUNNING phase."""
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


def _compute_duration_seconds(status: dict[str, Any]) -> float | None:
    start_time = status.get("startTime")
    if not start_time:
        return None
    try:
        start_dt = parse_timestamp(start_time)
        return (datetime.now(timezone.utc) - start_dt).total_seconds()
    except (ValueError, TypeError):
        return None


def _record_results_on_status(
    *,
    body: dict[str, Any],
    namespace: str,
    job_id: str,
    result: ControllerFetchResult,
    sb: StatusBuilder,
    has_metrics: bool,
    has_files: bool,
) -> None:
    """Populate metrics/summary/resultsPath on the status patch."""
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


def _set_results_phase_and_condition(
    *,
    body: dict[str, Any],
    jobset_name: str,
    result: ControllerFetchResult,
    sb: StatusBuilder,
    has_metrics: bool,
    has_files: bool,
    has_error: bool,
    success: bool,
) -> None:
    """Set phase + RESULTS_AVAILABLE condition; emit failure event on failure.

    Result files are the authoritative source - /api/metrics is a convenience
    that duplicates what's derivable from the files. Files alone = full success,
    but only if ControllerFetchResult.error is empty: a partial fetch can set
    has_files while still reporting an error for missing key artifacts.
    """
    if success:
        if has_metrics:
            msg = f"Metrics and {len(result.downloaded)} result files stored"
        else:
            msg = f"{len(result.downloaded)} result files stored"
            logger.info(
                f"Metrics fetch skipped/failed for {jobset_name} - "
                f"result files are sufficient"
            )
        sb.set_phase(Phase.COMPLETED)
        sb.conditions.set_true(ConditionType.RESULTS_AVAILABLE, "ResultsStored", msg)
        return

    sb.set_phase(Phase.FAILED)
    failure_msg = (
        result.error
        if has_error
        else "Failed to fetch complete result files from controller"
    )
    sb.conditions.set_false(
        ConditionType.RESULTS_AVAILABLE,
        "ResultsFetchFailed",
        failure_msg,
    )
    if has_files and has_error:
        logger.warning(
            f"Partial results for {jobset_name}: key files present but "
            f"fetch reported error: {result.error}"
        )
    elif has_metrics:
        logger.warning(
            f"Metrics were fetched for {jobset_name}, "
            "but complete result files were not available"
        )
    else:
        logger.warning(f"No result files downloaded for {jobset_name}")
    events.results_failed(body, failure_msg)


async def _update_job_index_safe(
    *,
    body: dict[str, Any],
    namespace: str,
    job_id: str,
    result: ControllerFetchResult,
    sb: StatusBuilder,
    success: bool,
) -> None:
    """Update the job index; on failure, degrade gracefully.

    Results are already persisted to disk, so a failure here only affects
    discoverability via the index/history API - don't retry the whole
    completion handler, but set a status condition and event so operators
    can see the gap.
    """
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


async def _delete_backing_jobset(namespace: str, jobset_name: str) -> None:
    try:
        async with k8s_client() as api:
            await client.CustomObjectsApi(api).delete_namespaced_custom_object(
                group=JOBSET_GROUP,
                version=JOBSET_VERSION,
                plural=JOBSET_PLURAL,
                namespace=namespace,
                name=jobset_name,
            )
        logger.info(f"Deleted JobSet {jobset_name} after results stored")
    except ApiException as e:
        if e.status != 404:
            logger.warning(
                f"Failed to delete JobSet {jobset_name} after completion: {e}"
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
    dest_dir = OperatorEnvironment.RESULTS.DIR / namespace / job_id

    try:
        for path in _metric_file_candidates(dest_dir, downloaded):
            data = _load_metrics_payload(path)
            if data is None:
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
    except (OSError, ValueError, orjson.JSONDecodeError, zstandard.ZstdError) as e:
        logger.warning(f"Failed to parse metrics from {dest_dir}: {e}")
    return None


def _metric_file_candidates(dest_dir: Path, downloaded: list[str]) -> list[Path]:
    """Return a de-duplicated, existence-checked, .zst-first candidate list."""
    candidates: list[Path] = [dest_dir / name for name in downloaded]
    candidates.extend(
        [
            dest_dir / "profile_export_aiperf.json.zst",
            dest_dir / "profile_export_aiperf.json",
        ]
    )
    candidates.sort(key=lambda p: 0 if p.suffix == ".zst" else 1)

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _load_metrics_payload(path: Path) -> dict[str, Any] | None:
    """Load + decode a metrics payload. Returns None if it isn't a dict."""
    if path.suffix == ".zst":
        raw = (
            zstandard.ZstdDecompressor()
            .stream_reader(io.BytesIO(path.read_bytes()))
            .read()
        )
        data = orjson.loads(raw)
    else:
        data = orjson.loads(path.read_bytes())
    return data if isinstance(data, dict) else None
