# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Completion handling and result fetching for AIPerfJob."""

from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson
import zstandard
from kubernetes_asyncio import client
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.common.finite import scrub_non_finite
from aiperf.kubernetes.client import k8s_client
from aiperf.kubernetes.cr_refs import JOBSET_GROUP, JOBSET_PLURAL, JOBSET_VERSION
from aiperf.kubernetes.jobset import controller_dns_name
from aiperf.kubernetes.results_sidecar import write_ready_marker
from aiperf.operator import events, runs_index
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
from aiperf.operator.handlers._completion_retry import (
    maybe_raise_for_transient_fetch_failure,
)
from aiperf.operator.models import ControllerFetchResult, MetricsSummary
from aiperf.operator.progress_client import ProgressClient  # re-exported for tests
from aiperf.operator.results_layout import (
    enforce_retention,
    epoch_key_from_body,
    run_dir,
    schedule_index_drops,
    write_latest,
)
from aiperf.operator.status import ConditionType, Phase, StatusBuilder, parse_timestamp

__all__ = [
    "ProgressClient",
    "_IncompleteResultsError",
    "_NO_PROGRESS_STAGNATION_LIMIT",
    "_fetch_with_progress_aware_retry",
    "_parse_metrics_from_files",
    "_record_results_on_status",
    "fetch_results_with_retry",
    "get_or_create_progress_client",
    "handle_completion",
]

logger = logging.getLogger(__name__)

_KEY_RESULT_FILES = frozenset(
    {"profile_export_aiperf.json", "profile_export_aiperf.csv"}
)


def _has_key_result_files(paths: list[str] | None) -> bool:
    """Return True when the authoritative AIPerf exports are present.

    Accept both raw and on-disk-compressed names. The operator stores final
    artifacts as ``*.zst`` when COMPRESS_ON_DISK is enabled, but the completion
    classifier still needs to recognize those files as authoritative results.
    """
    names = set(paths or [])
    return any(key in names or f"{key}.zst" in names for key in _KEY_RESULT_FILES)


def _key_files_materialized(namespace: str, job_id: str, epoch: str) -> bool:
    """Return True when an authoritative export is actually on disk for this run.

    The controller's ``downloaded`` list claims which files it pushed, but the
    operator must not advance ``latest.txt``/``runEpoch``/the in-DB latest
    pointer (or even create the run dir) until a key export is materialized on
    its own PVC — otherwise a transport race that reports the file without
    landing it would point readers at an empty directory. Checks both the raw
    and ``.zst`` on-disk names, mirroring :func:`_has_key_result_files`.

    Existence alone is NOT sufficient: a mid-write disk-full leaves a truncated
    file on disk, and serving it as a complete result is a data-integrity bug.
    A key artifact only counts as materialized when :func:`_key_artifact_valid`
    confirms it is non-empty and (for the JSON export) parses to a non-empty
    dict, mirroring the wave-9/10 JSONL-degradation and harvest marker-parse
    hardening. Returns True on the FIRST valid key so a csv-authoritative run
    still succeeds without a readable JSON summary.
    """
    dest_dir = run_dir(OperatorEnvironment.RESULTS.DIR, namespace, job_id, epoch)
    if not dest_dir.exists():
        return False
    for key in _KEY_RESULT_FILES:
        for candidate in ((dest_dir / key), (dest_dir / f"{key}.zst")):
            if candidate.is_file() and _key_artifact_valid(candidate):
                return True
    return False


def _key_artifact_valid(path: Path) -> bool:
    """Return True when a key result artifact is fully materialized (not truncated).

    A truncated ENOSPC write leaves a non-empty-but-corrupt file on disk, so
    existence is not enough. Validation:

    - Empty file (0 bytes) → invalid.
    - ``.json`` / ``.json.zst`` → must decode (zstd, if compressed) and
      ``orjson.loads`` to a non-empty dict.
    - ``.csv`` / ``.csv.zst`` → non-empty is sufficient (no cheap structural
      parse; the JSON export is the operator's authoritative summary).

    A truncated/unparsable JSON export MUST NOT count as materialized so the
    operator neither advances ``latest.txt`` nor serves corrupt results.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return False
    if not raw:
        return False

    is_zst = path.suffix == ".zst"
    logical_name = path.name[: -len(".zst")] if is_zst else path.name

    # CSV export: non-empty payload is sufficient. Do NOT require a valid zstd
    # frame here — treat the compressed CSV as an opaque, present artifact.
    if logical_name.endswith(".csv"):
        return True

    if is_zst:
        try:
            raw = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)).read()
        except (zstandard.ZstdError, OSError):
            return False

    try:
        data = orjson.loads(raw)
    except (orjson.JSONDecodeError, ValueError):
        return False
    return isinstance(data, dict) and bool(data)


def _recover_result_from_disk(
    *,
    body: dict[str, Any],
    namespace: str,
    job_id: str,
    result: ControllerFetchResult,
) -> ControllerFetchResult:
    """Promote already-downloaded final exports from disk into the fetch result.

    A controller-side transport race can leave ``result.downloaded`` empty even
    though the operator's results dir already contains the final compressed
    exports. In that case the on-disk files are authoritative and completion
    should recover from them instead of stamping ``ResultsFetchFailed``.
    """
    epoch = epoch_key_from_body(body)
    dest_dir = run_dir(OperatorEnvironment.RESULTS.DIR, namespace, job_id, epoch)
    if not dest_dir.exists():
        return result

    on_disk = sorted(
        str(path.relative_to(dest_dir))
        for path in dest_dir.rglob("*")
        if path.is_file() and path.name != "latest.txt"
    )
    if not _has_key_result_files(on_disk):
        return result

    metrics = result.metrics or _parse_metrics_from_files(
        on_disk, namespace, job_id, epoch=epoch
    )
    return ControllerFetchResult(
        metrics=metrics,
        downloaded=on_disk,
        checkpoints=result.checkpoints,
        error="",
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
    backing JobSet on success. Short-circuits if ``on_delete`` has already
    requested cancellation. ``result`` lets the salvage path skip the HTTP
    round-trip.
    """
    # on_delete cancellation: skip fetch/JobSet-delete/status patches so the
    # CR delete doesn't block on retry backoff.
    if _completion_cancelled(namespace, job_id):
        return

    duration_sec = _compute_duration_seconds(status)

    if result is None:
        host = controller_dns_name(jobset_name, namespace)
        result = await fetch_results_with_retry(host, namespace, job_id, body=body)

    if _completion_cancelled(namespace, job_id):
        return

    result = _recover_result_from_disk(
        body=body,
        namespace=namespace,
        job_id=job_id,
        result=result,
    )
    flags = _compute_result_flags(result, job_id)
    # Race retry: see _completion_retry for the gate; raises kopf.TemporaryError.
    maybe_raise_for_transient_fetch_failure(
        body=body,
        namespace=namespace,
        job_id=job_id,
        result=result,
        flags=flags,
    )
    if _completion_cancelled(namespace, job_id):
        return

    staged_patch = _StagedStatusPatch(status={})
    staged_sb = StatusBuilder(staged_patch, status)
    _backfill_pre_completion_conditions(status, staged_sb)
    staged_sb.set_completion_time()
    await _apply_completion_results(
        body=body,
        namespace=namespace,
        jobset_name=jobset_name,
        job_id=job_id,
        result=result,
        sb=staged_sb,
        flags=flags,
    )

    if _completion_cancelled(namespace, job_id):
        return

    staged_sb.finalize()
    sb._patch.status.update(staged_patch.status)
    if flags.success:
        events.completed(body, job_id, duration_sec)

    await _maybe_delete_jobset_after_success(namespace, jobset_name, job_id, flags)


def _completion_cancelled(namespace: str, job_id: str) -> bool:
    """Return True and log when a completion path should stop mutating status."""
    if not is_cancellation_requested(job_key(namespace, job_id)):
        return False
    logger.info(
        f"Cancellation requested for {namespace}/{job_id}, skipping completion handling"
    )
    return True


async def _apply_completion_results(
    *,
    body: dict[str, Any],
    namespace: str,
    jobset_name: str,
    job_id: str,
    result: ControllerFetchResult,
    sb: StatusBuilder,
    flags: _ResultFlags,
) -> None:
    """Stamp results/phase/condition + update index. Index is updated BEFORE
    ``sb.finalize()`` so its failure path can queue an INDEX_UPDATED=False
    condition without racing the single finalize() pass.
    """
    _record_results_on_status(
        body=body,
        namespace=namespace,
        job_id=job_id,
        result=result,
        sb=sb,
        has_metrics=flags.has_metrics,
        has_files=flags.has_files,
    )
    epoch = epoch_key_from_body(body)
    # Retention lives here (not in the sync _record_results_on_status) so the
    # rmtree walk can run off-loop; the materialized gate mirrors the
    # latest.txt/runEpoch gate inside _record_results_on_status.
    if flags.has_files and _key_files_materialized(namespace, job_id, epoch):
        await _run_retention_pass(namespace, job_id, epoch)
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
    summary_blob, mtime_epoch, end_time, total_size_bytes = _gather_index_inputs(
        namespace, job_id, epoch
    )
    phase = "Succeeded" if flags.success else "Failed"
    # On the file-metrics path (API metrics empty but key exports present), feed
    # the index the same metrics ``_record_results_on_status`` stamped on the CR
    # so the narrow compare columns match status.summary / the on-disk JSON.
    # Without this, sub-second / CompletedBeforeMonitor jobs write all-NULL
    # narrow columns because result.metrics is None.
    if not flags.has_metrics and flags.has_files:
        index_metrics = _parse_metrics_from_files(
            result.downloaded, namespace, job_id, epoch=epoch
        )
    else:
        index_metrics = result.metrics
    await _update_job_index_safe(
        namespace=namespace,
        job_id=job_id,
        epoch=epoch,
        body=body,
        sb=sb,
        phase=phase,
        summary_blob=summary_blob,
        metrics=scrub_non_finite(index_metrics),
        downloaded_files=result.downloaded,
        error=result.error or None,
        mtime_epoch=mtime_epoch,
        end_time=end_time,
        total_size_bytes=total_size_bytes,
    )


def _gather_index_inputs(
    namespace: str,
    job_id: str,
    epoch: str,
) -> tuple[bytes | None, int, str | None, int]:
    """Read the on-disk summary file and compute (summary_blob, mtime_epoch,
    end_time, total_size_bytes) for the runs_index upsert. Returns
    (None, 0, None, 0) if nothing on disk yet (e.g. fetch failed).

    summary_blob is always the zstd-compressed bytes of the
    profile_export_aiperf.json payload — matches the on-disk .json.zst when
    present, or compresses the raw .json otherwise.
    """
    dest_dir = run_dir(OperatorEnvironment.RESULTS.DIR, namespace, job_id, epoch)
    if not dest_dir.exists():
        return None, 0, None, 0

    summary_blob: bytes | None = None
    end_time: str | None = None

    summary_zst = dest_dir / "profile_export_aiperf.json.zst"
    summary_raw = dest_dir / "profile_export_aiperf.json"
    try:
        if summary_zst.exists():
            blob = summary_zst.read_bytes()
            metrics = orjson.loads(runs_index.zstd_decompress(blob))
            summary_blob = blob
            end_time = metrics.get("end_time")
        elif summary_raw.exists():
            raw = summary_raw.read_bytes()
            metrics = orjson.loads(raw)
            summary_blob = zstandard.ZstdCompressor().compress(raw)
            end_time = metrics.get("end_time")
    except (OSError, orjson.JSONDecodeError, zstandard.ZstdError) as exc:
        logger.warning(
            "completion: cannot read summary at %s for index update: %s",
            dest_dir,
            exc,
        )

    try:
        files = [f for f in dest_dir.iterdir() if f.is_file()]
        total_size = sum(f.stat().st_size for f in files)
        mtime_epoch = int(dest_dir.stat().st_mtime)
    except OSError:
        total_size = 0
        mtime_epoch = 0

    return summary_blob, mtime_epoch, end_time, total_size


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


@dataclass(slots=True)
class _StagedStatusPatch:
    """Minimal patch object for staging completion status until cancellation-safe."""

    status: dict[str, Any]


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
    has_files = _has_key_result_files(result.downloaded)
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
        return (datetime.now(UTC) - start_dt).total_seconds()
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
    epoch = epoch_key_from_body(body)
    if has_metrics:
        metrics_for_status = scrub_non_finite(result.metrics)
        sb.set_results(metrics_for_status)
        summary = MetricsSummary.from_metrics(metrics_for_status)
        summary_dict = scrub_non_finite(summary.to_status_dict())
        if summary_dict:
            sb.set_summary(summary_dict)
    elif has_files:
        # API metrics empty/unavailable but files downloaded.
        # Parse metrics from the JSON export file and store in CR.
        file_metrics = _parse_metrics_from_files(
            result.downloaded, namespace, job_id, epoch=epoch
        )
        if file_metrics:
            file_metrics_for_status = scrub_non_finite(file_metrics)
            sb.set_results(file_metrics_for_status)
            # Also derive ``status.summary`` from file_metrics so kube-list /
            # operator UI show throughput / latency on jobs that finished
            # before the controller progress poll could land. Without this,
            # ``status.summary`` stays empty even when ``status.results`` is
            # fully populated, and kube-list shows '-' for THROUGHPUT/LATENCY.
            file_summary = MetricsSummary.from_metrics(file_metrics_for_status)
            file_summary_dict = scrub_non_finite(file_summary.to_status_dict())
            if file_summary_dict:
                sb.set_summary(file_summary_dict)
            logger.info(f"Parsed metrics from result files for {job_id}")

    if has_files and _key_files_materialized(namespace, job_id, epoch):
        dest_dir = run_dir(OperatorEnvironment.RESULTS.DIR, namespace, job_id, epoch)
        write_ready_marker(dest_dir)
        sb.set_results_path(str(dest_dir))
        write_latest(OperatorEnvironment.RESULTS.DIR, namespace, job_id, epoch)
        sb.set_run_epoch(int(epoch))
        events.results_stored(body, str(dest_dir), len(result.downloaded))
        logger.info(f"Downloaded {len(result.downloaded)} result files to {dest_dir}")


async def _run_retention_pass(namespace: str, job_id: str, epoch: str) -> None:
    """Trim old run dirs after a successful write; never fatal on failure.

    The rmtree walk runs in a worker thread so a slow PVC prune cannot stall
    the kopf event loop. Index-drop scheduling happens back on the loop via
    ``schedule_index_drops`` — ``asyncio.get_running_loop()`` raises inside
    ``asyncio.to_thread``, so drops scheduled from the worker thread would
    silently be skipped.
    """
    try:
        deleted = await asyncio.to_thread(
            enforce_retention,
            OperatorEnvironment.RESULTS.DIR,
            namespace,
            job_id,
            keep=OperatorEnvironment.RESULTS.RETAIN_RUNS,
            protect_epoch=epoch,
            retain_days=OperatorEnvironment.RESULTS.RETAIN_DAYS,
        )
    except Exception:  # noqa: BLE001 - retention is best-effort; never fail completion on disk I/O
        logger.warning(
            "retention pass failed for %s/%s; continuing",
            namespace,
            job_id,
            exc_info=True,
        )
        return
    schedule_index_drops(namespace, job_id, deleted)
    if deleted:
        logger.info(
            "retention: trimmed %d old runs for %s/%s",
            len(deleted),
            namespace,
            job_id,
        )


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
    namespace: str,
    job_id: str,
    epoch: str,
    body: dict[str, Any],
    sb: StatusBuilder,
    phase: str,
    summary_blob: bytes | None,
    metrics: dict[str, Any] | None,
    downloaded_files: list[str],
    error: str | None,
    mtime_epoch: int,
    end_time: str | None,
    total_size_bytes: int,
) -> None:
    """Update the runs_index; on failure, degrade gracefully.

    Results are already persisted to disk, so a failure here only affects
    discoverability via the index/history API - don't retry the whole
    completion handler, but set a status condition and event so operators
    can see the gap. Always calls ``set_latest`` after a successful upsert
    so the in-DB latest pointer matches latest.txt on disk.
    """
    try:
        if phase in ("Succeeded", "PartiallyFailed"):
            # Completion is keyed on JSON-OR-CSV (``_KEY_RESULT_FILES``), so a
            # csv-authoritative run can succeed with no readable JSON summary
            # blob. Record it as completed anyway: routing a success verdict to
            # ``upsert_run_failed`` would stamp ``error="unknown"`` and zero
            # metrics, contradicting the CR's Succeeded/ResultsAvailable status
            # and the disk-fallback path (``results_db._index_from_disk``,
            # which records the same run as Succeeded/error=None).
            await runs_index.upsert_run_completed(
                namespace,
                job_id,
                epoch,
                summary_blob=summary_blob if summary_blob is not None else b"",
                metrics=metrics or {},
                files=downloaded_files,
                mtime_epoch=mtime_epoch,
                end_time=end_time,
                total_size_bytes=total_size_bytes,
                phase=phase,
            )
        else:
            await runs_index.upsert_run_failed(
                namespace,
                job_id,
                epoch,
                error=error or "unknown",
                phase=phase,
            )
        # Only advance the in-DB latest pointer once the authoritative export
        # is materialized on disk. A row whose key files never landed must not
        # become the discoverable latest run (mirrors the latest.txt gate in
        # ``_record_results_on_status``).
        if _key_files_materialized(namespace, job_id, epoch):
            await runs_index.set_latest(namespace, job_id, epoch)
    except Exception as e:
        logger.exception(f"Failed to update runs_index for {job_id}")
        sb.conditions.set_false(
            ConditionType.INDEX_UPDATED,
            "IndexUpdateFailed",
            f"Index write failed: {e}",
        )
        events.index_update_failed(body, str(e))


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
    *,
    epoch: str,
) -> dict[str, Any] | None:
    """Parse metrics from downloaded result files.

    Looks for profile_export_aiperf.json (or .json.zst) which contains the
    full benchmark results in a format compatible with the CR status.

    Per-candidate failures (non-JSON .zst siblings such as
    ``profile_export.jsonl.zst`` or ``server_metrics_export.parquet.zst``)
    are caught and skipped — the candidate sort puts ``.zst`` first, so a
    bail-out on the first unparsable file would silently swallow the
    valid ``profile_export_aiperf.json`` that follows it.
    """
    dest_dir = run_dir(OperatorEnvironment.RESULTS.DIR, namespace, job_id, epoch)

    for path in _metric_file_candidates(dest_dir, downloaded):
        try:
            data = _load_metrics_payload(path)
        except (OSError, ValueError, orjson.JSONDecodeError, zstandard.ZstdError) as e:
            logger.debug(
                f"completion: skipping unparsable candidate {path} for "
                f"{namespace}/{job_id} epoch={epoch} "
                f"({type(e).__name__}: {e})"
            )
            continue
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
