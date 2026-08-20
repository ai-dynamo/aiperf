# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Completion handling and result fetching for AIPerfJob."""

from __future__ import annotations

import asyncio
import copy
import io
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson
import zstandard

from aiperf.common.finite import is_finite_value, scrub_non_finite
from aiperf.common.results_markers import ready_marker_path, write_ready_marker
from aiperf.kubernetes.crd_models import ControllerFetchResult, MetricsSummary
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.jobset import controller_dns_name
from aiperf.kubernetes.phase import Phase, parse_timestamp
from aiperf.kubernetes.spec_converter import (
    DEFAULT_KEY_EXPORT_NAMES,
    KeyExportNames,
    key_export_names_from_body,
)
from aiperf.operator import events, runs_index
from aiperf.operator.client_cache import (
    get_cancellation_event,
    get_or_create_progress_client,
    is_cancellation_requested,
    job_key,
)
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.handlers._completion_fetch import (
    _NO_PROGRESS_STAGNATION_LIMIT,  # re-exported for tests
    _await_or_cancel,
    _fetch_with_progress_aware_retry,  # re-exported for tests
    _FetchCancelled,
    _IncompleteResultsError,  # re-exported for tests/monitor
    fetch_results_with_retry,
)
from aiperf.operator.handlers._completion_retry import (
    maybe_raise_for_transient_fetch_failure,
)
from aiperf.operator.handlers._job_identity import (
    StaleAIPerfJobCallback,
    body_name,
    body_uid,
    current_aiperfjob_resource_version,
    delete_owned_aiperfjob_jobset,
)
from aiperf.operator.progress_client import ProgressClient  # re-exported for tests
from aiperf.operator.progress_models import JobProgress
from aiperf.operator.results_layout import (
    enforce_retention,
    epoch_key_from_body,
    reconcile_latest,
    resolve_latest,
    run_dir,
    schedule_index_drops,
    write_latest,
)
from aiperf.operator.status import ConditionType, StatusBuilder

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

_KEY_RESULT_FILES = DEFAULT_KEY_EXPORT_NAMES.names
_PHASE_MANIFEST_NAME = "phase_manifest.json"
_PHASE_MANIFEST_SCHEMA_VERSION = 1


def _has_key_result_files(
    paths: list[str] | None,
    *,
    key_names: frozenset[str] = _KEY_RESULT_FILES,
) -> bool:
    """Return True when the authoritative AIPerf exports are present.

    Accept both raw and on-disk-compressed names. The operator stores final
    artifacts as ``*.zst`` when COMPRESS_ON_DISK is enabled, but the completion
    classifier still needs to recognize those files as authoritative results.
    """
    names = set(paths or [])
    return any(key in names or f"{key}.zst" in names for key in key_names)


def _key_files_materialized(
    namespace: str,
    job_id: str,
    epoch: str,
    *,
    key_names: frozenset[str] = _KEY_RESULT_FILES,
) -> bool:
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
    for key in key_names:
        for candidate in ((dest_dir / key), (dest_dir / f"{key}.zst")):
            if candidate.is_file() and _key_artifact_valid(candidate):
                return True
    return False


def _key_artifact_valid(path: Path) -> bool:
    """Return True when a key result artifact is fully materialized (not truncated).

    A truncated ENOSPC write leaves a non-empty-but-corrupt file on disk, so
    existence is not enough. Validation:

    - Empty file (0 bytes) → invalid.
    - ``.json`` / ``.json.zst`` → must decode (a complete zstd frame, if
      compressed) and ``orjson.loads`` to a non-empty dict.
    - ``.csv`` → non-empty is sufficient (no cheap structural parse; the
      JSON export is the operator's authoritative summary).
    - ``.csv.zst`` → must contain a complete, non-empty zstd frame.

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

    if is_zst:
        try:
            decompressor = zstandard.ZstdDecompressor().decompressobj()
            raw = decompressor.decompress(raw)
        except zstandard.ZstdError:
            return False
        if not decompressor.eof or decompressor.unused_data or not raw:
            return False

    if logical_name.endswith(".csv"):
        return True

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
    key_names: KeyExportNames | None = None,
) -> ControllerFetchResult:
    """Promote already-downloaded final exports from disk into the fetch result.

    A controller-side transport race can leave ``result.downloaded`` empty even
    though the operator's results dir already contains the final compressed
    exports. In that case the on-disk files are authoritative and completion
    should recover from them instead of stamping ``ResultsFetchFailed``.
    """
    key_names = key_names or key_export_names_from_body(body)
    epoch = epoch_key_from_body(body)
    dest_dir = run_dir(OperatorEnvironment.RESULTS.DIR, namespace, job_id, epoch)
    if not dest_dir.exists():
        return result

    on_disk = sorted(
        str(path.relative_to(dest_dir))
        for path in dest_dir.rglob("*")
        if path.is_file() and path.name != "latest.txt"
    )
    if not _has_key_result_files(on_disk, key_names=key_names.names):
        return result
    if not _key_files_materialized(
        namespace,
        job_id,
        epoch,
        key_names=key_names.names,
    ):
        return result

    metrics = result.metrics or _parse_metrics_from_files(
        on_disk, namespace, job_id, epoch=epoch, json_name=key_names.json_name
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
    expected_parent_uid: str | None = None,
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
    parent_uid = expected_parent_uid or body_uid(body)
    parent_name = body_name(body, job_id)
    try:
        await current_aiperfjob_resource_version(namespace, parent_name, parent_uid)
    except StaleAIPerfJobCallback as exc:
        logger.info("Skipping stale completion callback: %s", exc)
        return

    # on_delete cancellation: skip fetch/JobSet-delete/status patches so the
    # CR delete doesn't block on retry backoff.
    if _completion_cancelled(namespace, job_id, parent_uid):
        return

    duration_sec = _compute_duration_seconds(status)
    key_names = key_export_names_from_body(body)

    if result is None:
        host = controller_dns_name(jobset_name, namespace)
        result = await fetch_results_with_retry(host, namespace, job_id, body=body)

    if _completion_cancelled(namespace, job_id, parent_uid):
        return

    result = _recover_result_from_disk(
        body=body,
        namespace=namespace,
        job_id=job_id,
        result=result,
        key_names=key_names,
    )
    flags = _compute_result_flags(result, job_id, key_names=key_names)
    flags = _demote_unmaterialized_result_files(
        body=body,
        namespace=namespace,
        job_id=job_id,
        flags=flags,
        key_names=key_names,
    )
    # Race retry: see _completion_retry for the gate; raises kopf.TemporaryError.
    maybe_raise_for_transient_fetch_failure(
        body=body,
        namespace=namespace,
        job_id=job_id,
        result=result,
        flags=flags,
    )
    if _completion_cancelled(namespace, job_id, parent_uid):
        return

    staged_patch = _StagedStatusPatch(status={})
    staged_sb = StatusBuilder(staged_patch, status)
    await _refresh_final_phase_progress(
        namespace=namespace,
        jobset_name=jobset_name,
        job_id=job_id,
        patch=staged_patch,
        expected_parent_uid=parent_uid,
    )
    _backfill_pre_completion_conditions(status, staged_sb)
    staged_sb.set_completion_time()

    if _completion_cancelled(namespace, job_id, parent_uid):
        return

    # The fetched artifacts are already durable on disk, but terminal status,
    # ready/latest publication, index updates, and success events are not
    # committed until this final cluster await is cancellation-safe. If the
    # operator crashes after deletion, the durable claim + on-disk artifacts
    # are recovered by the orphan-claim path.
    if not await _maybe_delete_jobset_after_success(
        namespace,
        jobset_name,
        job_id,
        flags,
        parent_name=parent_name,
        parent_uid=parent_uid,
    ):
        return

    if _completion_cancelled(namespace, job_id, parent_uid):
        return

    if flags.has_files and not _key_files_materialized(
        namespace,
        job_id,
        epoch_key_from_body(body),
        key_names=key_names.names,
    ):
        logger.error(
            "Key exports for %s/%s disappeared while deleting JobSet %s; "
            "publishing a failed result state",
            namespace,
            job_id,
            jobset_name,
        )
        flags = replace(
            flags,
            has_files=False,
            success=False,
            benchmark_failure=None,
        )

    await _publish_completion_after_jobset_delete(
        body=body,
        namespace=namespace,
        jobset_name=jobset_name,
        job_id=job_id,
        result=result,
        staged_sb=staged_sb,
        target_sb=sb,
        status=status,
        flags=flags,
        key_names=key_names,
        parent_name=parent_name,
        parent_uid=parent_uid,
        duration_sec=duration_sec,
    )


async def _publish_completion_after_jobset_delete(
    *,
    body: dict[str, Any],
    namespace: str,
    jobset_name: str,
    job_id: str,
    result: ControllerFetchResult,
    staged_sb: StatusBuilder,
    target_sb: StatusBuilder,
    status: dict[str, Any],
    flags: _ResultFlags,
    key_names: KeyExportNames,
    parent_name: str,
    parent_uid: str | None,
    duration_sec: float | None,
) -> None:
    """Publish completion only while the immutable parent identity remains live."""
    try:
        await current_aiperfjob_resource_version(namespace, parent_name, parent_uid)
    except StaleAIPerfJobCallback as exc:
        logger.info("Skipping stale completion publication: %s", exc)
        return

    flags, artifact_fingerprint = await _apply_completion_results(
        body=body,
        namespace=namespace,
        jobset_name=jobset_name,
        job_id=job_id,
        result=result,
        sb=staged_sb,
        status=status,
        flags=flags,
        key_names=key_names,
        parent_name=parent_name,
        parent_uid=parent_uid,
    )
    if _completion_cancelled(namespace, job_id, parent_uid):
        return

    try:
        await current_aiperfjob_resource_version(namespace, parent_name, parent_uid)
    except StaleAIPerfJobCallback as exc:
        logger.info("Discarding stale completion publication: %s", exc)
        await _drop_index_row(namespace, job_id, epoch_key_from_body(body))
        return

    had_files = flags.has_files
    flags, final_artifacts_materialized = await _verify_final_artifact_publication(
        namespace=namespace,
        job_id=job_id,
        epoch=epoch_key_from_body(body),
        flags=flags,
        expected_fingerprint=artifact_fingerprint,
        key_names=key_names,
        sb=staged_sb,
    )
    if had_files and not final_artifacts_materialized:
        _set_results_phase_and_condition(
            body=body,
            jobset_name=jobset_name,
            result=result,
            sb=staged_sb,
            has_metrics=flags.has_metrics,
            has_files=flags.has_files,
            has_error=flags.has_error,
            success=flags.success,
            benchmark_failure=flags.benchmark_failure,
            emit_event=False,
        )

    staged_sb.finalize()
    # Do NOT call fence_status_patch here. The fence writes metadata.resourceVersion
    # into the kopf Patch, which causes kopf's single MERGE PATCH (metadata+status) to
    # fail with 409 Conflict on any concurrent CR write. The status update is then
    # silently dropped — and because try_claim_completion's durable annotation blocks
    # re-entry, the phase never transitions to Completed. Stale-write protection in the
    # completion path already comes from try_claim_completion + UID fences.
    target_sb._patch.status.update(staged_sb._patch.status)
    _emit_accepted_completion_events(
        body=body,
        namespace=namespace,
        jobset_name=jobset_name,
        job_id=job_id,
        result=result,
        status_patch=staged_sb._patch.status,
        flags=flags,
        key_names=key_names,
        duration_sec=duration_sec,
    )


def _emit_accepted_completion_events(
    *,
    body: dict[str, Any],
    namespace: str,
    jobset_name: str,
    job_id: str,
    result: ControllerFetchResult,
    status_patch: dict[str, Any],
    flags: _ResultFlags,
    key_names: KeyExportNames,
    duration_sec: float | None,
) -> None:
    """Emit completion events only after the final UID/resourceVersion fence."""
    epoch = epoch_key_from_body(body)
    if flags.has_files and _key_files_materialized(
        namespace, job_id, epoch, key_names=key_names.names
    ):
        dest_dir = run_dir(OperatorEnvironment.RESULTS.DIR, namespace, job_id, epoch)
        events.results_stored(body, str(dest_dir), len(result.downloaded))
    if flags.success:
        events.completed(body, job_id, duration_sec)
    elif flags.benchmark_failure is not None:
        events.failed(body, jobset_name, flags.benchmark_failure)
    else:
        failure_msg = (
            result.error or "Failed to fetch complete result files from controller"
        )
        events.results_failed(body, failure_msg)

    for condition in status_patch.get("conditions") or []:
        if (
            condition.get("type") == ConditionType.INDEX_UPDATED
            and condition.get("status") == "False"
            and condition.get("reason") == "IndexUpdateFailed"
        ):
            events.index_update_failed(body, str(condition.get("message") or ""))
            break


def _completion_cancelled(
    namespace: str, job_id: str, parent_uid: str | None = None
) -> bool:
    """Return True and log when a completion path should stop mutating status."""
    if not is_cancellation_requested(job_key(namespace, job_id, parent_uid)):
        return False
    logger.info(
        f"Cancellation requested for {namespace}/{job_id}, skipping completion handling"
    )
    return True


async def _parent_identity_is_current(
    namespace: str,
    parent_name: str | None,
    parent_uid: str | None,
    *,
    context: str,
) -> bool:
    """Return False for a stale parent while preserving transient retry errors."""
    if parent_name is None or parent_uid is None:
        return True
    try:
        await current_aiperfjob_resource_version(namespace, parent_name, parent_uid)
    except StaleAIPerfJobCallback as exc:
        logger.info("Skipping stale %s: %s", context, exc)
        return False
    return True


def _select_single_results_phase(
    phases: dict[str, Any],
) -> dict[str, Any] | None:
    """Select one profiling-kind status block, with legacy compatibility."""
    explicit_results = [
        phase
        for phase in phases.values()
        if isinstance(phase, dict) and phase.get("phaseKind") == "profiling"
    ]
    if len(explicit_results) == 1:
        return explicit_results[0]
    if explicit_results:
        return None
    if any(
        isinstance(phase, dict) and phase.get("phaseKind") is not None
        for phase in phases.values()
    ):
        return None

    legacy = phases.get("profiling")
    if isinstance(legacy, dict):
        return legacy
    legacy_results = [
        phase
        for phase_name, phase in phases.items()
        if phase_name != "warmup" and isinstance(phase, dict)
    ]
    return legacy_results[0] if len(legacy_results) == 1 else None


def _repair_results_phase_counts(
    results_phase: dict[str, Any], successful: int, errors: int = 0
) -> bool:
    """Repair lagging status counters from authoritative exported counts."""
    completed = successful + errors
    total = results_phase.get("requestsTotal")
    if isinstance(total, int) and total > 0 and completed > total:
        return False

    changed = False
    if (results_phase.get("requestsSent") or 0) < completed:
        results_phase["requestsSent"] = completed
        changed = True
    if not results_phase.get("sendingComplete"):
        results_phase["sendingComplete"] = True
        changed = True
    if (results_phase.get("requestsCompleted") or 0) < completed:
        results_phase["requestsCompleted"] = completed
        changed = True
    if (results_phase.get("requestsErrors") or 0) < errors:
        results_phase["requestsErrors"] = errors
        changed = True
    if not results_phase.get("isRequestsComplete"):
        results_phase["isRequestsComplete"] = True
        changed = True
    if (results_phase.get("recordsSuccess") or 0) < successful:
        results_phase["recordsSuccess"] = successful
        changed = True
    if (results_phase.get("recordsError") or 0) < errors:
        results_phase["recordsError"] = errors
        changed = True
    if not results_phase.get("isRecordsComplete"):
        results_phase["isRecordsComplete"] = True
        results_phase["recordsProgressPercent"] = 100
        changed = True
    return changed


def _manifest_count(entry: dict[str, Any], field: str) -> int | None:
    """Return one non-negative integer manifest count."""
    value = entry.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _phase_counts_from_manifest(
    manifest: dict[str, Any],
) -> dict[str, tuple[int, int]] | None:
    """Validate manifest v1 and return profiling counts keyed by phase name."""
    schema_version = manifest.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != _PHASE_MANIFEST_SCHEMA_VERSION
    ):
        return None
    entries = manifest.get("phases")
    if not isinstance(entries, list):
        return None

    counts: dict[str, tuple[int, int]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            return None
        phase_kind = entry.get("phase_kind")
        if phase_kind not in {"warmup", "profiling"}:
            return None
        if phase_kind != "profiling":
            continue
        phase_name = entry.get("phase_name")
        successful = _manifest_count(entry, "successful_request_count")
        errors = _manifest_count(entry, "error_request_count")
        total = _manifest_count(entry, "total_request_count")
        if (
            not isinstance(phase_name, str)
            or not phase_name
            or successful is None
            or errors is None
            or total != successful + errors
            or phase_name in counts
        ):
            return None
        counts[phase_name] = (successful, errors)
    return counts


def _load_phase_manifest_payload(dest_dir: Path) -> dict[str, Any] | None:
    """Read a raw or legacy-compressed phase manifest from one run directory."""
    candidates = [
        dest_dir / _PHASE_MANIFEST_NAME,
        dest_dir / f"{_PHASE_MANIFEST_NAME}.zst",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = _load_metrics_payload(path)
        except (OSError, ValueError, orjson.JSONDecodeError, zstandard.ZstdError) as e:
            logger.warning(
                "completion: ignoring unreadable phase manifest %s (%s: %s)",
                path,
                type(e).__name__,
                e,
            )
            continue
        if payload is not None:
            return payload
    return None


def _reconcile_phases_from_manifest(
    phases: dict[str, Any], manifest: dict[str, Any]
) -> bool | None:
    """Reconcile named profiling phases, or return None for an invalid manifest."""
    counts = _phase_counts_from_manifest(manifest)
    if counts is None:
        return None

    changed = False
    for phase_name, (successful, errors) in counts.items():
        phase = phases.get(phase_name)
        if not isinstance(phase, dict):
            continue
        status_phase_name = phase.get("phaseName")
        if status_phase_name is not None and status_phase_name != phase_name:
            continue
        status_phase_kind = phase.get("phaseKind")
        if status_phase_kind is not None and status_phase_kind != "profiling":
            continue
        changed |= _repair_results_phase_counts(phase, successful, errors)
    return changed


def _phase_status_snapshot(
    sb: StatusBuilder, status: dict[str, Any]
) -> dict[str, Any] | None:
    """Return staged phases, or a mutable copy of the current CR phases."""
    phases = sb._patch.status.get("phases")
    if isinstance(phases, dict):
        return phases
    existing = status.get("phases")
    return copy.deepcopy(existing) if isinstance(existing, dict) else None


def _reconcile_phase_counts_from_results(
    *,
    sb: StatusBuilder,
    status: dict[str, Any],
    result: ControllerFetchResult,
    flags: _ResultFlags,
    phase_manifest: dict[str, Any] | None = None,
) -> _ResultFlags:
    """Trust final exports over a dying controller's sampled phase counters.

    ``status.phases`` mirrors live controller progress, and by completion the
    controller pod is racing its own shutdown -- sampling it is inherently a
    race. Observed across three consecutive live gemma sweeps, the second
    variation landed on 284/300, then 300/300, then 240/300 for identical
    work whose exports every time contained all 300 records. Re-sampling only
    narrows that window; it cannot close it.

    ``phase_manifest.json`` owns exact per-phase successful and error counts,
    keyed by the canonical user-provided phase name. When it is unavailable,
    the aggregate ``request_count`` metric remains a compatibility fallback
    for a run with exactly one identifiable profiling-kind phase.

    Top-level ``request_count`` aggregates results when a run has multiple
    profiling-kind phases. It cannot safely repair one member of that set, so
    the compatibility fallback remains limited to one results phase.

    The aggregate fallback is gated on ``has_metrics`` rather than
    ``has_files``: a parsed ``request_count`` only exists because an
    authoritative export was read, and the download list is NOT a reliable
    proxy for that. Sweep children routinely finish with
    ``ResultsAvailable=True``, a populated summary, and an empty
    ``results.downloaded`` (the exports were promoted from the operator's own
    disk rather than transferred in this call), which is exactly the case that
    needs reconciling.

    Still scoped to successful runs: a failed or partial run must keep the
    counters that show how far it actually got.
    """
    if not flags.success:
        return

    # Read the CR's CURRENT phases, not the patch. The final progress
    # re-sample only writes into the patch when the controller was still
    # answering, and by completion it frequently is not -- so on exactly the
    # runs that need reconciling, the patch has no phases key at all and the
    # stale block lives only in the existing status. Copy it forward so the
    # corrected values are what gets merged.
    phases = _phase_status_snapshot(sb, status)
    if phases is None:
        return

    if phase_manifest is not None:
        manifest_changed = _reconcile_phases_from_manifest(phases, phase_manifest)
        if manifest_changed is not None:
            if manifest_changed:
                sb._patch.status["phases"] = phases
            return

    if not flags.has_metrics:
        return
    summary = MetricsSummary.from_metrics(result.metrics).data
    request_count = (summary.get("request_count") or {}).get("avg")
    if request_count is None:
        return
    results_phase = _select_single_results_phase(phases)
    if results_phase is None:
        return

    authoritative = int(request_count)
    if _repair_results_phase_counts(results_phase, authoritative):
        sb._patch.status["phases"] = phases


async def _apply_completion_results(
    *,
    body: dict[str, Any],
    namespace: str,
    jobset_name: str,
    job_id: str,
    result: ControllerFetchResult,
    sb: StatusBuilder,
    status: dict[str, Any],
    flags: _ResultFlags,
    key_names: KeyExportNames = DEFAULT_KEY_EXPORT_NAMES,
    parent_name: str | None = None,
    parent_uid: str | None = None,
) -> tuple[_ResultFlags, tuple[_KeyArtifactFingerprint, ...]]:
    """Stamp results/phase/condition + update index. Index is updated BEFORE
    ``sb.finalize()`` so its failure path can queue an INDEX_UPDATED=False
    condition without racing the single finalize() pass.
    """
    epoch = epoch_key_from_body(body)
    flags = _demote_missing_publication_artifacts(
        namespace=namespace,
        job_id=job_id,
        epoch=epoch,
        flags=flags,
        key_names=key_names,
    )
    phase_manifest = None
    if flags.success:
        phase_manifest = await asyncio.to_thread(
            _load_phase_manifest_payload,
            run_dir(OperatorEnvironment.RESULTS.DIR, namespace, job_id, epoch),
        )
        if _completion_cancelled(namespace, job_id, parent_uid):
            return flags, ()

    if not await _parent_identity_is_current(
        namespace,
        parent_name,
        parent_uid,
        context="results publication",
    ):
        return flags, ()

    flags, artifacts_materialized, artifact_fingerprint = (
        _capture_publication_artifacts(
            namespace=namespace,
            job_id=job_id,
            epoch=epoch,
            flags=flags,
            key_names=key_names,
        )
    )
    catalog_marker = (
        runs_index.begin_catalog_update(
            OperatorEnvironment.RESULTS.DIR, namespace, job_id, epoch
        )
        if artifacts_materialized
        else None
    )
    phase = "Succeeded" if flags.success else "Failed"
    terminal_error = (
        None
        if flags.success
        else (
            flags.benchmark_failure
            or result.error
            or "Failed to fetch complete result files from controller"
        )
    )
    _record_results_on_status(
        body=body,
        namespace=namespace,
        job_id=job_id,
        result=result,
        sb=sb,
        has_metrics=flags.has_metrics,
        has_files=flags.has_files,
        terminal_phase=phase,
        terminal_error=terminal_error,
        key_names=key_names,
        emit_event=False,
    )
    _reconcile_phase_counts_from_results(
        sb=sb,
        status=status,
        result=result,
        flags=flags,
        phase_manifest=phase_manifest,
    )
    # Retention lives here (not in the sync _record_results_on_status) so the
    # rmtree walk can run off-loop; the materialized gate mirrors the
    # latest.txt/runEpoch gate inside _record_results_on_status.
    if artifacts_materialized:
        if not await _parent_identity_is_current(
            namespace,
            parent_name,
            parent_uid,
            context="retention pass",
        ):
            return flags, artifact_fingerprint
        await _run_retention_pass(namespace, job_id, epoch)
        if _completion_cancelled(namespace, job_id, parent_uid):
            return flags, artifact_fingerprint
    flags, _ = await _verify_final_artifact_publication(
        namespace=namespace,
        job_id=job_id,
        epoch=epoch,
        flags=flags,
        expected_fingerprint=artifact_fingerprint,
        key_names=key_names,
        sb=sb,
    )
    summary_blob, mtime_epoch, end_time, total_size_bytes = _gather_index_inputs(
        namespace, job_id, epoch, json_name=key_names.json_name
    )
    # On the file-metrics path (API metrics empty but key exports present), feed
    # the index the same metrics ``_record_results_on_status`` stamped on the CR
    # so the narrow compare columns match status.summary / the on-disk JSON.
    # Without this, sub-second / CompletedBeforeMonitor jobs write all-NULL
    # narrow columns because result.metrics is None.
    if not flags.has_metrics and flags.has_files:
        index_metrics = _parse_metrics_from_files(
            result.downloaded,
            namespace,
            job_id,
            epoch=epoch,
            json_name=key_names.json_name,
        )
    else:
        index_metrics = result.metrics
    phase = "Succeeded" if flags.success else "Failed"
    index_updated = await _update_job_index_safe(
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
        key_names=key_names,
        parent_name=parent_name,
        parent_uid=parent_uid,
        emit_event=False,
    )
    flags, final_artifacts_materialized = await _verify_final_artifact_publication(
        namespace=namespace,
        job_id=job_id,
        epoch=epoch,
        flags=flags,
        expected_fingerprint=artifact_fingerprint,
        key_names=key_names,
        sb=sb,
    )
    if catalog_marker is not None and (
        not final_artifacts_materialized or index_updated
    ):
        runs_index.finish_catalog_update(catalog_marker)
    _set_results_phase_and_condition(
        body=body,
        jobset_name=jobset_name,
        result=result,
        sb=sb,
        has_metrics=flags.has_metrics,
        has_files=flags.has_files,
        has_error=flags.has_error,
        success=flags.success,
        benchmark_failure=flags.benchmark_failure,
        emit_event=False,
    )
    if _completion_cancelled(namespace, job_id, parent_uid):
        return flags, artifact_fingerprint
    return flags, artifact_fingerprint


def _demote_missing_publication_artifacts(
    *,
    namespace: str,
    job_id: str,
    epoch: str,
    flags: _ResultFlags,
    key_names: KeyExportNames,
) -> _ResultFlags:
    """Reject claimed artifacts that disappeared before publication starts."""
    if not flags.has_files or _key_files_materialized(
        namespace, job_id, epoch, key_names=key_names.names
    ):
        return flags
    logger.error(
        "Key exports for %s/%s disappeared before completion publication",
        namespace,
        job_id,
    )
    return replace(
        flags,
        has_files=False,
        success=False,
        benchmark_failure=None,
    )


def _capture_publication_artifacts(
    *,
    namespace: str,
    job_id: str,
    epoch: str,
    flags: _ResultFlags,
    key_names: KeyExportNames,
) -> tuple[_ResultFlags, bool, tuple[_KeyArtifactFingerprint, ...]]:
    """Capture the stable key-export snapshot before completion publication."""
    artifacts_materialized = flags.has_files and _key_files_materialized(
        namespace, job_id, epoch, key_names=key_names.names
    )
    fingerprint = _key_artifact_fingerprint(
        namespace, job_id, epoch, key_names=key_names.names
    )
    if not artifacts_materialized or fingerprint:
        return flags, artifacts_materialized, fingerprint
    return (
        _demote_missing_publication_artifacts(
            namespace=namespace,
            job_id=job_id,
            epoch=epoch,
            flags=flags,
            key_names=key_names,
        ),
        False,
        fingerprint,
    )


def _key_artifact_fingerprint(
    namespace: str,
    job_id: str,
    epoch: str,
    *,
    key_names: frozenset[str],
) -> tuple[_KeyArtifactFingerprint, ...]:
    """Return stable fingerprints for every valid JSON-or-CSV key export."""
    dest_dir = run_dir(OperatorEnvironment.RESULTS.DIR, namespace, job_id, epoch)
    fingerprints: list[_KeyArtifactFingerprint] = []
    for key in key_names:
        for candidate in (dest_dir / key, dest_dir / f"{key}.zst"):
            if not candidate.is_file() or not _key_artifact_valid(candidate):
                continue
            try:
                stat = candidate.stat()
            except OSError:
                continue
            if stat.st_size > 0:
                fingerprints.append(
                    _KeyArtifactFingerprint(
                        name=candidate.name,
                        size=stat.st_size,
                        mtime_ns=stat.st_mtime_ns,
                    )
                )
    return tuple(sorted(fingerprints, key=lambda fingerprint: fingerprint.name))


async def _verify_final_artifact_publication(
    *,
    namespace: str,
    job_id: str,
    epoch: str,
    flags: _ResultFlags,
    expected_fingerprint: tuple[_KeyArtifactFingerprint, ...],
    key_names: KeyExportNames,
    sb: StatusBuilder,
) -> tuple[_ResultFlags, bool]:
    """Fail closed when final artifacts vanish during index publication."""
    dest_dir = run_dir(OperatorEnvironment.RESULTS.DIR, namespace, job_id, epoch)
    final_artifacts_materialized = (
        flags.has_files
        and bool(expected_fingerprint)
        and _key_files_materialized(namespace, job_id, epoch, key_names=key_names.names)
        and _key_artifact_fingerprint(
            namespace, job_id, epoch, key_names=key_names.names
        )
        == expected_fingerprint
        and ready_marker_path(dest_dir).is_file()
    )
    if not flags.has_files or final_artifacts_materialized:
        return flags, final_artifacts_materialized

    logger.error(
        "Final result artifacts for %s/%s disappeared during publication; "
        "discarding terminal success state",
        namespace,
        job_id,
    )
    await _drop_index_row(namespace, job_id, epoch)
    ready_marker_path(dest_dir).unlink(missing_ok=True)
    if resolve_latest(OperatorEnvironment.RESULTS.DIR, namespace, job_id) == epoch:
        reconcile_latest(OperatorEnvironment.RESULTS.DIR, namespace, job_id)
    for field in ("results", "summary", "resultsPath", "runEpoch"):
        sb._patch.status.pop(field, None)
    return (
        replace(
            flags,
            has_files=False,
            success=False,
            benchmark_failure=None,
        ),
        False,
    )


def _gather_index_inputs(
    namespace: str,
    job_id: str,
    epoch: str,
    *,
    json_name: str = DEFAULT_KEY_EXPORT_NAMES.json_name,
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

    summary_zst = dest_dir / f"{json_name}.zst"
    summary_raw = dest_dir / json_name
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
    namespace: str,
    jobset_name: str,
    job_id: str,
    flags: _ResultFlags,
    *,
    parent_name: str | None = None,
    parent_uid: str | None = None,
) -> bool:
    """Delete the backing JobSet to free cluster resources once results are stored.

    Keep pods alive for retry on the next monitor tick if fetch failed or only
    partial/non-authoritative artifacts were available. Skip the delete on
    cancellation — K8s GC via ownerReferences will reap the JobSet.
    """
    if not flags.success or is_cancellation_requested(
        job_key(namespace, job_id, parent_uid)
    ):
        return True
    if parent_name is None or parent_uid is None:
        logger.warning(
            "Skipping unfenced completion JobSet delete for %s/%s",
            namespace,
            jobset_name,
        )
        return True
    return await _delete_backing_jobset(
        namespace,
        jobset_name,
        parent_name=parent_name,
        parent_uid=parent_uid,
    )


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
    # Set when the fetch succeeded but the benchmark itself failed (e.g. every
    # request errored). Distinct from has_error, which describes the fetch.
    benchmark_failure: str | None = None


@dataclass(frozen=True, slots=True)
class _KeyArtifactFingerprint:
    """Stable identity of one valid authoritative artifact on the operator PVC."""

    name: str
    size: int
    mtime_ns: int


def _demote_unmaterialized_result_files(
    *,
    body: dict[str, Any],
    namespace: str,
    job_id: str,
    flags: _ResultFlags,
    key_names: KeyExportNames,
) -> _ResultFlags:
    """Reject key filenames that were not durably materialized on the operator PVC.

    ``ControllerFetchResult.downloaded`` is transport metadata, not durability
    proof. Keeping ``has_files=False`` routes the result through the bounded
    transient-retry gate and prevents terminal publication or JobSet deletion
    until a custom-prefix-aware key export validates on disk.
    """
    if not flags.has_files:
        return flags
    epoch = epoch_key_from_body(body)
    if _key_files_materialized(
        namespace,
        job_id,
        epoch,
        key_names=key_names.names,
    ):
        return flags
    logger.warning(
        "Controller reported key exports for %s/%s, but no valid key artifact "
        "was materialized for run %s; preserving the JobSet for retry",
        namespace,
        job_id,
        epoch,
    )
    return replace(
        flags,
        has_files=False,
        success=False,
        benchmark_failure=None,
    )


def _compute_result_flags(
    result: ControllerFetchResult,
    job_id: str,
    *,
    key_names: KeyExportNames = DEFAULT_KEY_EXPORT_NAMES,
) -> _ResultFlags:
    """Derive has_metrics/has_files/has_error/success flags and log a summary.

    A partial fetch can set has_files=True but still populate result.error
    (e.g. checkpoints saved but key export files missing). Treat error as
    authoritative so a false-success Completed phase never overwrites the
    real failure signal.
    """
    has_metrics = bool(result.metrics and result.metrics.get("metrics"))
    has_files = _has_key_result_files(result.downloaded, key_names=key_names.names)
    has_error = bool(result.error)
    success = has_files and not has_error

    # Files existing is not a success signal. A run in which every request
    # errored still writes profile_export_aiperf.json, so keying success on
    # file presence alone reports Completed for a benchmark that measured
    # nothing. No mature orchestrator infers success from artifacts: Tekton
    # keys on step exit codes, KubeRay reads the application's own verdict, and
    # Argo's artifact check can only demote a run that already succeeded --
    # presence never promotes. Require an affirmative signal from the results.
    benchmark_failure: str | None = None
    if success:
        rate, errors, requests = _result_error_rate(result)
        if requests > 0 and rate >= K8sEnvironment.DIAGNOSIS.FAIL_ABOVE_ERROR_RATE:
            success = False
            benchmark_failure = (
                f"Benchmark failed: {rate:.1%} of requests errored "
                f"({errors}/{requests})"
            )

    logger.info(
        f"Results for {job_id}: has_metrics={has_metrics}, has_files={has_files}, "
        f"metrics_keys={list(result.metrics.keys()) if result.metrics else []}"
    )
    return _ResultFlags(
        has_metrics=has_metrics,
        has_files=has_files,
        has_error=has_error,
        success=success,
        benchmark_failure=benchmark_failure,
    )


def _result_error_rate(result: ControllerFetchResult) -> tuple[float, int, int]:
    """Return (error_rate, errors, requests) from the fetched final metrics.

    Mirrors ``aiperf.kubernetes.benchmark_diagnosis.error_rate`` but reads the
    authoritative post-run metrics rather than the sampled liveMetrics window,
    so the counts are exact instead of averaged across staggered samples.

    Returns ``(0.0, 0, 0)`` when metrics are absent, which callers must treat as
    "unknown", never as "no errors" -- a missing payload is not evidence of a
    healthy run.
    """
    metrics = (result.metrics or {}).get("metrics")
    if not isinstance(metrics, dict):
        return 0.0, 0, 0

    def _avg(key: str) -> float | None:
        entry = metrics.get(key, {})
        value = entry.get("avg", 0.0) if isinstance(entry, dict) else entry
        return float(value) if is_finite_value(value) else None

    requests = _avg("request_count")
    errors = _avg("error_count")
    if requests is None or errors is None or requests <= 0:
        return 0.0, 0, 0
    rate = min(1.0, max(0.0, errors / requests))
    return rate, int(errors), int(requests)


def _raise_if_phase_refresh_cancelled(cancellation_event: asyncio.Event) -> None:
    """Abort final progress sampling after a cooperative cancellation signal."""
    if cancellation_event.is_set():
        raise _FetchCancelled


async def _get_phase_refresh_client(
    key: str, cancellation_event: asyncio.Event
) -> ProgressClient:
    """Acquire the cached progress client without blocking cancellation."""
    _raise_if_phase_refresh_cancelled(cancellation_event)
    progress_client = await _await_or_cancel(
        get_or_create_progress_client(key), cancellation_event
    )
    _raise_if_phase_refresh_cancelled(cancellation_event)
    return progress_client


def _build_final_phase_sample(progress: JobProgress) -> dict[str, Any]:
    """Convert one controller progress response to the CR status shape."""
    from aiperf.operator.handlers.monitor import _build_phase_progress

    sample: dict[str, Any] = {}
    for phase, stats in progress.phases.items():
        if phase_progress := _build_phase_progress(stats):
            sample[phase] = phase_progress.to_k8s_dict()
    return sample


def _final_phases_still_settling(phases: dict[str, Any]) -> bool:
    """Return whether any final phase snapshot still reports unfinished work.

    The benchmark has already ended, so every phase should read complete.
    Records can trail the last request briefly, and the JobSet terminal event
    can arrive before the controller publishes its final request counts. Check
    both completion flags so a phase stuck at 284/300 is sampled again too.
    """
    return any(
        not (phase.get("isRequestsComplete") and phase.get("isRecordsComplete"))
        for phase in phases.values()
    )


async def _collect_final_phase_progress(
    *,
    progress_client: ProgressClient,
    host: str,
    cancellation_event: asyncio.Event,
) -> dict[str, Any]:
    """Collect the latest complete phase snapshot within the settle budget."""
    attempts = OperatorEnvironment.RESULTS.PHASE_SETTLE_ATTEMPTS
    phases_data: dict[str, Any] = {}
    for attempt in range(attempts + 1):
        progress = await _await_or_cancel(
            progress_client.get_progress(host), cancellation_event
        )
        _raise_if_phase_refresh_cancelled(cancellation_event)
        if progress.connection_error:
            break
        sample = _build_final_phase_sample(progress)
        if sample:
            phases_data = sample
        if (
            not sample
            or not _final_phases_still_settling(sample)
            or attempt == attempts
        ):
            break
        await _await_or_cancel(
            asyncio.sleep(OperatorEnvironment.RESULTS.PHASE_SETTLE_DELAY_SEC),
            cancellation_event,
        )
        _raise_if_phase_refresh_cancelled(cancellation_event)
    return phases_data


async def _refresh_final_phase_progress(
    *,
    namespace: str,
    jobset_name: str,
    job_id: str,
    patch: Any,
    expected_parent_uid: str | None = None,
) -> None:
    """Take one last progress sample while the controller is still reachable.

    ``status.phases`` mirrors the controller's live progress, sampled once per
    monitor tick. A benchmark shorter than the tick interval therefore freezes
    the CR at whatever the last tick saw, and the job reports Completed while
    its own phase counters say otherwise -- observed on a live 21s gemma sweep
    child: ``requestsCompleted 284/300`` and ``isRecordsComplete false`` on a
    run whose export contains all 300 records.

    Completion runs before the controller is torn down (results were just
    fetched from it), so one more sample here closes the gap. Best-effort:
    the counters are a mirror, and failing to refresh them must never fail an
    otherwise-successful completion.
    """
    try:
        key = job_key(namespace, job_id, expected_parent_uid)
        cancellation_event = get_cancellation_event(key)
        progress_client = await _get_phase_refresh_client(key, cancellation_event)
        host = controller_dns_name(jobset_name, namespace)
        phases_data = await _collect_final_phase_progress(
            progress_client=progress_client,
            host=host,
            cancellation_event=cancellation_event,
        )
        if phases_data:
            patch.status["phases"] = phases_data
    except _FetchCancelled:
        logger.info(
            "Cancellation interrupted final phase-progress refresh for %s/%s",
            namespace,
            job_id,
        )
    except Exception as e:  # noqa: BLE001 - mirror refresh must not fail completion
        logger.debug(
            "final phase-progress refresh unavailable for %s/%s: %s",
            namespace,
            job_id,
            e,
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
    terminal_phase: str | None = None,
    terminal_error: str | None = None,
    key_names: KeyExportNames = DEFAULT_KEY_EXPORT_NAMES,
    emit_event: bool = True,
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
            result.downloaded,
            namespace,
            job_id,
            epoch=epoch,
            json_name=key_names.json_name,
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

    if has_files and _key_files_materialized(
        namespace, job_id, epoch, key_names=key_names.names
    ):
        dest_dir = run_dir(OperatorEnvironment.RESULTS.DIR, namespace, job_id, epoch)
        write_ready_marker(
            dest_dir,
            terminal_phase=terminal_phase,
            terminal_error=terminal_error,
        )
        sb.set_results_path(str(dest_dir))
        write_latest(OperatorEnvironment.RESULTS.DIR, namespace, job_id, epoch)
        sb.set_run_epoch(int(epoch))
        if emit_event:
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
    benchmark_failure: str | None = None,
    emit_event: bool = True,
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

    if benchmark_failure is not None:
        # The fetch worked and the artifacts are on disk -- the benchmark itself
        # failed. Keep ResultsAvailable true so the run stays downloadable and
        # debuggable, but fail the job with a reason that names the real cause
        # rather than the fetch-failure text below.
        sb.set_phase(Phase.FAILED).set_error(benchmark_failure)
        sb.conditions.set_true(
            ConditionType.RESULTS_AVAILABLE,
            "ResultsStored",
            f"{len(result.downloaded)} result files stored; {benchmark_failure}",
        )
        if emit_event:
            events.failed(body, jobset_name, benchmark_failure)
        logger.error(f"{jobset_name}: {benchmark_failure}")
        return

    failure_msg = (
        result.error
        if has_error
        else "Failed to fetch complete result files from controller"
    )
    # set_error is required, not decorative. Without it _derive_terminal_conditions
    # falls back to `status.error or "Job failed"`, so .status.error is either empty
    # or -- worse -- still holds an unrelated controller-side error staged earlier in
    # this same tick by _fetch_progress, which the completion patch merge does not
    # clear. Every other FAILED path in the operator pairs set_phase with set_error.
    sb.set_phase(Phase.FAILED).set_error(failure_msg)
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
    if emit_event:
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
    key_names: KeyExportNames = DEFAULT_KEY_EXPORT_NAMES,
    parent_name: str | None = None,
    parent_uid: str | None = None,
    emit_event: bool = True,
) -> bool:
    """Update the runs_index; on failure, degrade gracefully.

    Results are already persisted to disk, so a failure here only affects
    discoverability via the index/history API - don't retry the whole
    completion handler, but set a status condition and event so operators
    can see the gap. Advances the in-DB latest pointer only when ``latest.txt``
    accepted this exact epoch.
    """
    if _completion_cancelled(namespace, job_id, parent_uid):
        return False

    if not await _parent_identity_is_current(
        namespace,
        parent_name,
        parent_uid,
        context="runs_index publication",
    ):
        return False

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
        if await _delete_index_row_if_inactive(
            namespace,
            job_id,
            epoch,
            parent_name=parent_name,
            parent_uid=parent_uid,
        ):
            return False
        # Only advance the in-DB latest pointer once the authoritative export
        # is materialized on disk. A row whose key files never landed must not
        # become the discoverable latest run (mirrors the latest.txt gate in
        # ``_record_results_on_status``).
        if (
            _key_files_materialized(namespace, job_id, epoch, key_names=key_names.names)
            and resolve_latest(OperatorEnvironment.RESULTS.DIR, namespace, job_id)
            == epoch
        ):
            await runs_index.set_latest(namespace, job_id, epoch)
            if await _delete_index_row_if_inactive(
                namespace,
                job_id,
                epoch,
                parent_name=parent_name,
                parent_uid=parent_uid,
            ):
                return False
    except Exception as e:
        if await _delete_index_row_if_inactive(
            namespace,
            job_id,
            epoch,
            parent_name=parent_name,
            parent_uid=parent_uid,
        ):
            return False
        logger.exception(f"Failed to update runs_index for {job_id}")
        sb.conditions.set_false(
            ConditionType.INDEX_UPDATED,
            "IndexUpdateFailed",
            f"Index write failed: {e}",
        )
        _emit_index_update_failed(body, str(e), enabled=emit_event)
        return False
    return True


def _emit_index_update_failed(
    body: dict[str, Any], error: str, *, enabled: bool
) -> None:
    """Emit an index failure immediately only for non-staged callers."""
    if enabled:
        events.index_update_failed(body, error)


async def _delete_index_row_if_cancelled(
    namespace: str,
    job_id: str,
    epoch: str,
    *,
    parent_uid: str | None = None,
) -> bool:
    """Compensate an index write that raced AIPerfJob deletion.

    The delete handler can finish its index sweep while a completion upsert is
    still awaiting SQLite. Rechecking after every index await and deleting the
    exact epoch here makes the later completion writer responsible for removing
    any row it may have recreated.
    """
    if not _completion_cancelled(namespace, job_id, parent_uid):
        return False
    await _drop_index_row(namespace, job_id, epoch)
    return True


async def _drop_index_row(namespace: str, job_id: str, epoch: str) -> None:
    """Best-effort compensation for a completion row that lost its CR identity."""
    try:
        await runs_index.delete_run(namespace, job_id, epoch)
        accepted_latest = resolve_latest(
            OperatorEnvironment.RESULTS.DIR, namespace, job_id
        )
        if (
            accepted_latest is not None
            and accepted_latest != epoch
            and await runs_index.get_run(namespace, job_id, accepted_latest) is not None
        ):
            await runs_index.set_latest(namespace, job_id, accepted_latest)
    except Exception:  # noqa: BLE001 - deletion remains best-effort when the index itself is unavailable
        logger.warning(
            "runs_index.delete_run failed for cancelled completion %s/%s/%s",
            namespace,
            job_id,
            epoch,
            exc_info=True,
        )


async def _delete_index_row_if_inactive(
    namespace: str,
    job_id: str,
    epoch: str,
    *,
    parent_name: str | None,
    parent_uid: str | None,
) -> bool:
    """Remove a row written after cancellation or same-name replacement."""
    if await _delete_index_row_if_cancelled(
        namespace, job_id, epoch, parent_uid=parent_uid
    ):
        return True
    if parent_name is None or parent_uid is None:
        return False
    try:
        await current_aiperfjob_resource_version(namespace, parent_name, parent_uid)
    except StaleAIPerfJobCallback as exc:
        logger.info("Removing stale runs_index publication: %s", exc)
        await _drop_index_row(namespace, job_id, epoch)
        return True
    return False


async def _delete_backing_jobset(
    namespace: str,
    jobset_name: str,
    *,
    parent_name: str | None = None,
    parent_uid: str | None = None,
) -> bool:
    deleted = await delete_owned_aiperfjob_jobset(
        namespace,
        jobset_name,
        parent_name=parent_name or jobset_name,
        parent_uid=parent_uid,
        context="results stored",
    )
    if deleted:
        logger.info(f"Deleted JobSet {jobset_name} after results stored")
    return deleted


def _parse_metrics_from_files(
    downloaded: list[str],
    namespace: str,
    job_id: str,
    *,
    epoch: str,
    json_name: str = DEFAULT_KEY_EXPORT_NAMES.json_name,
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

    for path in _metric_file_candidates(dest_dir, downloaded, json_name=json_name):
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


def _metric_file_candidates(
    dest_dir: Path,
    downloaded: list[str],
    *,
    json_name: str = DEFAULT_KEY_EXPORT_NAMES.json_name,
) -> list[Path]:
    """Return a de-duplicated, existence-checked, .zst-first candidate list."""
    candidates: list[Path] = [dest_dir / name for name in downloaded]
    candidates.extend([dest_dir / f"{json_name}.zst", dest_dir / json_name])
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
