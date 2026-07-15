# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified view of AIPerfJobs: cluster CRs + PVC result directories.

The operator UI treats "a job" as a single logical concept, but the data lives
in two planes:

1. Cluster CRs (ephemeral, live state: workers, pods, phase).
2. PVC result directories (persistent, historical state: metrics, config).

This module joins the two by `(namespace, name)` and stamps each entry with a
`source` field so callers can reason about provenance.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson

# Import at module level so tests can monkeypatch these bindings directly.
from aiperf.kubernetes.client import find_aiperf_job, list_aiperf_jobs
from aiperf.kubernetes.models import AIPerfJobInfo
from aiperf.operator._archived_stubs import archived_stub
from aiperf.operator.models import MetricsSummary
from aiperf.operator.results_layout import resolve_run_dir
from aiperf.operator.runs_index import zstd_decompress

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient

logger = logging.getLogger("aiperf.operator.job_union")

# Filename the operator writes once a run is persisted to the PVC. Its presence
# marks a directory as "has a real completed run" for the union.
_SUMMARY_FILE = "profile_export_aiperf.json"

# Marker filename the sweep-controller drops into each child's results dir at
# CR-create time. Persisting the sweep linkage on disk preserves the parent
# pointer after the AIPerfJob CR is TTL-reaped.
_SWEEP_MARKER_FILE = "sweep.json"


def _sweep_linkage_from_marker(
    job_dir: Path,
) -> tuple[str | None, int | None, str | None]:
    """Read sweep linkage from the per-child ``sweep.json`` marker.

    Returns ``(None, None, None)`` if the marker is absent or unreadable —
    this is the expected path for standalone (non-sweep) jobs.
    """
    marker = job_dir / _SWEEP_MARKER_FILE
    if not marker.is_file():
        return None, None, None
    try:
        doc = orjson.loads(marker.read_bytes())
    except (OSError, orjson.JSONDecodeError) as e:
        logger.warning(f"sweep.json unreadable at {marker}: {e}")
        return None, None, None
    sweep_name = doc.get("sweep_name") or None
    raw_idx = doc.get("variation_index")
    try:
        variation_index = int(raw_idx) if raw_idx is not None else None
    except (TypeError, ValueError):
        variation_index = None
    variation_label = doc.get("variation_label") or None
    return sweep_name, variation_index, variation_label


def _read_summary(path: Path) -> dict[str, Any] | None:
    """Load a profile export summary or return None if unreadable."""
    try:
        payload = path.read_bytes()
        if path.suffix == ".zst":
            payload = zstd_decompress(payload)
        return orjson.loads(payload)
    except (OSError, orjson.JSONDecodeError, ValueError) as e:
        logger.warning(f"Skipping unreadable summary {path}: {e}")
        return None


def _summary_path(run: Path) -> Path | None:
    for candidate in (run / f"{_SUMMARY_FILE}.zst", run / _SUMMARY_FILE):
        if candidate.is_file():
            return candidate
    return None


def _mtime_iso(path: Path) -> str:
    """Return the file's mtime as an ISO-8601 UTC timestamp with a Z suffix."""
    return (
        datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _kpi_fields_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Project the KPI fields the live ``AIPerfJobCR.to_info`` path emits.

    Returns a dict with ``ttft_ms``, ``output_token_throughput_tps``,
    ``inter_token_latency_ms``, ``total_requests``, ``error_rate``, all
    ``None`` when the corresponding metric / scalar is absent. The lookup
    shape matches live ``status.summary`` because we project through
    :class:`MetricsSummary` first.
    """
    projected = MetricsSummary.from_metrics(summary).to_status_dict()

    def _stat(tag: str, stat: str) -> float | None:
        entry = projected.get(tag)
        if not isinstance(entry, dict):
            return None
        val = entry.get(stat)
        return float(val) if isinstance(val, (int, float)) else None

    total_requests: int | None = None
    raw_total = projected.get("total_requests")
    if isinstance(raw_total, (int, float)):
        total_requests = int(raw_total)
    if total_requests is None:
        rc = _stat("request_count", "avg")
        if rc is not None:
            total_requests = int(rc)

    error_rate: float | None = None
    raw_err = projected.get("error_rate")
    if isinstance(raw_err, (int, float)):
        error_rate = float(raw_err)

    return {
        "ttft_ms": _stat("time_to_first_token", "avg"),
        "output_token_throughput_tps": _stat("output_token_throughput", "avg"),
        "inter_token_latency_ms": _stat("inter_token_latency", "avg"),
        "total_requests": total_requests,
        "error_rate": error_rate,
    }


def _archived_from_summary(
    namespace: str,
    name: str,
    summary: dict[str, Any],
    *,
    mtime_iso: str,
    name_dir: Path | None = None,
) -> AIPerfJobInfo:
    """Build an archived ``AIPerfJobInfo`` from a summary JSON dict.

    When ``name_dir`` is supplied, also reads ``sweep.json`` from that
    directory to populate sweep linkage for archived sweep children whose
    parent CR has been TTL-reaped.
    """
    phase = str(summary.get("status") or "Archived")
    start_time = summary.get("start_time") or None
    end_time = summary.get("end_time") or None

    rt = summary.get("request_throughput") or {}
    throughput = rt.get("avg") if isinstance(rt, dict) else None
    lat = summary.get("request_latency") or {}
    latency_p99 = lat.get("p99") if isinstance(lat, dict) else None

    kpi = _kpi_fields_from_summary(summary)

    ic = summary.get("input_config") or {}
    models = (ic.get("models") or {}).get("items") or []
    model: str | None = None
    if models:
        first = models[0]
        model = first.get("name") if isinstance(first, dict) else first
    endpoint = ic.get("endpoint") or {}
    urls = endpoint.get("urls") or []
    endpoint_url = urls[0] if urls else None

    sweep_name: str | None = None
    variation_index: int | None = None
    variation_label: str | None = None
    if name_dir is not None:
        sweep_name, variation_index, variation_label = _sweep_linkage_from_marker(
            name_dir
        )

    return AIPerfJobInfo(
        name=name,
        namespace=namespace,
        phase=phase,
        job_id=name,
        workers_ready=0,
        workers_total=0,
        current_phase=None,
        error=None,
        start_time=start_time,
        completion_time=end_time,
        created=start_time or mtime_iso,
        progress_percent=100.0,
        throughput_rps=float(throughput) if throughput is not None else None,
        latency_p99_ms=float(latency_p99) if latency_p99 is not None else None,
        ttft_ms=kpi["ttft_ms"],
        output_token_throughput_tps=kpi["output_token_throughput_tps"],
        inter_token_latency_ms=kpi["inter_token_latency_ms"],
        total_requests=kpi["total_requests"],
        error_rate=kpi["error_rate"],
        model=model,
        endpoint=endpoint_url,
        source="archived",
        sweep_name=sweep_name,
        variation_index=variation_index,
        variation_label=variation_label,
    )


def _scan_pvc_jobs(
    base_dir: Path,
    *,
    namespace: str | None = None,
) -> list[AIPerfJobInfo]:
    """Walk ``<base>/<ns>/<job>/profile_export_aiperf.json`` and build entries.

    Skips namespaces other than ``namespace`` if supplied; skips dirs that
    lack a summary JSON; logs + skips unreadable summaries.
    """
    if not base_dir.exists() or not base_dir.is_dir():
        return []

    out: list[AIPerfJobInfo] = []
    for ns_dir in sorted(base_dir.iterdir()):
        if not ns_dir.is_dir():
            continue
        if namespace is not None and ns_dir.name != namespace:
            continue
        for name_dir in sorted(ns_dir.iterdir()):
            if not name_dir.is_dir():
                continue
            run = resolve_run_dir(base_dir, ns_dir.name, name_dir.name)
            if run is None:
                continue
            summary_path = _summary_path(run)
            if summary_path is None:
                continue
            summary = _read_summary(summary_path)
            if summary is None:
                continue
            out.append(
                _archived_from_summary(
                    ns_dir.name,
                    name_dir.name,
                    summary,
                    mtime_iso=_mtime_iso(summary_path),
                    name_dir=name_dir,
                )
            )
    return out


# Map from profile_export nested metric path -> flat status.summary key.
# Each entry: (flat_key, summary_key, nested_field). A source of None
# (value absent or nested missing) means the key is OMITTED from the output,
# so downstream UI code that does ``summary.throughput_rps ?? null`` picks
# up the fall-through instead of a silently-null key.
def _summary_from_profile_export(summary: dict[str, Any]) -> dict[str, Any]:
    """Project a curated nested metric view from a profile_export summary.

    ``profile_export_aiperf.json`` stores AIPerf metric tags
    (``request_throughput``, ``request_latency``, ...) at the top level of
    the dict. This helper passes that through ``MetricsSummary.from_metrics``
    so archived jobs land in ``status.summary`` with the same nested
    ``{tag: {avg, p50, p99, ...}}`` shape as live CRs.
    """
    return MetricsSummary.from_metrics(summary).to_status_dict()


def synthesize_status_from_summary(
    namespace: str,
    name: str,
    summary: dict[str, Any],
    conditions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a ``status``-shaped dict for an archived (PVC-only) job.

    The returned dict matches the schema a live CR's ``status`` subresource
    exposes (jobId, phase, workers, conditions, phases, summary, timestamps),
    so the UI can consume it through the same code path as live jobs — just
    with pods/workers empty. ``status.summary`` is a curated nested
    ``{metric_tag: {avg, p50, p99, ...}}`` projection of the
    ``profile_export_aiperf.json`` payload via :func:`_summary_from_profile_export`.

    Args:
        namespace: Kubernetes namespace of the original job (retained for
            symmetry with the live-CR caller; not currently embedded).
        name: AIPerfJob name — written into ``status.jobId``.
        summary: The parsed ``profile_export_aiperf.json`` dict.
        conditions: Optional list of condition dicts to pass through verbatim.
            When omitted, a single ``{type: <phase>, status: "True"}`` entry
            is synthesized from ``summary["status"]``.

    Returns:
        A dict with keys ``jobId, phase, startTime, completionTime,
        currentPhase, workers, conditions, phases, summary``. Archived is
        always past-completion, so ``currentPhase`` is hardcoded to
        ``"completed"`` and the synthetic ``phases.benchmark`` entry reports
        100 % progress.
    """
    del namespace  # reserved for future use
    phase = str(summary.get("status") or "Archived")
    if conditions is None:
        conditions = [{"type": phase, "status": "True"}]

    # Request count for the phases bar. ``profile_export_aiperf.json``
    # carries ``request_count`` as a metric dict ``{unit, avg}`` (the
    # canonical MetricsSummary shape); older / hand-rolled summaries
    # may carry it as a bare int. Accept both.
    rc = summary.get("request_count")
    if isinstance(rc, dict):
        rc = rc.get("avg") or rc.get("count") or 0
    try:
        request_count = int(rc or 0)
    except (TypeError, ValueError):
        request_count = 0

    return {
        "jobId": name,
        "phase": phase,
        "startTime": summary.get("start_time"),
        "completionTime": summary.get("end_time"),
        "currentPhase": "completed",
        "workers": {"ready": 0, "total": 0},
        "conditions": conditions,
        "phases": {
            "benchmark": {
                "requestsCompleted": request_count,
                "requestsTotal": request_count,
                "requestsProgressPercent": 100,
            }
        },
        "summary": _summary_from_profile_export(summary),
    }


def _parse_sort_ts(ts: str | None) -> float:
    if not ts:
        return 0.0
    candidate = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
    try:
        return datetime.fromisoformat(candidate).astimezone(UTC).timestamp()
    except ValueError:
        return 0.0


def _backfill_cr_from_archived(cr: AIPerfJobInfo, pvc: AIPerfJobInfo) -> None:
    """Promote a live CR entry to ``source="both"`` and backfill PVC-only fields.

    Mutates ``cr`` in place. The CR wins on every field it already carries
    (it has live worker/phase data); the archived half only fills the
    historical fields the CR is silent about.
    """
    cr.source = "both"
    if cr.throughput_rps is None:
        cr.throughput_rps = pvc.throughput_rps
    if cr.latency_p99_ms is None:
        cr.latency_p99_ms = pvc.latency_p99_ms
    if cr.model is None:
        cr.model = pvc.model
    if cr.endpoint is None:
        cr.endpoint = pvc.endpoint
    if cr.sweep_name is None:
        cr.sweep_name = pvc.sweep_name
    if cr.variation_index is None:
        cr.variation_index = pvc.variation_index
    if cr.variation_label is None:
        cr.variation_label = pvc.variation_label


async def list_all_jobs(
    api: ApiClient | None,
    results_dir: Path,
    *,
    all_namespaces: bool = True,
    namespace: str | None = None,
) -> list[AIPerfJobInfo]:
    """Return the union of cluster CRs and PVC result directories.

    Keyed by (namespace, name). Overlap entries are tagged ``source="both"``
    using the CR's values as the base (it has live worker/phase data) and
    letting PVC fields through only where the CR doesn't already carry them.
    """
    cr_jobs: list[AIPerfJobInfo] = []
    try:
        cr_jobs = await list_aiperf_jobs(
            api,
            all_namespaces=all_namespaces,
            namespace=namespace,
        )
    except Exception as e:  # noqa: BLE001 - broad by design: PVC still usable
        logger.warning(f"list_aiperf_jobs failed, continuing PVC-only: {e}")
        cr_jobs = []
    # Freshly stamped source=live (even though the default is live, make it
    # explicit so the "both" promotion below is unambiguous).
    for j in cr_jobs:
        j.source = "live"

    # Full PVC walk (iterdir/stat/read per run dir) — pure filesystem work,
    # so offload it; the UI polls this endpoint every few seconds and a
    # synchronous scan would stall the event loop on large PVCs.
    pvc_jobs = await asyncio.to_thread(_scan_pvc_jobs, results_dir, namespace=namespace)

    cr_keys = {(j.namespace, j.name) for j in cr_jobs}
    out: list[AIPerfJobInfo] = list(cr_jobs)
    for pj in pvc_jobs:
        key = (pj.namespace, pj.name)
        if key in cr_keys:
            for cj in out:
                if (cj.namespace, cj.name) == key:
                    _backfill_cr_from_archived(cj, pj)
                    break
        else:
            out.append(pj)
    return sorted(out, key=lambda j: _parse_sort_ts(j.created), reverse=True)


async def _find_live_cr(
    api: ApiClient | None, namespace: str, name: str
) -> AIPerfJobInfo | None:
    """Look up the live CR half of a job, returning None on lookup failure."""
    try:
        cr = await find_aiperf_job(api, name, namespace)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"find_aiperf_job failed, falling back to PVC: {e}")
        return None
    if cr is not None:
        cr.source = "live"
    return cr


def _find_archived_job(
    results_dir: Path,
    namespace: str,
    name: str,
    epoch: str | None,
) -> AIPerfJobInfo | None:
    """Load the archived (PVC) half of a job, or None when absent/unreadable."""
    run = resolve_run_dir(results_dir, namespace, name, epoch=epoch)
    if run is None:
        return None
    summary_path = _summary_path(run)
    if summary_path is None:
        if epoch is not None and epoch != "latest":
            # Pinned epoch dir exists on disk but has no profile_export summary;
            # ``/epochs`` still lists this epoch (see ``_archived_stubs``).
            return archived_stub(namespace, name, run_dir=run, name_dir=run.parent)
        return None
    data = _read_summary(summary_path)
    if data is None:
        return None
    # ``run`` is the epoch-specific dir; the sweep marker lives at the
    # per-name root one level up since sweep linkage is fixed for a
    # given child name (not per-epoch). ``run.parent`` resolves to
    # ``<results_dir>/<ns>/<name>`` for any epoch, latest or pinned.
    return _archived_from_summary(
        namespace,
        name,
        data,
        mtime_iso=_mtime_iso(summary_path),
        name_dir=run.parent,
    )


async def find_any_job(
    api: ApiClient | None,
    results_dir: Path,
    namespace: str,
    name: str,
    *,
    epoch: str | None = None,
) -> AIPerfJobInfo | None:
    """Return the unified view of a single job or None if neither source has it.

    If both sources have it, CR wins on live fields and ``source="both"``.

    When ``epoch`` is supplied, the archived half is pinned to
    ``<results_dir>/<ns>/<name>/<epoch>/profile_export_aiperf.json`` rather
    than the ``latest.txt`` pointer, and the live CR half is dropped — the
    live CR always reflects the *current* run, so merging it into a request
    for a historical epoch would conflate epochs. ``epoch`` of ``"latest"``
    or ``None`` falls through to ``latest.txt`` (legacy behavior).
    """
    cr = await _find_live_cr(api, namespace, name)
    pvc = _find_archived_job(results_dir, namespace, name, epoch)

    # Caller asked for a specific historical epoch: never merge the live CR.
    if epoch is not None and epoch != "latest":
        return pvc

    if cr is None:
        return pvc
    if pvc is None:
        return cr
    # Both present: backfill missing CR fields from PVC.
    _backfill_cr_from_archived(cr, pvc)
    return cr
