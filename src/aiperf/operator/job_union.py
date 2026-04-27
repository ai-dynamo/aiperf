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

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson

# Import at module level so tests can monkeypatch these bindings directly.
from aiperf.kubernetes.client import find_aiperf_job, list_aiperf_jobs
from aiperf.kubernetes.models import AIPerfJobInfo
from aiperf.operator.models import MetricsSummary
from aiperf.operator.results_layout import resolve_run_dir

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
    """Load a ``profile_export_aiperf.json`` or return None if unreadable."""
    try:
        return orjson.loads(path.read_bytes())
    except (OSError, orjson.JSONDecodeError) as e:
        logger.warning(f"Skipping unreadable summary {path}: {e}")
        return None


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
    import datetime as _dt

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
            summary_path = run / _SUMMARY_FILE
            if not summary_path.is_file():
                continue
            summary = _read_summary(summary_path)
            if summary is None:
                continue
            mtime_iso = (
                _dt.datetime.fromtimestamp(
                    summary_path.stat().st_mtime,
                    tz=_dt.timezone.utc,
                )
                .isoformat()
                .replace("+00:00", "Z")
            )
            out.append(
                _archived_from_summary(
                    ns_dir.name,
                    name_dir.name,
                    summary,
                    mtime_iso=mtime_iso,
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

    # Request count for the phases bar: prefer explicit, else fall back to 0.
    request_count = int(summary.get("request_count") or 0)

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

    pvc_jobs = _scan_pvc_jobs(results_dir, namespace=namespace)

    cr_keys = {(j.namespace, j.name) for j in cr_jobs}
    out: list[AIPerfJobInfo] = list(cr_jobs)
    for pj in pvc_jobs:
        key = (pj.namespace, pj.name)
        if key in cr_keys:
            # Promote the matching CR entry to source="both" and backfill any
            # historical-only fields the CR is silent about.
            for cj in out:
                if (cj.namespace, cj.name) == key:
                    cj.source = "both"
                    if cj.throughput_rps is None:
                        cj.throughput_rps = pj.throughput_rps
                    if cj.latency_p99_ms is None:
                        cj.latency_p99_ms = pj.latency_p99_ms
                    if cj.model is None:
                        cj.model = pj.model
                    if cj.endpoint is None:
                        cj.endpoint = pj.endpoint
                    if cj.sweep_name is None:
                        cj.sweep_name = pj.sweep_name
                    if cj.variation_index is None:
                        cj.variation_index = pj.variation_index
                    if cj.variation_label is None:
                        cj.variation_label = pj.variation_label
                    break
        else:
            out.append(pj)
    return out


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
    cr: AIPerfJobInfo | None = None
    try:
        cr = await find_aiperf_job(api, name, namespace)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"find_aiperf_job failed, falling back to PVC: {e}")
        cr = None
    if cr is not None:
        cr.source = "live"

    run = resolve_run_dir(results_dir, namespace, name, epoch=epoch)
    summary_path = run / _SUMMARY_FILE if run is not None else None
    pvc: AIPerfJobInfo | None = None
    if summary_path is not None and summary_path.is_file():
        import datetime as _dt

        data = _read_summary(summary_path)
        if data is not None:
            mtime_iso = (
                _dt.datetime.fromtimestamp(
                    summary_path.stat().st_mtime,
                    tz=_dt.timezone.utc,
                )
                .isoformat()
                .replace("+00:00", "Z")
            )
            # ``run`` is the epoch-specific dir; the sweep marker lives at the
            # per-name root one level up since sweep linkage is fixed for a
            # given child name (not per-epoch). ``run.parent`` resolves to
            # ``<results_dir>/<ns>/<name>`` for any epoch, latest or pinned.
            pvc = _archived_from_summary(
                namespace,
                name,
                data,
                mtime_iso=mtime_iso,
                name_dir=run.parent,
            )

    # Caller asked for a specific historical epoch: never merge the live CR.
    if epoch is not None and epoch != "latest":
        return pvc

    if cr is None and pvc is None:
        return None
    if cr is None:
        return pvc
    if pvc is None:
        return cr
    # Both present: backfill missing CR fields from PVC.
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
    return cr
