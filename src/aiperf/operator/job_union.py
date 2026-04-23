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

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient

logger = logging.getLogger("aiperf.operator.job_union")

# Filename the operator writes once a run is persisted to the PVC. Its presence
# marks a directory as "has a real completed run" for the union.
_SUMMARY_FILE = "profile_export_aiperf.json"


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
) -> AIPerfJobInfo:
    """Build an archived ``AIPerfJobInfo`` from a summary JSON dict."""
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
        for job_dir in sorted(ns_dir.iterdir()):
            if not job_dir.is_dir():
                continue
            summary_path = job_dir / _SUMMARY_FILE
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
                    job_dir.name,
                    summary,
                    mtime_iso=mtime_iso,
                )
            )
    return out


# Map from profile_export nested metric path -> flat status.summary key.
# Each entry: (flat_key, summary_key, nested_field). A source of None
# (value absent or nested missing) means the key is OMITTED from the output,
# so downstream UI code that does ``summary.throughput_rps ?? null`` picks
# up the fall-through instead of a silently-null key.
_FLAT_METRIC_MAP: tuple[tuple[str, str, str], ...] = (
    ("throughput_rps", "request_throughput", "avg"),
    ("latency_avg_ms", "request_latency", "avg"),
    ("latency_p99_ms", "request_latency", "p99"),
    ("ttft_avg_ms", "time_to_first_token", "avg"),
    ("ttft_p99_ms", "time_to_first_token", "p99"),
    ("itl_avg_ms", "inter_token_latency", "avg"),
    ("itl_p99_ms", "inter_token_latency", "p99"),
    ("output_token_throughput_tps", "output_token_throughput", "avg"),
)


def _flat_summary_from_profile_export(summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten a profile_export summary into the camel-ish keys the UI reads.

    The UI's job-detail page reads KPIs from ``status.summary.{throughput_rps,
    ttft_avg_ms, latency_p99_ms, output_token_throughput_tps, error_rate,
    total_requests}``. Live CRs write these flat keys directly; archived jobs
    only have nested ``profile_export_aiperf.json`` metrics, which this helper
    translates. Keys whose source is missing/None are omitted.
    """
    flat: dict[str, Any] = {}
    for flat_key, nested_key, field in _FLAT_METRIC_MAP:
        nested = summary.get(nested_key)
        if not isinstance(nested, dict):
            continue
        value = nested.get(field)
        if value is None:
            continue
        flat[flat_key] = value

    request_count = summary.get("request_count")
    if request_count is not None:
        flat["total_requests"] = request_count
    # error_rate is always present in the UI's expected schema; default to 0.0
    # when the source summary is silent so cards don't render "---" for a
    # benchmark that simply had zero errors.
    flat["error_rate"] = summary.get("error_rate") or 0.0
    return flat


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
    with pods/workers empty. The flat ``status.summary.*`` keys (``throughput_rps``,
    ``latency_p99_ms``, ...) are derived from the nested ``profile_export_aiperf.json``
    metrics via :func:`_flat_summary_from_profile_export`.

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
        "summary": _flat_summary_from_profile_export(summary),
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
                    break
        else:
            out.append(pj)
    return out


async def find_any_job(
    api: ApiClient | None,
    results_dir: Path,
    namespace: str,
    name: str,
) -> AIPerfJobInfo | None:
    """Return the unified view of a single job or None if neither source has it.

    If both sources have it, CR wins on live fields and ``source="both"``.
    """
    cr: AIPerfJobInfo | None = None
    try:
        cr = await find_aiperf_job(api, name, namespace)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"find_aiperf_job failed, falling back to PVC: {e}")
        cr = None
    if cr is not None:
        cr.source = "live"

    summary_path = results_dir / namespace / name / _SUMMARY_FILE
    pvc: AIPerfJobInfo | None = None
    if summary_path.is_file():
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
            pvc = _archived_from_summary(
                namespace,
                name,
                data,
                mtime_iso=mtime_iso,
            )

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
    return cr
