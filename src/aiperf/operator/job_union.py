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
from typing import Any

import orjson

from aiperf.kubernetes.models import AIPerfJobInfo

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
    namespace: str, name: str, summary: dict[str, Any], *, mtime_iso: str,
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
    base_dir: Path, *, namespace: str | None = None,
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
            mtime_iso = _dt.datetime.fromtimestamp(
                summary_path.stat().st_mtime, tz=_dt.timezone.utc,
            ).isoformat().replace("+00:00", "Z")
            out.append(
                _archived_from_summary(
                    ns_dir.name, job_dir.name, summary, mtime_iso=mtime_iso,
                )
            )
    return out
