# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified live + archived view over AIPerfSweep records.

Mirrors :mod:`aiperf.operator.job_union` for sweeps. Live state comes
from the apiserver; archived state comes from
``<results_dir>/<ns>/sweeps/<name>/aggregate.json`` which is written
by the sweep-controller at terminal phase. Records are joined by
``(namespace, name)`` and tagged ``source = "live" | "archived" | "both"``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson

from aiperf.kubernetes.client import find_aiperfsweep, list_aiperfsweeps
from aiperf.operator.results_layout import (
    epoch_key_seconds,
    resolve_sweep_dir,
    resolve_sweep_latest,
)

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient

logger = logging.getLogger("aiperf.operator.sweep_union")

_AGGREGATE_FILE = "aggregate.json"


@dataclass
class SweepRecord:
    namespace: str
    name: str
    source: str  # Literal["live", "archived", "both"]
    phase: str
    total_variations: int
    completed_runs: int
    failed_runs: int
    age_seconds: int
    model: str | None
    aggregate_path: str | None = None
    raw_status: dict[str, Any] = field(default_factory=dict)
    raw_spec: dict[str, Any] = field(default_factory=dict)
    aggregate_doc: dict[str, Any] | None = None
    # Extended sweep-status fields. ``cancelled_runs`` is its own bucket, not
    # rolled into ``failed_runs`` — UI surfaces it separately so user-
    # cancelled children don't masquerade as failures.
    cancelled_runs: int = 0
    # Pointer to the in-flight child for live drill-down (``status.currentChildRef``).
    current_child_ref: dict[str, Any] | None = None
    # Stamped on the AIPerfSweep alongside terminal transitions.
    started_at: str | None = None
    completed_at: str | None = None
    # Operator base URL for cross-process result fetches.
    api_url: str | None = None
    results_available: bool = False
    # Per-state run counts rolled up from children (``status.runStates``).
    # Keyed by run-state bucket: pending / running / completed / failed / cancelled.
    run_states: dict[str, int] = field(default_factory=dict)


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


def _parse_creation_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    # Accept both ``2026-05-04T14:23:45Z`` and ``2026-05-04T14:23:45.123456Z``;
    # archived ``aggregate.json`` writes microsecond-precision timestamps and
    # the strict "%S" parser rejects them, leaving every archived sweep at age=0.
    candidate = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
    try:
        return datetime.fromisoformat(candidate).astimezone(UTC)
    except ValueError:
        return None


def _age_seconds(created: datetime | None) -> int:
    if created is None:
        return 0
    delta = (_now_utc() - created).total_seconds()
    return max(0, int(delta))


def _first_model_name(models: Any) -> str | None:
    """Return the first model name from any CRD-accepted model shape."""
    if isinstance(models, str):
        return models or None
    if isinstance(models, dict):
        items = models.get("items") or models.get("modelNames") or []
    elif isinstance(models, list):
        items = models
    else:
        return None
    if not items:
        return None
    first = items[0]
    if isinstance(first, dict):
        return first.get("name")
    if isinstance(first, str):
        return first or None
    return None


def _model_from_spec(spec: dict[str, Any]) -> str | None:
    """Resolve a model from the actual ``spec.benchmark`` workload shape."""
    benchmark = spec.get("benchmark") or {}
    if not isinstance(benchmark, dict):
        return None
    return _first_model_name(benchmark.get("models")) or _first_model_name(
        benchmark.get("model")
    )


def sanitize_run_states(raw: Any) -> dict[str, int]:
    """Return numeric run-state counters from an untrusted status payload."""
    if not isinstance(raw, dict):
        return {}
    return {
        k: int(v)
        for k, v in raw.items()
        if isinstance(k, str) and isinstance(v, (int, float))
    }


def sanitize_current_child_ref(raw: Any) -> dict[str, Any] | None:
    """Return a current-child reference only when all display fields exist."""
    if not isinstance(raw, dict):
        return None
    if not {"name", "index", "label"}.issubset(raw):
        return None
    return raw


def _record_from_live(cr: dict[str, Any]) -> SweepRecord:
    meta = cr.get("metadata") or {}
    spec = cr.get("spec") or {}
    status = cr.get("status") or {}
    created = _parse_creation_ts(meta.get("creationTimestamp"))
    run_states = sanitize_run_states(status.get("runStates"))
    return SweepRecord(
        namespace=meta.get("namespace") or "",
        name=meta.get("name") or "",
        source="live",
        phase=str(status.get("phase") or "Unknown"),
        total_variations=int(status.get("totalVariations") or 0),
        completed_runs=int(status.get("completedRuns") or 0),
        failed_runs=int(status.get("failedRuns") or 0),
        cancelled_runs=int(run_states.get("cancelled") or 0),
        age_seconds=_age_seconds(created),
        model=_model_from_spec(spec),
        aggregate_path=None,
        raw_status=status,
        raw_spec=spec,
        aggregate_doc=None,
        current_child_ref=sanitize_current_child_ref(status.get("currentChildRef")),
        started_at=status.get("startedAt") or None,
        completed_at=status.get("completedAt") or None,
        api_url=status.get("apiUrl") or None,
        results_available=bool(status.get("resultsAvailable")),
        run_states=run_states,
    )


def _read_aggregate_doc(path: Path) -> dict[str, Any] | None:
    try:
        return orjson.loads(path.read_bytes())
    except (OSError, orjson.JSONDecodeError) as e:
        logger.warning("aggregate.json unreadable at %s: %s", path, e)
        return None


def _record_from_archive(
    namespace: str, name: str, sweep_dir: Path
) -> SweepRecord | None:
    """Build a SweepRecord from an on-disk archive epoch dir.

    ``sweep_dir`` MUST be the per-epoch dir
    (``<base>/<ns>/sweeps/<name>/<epoch>/``), not the per-name root.
    Callers resolve the epoch via :func:`resolve_sweep_dir` (explicit
    epoch) or :func:`resolve_sweep_latest` (latest pointer).
    """
    agg_path = sweep_dir / _AGGREGATE_FILE
    if not agg_path.is_file():
        return None
    doc = _read_aggregate_doc(agg_path)
    if doc is None:
        # Surface as Unknown so corrupt sweeps still appear and operators see them.
        try:
            mtime = datetime.fromtimestamp(agg_path.stat().st_mtime, tz=UTC)
        except OSError:
            mtime = None
        return SweepRecord(
            namespace=namespace,
            name=name,
            source="archived",
            phase="Unknown",
            total_variations=0,
            completed_runs=0,
            failed_runs=0,
            age_seconds=_age_seconds(mtime),
            model=None,
            aggregate_path=str(agg_path),
            aggregate_doc=None,
        )
    completed_at = _parse_creation_ts(doc.get("completedAt"))
    started_at = _parse_creation_ts(doc.get("startedAt"))
    # The directory name is an epoch KEY, not a POSIX timestamp: it may carry a
    # six-digit suffix (see results_layout.epoch_key_from_body). Reading it raw
    # made one suffixed directory raise "year 56594345 is out of range" and 422
    # the entire sweeps list, older sweeps included.
    epoch_seconds = epoch_key_seconds(sweep_dir.name)
    epoch_at = (
        datetime.fromtimestamp(epoch_seconds, tz=UTC)
        if epoch_seconds is not None
        else None
    )
    age_anchor = completed_at or started_at or epoch_at or mtime
    run_states = sanitize_run_states(doc.get("runStates"))
    return SweepRecord(
        namespace=namespace,
        name=name,
        source="archived",
        phase=str(doc.get("phase") or "Archived"),
        total_variations=int(doc.get("totalVariations") or 0),
        completed_runs=int(doc.get("completedRuns") or 0),
        failed_runs=int(doc.get("failedRuns") or 0),
        cancelled_runs=int(run_states.get("cancelled") or 0),
        age_seconds=_age_seconds(age_anchor),
        model=doc.get("model") or _model_from_spec(doc.get("specSnapshot") or {}),
        aggregate_path=str(agg_path),
        aggregate_doc=doc,
        started_at=doc.get("startedAt") or None,
        completed_at=doc.get("completedAt") or None,
        results_available=True,
        run_states=run_states,
    )


def _latest_archived_record(
    base_dir: Path, ns_name: str, sweep_dir: Path
) -> SweepRecord | None:
    """Build the latest-epoch archived record for one sweep dir, if resolvable.

    The list page is latest-only: sweeps without a latest pointer (or whose
    pointed-at epoch dir is missing) return None and are skipped — cluster
    operators must run the wipe script first.
    """
    latest = resolve_sweep_latest(base_dir, ns_name, sweep_dir.name)
    if latest is None:
        return None
    epoch_dir = sweep_dir / latest
    if not epoch_dir.is_dir():
        return None
    return _record_from_archive(ns_name, sweep_dir.name, epoch_dir)


def _scan_archived(base_dir: Path, namespace: str | None = None) -> list[SweepRecord]:
    if not base_dir.exists() or not base_dir.is_dir():
        return []
    out: list[SweepRecord] = []
    for ns_dir in sorted(base_dir.iterdir()):
        if not ns_dir.is_dir():
            continue
        if namespace is not None and ns_dir.name != namespace:
            continue
        sweeps_root = ns_dir / "sweeps"
        if not sweeps_root.is_dir():
            continue
        for sweep_dir in sorted(sweeps_root.iterdir()):
            if not sweep_dir.is_dir():
                continue
            rec = _latest_archived_record(base_dir, ns_dir.name, sweep_dir)
            if rec is not None:
                out.append(rec)
    return out


def _merge(live: list[SweepRecord], archived: list[SweepRecord]) -> list[SweepRecord]:
    by_key: dict[tuple[str, str], SweepRecord] = {}
    for rec in archived:
        by_key[(rec.namespace, rec.name)] = rec
    for live_rec in live:
        key = (live_rec.namespace, live_rec.name)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = live_rec
            continue
        # Both sources present: the live CR is authoritative for the
        # currently-tracked run, so its numeric counters and run-state map
        # are taken unconditionally — a re-running sweep reports 0 completed
        # and must NOT fall back to a previous archived run's counts (that
        # produced impossible completedRuns > totalVariations). Archived only
        # backfills None-able identity/string fields the live half omits.
        merged = SweepRecord(
            namespace=live_rec.namespace,
            name=live_rec.name,
            source="both",
            phase=live_rec.phase or existing.phase,
            total_variations=live_rec.total_variations,
            completed_runs=live_rec.completed_runs,
            failed_runs=live_rec.failed_runs,
            cancelled_runs=live_rec.cancelled_runs,
            age_seconds=live_rec.age_seconds or existing.age_seconds,
            model=live_rec.model or existing.model,
            aggregate_path=existing.aggregate_path,
            raw_status=live_rec.raw_status,
            raw_spec=live_rec.raw_spec,
            aggregate_doc=existing.aggregate_doc,
            current_child_ref=live_rec.current_child_ref,
            started_at=live_rec.started_at or existing.started_at,
            completed_at=live_rec.completed_at or existing.completed_at,
            api_url=live_rec.api_url or existing.api_url,
            results_available=live_rec.results_available or existing.results_available,
            run_states=live_rec.run_states,
        )
        by_key[key] = merged
    return sorted(by_key.values(), key=lambda r: r.age_seconds)


async def list_all_sweeps(
    api: ApiClient,
    base_dir: Path,
    *,
    namespace: str | None = None,
    all_namespaces: bool = True,
) -> list[SweepRecord]:
    """Return the joined live + archived sweep view, source-tagged."""
    try:
        live_crs = await list_aiperfsweeps(
            api, namespace=namespace, all_namespaces=all_namespaces
        )
    except Exception as e:  # noqa: BLE001 — list endpoint is best-effort like jobs
        logger.warning("list_aiperfsweeps failed; live half empty: %s", e)
        live_crs = []
    live = [_record_from_live(cr) for cr in live_crs]
    # Full archive walk (iterdir + aggregate.json reads) — pure filesystem
    # work, so offload it off the event loop; the UI polls this per request.
    archived = await asyncio.to_thread(_scan_archived, base_dir, namespace=namespace)
    return _merge(live, archived)


async def find_any_sweep(
    api: ApiClient,
    base_dir: Path,
    namespace: str,
    name: str,
    *,
    epoch: str | None = None,
) -> SweepRecord | None:
    """Resolve a single sweep across live and archived state.

    When ``epoch`` is given, the historical aggregate at
    ``<base>/<ns>/sweeps/<name>/<epoch>/aggregate.json`` is returned
    without mixing in the live CR (a per-epoch view of a finished run).
    When ``epoch`` is ``None``, ``latest.txt`` is consulted and the
    archived half (if any) is merged with the live CR. Returns
    ``None`` when neither half is present.
    """
    cr = await find_aiperfsweep(api, namespace, name)
    archive_dir = resolve_sweep_dir(base_dir, namespace, name, epoch=epoch)
    archived = (
        _record_from_archive(namespace, name, archive_dir)
        if archive_dir is not None
        else None
    )
    if cr is None and archived is None:
        return None
    if epoch is not None:
        # Historical lookup: ignore the live half — the CR describes the
        # current/most-recent run, not the requested epoch.
        return archived
    if cr is None:
        return archived
    live = _record_from_live(cr)
    if archived is None:
        return live
    return _merge([live], [archived])[0]


def synthesize_sweep_status_from_aggregate(
    namespace: str,
    name: str,
    aggregate: dict[str, Any],
    conditions: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Build a status-shaped dict from an archived sweep's aggregate.json.

    Returns a dict the UI consumes the same way as a live ``.status`` subresource.
    """
    return {
        "phase": str(aggregate.get("phase") or "Archived"),
        "totalVariations": int(aggregate.get("totalVariations") or 0),
        "completedRuns": int(aggregate.get("completedRuns") or 0),
        "failedRuns": int(aggregate.get("failedRuns") or 0),
        "maxTotalRuns": int(aggregate.get("maxTotalRuns") or 0),
        "completedAt": aggregate.get("completedAt"),
        "conditions": conditions or [],
        "aggregateRef": aggregate.get("aggregateRef"),
    }
