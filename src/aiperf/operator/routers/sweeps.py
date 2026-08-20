# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI router for /api/v1/sweeps* — read-only AIPerfSweep view.

Dual-backed via :mod:`aiperf.operator.sweep_union`: every endpoint
returns the same shape regardless of whether the parent CR exists or
the data is reconstructed from the archived ``aggregate.json``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import orjson
from fastapi import APIRouter, HTTPException, Request
from fastapi.params import Depends as DependsParam
from fastapi.responses import Response
from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException
from pydantic import ValidationError
from zstandard import ZstdError

from aiperf.common.endpoint_credentials import (
    redact_sweep_display_label,
    redact_sweep_public_data,
)
from aiperf.common.redact import redact_endpoint_spec
from aiperf.common.results_markers import EPOCH_RE
from aiperf.kubernetes.models import AIPerfJobInfo
from aiperf.operator.job_union import list_all_jobs
from aiperf.operator.results_layout import (
    list_sweep_epochs_async,
    resolve_sweep_dir,
)
from aiperf.operator.routers._etag import etag_response
from aiperf.operator.routers._path_params import validate_results_path_params
from aiperf.operator.routers._sweeps_artifacts import register_sweep_artifact_routes
from aiperf.operator.routers._sweeps_diagnostics import (
    fetch_sweep_pod_summaries,
    register_diagnostics_routes,
)
from aiperf.operator.routers._sweeps_live import children_manifest_from_live_aiperfjobs
from aiperf.operator.routers._sweeps_spec import (
    spec_summary_from_record,
)
from aiperf.operator.routers.sweeps_models import (
    MAX_BEST_TRIALS,
    CellAggregatesResponse,
    CellEntry,
    ChildJobRef,
    ChildrenManifestEntry,
    ChildrenManifestResponse,
    CreateSweepRequest,
    CreateSweepResponse,
    SearchBestTrial,
    SearchBoundaryEdge,
    SearchBoundarySummary,
    SearchObjective,
    SearchSLABreach,
    SweepDetailResponse,
    SweepEpochsResponse,
    SweepEpochSummary,
    SweepListResponse,
    SweepSearchSummary,
    SweepSummary,
)
from aiperf.operator.runs_index import zstd_decompress
from aiperf.operator.sweep_union import (
    SweepRecord,
    find_any_sweep,
    list_all_sweeps,
    sanitize_current_child_ref,
    sanitize_run_states,
    synthesize_sweep_status_from_aggregate,
)

logger = logging.getLogger("aiperf.operator.ui")


def _summary(rec: SweepRecord) -> SweepSummary:
    return SweepSummary(
        namespace=rec.namespace,
        name=rec.name,
        source=rec.source,  # type: ignore[arg-type]
        phase=rec.phase,
        total_variations=rec.total_variations,
        completed_runs=rec.completed_runs,
        failed_runs=rec.failed_runs,
        cancelled_runs=rec.cancelled_runs,
        age_seconds=rec.age_seconds,
        model=rec.model,
        started_at=rec.started_at,
        completed_at=rec.completed_at,
        api_url=rec.api_url,
        results_available=rec.results_available,
        current_child_ref=redact_sweep_public_data(
            sanitize_current_child_ref(rec.current_child_ref)
        ),
        run_states=sanitize_run_states(rec.run_states),
    )


def _read_conditions(sweep_dir_path: str | None) -> list[dict[str, Any]]:
    if not sweep_dir_path:
        return []
    p = Path(sweep_dir_path).parent / "conditions.json"
    if not p.is_file():
        return []
    try:
        raw = orjson.loads(p.read_bytes())
    except (OSError, orjson.JSONDecodeError) as e:
        logger.warning("conditions.json unreadable at %s: %s", p, e)
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("conditions"), list):
        return raw["conditions"]
    return []


SEARCH_HISTORY_FILENAME = "search_history.json"

# Stop reasons carry planner-specific strings that keep growing (each new 1-D
# SLA planner adds its own). Classifying them once here keeps every client off
# the business of pattern-matching a moving enum.
#
# "unknown" is the orchestrator's clean-terminal fallback: ``ask()`` returned
# None -- the planner DID decide to stop -- but recorded no structured reason.
# That is a convergence, just an unlabelled one, so it maps to "converged"
# rather than to "incomplete", which is reserved for a null reason (mid-loop
# write, cancellation, or crash).
_BUDGET_EXHAUSTED_REASON = "max_iterations"


def _classify_stop(convergence_reason: str | None) -> str:
    if convergence_reason is None:
        return "incomplete"
    if convergence_reason == _BUDGET_EXHAUSTED_REASON:
        return "budget_exhausted"
    return "converged"


def _as_float(raw: Any) -> float | None:
    """Coerce a JSON number to float, or None for anything non-numeric."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


def _as_int(raw: Any) -> int | None:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return int(raw)


def _breach_from_doc(raw: Any) -> SearchSLABreach | None:
    if not isinstance(raw, dict):
        return None
    return SearchSLABreach(
        metric_tag=raw.get("metric_tag") or None,
        stat=raw.get("stat") or None,
        op=raw.get("op") or None,
        threshold=_as_float(raw.get("threshold")),
        observed=_as_float(raw.get("observed")),
    )


def _boundary_edge_from_doc(raw: Any) -> SearchBoundaryEdge | None:
    if not isinstance(raw, dict):
        return None
    return SearchBoundaryEdge(
        value=_as_float(raw.get("value")),
        iteration_idx=_as_int(raw.get("iteration_idx")),
        objective_value=_as_float(raw.get("objective_value")),
        first_breach=_breach_from_doc(raw.get("first_breach")),
    )


def _boundary_from_doc(raw: Any) -> SearchBoundarySummary | None:
    """Project ``boundary_summary``; None when the block is absent or unusable.

    ``swept_dim_path`` is the one required key -- a boundary with no axis to
    name is not renderable, so drop the whole block rather than emit a
    half-labelled one.
    """
    if not isinstance(raw, dict):
        return None
    swept = raw.get("swept_dim_path")
    if not isinstance(swept, str) or not swept:
        return None
    boundary_type = raw.get("boundary_type")
    binding = raw.get("binding_constraint")
    return SearchBoundarySummary(
        swept_dim_path=swept,
        feasible_max=_boundary_edge_from_doc(raw.get("feasible_max")),
        infeasible_min=_boundary_edge_from_doc(raw.get("infeasible_min")),
        boundary_type=boundary_type if isinstance(boundary_type, str) else None,
        binding_constraint=binding if isinstance(binding, str) else None,
    )


def _objectives_from_doc(raw: Any) -> list[SearchObjective]:
    """Project ``config.objectives``, normalizing direction case.

    The artifact writes ``OptimizationDirection.name`` (``"MAXIMIZE"``) while
    ``SpecSummary.objectives`` writes the enum value (``"maximize"``); the same
    response would otherwise carry both spellings.
    """
    out: list[SearchObjective] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        metric = item.get("metric")
        if not isinstance(metric, str) or not metric:
            continue
        direction = item.get("direction")
        out.append(
            SearchObjective(
                metric=metric,
                stat=str(item.get("stat") or "avg"),
                direction=str(direction or "maximize").lower(),
            )
        )
    return out


def _count_sla_filters(raw: Any) -> int:
    """Count configured SLA filters; zero makes every feasibility flag vacuous.

    ``write_search_history`` defaults each iteration's ``feasible`` to true when
    nothing constrains it, so without this count a client cannot tell "passed
    every SLA" from "there were no SLAs".
    """
    if not isinstance(raw, list):
        return 0
    return sum(1 for item in raw if isinstance(item, dict))


def _best_trials_from_doc(raw: Any) -> tuple[list[SearchBestTrial], bool]:
    """Project ``best_trials``; returns the (possibly truncated) list + flag."""
    trials: list[SearchBestTrial] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        item = redact_sweep_public_data(item)
        idx = _as_int(item.get("iteration_idx"))
        if idx is None:
            continue
        values = item.get("objective_values")
        # Nulls are LOAD-BEARING and must survive the projection. The exporter
        # writes them deliberately -- scrub_non_finite maps a NaN score to null
        # so the artifact keeps "the scorer returned NaN for this objective"
        # distinct from "this iteration was never scored"
        # (exporters/search_history.py:135-138). ``objective_values`` is
        # positional against ``objectives``, so dropping one entry silently
        # relabels every later value: on a two-objective run whose second score
        # was NaN, objective #2's slot renders objective #1's number.
        objective_values = (
            [_as_float(x) for x in values] if isinstance(values, list) else None
        )
        variation_values = item.get("variation_values")
        trials.append(
            SearchBestTrial(
                iteration_idx=idx,
                objective_values=objective_values,
                variation_values=variation_values
                if isinstance(variation_values, dict)
                else {},
                feasible=bool(item.get("feasible")),
                feasible_count=_as_int(item.get("feasible_count")) or 0,
                pareto_rank=_as_int(item.get("pareto_rank")) or 0,
            )
        )
    truncated = len(trials) > MAX_BEST_TRIALS
    return trials[:MAX_BEST_TRIALS], truncated


def _search_summary_from_doc(doc: dict[str, Any]) -> SweepSearchSummary:
    """Project a parsed ``search_history.json`` into the response model."""
    doc = redact_sweep_public_data(doc)
    iterations = doc.get("iterations")
    iterations = iterations if isinstance(iterations, list) else []
    config = doc.get("config")
    config = config if isinstance(config, dict) else {}
    reason = doc.get("convergence_reason")
    reason = reason if isinstance(reason, str) and reason else None
    best_trials, truncated = _best_trials_from_doc(doc.get("best_trials"))
    recipe = doc.get("recipe")
    return SweepSearchSummary(
        convergence_reason=reason,
        stop_kind=_classify_stop(reason),  # type: ignore[arg-type]
        iteration_count=len(iterations),
        feasible_iteration_count=sum(
            1 for it in iterations if isinstance(it, dict) and it.get("feasible")
        ),
        objectives=_objectives_from_doc(config.get("objectives")),
        best_trials=best_trials,
        best_trials_truncated=truncated,
        sla_filter_count=_count_sla_filters(config.get("sla_filters")),
        boundary_summary=_boundary_from_doc(doc.get("boundary_summary")),
        recipe=recipe if isinstance(recipe, str) and recipe else None,
    )


def _read_search_summary(aggregate_path: str | None) -> SweepSearchSummary | None:
    """Read the adaptive trajectory sitting beside a sweep epoch's aggregate.

    The epoch dir is derived from ``aggregate_path`` (not re-resolved from
    ``latest.txt``) for the same reason :func:`_read_conditions` does it: the
    rest of the response describes exactly that epoch, and a re-running sweep
    whose current epoch has not been harvested yet must not be captioned with
    the *previous* run's trajectory.

    ``search_history.json`` is written by the sweep-controller into its own
    pod-local emptyDir and only reaches the operator's PVC once
    ``handlers/sweep/_aggregate_fetch.fetch_sweep_aggregate_to_disk`` harvests
    it, so this is a cheap pair of ``is_file`` misses for the whole live phase
    of a sweep and one small parse afterwards. That is far below the cost the
    route already pays for ``list_all_jobs``, whose PVC half walks every run
    directory in the namespace (``job_union.py:481``).

    Absent for every grid-family sweep and for adaptive sweeps that died before
    the first write; unreadable when a torn write is observed (the exporter
    does a bare ``write_bytes`` with no temp-and-rename). All of those degrade
    to ``None`` -- per-file resilience in the shape of
    ``handlers/completion.py:_parse_metrics_from_files``.
    """
    if not aggregate_path:
        return None
    epoch_dir = Path(aggregate_path).parent
    for candidate in (
        epoch_dir / f"{SEARCH_HISTORY_FILENAME}.zst",
        epoch_dir / SEARCH_HISTORY_FILENAME,
    ):
        if not candidate.is_file():
            continue
        try:
            payload = candidate.read_bytes()
            if candidate.suffix == ".zst":
                payload = zstd_decompress(payload)
            doc = orjson.loads(payload)
        except (OSError, ValueError, orjson.JSONDecodeError, ZstdError) as e:
            logger.warning("search_history unreadable at %s: %s", candidate, e)
            continue
        if not isinstance(doc, dict):
            logger.warning("search_history at %s is not a JSON object", candidate)
            continue
        try:
            return _search_summary_from_doc(doc)
        except (ValidationError, TypeError, ValueError) as e:
            logger.warning("search_history at %s rejected: %s", candidate, e)
            continue
    return None


async def _list_sweeps_impl(api: ApiClient, base_dir: Path) -> SweepListResponse:
    records = await list_all_sweeps(api, base_dir, all_namespaces=True)
    return SweepListResponse(sweeps=[_summary(r) for r in records])


async def _get_sweep_impl(
    api: ApiClient,
    base_dir: Path,
    namespace: str,
    name: str,
    *,
    epoch: str | None = None,
) -> SweepDetailResponse:
    rec = await find_any_sweep(api, base_dir, namespace, name, epoch=epoch)
    if rec is None:
        raise HTTPException(404, f"Sweep {namespace}/{name} not found")

    if rec.source == "archived" and rec.aggregate_doc is not None:
        status = synthesize_sweep_status_from_aggregate(
            namespace, name, rec.aggregate_doc, _read_conditions(rec.aggregate_path)
        )
    elif rec.source == "archived":
        status = {"phase": "Unknown", "conditions": []}
    else:
        status = rec.raw_status or {}
    status = redact_sweep_public_data(status)

    spec_summary = spec_summary_from_record(rec)

    children_records = (
        []
        if epoch is not None
        else await list_all_jobs(
            api, base_dir, all_namespaces=False, namespace=namespace
        )
    )
    children = redact_sweep_public_data(
        [
            j.model_dump(by_alias=True)
            for j in children_records
            if getattr(j, "sweep_name", None) == name and j.namespace == namespace
        ]
    )

    pods = await fetch_sweep_pod_summaries(api, namespace, name, rec.source)

    return SweepDetailResponse(
        sweep=_summary(rec),
        status=status,
        spec_summary=spec_summary,
        search_summary=await asyncio.to_thread(
            _read_search_summary, rec.aggregate_path
        ),
        children=children,
        pods=pods,
    )


def _cells_from_aggregate(doc: dict[str, Any]) -> list[CellEntry]:
    raw_cells = doc.get("per_cell_aggregates") or []
    out: list[CellEntry] = []
    for c in raw_cells:
        if not isinstance(c, dict):
            continue
        c = redact_sweep_public_data(c)
        children_raw = c.get("children") or []
        children = [
            ChildJobRef(
                namespace=child.get("namespace") or "",
                name=child.get("name") or "",
                trial_index=child.get("trial_index"),
                phase=child.get("phase"),
            )
            for child in children_raw
            if isinstance(child, dict)
        ]
        out.append(
            CellEntry(
                variation_index=int(c.get("variation_index") or 0),
                variation_label=str(c.get("variation_label") or ""),
                values=dict(c.get("values") or {}),
                trials_completed=int(c.get("trials_completed") or 0),
                trials_failed=int(c.get("trials_failed") or 0),
                metrics=dict(c.get("metrics") or {}),
                children=children,
            )
        )
    return sorted(out, key=lambda x: x.variation_index)


def _fold_child_into_bucket(bucket: dict[str, Any], job: AIPerfJobInfo) -> None:
    """Fold one child job into its variation bucket (counts, metrics, refs).

    Status mapping: only count terminal children towards aggregates.
    """
    phase = (job.phase or "").lower()
    if phase in {"succeeded", "completed"}:
        bucket["trials_completed"] += 1
        if job.throughput_rps is not None:
            bucket["throughputs"].append(float(job.throughput_rps))
        if job.latency_p99_ms is not None:
            bucket["p99_latencies"].append(float(job.latency_p99_ms))
    elif phase in {"failed", "cancelled", "partiallyfailed"}:
        bucket["trials_failed"] += 1
    bucket["children"].append(
        ChildJobRef(
            namespace=job.namespace,
            name=job.name,
            trial_index=None,
            phase=job.phase,
        )
    )


def _avg(xs: list[float]) -> float | None:
    """Arithmetic mean of ``xs``, or None when the list is empty."""
    return (sum(xs) / len(xs)) if xs else None


def _cell_entry_from_bucket(idx: int, bucket: dict[str, Any]) -> CellEntry:
    """Materialize the response-facing CellEntry for one variation bucket."""
    metrics: dict[str, dict[str, float]] = {}
    thr_avg = _avg(bucket["throughputs"])
    if thr_avg is not None:
        metrics["request_throughput"] = {"avg": thr_avg}
    lat_avg = _avg(bucket["p99_latencies"])
    if lat_avg is not None:
        metrics["request_latency_p99"] = {"avg": lat_avg}
    return CellEntry(
        variation_index=idx,
        variation_label=redact_sweep_display_label(bucket["variation_label"]),
        values={},  # structured values come from spec; live path leaves empty
        trials_completed=bucket["trials_completed"],
        trials_failed=bucket["trials_failed"],
        metrics=metrics,
        children=bucket["children"],
    )


async def _cells_from_live_children(
    api: ApiClient,
    base_dir: Path,
    namespace: str,
    sweep_name: str,
) -> list[CellEntry]:
    """Compute per-cell aggregates by grouping children by variation_index.

    Used when the sweep is live and has no aggregate.json yet (mid-run).
    Reads each child's profile_export_aiperf.json from the PVC if present.
    Returns an empty list if no terminal children are persisted yet.
    """
    children_records = await list_all_jobs(
        api, base_dir, all_namespaces=False, namespace=namespace
    )
    matched = [
        j
        for j in children_records
        if getattr(j, "sweep_name", None) == sweep_name and j.namespace == namespace
    ]
    by_cell: dict[int, dict[str, Any]] = {}
    for j in matched:
        idx = getattr(j, "variation_index", None)
        if idx is None:
            continue
        bucket = by_cell.setdefault(
            int(idx),
            {
                "variation_label": getattr(j, "variation_label", "") or "",
                "trials_completed": 0,
                "trials_failed": 0,
                "throughputs": [],
                "p99_latencies": [],
                "children": [],
            },
        )
        _fold_child_into_bucket(bucket, j)

    return [_cell_entry_from_bucket(idx, b) for idx, b in sorted(by_cell.items())]


async def _get_cells_impl(
    api: ApiClient,
    base_dir: Path,
    namespace: str,
    name: str,
    *,
    epoch: str | None = None,
) -> CellAggregatesResponse:
    rec = await find_any_sweep(api, base_dir, namespace, name, epoch=epoch)
    if rec is None:
        raise HTTPException(404, f"Sweep {namespace}/{name} not found")
    spec_summary = spec_summary_from_record(rec)
    if rec.aggregate_doc is not None:
        cells = _cells_from_aggregate(rec.aggregate_doc)
        source = rec.source
    elif epoch is None:
        cells = await _cells_from_live_children(api, base_dir, namespace, name)
        source = "live"
    else:
        cells = []
        source = rec.source
    return CellAggregatesResponse(
        dimensions=spec_summary.dimensions,
        cells=cells,
        source=source,  # type: ignore[arg-type]
    )


async def _list_sweep_epochs_impl(
    base_dir: Path, namespace: str, name: str
) -> SweepEpochsResponse:
    runs = await list_sweep_epochs_async(base_dir, namespace, name)
    return SweepEpochsResponse(
        epochs=[
            SweepEpochSummary(
                epoch=r.epoch,
                is_latest=r.is_latest,
                mtime_epoch=r.mtime_epoch,
                file_count=r.file_count,
            )
            for r in runs
        ]
    )


def _children_manifest_from_doc(
    doc: dict[str, Any], epoch: str | None
) -> ChildrenManifestResponse:
    """Build a ChildrenManifestResponse from a ``children.json``-shaped dict.

    Accepts the disk envelope ``{"sweep_run_epoch": "...", "children": [...]}``
    that is also embedded verbatim into ``status.aggregate.children`` on the
    live CR — both the live (CR) and archived (PVC) read paths converge on
    this shape.
    """
    doc = redact_sweep_public_data(doc)
    return ChildrenManifestResponse(
        sweep_run_epoch=str(doc.get("sweep_run_epoch") or epoch or ""),
        children=[
            ChildrenManifestEntry(
                namespace=c.get("namespace", ""),
                name=c.get("name", ""),
                variation_index=int(c.get("variation_index") or 0),
                variation_label=c.get("variation_label") or "",
                variation_values=str(c.get("variation_values") or ""),
                trial_index=c.get("trial_index"),
                child_run_epoch=str(c.get("child_run_epoch") or ""),
            )
            for c in (doc.get("children") or [])
            if isinstance(c, dict)
        ],
    )


async def _get_children_impl(
    api: ApiClient,
    base_dir: Path,
    *,
    namespace: str,
    name: str,
    epoch: str | None,
) -> ChildrenManifestResponse:
    """Resolve the per-epoch children manifest, preferring the live CR.

    The sweep-controller writes ``children.json`` to its own pod-local emptyDir
    *and* embeds the same envelope at ``status.aggregate.children`` on the
    parent AIPerfSweep CR. The operator pod reading this route does NOT
    share a PVC with the controller pod, so the on-disk file is invisible
    here for live sweeps — the CR is the only source the operator can
    actually observe. Read the CR first; fall back to disk only for
    archived (post-TTL) sweeps where the CR is gone but the controller's
    PVC was promoted to a shared archive.

    Returns 404 only when neither half has data — the prior disk-only
    implementation 404'd every live sweep regardless of CR state.
    """
    if epoch is None:
        rec = await find_any_sweep(api, base_dir, namespace, name)
        if rec is not None and rec.raw_status:
            aggregate = rec.raw_status.get("aggregate")
            if isinstance(aggregate, dict):
                children_doc = aggregate.get("children")
                if isinstance(children_doc, dict) and isinstance(
                    children_doc.get("children"), list
                ):
                    return _children_manifest_from_doc(children_doc, epoch=epoch)

        if rec is not None:
            live = await children_manifest_from_live_aiperfjobs(api, namespace, name)
            if live is not None:
                return live

    sweep_dir = resolve_sweep_dir(base_dir, namespace, name, epoch=epoch)
    if sweep_dir is None:
        raise HTTPException(
            404, f"Sweep epoch not found: {namespace}/{name} epoch={epoch}"
        )
    p = sweep_dir / "children.json"
    if not p.is_file():
        raise HTTPException(
            404, f"children.json missing for {namespace}/{name} epoch={epoch}"
        )
    try:
        doc = orjson.loads(p.read_bytes())
    except (OSError, orjson.JSONDecodeError) as e:
        raise HTTPException(503, f"children.json unreadable: {e}") from e
    return _children_manifest_from_doc(doc, epoch=epoch)


def _register_sweep_read_routes(
    router: APIRouter,
    require_api: Callable[[], ApiClient],
    base_dir: Path,
) -> None:
    """Register the sweep list, detail, and epoch-listing endpoints."""

    @router.get("/sweeps", response_model=SweepListResponse)
    async def list_sweeps(request: Request) -> Response:
        result = await _list_sweeps_impl(require_api(), base_dir)
        return etag_response(request, result.model_dump(mode="json", by_alias=True))

    @router.get("/sweeps/{namespace}/{name}", response_model=SweepDetailResponse)
    async def get_sweep(
        request: Request, namespace: str, name: str, epoch: str | None = None
    ) -> Response:
        validate_results_path_params(namespace, name)
        if epoch is not None and not EPOCH_RE.match(epoch):
            raise HTTPException(400, f"Invalid epoch: {epoch!r}")
        result = await _get_sweep_impl(
            require_api(), base_dir, namespace, name, epoch=epoch
        )
        return etag_response(request, result.model_dump(mode="json", by_alias=True))

    @router.get(
        "/sweeps/{namespace}/{name}/epochs",
        response_model=SweepEpochsResponse,
        response_model_by_alias=True,
    )
    async def list_sweep_epochs_endpoint(
        namespace: str, name: str
    ) -> SweepEpochsResponse:
        validate_results_path_params(namespace, name)
        return await _list_sweep_epochs_impl(base_dir, namespace, name)


def _register_sweep_cell_routes(
    router: APIRouter,
    require_api: Callable[[], ApiClient],
    base_dir: Path,
) -> None:
    """Register the per-cell aggregate and children-manifest endpoints."""

    @router.get(
        "/sweeps/{namespace}/{name}/cells",
        response_model=CellAggregatesResponse,
    )
    async def get_sweep_cells(
        namespace: str, name: str, epoch: str | None = None
    ) -> CellAggregatesResponse:
        validate_results_path_params(namespace, name)
        if epoch is not None and not EPOCH_RE.match(epoch):
            raise HTTPException(400, f"Invalid epoch: {epoch!r}")
        return await _get_cells_impl(
            require_api(), base_dir, namespace, name, epoch=epoch
        )

    @router.get(
        "/sweeps/{namespace}/{name}/children",
        response_model=ChildrenManifestResponse,
        response_model_by_alias=True,
    )
    async def get_sweep_children(
        namespace: str, name: str, epoch: str | None = None
    ) -> ChildrenManifestResponse:
        validate_results_path_params(namespace, name)
        if epoch is not None and not EPOCH_RE.match(epoch):
            raise HTTPException(400, f"Invalid epoch: {epoch!r}")
        return await _get_children_impl(
            require_api(), base_dir, namespace=namespace, name=name, epoch=epoch
        )


async def _create_sweep_impl(
    api: ApiClient,
    manifest: dict[str, Any],
) -> CreateSweepResponse:
    """Body of POST /api/v1/sweeps: create an AIPerfSweep CR from a manifest dict."""
    if not isinstance(manifest, dict):
        raise HTTPException(400, "Manifest must be a JSON/YAML object.")

    manifest = dict(manifest)
    manifest.setdefault("apiVersion", "aiperf.nvidia.com/v1alpha1")
    manifest.setdefault("kind", "AIPerfSweep")
    metadata = manifest.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise HTTPException(400, "metadata must be an object.")
    name = metadata.get("name")
    if not name:
        raise HTTPException(400, "metadata.name is required.")
    namespace = metadata.get("namespace") or "default"
    metadata["namespace"] = namespace
    manifest["metadata"] = metadata

    co = client.CustomObjectsApi(api)
    try:
        created = await co.create_namespaced_custom_object(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            namespace=namespace,
            plural="aiperfsweeps",
            body=manifest,
        )
    except ApiException as e:
        detail = e.body or e.reason or "Kubernetes API error"
        raise HTTPException(e.status or 500, detail) from e

    uid = (created.get("metadata") or {}).get("uid")
    return CreateSweepResponse(namespace=namespace, name=name, uid=uid)


def _register_sweep_mutating_routes(
    router: APIRouter,
    require_api: Callable[[], ApiClient],
    base_dir: Path,
    mutating_dependencies: Sequence[DependsParam],
) -> None:
    """Register POST /sweeps (create) and GET /sweeps/{ns}/{name}/config."""

    @router.get(
        "/sweeps/{namespace}/{name}/config",
        response_model=None,
    )
    async def get_sweep_config(
        namespace: str, name: str, epoch: str | None = None
    ) -> dict[str, Any]:
        validate_results_path_params(namespace, name)
        rec = await find_any_sweep(
            require_api(), base_dir, namespace, name, epoch=epoch
        )
        spec = rec.raw_spec if rec is not None else None
        if epoch is not None and rec is not None and rec.aggregate_doc is not None:
            spec = rec.aggregate_doc.get("specSnapshot")
        if rec is None or not isinstance(spec, dict):
            raise HTTPException(
                404, f"Sweep {namespace}/{name} not found or spec unavailable."
            )
        return {
            "apiVersion": "aiperf.nvidia.com/v1alpha1",
            "kind": "AIPerfSweep",
            "metadata": {"name": rec.name, "namespace": rec.namespace},
            "spec": redact_endpoint_spec(spec),
        }

    @router.post(
        "/sweeps",
        response_model=CreateSweepResponse,
        status_code=201,
        dependencies=list(mutating_dependencies),
    )
    async def create_sweep(body: CreateSweepRequest) -> CreateSweepResponse:
        return await _create_sweep_impl(require_api(), body.manifest)


def create_sweeps_router(
    api_holder: list[ApiClient | None] | None = None,
    results_dir: Path | None = None,
    mutating_dependencies: Sequence[DependsParam] = (),
) -> APIRouter:
    """Build the sweeps router. Mirrors :func:`create_jobs_router`'s shape."""
    _holder = api_holder if api_holder is not None else [None]
    _base_dir = results_dir if results_dir is not None else Path("/data")
    router = APIRouter(prefix="/api/v1", tags=["sweeps"])

    def _require_api() -> ApiClient:
        api = _holder[0] if _holder else None
        if api is None:
            raise HTTPException(
                503,
                "Kubernetes API client not yet initialized by FastAPI lifespan; "
                "retry in a few seconds or check /healthz",
            )
        return api

    _register_sweep_read_routes(router, _require_api, _base_dir)
    register_sweep_artifact_routes(router, _base_dir)
    _register_sweep_cell_routes(router, _require_api, _base_dir)
    register_diagnostics_routes(router, _require_api)
    _register_sweep_mutating_routes(
        router, _require_api, _base_dir, mutating_dependencies
    )

    return router
