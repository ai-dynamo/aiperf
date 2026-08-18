# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep-controller pod entry point.

Reads its target AIPerfSweep CR from the apiserver, builds a BenchmarkPlan,
runs MultiRunOrchestrator with K8sChildJobExecutor, runs aggregate_and_export
once all variations are done, publishes terminal status, and exits.

Idempotent: a restart re-reads the CR, sees existing terminal children
(ownerRef + label match), and resumes from the first non-existent variation.
If a complete epoch bundle is already marked ready, the controller republishes
status directly without replaying children or rebuilding temporary artifacts.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson

from aiperf.common.endpoint_credentials import (
    redact_sweep_display_label,
    redact_sweep_public_data,
)
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.jobset import controller_dns_name
from aiperf.kubernetes.results_artifacts import API_RESULTS_FILES_PATH
from aiperf.kubernetes.results_sidecar import READY_MARKER_NAME, write_ready_marker

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkPlan
    from aiperf.config.sweep import AdaptiveSearchSweep
    from aiperf.orchestrator.models import RunResult
    from aiperf.sweep_controller.k8s_executor import K8sChildJobExecutor
    from aiperf.sweep_controller.status_writer import SweepStatusWriter

logger = logging.getLogger(__name__)

AGGREGATE_READY_MARKER = READY_MARKER_NAME
RESULTS_DIR = Path("/results")
AGGREGATE_SUBDIR = "aggregate"
SWEEP_AGGREGATE_SUBDIR = "sweep_aggregate"
SEARCH_HISTORY_FILENAME = "search_history.json"
SAMPLING_DESIGN_FILENAME = "sampling_design.json"
SWEEP_CONTROLLER_RESULTS_SIDECAR_PORT = K8sEnvironment.PORTS.RESULTS_SIDECAR
CANCEL_POLL_INTERVAL_SECONDS = 10.0
_TERMINAL_SWEEP_PHASES = frozenset(
    {"Succeeded", "Failed", "PartiallyFailed", "Cancelled"}
)


def _is_cancelled_result(result: Any) -> bool:
    """True if ``result`` is a child whose terminal phase was ``Cancelled``."""
    return not result.success and bool(getattr(result, "was_cancelled", False))


# K8s rejects CR patches > ~1 MiB with HTTP 413; the inline aggregate budget
# lives on K8sEnvironment.JOBSET.SWEEP_AGGREGATE_INLINE_MAX_BYTES. Bound once
# at module scope so every fit decision in a run uses the same cap.
_AGGREGATE_INLINE_MAX_BYTES = K8sEnvironment.JOBSET.SWEEP_AGGREGATE_INLINE_MAX_BYTES


def aggregate_marker_exists(base_dir: Path) -> bool:
    """Return True iff the aggregation ready marker is present."""
    return (base_dir / AGGREGATE_READY_MARKER).exists()


def _has_durable_sweep_aggregate(status: dict[str, Any]) -> bool:
    """Return whether the operator has published its PVC-backed reference."""
    aggregate_ref = status.get("aggregateRef") or {}
    return (
        status.get("resultsAvailable") is True
        and isinstance(aggregate_ref, dict)
        and bool(aggregate_ref.get("url"))
    )


def resolve_terminal_phase(
    *,
    completed: int,
    failed: int,
    max_failures: int,
    cancel_requested: bool = False,
    cancelled: int = 0,
    on_child_failure: str = "continue",
) -> str:
    """Resolve the AIPerfSweep terminal ``status.phase`` from child outcomes.

    Three-way classification keeps a single bad trial in a 6-trial sweep from
    masquerading as a total run-failure:

    * ``Cancelled`` — the parent CR requested cancellation, OR no genuine
      failures occurred but at least one child was cancelled and none
      succeeded; partial child results still feed aggregate artifacts.
    * ``Succeeded`` — no failures.
    * ``Failed`` — every result failed (no successful trial), OR
      ``max_failures > 0`` and ``failed >= max_failures`` (explicit budget),
      OR ``on_child_failure == "abort"`` and any genuine failure occurred
      (the abort policy is terminal-fatal even when a prior child succeeded).
    * ``PartiallyFailed`` — some failed, some succeeded, and neither the
      explicit budget nor the abort policy was tripped.

    The CRD enum (``crd-aiperfsweep.yaml``) has carried ``PartiallyFailed``
    since the schema was first written, but every prior call site collapsed
    "any failure" → ``Failed``. ``aiperf kube list`` already
    accept the enum verbatim because the CRD declared it.

    Cancelled children (a user cancelling individual child AIPerfJobs out of
    band, so the parent's ``spec.cancel`` never flips) are NOT failures: they
    are counted separately via ``cancelled`` and excluded from ``failed`` by
    the caller, mirroring the operator rollup's distinct ``cancelled`` bucket
    (``child_rollup``). Folding them into ``failed`` let an externally
    cancelled sweep trip ``failed >= max_failures`` and resolve ``Failed``.

    Args:
        completed: Count of successful child results across all (variation,
            trial) cells. Sourced from ``RunResult.success`` truthiness.
        failed: Count of genuinely failed child results across all cells
            (child Job ``Failed``). Cancelled children are excluded — they
            are passed via ``cancelled`` instead.
        max_failures: ``spec.failurePolicy.maxFailures`` from the CR.
            ``0`` = unbounded (no explicit threshold; use the all-failed
            rule). ``>0`` = treat ``failed >= max_failures`` as
            non-recoverable.
        cancel_requested: Whether ``spec.cancel`` was observed during the run.
        cancelled: Count of child results whose terminal phase was
            ``Cancelled`` (out-of-band per-child cancellation).
        on_child_failure: ``spec.failurePolicy.onChildFailure`` from the CR.
            ``"abort"`` makes the first genuine failure terminal-fatal — the
            orchestrator stops issuing further children, so the sweep is
            ``Failed`` even with a prior success and the default
            ``max_failures=0``. ``"continue"`` (default) leaves resolution to
            the all-failed / budget rules above.

    Returns:
        One of ``"Cancelled"``, ``"Succeeded"``, ``"PartiallyFailed"``,
        ``"Failed"`` — members of ``PARENT_TERMINAL_PHASES`` in
        ``aiperf.operator.handlers.sweep.child_rollup``.

    Example:
        >>> resolve_terminal_phase(completed=5, failed=1, max_failures=0)
        'PartiallyFailed'
        >>> resolve_terminal_phase(completed=0, failed=6, max_failures=0)
        'Failed'
        >>> resolve_terminal_phase(completed=6, failed=0, max_failures=0)
        'Succeeded'
        >>> resolve_terminal_phase(completed=4, failed=2, max_failures=2)
        'Failed'
        >>> resolve_terminal_phase(completed=1, failed=0, max_failures=0, cancel_requested=True)
        'Cancelled'
        >>> resolve_terminal_phase(completed=0, failed=0, max_failures=2, cancelled=4)
        'Cancelled'
    """
    if cancel_requested:
        return "Cancelled"
    if failed <= 0:
        if cancelled > 0 and completed <= 0:
            return "Cancelled"
        return "Succeeded"
    if max_failures > 0 and failed >= max_failures:
        return "Failed"
    if completed <= 0:
        return "Failed"
    if on_child_failure == "abort":
        # Abort policy stops the sweep on the first genuine failure, so the
        # run never reaches a recoverable partial state even when an earlier
        # child succeeded. The orchestrator already halted (see
        # MultiRunOrchestrator._sweep_failure_threshold_exceeded); resolving
        # PartiallyFailed here would contradict the documented terminal phase.
        return "Failed"
    return "PartiallyFailed"


def sweep_controller_host(sweep_name: str, namespace: str) -> str:
    """Return the sweep-controller pod's cluster-resolvable DNS name."""
    return controller_dns_name(f"aiperf-{sweep_name}", namespace)


def sweep_aggregate_api_path(
    *, namespace: str, sweep_name: str, sweep_run_epoch: str
) -> str:
    """Return the results-sidecar route for the durable parent aggregate."""
    return (
        f"{API_RESULTS_FILES_PATH}/{namespace}/sweeps/{sweep_name}/"
        f"{sweep_run_epoch}/aggregate.json"
    )


def _adaptive_search_log_summary(adaptive: AdaptiveSearchSweep) -> str:
    objectives = ", ".join(
        f"{objective.metric}:{objective.stat}:{objective.direction}"
        for objective in adaptive.objectives
    )
    return (
        f"planner={adaptive.planner}, max_iterations={adaptive.max_iterations}, "
        f"objectives={objectives}"
    )


def write_aggregate_marker(base_dir: Path) -> None:
    """Durably publish the canonical aggregation ready marker."""
    write_ready_marker(base_dir)


async def _mark_sweep_aggregate_ready(
    *,
    base_dir: Path,
    namespace: str,
    sweep_name: str,
    sweep_run_epoch: str,
    auto_plot: bool,
    plot_required: bool,
    plot_envelope: Any,
) -> None:
    """Finish configured plotting before exposing the aggregate to readers."""
    if auto_plot:
        from aiperf.plot.auto_plot import run_auto_plot_async

        sweep_dir = base_dir / namespace / "sweeps" / sweep_name / sweep_run_epoch
        await run_auto_plot_async(
            artifact_dir=sweep_dir,
            input_paths=[base_dir],
            output_dir=sweep_dir / "plots",
            plot_required=plot_required,
            plot_envelope=plot_envelope,
        )
    await asyncio.to_thread(
        _prune_noncanonical_sweep_artifacts,
        base_dir=base_dir,
        namespace=namespace,
        sweep_name=sweep_name,
        sweep_run_epoch=sweep_run_epoch,
    )
    await asyncio.to_thread(write_aggregate_marker, base_dir)


def _prune_noncanonical_sweep_artifacts(
    *,
    base_dir: Path,
    namespace: str,
    sweep_name: str,
    sweep_run_epoch: str,
) -> None:
    """Leave only the ready marker and epoch bundle the operator should harvest.

    The shared orchestrator temporarily uses its normal local artifact layout
    directly under ``base_dir``. Those run caches and aggregate directories
    are inputs to the Kubernetes parent bundle, not durable operator results.
    The results sidecar enumerates the whole tree, so retaining them would
    copy unrelated top-level paths onto the operator's shared PVC and let one
    sweep overwrite another.

    Pruning happens immediately before the ready marker. A crash during this
    operation leaves the marker absent, causing the restart path to reconstruct
    the bundle from the durable child AIPerfJobs. Once the marker exists the
    artifact tree is immutable because the operator lists and downloads it in
    separate requests.
    """
    base_dir = Path(base_dir)
    if aggregate_marker_exists(base_dir):
        raise RuntimeError("refusing to prune a sweep artifact tree marked ready")
    canonical = base_dir / namespace / "sweeps" / sweep_name / sweep_run_epoch
    if not canonical.is_dir():
        raise FileNotFoundError(
            f"canonical sweep artifact directory is missing: {canonical}"
        )

    keep_chain = (
        (base_dir, base_dir / namespace),
        (base_dir / namespace, base_dir / namespace / "sweeps"),
        (
            base_dir / namespace / "sweeps",
            base_dir / namespace / "sweeps" / sweep_name,
        ),
        (base_dir / namespace / "sweeps" / sweep_name, canonical),
    )
    for parent, keep in keep_chain:
        for entry in parent.iterdir():
            if entry == keep:
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink(missing_ok=True)


async def _poll_cancel_flag(
    custom: Any,
    *,
    namespace: str,
    name: str,
    flag: dict[str, bool],
    expected_uid: str | None = None,
    interval: float = CANCEL_POLL_INTERVAL_SECONDS,
) -> None:
    """Background poller: set flag['requested']=True if parent CR's spec.cancel is set.

    Best-effort: apiserver hiccups are swallowed. The flag is monotonic — once set,
    it stays set, and the orchestrator/executor read it between cells/trials. A
    parent UID change also sets the flag so a stale controller winds down.
    """
    while not flag["requested"]:
        try:
            cr = await custom.get_namespaced_custom_object(
                group="aiperf.nvidia.com",
                version="v1alpha1",
                namespace=namespace,
                plural="aiperfsweeps",
                name=name,
            )
            if (
                expected_uid is not None
                and cr.get("metadata", {}).get("uid") != expected_uid
            ):
                logger.info(
                    "parent CR identity changed; stopping stale sweep-controller"
                )
                flag["requested"] = True
                return
            if bool((cr.get("spec") or {}).get("cancel", False)):
                logger.info("cancel observed on parent CR; signalling orchestrator")
                flag["requested"] = True
                return
        except Exception as e:  # noqa: BLE001 - best-effort poll, never crash the controller
            logger.debug(f"cancel-flag poll transient error: {e}")
        await asyncio.sleep(interval)


def _child_status(result: Any) -> str:
    """Terminal status string for one child, matching the rollup's buckets."""
    if result.success:
        return "Succeeded"
    return "Cancelled" if _is_cancelled_result(result) else "Failed"


def _write_aggregate_manifest(
    aggregate_dir: Path,
    sweep_cr: dict[str, Any],
    results: list,
    plan: Any,
) -> None:
    """Write the per-sweep manifest with epoch lineage of all child runs."""
    manifest = {
        "sweep": sweep_cr["metadata"]["name"],
        "sweep_namespace": sweep_cr["metadata"]["namespace"],
        "sweep_uid": sweep_cr["metadata"]["uid"],
        "sweep_epoch": sweep_cr.get("status", {}).get("runEpoch", ""),
        "total_variations": len(plan.configs),
        "completed_runs": sum(1 for r in results if r.success),
        # Same three-way split _write_sweep_parent_aggregate uses twenty lines
        # below. Counting cancelled children as failed here made two artifacts
        # generated from one list disagree, and contradicted the live rollup.
        "failed_runs": sum(
            1 for r in results if not r.success and not _is_cancelled_result(r)
        ),
        "cancelled_runs": sum(1 for r in results if _is_cancelled_result(r)),
        "child_runs": [
            {
                "label": redact_sweep_display_label(r.label),
                "status": _child_status(r),
                "error": redact_sweep_public_data(r.error or ""),
            }
            for r in results
        ],
    }
    (aggregate_dir / "manifest.json").write_bytes(
        orjson.dumps(manifest, option=orjson.OPT_INDENT_2)
    )


def _mirror_strategy_aggregate_to_sweep_dir(
    *,
    base_dir: Path,
    aggregate_dir: Path,
    namespace: str,
    sweep_name: str,
    sweep_run_epoch: str,
) -> None:
    sweep_aggregate_dir = (
        Path(base_dir)
        / namespace
        / "sweeps"
        / sweep_name
        / sweep_run_epoch
        / SWEEP_AGGREGATE_SUBDIR
    )
    sweep_aggregate_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(aggregate_dir.iterdir()):
        if source.is_file() and not source.is_symlink():
            shutil.copy2(source, sweep_aggregate_dir / source.name)


def _archive_search_history(
    *,
    base_dir: Path,
    namespace: str,
    sweep_name: str,
    sweep_run_epoch: str,
) -> Path | None:
    """Move the adaptive trajectory into its durable sweep-epoch directory.

    The canonical orchestrator writes ``search_history.json`` at its artifact
    root. A sweep-controller gets a private ``emptyDir`` root, but the operator
    later harvests that whole tree onto a shared PVC. Leaving the file at the
    root would let one adaptive sweep overwrite another and would put the
    history outside the sweep results API's epoch-scoped listing.

    Returns:
        The archived path, or ``None`` for non-adaptive sweeps.
    """
    source = base_dir / SEARCH_HISTORY_FILENAME
    if not source.is_file():
        return None
    destination = (
        base_dir
        / namespace
        / "sweeps"
        / sweep_name
        / sweep_run_epoch
        / SEARCH_HISTORY_FILENAME
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)
    return destination


def _archive_sampling_design(
    *,
    base_dir: Path,
    namespace: str,
    sweep_name: str,
    sweep_run_epoch: str,
) -> Path | None:
    """Move a QMC design beside the sweep epoch's aggregate artifacts."""
    source_dir = base_dir / SWEEP_AGGREGATE_SUBDIR
    source = source_dir / SAMPLING_DESIGN_FILENAME
    if not source.is_file():
        return None
    destination = (
        base_dir
        / namespace
        / "sweeps"
        / sweep_name
        / sweep_run_epoch
        / SWEEP_AGGREGATE_SUBDIR
        / SAMPLING_DESIGN_FILENAME
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)
    with contextlib.suppress(OSError):
        source_dir.rmdir()
    return destination


def _write_sweep_parent_aggregate(
    *,
    base_dir: Path,
    sweep_cr: dict[str, Any],
    spec: Any,
    results: list,
    child_runs: list[Any],
    plan: Any,
    sweep_run_epoch: str,
    terminal_phase: str | None = None,
) -> None:
    """Persist the durable parent ``aggregate.json`` under ``<base>/<ns>/sweeps/<name>/<epoch>/``.

    Anchors the dual-backed sweep API: while the controller pod is alive the
    operator can read live status from the CR; once the pod is gone the
    operator falls back to this directory. Also writes ``children.json``
    immediately after — the authoritative back-link from sweep epoch to each
    child AIPerfJob's name + child epoch, used by ``sweep_union`` to resolve
    archived sweeps after the parent CR has been TTL-reaped.

    Conditions are owned by the operator and not yet collected here, so we
    pass ``conditions=None`` and the ``conditions.json`` sibling is omitted.

    Two spec-derived keys are persisted: ``specSnapshot`` (the full
    ``AIPerfSweepSpec`` dump — the only durable copy of the spec after the CR
    is TTL-reaped) and ``specSummary`` (the purpose-built
    sweep_type/dimensions/multi_run/convergence dict built by
    ``spec_summary_snapshot``, which the operator's archived-sweep API reads
    back verbatim).
    """
    from aiperf.kubernetes.crd_models import AIPerfSweepSpec
    from aiperf.operator.routers._sweeps_spec import (
        SPEC_SUMMARY_KEY,
        spec_summary_snapshot,
    )
    from aiperf.sweep_controller.aggregator import (
        write_children_manifest,
        write_sweep_aggregate,
    )

    metadata = sweep_cr.get("metadata") or {}
    namespace = metadata["namespace"]
    name = metadata["name"]
    # Externally-cancelled children are their own bucket — they must not roll
    # into ``failedRuns`` here, mirroring the live CR rollup
    # (``child_rollup._tally_children``) and the archived read in
    # ``sweep_union`` (``runStates.cancelled``). Use the same
    # ``_is_cancelled_result`` discriminator so live and archived views agree.
    cancelled = sum(1 for r in results if _is_cancelled_result(r))
    failed = sum(1 for r in results if not r.success and not _is_cancelled_result(r))
    completed = len(results) - failed - cancelled
    is_adaptive = bool(getattr(plan, "is_adaptive_search", False))
    total_variations = (
        len({int(r.variation_index) for r in results})
        if is_adaptive
        else len(plan.configs)
    )
    children = redact_sweep_public_data([child.to_dict() for child in child_runs])
    doc: dict[str, Any] = {
        "phase": terminal_phase or ("Succeeded" if failed == 0 else "Failed"),
        "totalVariations": total_variations,
        "completedRuns": completed,
        "failedRuns": failed,
        "cancelledRuns": cancelled,
        "runStates": {
            "pending": 0,
            "running": 0,
            "completed": completed,
            "failed": failed,
            "cancelled": cancelled,
        },
        "specSnapshot": redact_sweep_public_data(
            spec.model_dump(mode="json", by_alias=True)
        )
        if hasattr(spec, "model_dump")
        else {},
        SPEC_SUMMARY_KEY: spec_summary_snapshot(spec)
        if isinstance(spec, AIPerfSweepSpec)
        else {},
        "childRuns": children,
    }
    write_sweep_aggregate(
        base_dir=base_dir,
        namespace=namespace,
        sweep_name=name,
        sweep_run_epoch=sweep_run_epoch,
        doc=doc,
        conditions=None,
        update_latest=False,
    )
    write_children_manifest(
        base_dir=base_dir,
        namespace=namespace,
        sweep_name=name,
        sweep_run_epoch=sweep_run_epoch,
        children=children,
    )


def _confidence_artifact_path(base_dir: Path, sweep_dir: Path) -> Path | None:
    """Locate the best aggregate statistics artifact available for CR mirroring."""
    mirrored = sweep_dir / SWEEP_AGGREGATE_SUBDIR
    for candidate in (
        mirrored / "profile_export_aiperf_sweep.json",
        mirrored / "profile_export_aiperf_aggregate.json",
        base_dir / AGGREGATE_SUBDIR / "profile_export_aiperf_aggregate.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def _load_aggregate_for_cr(
    base_dir: Path,
    namespace: str,
    sweep_name: str,
    sweep_run_epoch: str,
) -> dict[str, Any]:
    """Read the on-disk aggregate JSON files and bundle them for the CR patch.

    The sweep-controller writes aggregate artifacts under
    ``<base>/<ns>/sweeps/<name>/<epoch>/`` (parent ``aggregate.json``,
    ``children.json``) and the strategy-owned aggregate dir (typically
    ``<base>/aggregate/profile_export_aiperf_aggregate.json``). On small
    sweeps the bundle is ~50 KB and we embed everything inline on the CR
    to close the live half of the dual-backed sweep API contract
    documented in ``aggregator.py``.

    On large sweeps (many cells x metrics x percentiles) the strategy
    ``confidence`` payload grows linearly and the patch can exceed the
    apiserver's 1 MB CR size cap, returning 413 and stranding the parent
    at ``Aggregating``. We bound the inlined size: if the encoded bundle
    exceeds ``AIPERF_K8S_JOBSET_SWEEP_AGGREGATE_INLINE_MAX_BYTES`` we drop
    ``confidence`` first,
    then omit ``children`` and add a compact ``childrenTruncated`` marker
    if the post-drop payload still exceeds the budget. The disk-backed
    path served by the results sidecar still has the full document, so
    consumers fetching ``status.aggregateRef.apiPath`` see no loss; only
    the in-CR mirror is reduced.

    Missing files are silently skipped: this loader is best-effort and the
    primary signal (``aggregation.phase=Complete`` and ``terminal_phase``)
    is set by the caller regardless of which sub-files made it to disk.
    """
    sweep_dir = Path(base_dir) / namespace / "sweeps" / sweep_name / sweep_run_epoch
    bundle: dict[str, Any] = {}
    parent_path = sweep_dir / "aggregate.json"
    children_path = sweep_dir / "children.json"
    for key, path in (
        ("parent", parent_path),
        ("children", children_path),
        ("confidence", _confidence_artifact_path(Path(base_dir), sweep_dir)),
    ):
        if path is None:
            continue
        try:
            bundle[key] = redact_sweep_public_data(orjson.loads(path.read_bytes()))
        except FileNotFoundError:
            continue
        except (OSError, orjson.JSONDecodeError, ValueError) as exc:
            # A truncated or malformed file must not poison the bundle —
            # exit non-zero loses all three artifacts. Log + skip; the CR
            # patch carries whichever sub-files made it.
            logger.warning(
                "sweep aggregate: skipping %s (%s) — %s: %s",
                key,
                path,
                type(exc).__name__,
                exc,
            )
            continue

    _fit_aggregate_bundle_for_cr(bundle)
    return bundle


def _terminal_phase_from_aggregate_bundle(bundle: dict[str, Any]) -> str:
    """Return the validated parent terminal phase from a ready bundle."""
    parent = bundle.get("parent")
    phase = parent.get("phase") if isinstance(parent, dict) else None
    if phase not in _TERMINAL_SWEEP_PHASES:
        raise ValueError(
            f"ready sweep aggregate has no valid parent terminal phase: {phase!r}"
        )
    return phase


def _load_ready_terminal_phase(
    base_dir: Path,
    namespace: str,
    sweep_name: str,
    sweep_run_epoch: str,
) -> str:
    """Read terminal phase from the raw parent file before CR-size fitting.

    ``_load_aggregate_for_cr`` may intentionally replace an oversized inline
    bundle with ``aggregateTruncated``. The durable parent file remains whole
    and is therefore the authoritative phase source on a ready restart.
    """
    parent_path = (
        Path(base_dir)
        / namespace
        / "sweeps"
        / sweep_name
        / sweep_run_epoch
        / "aggregate.json"
    )
    try:
        parent = orjson.loads(parent_path.read_bytes())
    except (OSError, orjson.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"ready sweep parent aggregate is unreadable: {parent_path}"
        ) from exc
    return _terminal_phase_from_aggregate_bundle({"parent": parent})


def _fit_aggregate_bundle_for_cr(bundle: dict[str, Any]) -> None:
    """Mutate an aggregate bundle so its CR mirror fits the inline budget."""
    encoded_size = len(orjson.dumps(bundle))
    if encoded_size <= _AGGREGATE_INLINE_MAX_BYTES:
        return

    if "confidence" in bundle:
        logger.warning(
            "aggregate bundle is %d bytes (> %d cap); dropping `confidence` "
            "from CR mirror — full document remains at the disk-backed path",
            encoded_size,
            _AGGREGATE_INLINE_MAX_BYTES,
        )
        bundle.pop("confidence", None)
        encoded_size = len(orjson.dumps(bundle))

    if encoded_size <= _AGGREGATE_INLINE_MAX_BYTES:
        return

    children_doc = bundle.pop("children", None)
    if children_doc is not None:
        bundle["childrenTruncated"] = _children_truncated_marker(children_doc)
        logger.warning(
            "aggregate bundle is %d bytes (> %d cap) after dropping `confidence`; "
            "omitting `children` from CR mirror — full children manifest remains "
            "at the disk-backed path",
            encoded_size,
            _AGGREGATE_INLINE_MAX_BYTES,
        )
        encoded_size = len(orjson.dumps(bundle))

    if encoded_size <= _AGGREGATE_INLINE_MAX_BYTES:
        return

    original_keys = sorted(bundle)
    bundle.clear()
    bundle["aggregateTruncated"] = {
        "reason": "inline_status_budget_exceeded",
        "includedKeys": [],
        "omittedKeys": original_keys,
        "maxBytes": _AGGREGATE_INLINE_MAX_BYTES,
        "originalBytes": encoded_size,
    }
    if len(orjson.dumps(bundle)) > _AGGREGATE_INLINE_MAX_BYTES:
        bundle.clear()


def _children_truncated_marker(children_doc: Any) -> dict[str, Any]:
    total: int | None = None
    sweep_run_epoch = ""
    if isinstance(children_doc, dict):
        children = children_doc.get("children")
        sweep_run_epoch = str(children_doc.get("sweep_run_epoch") or "")
        if isinstance(children, list):
            total = len(children)
    elif isinstance(children_doc, list):
        total = len(children_doc)

    return {
        "reason": "inline_status_budget_exceeded",
        "total": total,
        "included": 0,
        "sweep_run_epoch": sweep_run_epoch,
    }


async def _handle_ready_restart(
    *,
    sweep_cr: dict[str, Any],
    status_writer: SweepStatusWriter,
    sweep_namespace: str,
    sweep_name: str,
    sweep_run_epoch: str,
) -> int | None:
    """Return an exit code when durable or ready state makes replay unnecessary."""
    if _has_durable_sweep_aggregate(sweep_cr.get("status") or {}):
        logger.info(
            "sweep already has an operator-backed aggregate reference; "
            "preserving durable status"
        )
        return 0
    if not aggregate_marker_exists(RESULTS_DIR):
        return None

    try:
        terminal_phase = _load_ready_terminal_phase(
            RESULTS_DIR,
            sweep_namespace,
            sweep_name,
            sweep_run_epoch,
        )
        aggregate_doc = _load_aggregate_for_cr(
            RESULTS_DIR,
            sweep_namespace,
            sweep_name,
            sweep_run_epoch,
        )
        await status_writer.aggregation_complete(
            aggregate_path=sweep_aggregate_api_path(
                namespace=sweep_namespace,
                sweep_name=sweep_name,
                sweep_run_epoch=sweep_run_epoch,
            ),
            controller_host=sweep_controller_host(sweep_name, sweep_namespace),
            port=SWEEP_CONTROLLER_RESULTS_SIDECAR_PORT,
            aggregate_doc=aggregate_doc,
            terminal_phase=terminal_phase,
        )
    except Exception:  # noqa: BLE001 - restartPolicy retries status publication
        logger.exception(
            "ready sweep bundle validation/status publication failed; "
            "exiting non-zero for restart"
        )
        return 1
    logger.info(
        "ready sweep bundle republished without orchestrator replay "
        f"(phase={terminal_phase})"
    )
    return 0


async def _recover_cancelled_terminal_results(
    *,
    cancel_requested: bool,
    executor: K8sChildJobExecutor,
    plan: BenchmarkPlan,
    results: list[RunResult],
) -> None:
    """Merge exact-identity terminal children after cancellation or restart."""
    if not cancel_requested:
        return
    results.extend(await executor.recover_terminal_results(plan))


async def main() -> int:
    """Run the sweep-controller pod: load CR, execute variations, and aggregate.

    Returns 0 on clean completion, 1 on unrecoverable error. Idempotent across
    pod restarts: existing terminal child jobs are reused, while a ready epoch
    bundle republishes terminal status without replaying the orchestrator.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    sweep_name = os.environ["AIPERF_SWEEP_NAME"]
    sweep_namespace = os.environ["AIPERF_SWEEP_NAMESPACE"]
    sweep_run_epoch = os.environ["AIPERF_SWEEP_EPOCH"]
    expected_sweep_uid = os.environ.get("AIPERF_SWEEP_UID")
    logger.info(f"sweep-controller starting for {sweep_namespace}/{sweep_name}")

    from kubernetes_asyncio.client import CustomObjectsApi

    from aiperf.cli_runner._aggregation_dispatch import aggregate_plan_results
    from aiperf.cli_runner._strategy import build_strategy
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.kubernetes.client import k8s_client
    from aiperf.kubernetes.spec_converter import validate_sweep_spec
    from aiperf.orchestrator.orchestrator import MultiRunOrchestrator
    from aiperf.orchestrator.search_planner import build_search_planner
    from aiperf.sweep_controller.k8s_executor import (
        K8sChildJobExecutor,
        needs_trial_suffix,
    )
    from aiperf.sweep_controller.plan_builder import build_plan_from_sweep
    from aiperf.sweep_controller.status_writer import SweepStatusWriter

    aiperf_logger = AIPerfLogger(__name__)

    async with k8s_client() as api:
        custom = CustomObjectsApi(api)
        sweep_cr = await custom.get_namespaced_custom_object(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            namespace=sweep_namespace,
            plural="aiperfsweeps",
            name=sweep_name,
        )
        current_sweep_uid = str((sweep_cr.get("metadata") or {}).get("uid") or "")
        if expected_sweep_uid is not None and current_sweep_uid != expected_sweep_uid:
            logger.warning(
                "refusing to run stale sweep-controller for %s/%s: expected uid %s, got %s",
                sweep_namespace,
                sweep_name,
                expected_sweep_uid,
                current_sweep_uid,
            )
            return 0
        spec = validate_sweep_spec(sweep_cr["spec"])

        status_writer = SweepStatusWriter(
            api,
            name=sweep_name,
            namespace=sweep_namespace,
            uid=expected_sweep_uid or current_sweep_uid,
        )
        restart_result = await _handle_ready_restart(
            sweep_cr=sweep_cr,
            status_writer=status_writer,
            sweep_namespace=sweep_namespace,
            sweep_name=sweep_name,
            sweep_run_epoch=sweep_run_epoch,
        )
        if restart_result is not None:
            return restart_result

        plan = build_plan_from_sweep(sweep_cr)
        cancel_flag: dict[str, bool] = {"requested": spec.cancel}
        cancel_task = asyncio.create_task(
            _poll_cancel_flag(
                custom,
                namespace=sweep_namespace,
                name=sweep_name,
                flag=cancel_flag,
                expected_uid=expected_sweep_uid or current_sweep_uid,
            )
        )
        try:
            # Promote `status.phase` from `Pending` to `Running` before the
            # orchestrator loop begins. The CRD declares Running but no other
            # writer ever set it, so parents jumped Pending -> Aggregating
            # directly. Atomic test/replace skips silently on pod restart or
            # if the rollup already advanced phase.
            await status_writer.parent_running()
            executor = K8sChildJobExecutor(
                api=api,
                sweep=sweep_cr,
                with_trial_suffix=needs_trial_suffix(
                    multi_run_trials=(
                        spec.multi_run.num_runs if spec.multi_run else None
                    ),
                    has_convergence=(
                        spec.multi_run is not None
                        and spec.multi_run.convergence is not None
                    ),
                ),
                base_dir=RESULTS_DIR,
                status_writer=status_writer,
                cancel_check=lambda: cancel_flag["requested"],
                sweep_run_epoch=sweep_run_epoch,
            )

            orchestrator = MultiRunOrchestrator(base_dir=RESULTS_DIR)
            search_planner = build_search_planner(plan)
            if search_planner is not None:
                from aiperf.config.sweep import AdaptiveSearchSweep

                adaptive = (
                    plan.sweep if isinstance(plan.sweep, AdaptiveSearchSweep) else None
                )
                if adaptive is not None:
                    logger.info(
                        "Cluster-side adaptive search active: "
                        f"{_adaptive_search_log_summary(adaptive)}"
                    )
            all_results = await orchestrator.execute(
                plan,
                executor,
                cancel_check=lambda: cancel_flag["requested"],
                search_planner=search_planner,
            )
            await _recover_cancelled_terminal_results(
                cancel_requested=cancel_flag["requested"],
                executor=executor,
                plan=plan,
                results=all_results,
            )
            archived_history = await asyncio.to_thread(
                _archive_search_history,
                base_dir=RESULTS_DIR,
                namespace=sweep_namespace,
                sweep_name=sweep_name,
                sweep_run_epoch=sweep_run_epoch,
            )
            if search_planner is not None and archived_history is None:
                raise RuntimeError(
                    "adaptive search completed without search_history.json; "
                    "refusing to publish an incomplete sweep aggregate"
                )
            archived_design = await asyncio.to_thread(
                _archive_sampling_design,
                base_dir=RESULTS_DIR,
                namespace=sweep_namespace,
                sweep_name=sweep_name,
                sweep_run_epoch=sweep_run_epoch,
            )
            from aiperf.config.sweep import LatinHypercubeSweep, SobolSweep

            if isinstance(plan.sweep, (LatinHypercubeSweep, SobolSweep)) and (
                archived_design is None
            ):
                raise RuntimeError(
                    "QMC sweep completed without sampling_design.json; refusing "
                    "to publish an incomplete sweep aggregate"
                )
        finally:
            cancel_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cancel_task

        cancelled_count = sum(1 for r in all_results if _is_cancelled_result(r))
        failed_count = sum(
            1 for r in all_results if not r.success and not _is_cancelled_result(r)
        )
        completed_count = len(all_results) - failed_count - cancelled_count
        terminal_phase = resolve_terminal_phase(
            completed=completed_count,
            failed=failed_count,
            max_failures=spec.failure_policy.max_failures,
            cancel_requested=cancel_flag["requested"],
            cancelled=cancelled_count,
            on_child_failure=spec.failure_policy.on_child_failure,
        )

        await status_writer.aggregation_running()
        try:
            # Top-level strategy mirrors cli_runner.py — only used for
            # aggregate-path resolution; per-cell strategies were rebuilt
            # inside the orchestrator.
            strategy = build_strategy(plan, aiperf_logger)
            artifact_dir = strategy.get_aggregate_path(RESULTS_DIR)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            if cancel_flag["requested"] and completed_count == 0:
                logger.info(
                    "cancellation requested before any successful child results; "
                    "skipping confidence aggregation"
                )
            else:
                artifact_dir = await aggregate_plan_results(
                    all_results,
                    plan,
                    strategy=strategy,
                    base_dir=RESULTS_DIR,
                    logger=aiperf_logger,
                )
                artifact_dir.mkdir(parents=True, exist_ok=True)
            _write_aggregate_manifest(artifact_dir, sweep_cr, all_results, plan)
            _mirror_strategy_aggregate_to_sweep_dir(
                base_dir=RESULTS_DIR,
                aggregate_dir=artifact_dir,
                namespace=sweep_namespace,
                sweep_name=sweep_name,
                sweep_run_epoch=sweep_run_epoch,
            )
            _write_sweep_parent_aggregate(
                base_dir=RESULTS_DIR,
                sweep_cr=sweep_cr,
                spec=spec,
                results=all_results,
                child_runs=list(executor.terminal_children),
                plan=plan,
                sweep_run_epoch=sweep_run_epoch,
                terminal_phase=terminal_phase,
            )
            await _mark_sweep_aggregate_ready(
                base_dir=RESULTS_DIR,
                namespace=sweep_namespace,
                sweep_name=sweep_name,
                sweep_run_epoch=sweep_run_epoch,
                auto_plot=spec.benchmark.artifacts.auto_plot,
                plot_required=spec.benchmark.artifacts.plot_required,
                plot_envelope=spec.plot,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("aggregation failed")
            await status_writer.aggregation_failed(
                error=str(redact_sweep_public_data(str(e)))
            )
            return 1

        # Idempotent across pod restarts: load disk artifacts and patch the CR
        # every time main() reaches this point. Without this, a sweep-controller
        # pod that aggregates once but fails to patch (apiserver hiccup, OOM,
        # crash before the patch) would never advance the parent CR — and the
        # restart path skips re-aggregation via aggregate_marker_exists, so
        # there is no second chance.
        controller_host = sweep_controller_host(sweep_name, sweep_namespace)
        try:
            aggregate_doc = _load_aggregate_for_cr(
                RESULTS_DIR, sweep_namespace, sweep_name, sweep_run_epoch
            )
            await status_writer.aggregation_complete(
                aggregate_path=sweep_aggregate_api_path(
                    namespace=sweep_namespace,
                    sweep_name=sweep_name,
                    sweep_run_epoch=sweep_run_epoch,
                ),
                controller_host=controller_host,
                port=SWEEP_CONTROLLER_RESULTS_SIDECAR_PORT,
                aggregate_doc=aggregate_doc,
                terminal_phase=terminal_phase,
            )
        except Exception:  # noqa: BLE001 - apiserver/disk failure path: log + exit non-zero so restartPolicy retries
            # Non-zero exit so the pod's `restartPolicy: OnFailure` restarts
            # us; the aggregate marker means re-aggregation is skipped, but
            # the CR-patch is retried fresh on next boot. Idling forever
            # leaks the pod (JobSet `completions=1` requires a clean exit
            # for the parent Job to complete and the CR-side TTL to fire).
            logger.exception("CR aggregate patch failed; exiting non-zero for restart")
            return 1

    # The controller container exits 0, but the pod's results-sidecar runs
    # uvicorn forever — so this Job never reaches `Succeeded` on its own and
    # the pod would linger until the parent CR's `ttlSecondsAfterFinished`
    # reaper deletes the CR (and the JobSet with it). The operator tears the
    # JobSet down promptly after harvesting the aggregate
    # (`on_aiperfsweep_aggregation_complete`), which stops the sidecar and
    # reaps this pod without waiting for CR TTL.
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
