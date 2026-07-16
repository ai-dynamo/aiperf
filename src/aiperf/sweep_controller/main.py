# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep-controller pod entry point.

Reads its target AIPerfSweep CR from the apiserver, builds a BenchmarkPlan,
runs MultiRunOrchestrator with K8sChildJobExecutor, runs aggregate_and_export
once all variations are done, and idles until the pod is reaped.

Idempotent: a restart re-reads the CR, sees existing terminal children
(ownerRef + label match), and resumes from the first non-existent variation.
Aggregation re-runs only if the ready marker is missing.
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

from aiperf.kubernetes.environment import K8sEnvironment

if TYPE_CHECKING:
    from aiperf.config.sweep import AdaptiveSearchSweep

logger = logging.getLogger(__name__)

AGGREGATE_READY_MARKER = ".aiperf_results_ready.json"
RESULTS_DIR = Path("/results")
AGGREGATE_SUBDIR = "aggregate"
# Port the sweep-controller pod's own results-sidecar listens on. DISTINCT
# from the operator's results-server port (8081); see the comment at
# ``sweep_controller.k8s_executor.SWEEP_CONTROLLER_RESULTS_SIDECAR_PORT``.
SWEEP_CONTROLLER_RESULTS_SIDECAR_PORT = 19090
CANCEL_POLL_INTERVAL_SECONDS = 10.0
# Suffix of ``RunResult.error`` for a child whose terminal phase was
# ``Cancelled``. Produced by ``K8sChildJobExecutor._collect_run_result``
# (``f"child terminal phase={phase}"``) when the operator wrote no
# top-level ``status.message`` — which it never does on the cancel path
# (``operator.handlers.sweep`` / ``operator.handlers.lifecycle`` stamp only
# ``status.phase`` and ``status.error``). A child cancelled out of band
# (user cancels the individual AIPerfJob, parent ``spec.cancel`` never set)
# returns ``success=False`` and must NOT be counted as a failure against
# ``max_failures``; see ``resolve_terminal_phase``.
_CANCELLED_CHILD_ERROR_SUFFIX = "phase=Cancelled"


def _is_cancelled_result(result: Any) -> bool:
    """True if ``result`` is a child whose terminal phase was ``Cancelled``."""
    error = result.error or ""
    return not result.success and error.endswith(_CANCELLED_CHILD_ERROR_SUFFIX)


# K8s rejects CR patches > ~1 MiB with HTTP 413; the inline aggregate budget
# lives on K8sEnvironment.JOBSET.SWEEP_AGGREGATE_INLINE_MAX_BYTES. Bound once
# at module scope so every fit decision in a run uses the same cap.
_AGGREGATE_INLINE_MAX_BYTES = K8sEnvironment.JOBSET.SWEEP_AGGREGATE_INLINE_MAX_BYTES


def aggregate_marker_exists(base_dir: Path) -> bool:
    """Return True iff the aggregation ready marker is present."""
    return (base_dir / AGGREGATE_READY_MARKER).exists()


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
    "any failure" → ``Failed``. ``aiperf kube watch`` and ``list`` already
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
    """Atomically write the aggregation ready marker."""
    marker = base_dir / AGGREGATE_READY_MARKER
    tmp = marker.with_suffix(".tmp")
    tmp.write_bytes(orjson.dumps({"ready": True}))
    tmp.rename(marker)


async def _poll_cancel_flag(
    custom: Any,
    *,
    namespace: str,
    name: str,
    flag: dict[str, bool],
    interval: float = CANCEL_POLL_INTERVAL_SECONDS,
) -> None:
    """Background poller: set flag['requested']=True if parent CR's spec.cancel is set.

    Best-effort: apiserver hiccups are swallowed. The flag is monotonic — once set,
    it stays set, and the orchestrator/executor read it between cells/trials.
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
            if bool((cr.get("spec") or {}).get("cancel", False)):
                logger.info("cancel observed on parent CR; signalling orchestrator")
                flag["requested"] = True
                return
        except Exception as e:  # noqa: BLE001 - best-effort poll, never crash the controller
            logger.debug(f"cancel-flag poll transient error: {e}")
        await asyncio.sleep(interval)


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
        "failed_runs": sum(1 for r in results if not r.success),
        "child_runs": [
            {
                "label": r.label,
                "status": "Succeeded" if r.success else "Failed",
                "error": r.error or "",
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
        / "sweep_aggregate"
    )
    sweep_aggregate_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(aggregate_dir.iterdir()):
        if source.is_file() and not source.is_symlink():
            shutil.copy2(source, sweep_aggregate_dir / source.name)


def _write_sweep_parent_aggregate(
    *,
    base_dir: Path,
    sweep_cr: dict[str, Any],
    spec: Any,
    results: list,
    plan: Any,
    sweep_run_epoch: str,
    with_trial_suffix: bool,
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
    from aiperf.operator.models import AIPerfSweepSpec
    from aiperf.operator.results_layout import list_run_epochs, write_sweep_latest
    from aiperf.operator.routers._sweeps_spec import (
        SPEC_SUMMARY_KEY,
        spec_summary_snapshot,
    )
    from aiperf.sweep_controller.aggregator import (
        write_children_manifest,
        write_sweep_aggregate,
    )
    from aiperf.sweep_controller.k8s_executor import build_child_name

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
    doc: dict[str, Any] = {
        "phase": terminal_phase or ("Succeeded" if failed == 0 else "Failed"),
        "totalVariations": len(plan.configs),
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
        "specSnapshot": spec.model_dump(mode="json")
        if hasattr(spec, "model_dump")
        else {},
        SPEC_SUMMARY_KEY: spec_summary_snapshot(spec)
        if isinstance(spec, AIPerfSweepSpec)
        else {},
        "childRuns": [
            {
                "label": r.label,
                "status": (
                    "Succeeded"
                    if r.success
                    else "Cancelled"
                    if _is_cancelled_result(r)
                    else "Failed"
                ),
                "error": r.error or "",
            }
            for r in results
        ],
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
    # Build children manifest by walking the actual results stream, not
    # plan.variations. For adaptive search plan.variations is a length-1
    # placeholder and the real variation set lives in `results` — each
    # RunResult carries its `variation_index` and `trial_index` directly
    # (stamped from the originating BenchmarkVariation.index). For grid/
    # repeated mode the results stream still preserves the same (var, trial)
    # ordering, so this single results-driven path handles both. Reading
    # `variation_index` straight from the result keeps the child name and
    # index aligned with the true variation that created the child AIPerfJob
    # even when the stream order is non-dense or labels collide (e.g. a BO
    # planner re-proposing the same config under a fresh index).
    children: list[dict[str, Any]] = []
    for r in results:
        var_idx = int(r.variation_index)
        trial_idx = int(r.trial_index)
        trial_for_name = trial_idx if with_trial_suffix else None
        child_name = build_child_name(
            sweep_name=name,
            sweep_run_epoch=sweep_run_epoch,
            variation_index=var_idx,
            trial_index=trial_for_name,
        )
        # The child's results dir is named by the child's OWN epoch (derived
        # from its creationTimestamp), never the sweep epoch. If the RunResult
        # epoch was never stamped (slow operator reconcile lost the race grace),
        # recover the true epoch from disk by the child's own job dir; falling
        # back to ``sweep_run_epoch`` would point runs_index at a guaranteed-
        # wrong directory and silently drop the variation. An empty string is
        # honest — ``_parse_child_sweep_manifest`` skips unknown-location rows.
        child_run_epoch = getattr(r, "child_run_epoch", "")
        if not child_run_epoch:
            disk_epochs = list_run_epochs(Path(base_dir), namespace, child_name)
            child_run_epoch = disk_epochs[-1] if disk_epochs else ""
        children.append(
            {
                "namespace": namespace,
                "name": child_name,
                "variation_index": var_idx,
                "variation_label": r.variation_label,
                "trial_index": trial_idx if with_trial_suffix else None,
                "child_run_epoch": child_run_epoch,
                "label": r.label,
                "status": (
                    "Succeeded"
                    if r.success
                    else "Cancelled"
                    if _is_cancelled_result(r)
                    else "Failed"
                ),
            }
        )
    write_children_manifest(
        base_dir=base_dir,
        namespace=namespace,
        sweep_name=name,
        sweep_run_epoch=sweep_run_epoch,
        children=children,
    )
    write_sweep_latest(Path(base_dir), namespace, name, sweep_run_epoch)


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
    confidence_path = (
        Path(base_dir) / "aggregate" / "profile_export_aiperf_aggregate.json"
    )
    for key, path in (
        ("parent", parent_path),
        ("children", children_path),
        ("confidence", confidence_path),
    ):
        try:
            bundle[key] = orjson.loads(path.read_bytes())
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


async def main() -> int:
    """Run the sweep-controller pod: load CR, execute variations, aggregate, idle.

    Returns 0 on clean completion, 1 on unrecoverable error. Idempotent across
    pod restarts (existing terminal child jobs are reused; aggregation re-runs
    only if the ready marker is missing).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    sweep_name = os.environ["AIPERF_SWEEP_NAME"]
    sweep_namespace = os.environ["AIPERF_SWEEP_NAMESPACE"]
    sweep_run_epoch = os.environ["AIPERF_SWEEP_EPOCH"]
    logger.info(f"sweep-controller starting for {sweep_namespace}/{sweep_name}")

    from kubernetes_asyncio.client import CustomObjectsApi

    from aiperf.cli_runner._aggregate import aggregate_and_export
    from aiperf.cli_runner._strategy import build_strategy
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.kubernetes.client import k8s_client
    from aiperf.operator.models import AIPerfSweepSpec
    from aiperf.orchestrator.orchestrator import MultiRunOrchestrator
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
        spec = AIPerfSweepSpec.model_validate(sweep_cr["spec"])

        plan = build_plan_from_sweep(sweep_cr)
        cancel_flag: dict[str, bool] = {"requested": False}
        cancel_task = asyncio.create_task(
            _poll_cancel_flag(
                custom,
                namespace=sweep_namespace,
                name=sweep_name,
                flag=cancel_flag,
            )
        )
        try:
            status_writer = SweepStatusWriter(
                api, name=sweep_name, namespace=sweep_namespace
            )
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
            from aiperf.cli_runner._strategy import _build_search_planner

            search_planner = _build_search_planner(plan)
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

        if not aggregate_marker_exists(RESULTS_DIR):
            await status_writer.aggregation_running()
            try:
                # Top-level strategy mirrors cli_runner.py — only used for
                # aggregate-path resolution; per-cell strategies were rebuilt
                # inside the orchestrator.
                strategy = build_strategy(plan, aiperf_logger)
                aggregate_dir = strategy.get_aggregate_path(RESULTS_DIR)
                aggregate_dir.mkdir(parents=True, exist_ok=True)
                if cancel_flag["requested"] and completed_count == 0:
                    logger.info(
                        "cancellation requested before any successful child results; "
                        "skipping confidence aggregation"
                    )
                else:
                    await aggregate_and_export(
                        all_results,
                        plan,
                        strategy=strategy,
                        base_dir=RESULTS_DIR,
                        logger=aiperf_logger,
                    )
                _write_aggregate_manifest(aggregate_dir, sweep_cr, all_results, plan)
                _mirror_strategy_aggregate_to_sweep_dir(
                    base_dir=RESULTS_DIR,
                    aggregate_dir=aggregate_dir,
                    namespace=sweep_namespace,
                    sweep_name=sweep_name,
                    sweep_run_epoch=sweep_run_epoch,
                )
                _write_sweep_parent_aggregate(
                    base_dir=RESULTS_DIR,
                    sweep_cr=sweep_cr,
                    spec=spec,
                    results=all_results,
                    plan=plan,
                    sweep_run_epoch=sweep_run_epoch,
                    with_trial_suffix=needs_trial_suffix(
                        multi_run_trials=(
                            spec.multi_run.num_runs if spec.multi_run else None
                        ),
                        has_convergence=(
                            spec.multi_run is not None
                            and spec.multi_run.convergence is not None
                        ),
                    ),
                    terminal_phase=terminal_phase,
                )
                write_aggregate_marker(RESULTS_DIR)
            except Exception as e:  # noqa: BLE001
                logger.exception("aggregation failed")
                await status_writer.aggregation_failed(error=str(e))
                return 1
        else:
            logger.info("aggregation already complete (marker present)")

        # Idempotent across pod restarts: load disk artifacts and patch the CR
        # every time main() reaches this point. Without this, a sweep-controller
        # pod that aggregates once but fails to patch (apiserver hiccup, OOM,
        # crash before the patch) would never advance the parent CR — and the
        # restart path skips re-aggregation via aggregate_marker_exists, so
        # there is no second chance.
        controller_host = os.environ.get("HOSTNAME", "")
        try:
            aggregate_doc = _load_aggregate_for_cr(
                RESULTS_DIR, sweep_namespace, sweep_name, sweep_run_epoch
            )
            await status_writer.aggregation_complete(
                aggregate_path=(
                    f"/api/v1/results/{sweep_namespace}/{sweep_name}/aggregate"
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
