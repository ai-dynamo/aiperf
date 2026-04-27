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
from pathlib import Path
from typing import Any

import orjson

logger = logging.getLogger(__name__)

AGGREGATE_READY_MARKER = ".aiperf_results_ready.json"
RESULTS_DIR = Path("/results")
AGGREGATE_SUBDIR = "aggregate"
RESULTS_SERVER_PORT = 19090
CANCEL_POLL_INTERVAL_SECONDS = 10.0


def aggregate_marker_exists(base_dir: Path) -> bool:
    """Return True iff the aggregation ready marker is present."""
    return (base_dir / AGGREGATE_READY_MARKER).exists()


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


def _write_sweep_parent_aggregate(
    *,
    base_dir: Path,
    sweep_cr: dict[str, Any],
    spec: Any,
    results: list,
    plan: Any,
    sweep_run_epoch: str,
    with_trial_suffix: bool,
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
    """
    from aiperf.sweep_controller.aggregator import (
        write_children_manifest,
        write_sweep_aggregate,
    )
    from aiperf.sweep_controller.k8s_executor import build_child_name

    metadata = sweep_cr.get("metadata") or {}
    namespace = metadata["namespace"]
    name = metadata["name"]
    completed = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    doc: dict[str, Any] = {
        "phase": "Succeeded" if failed == 0 else "Failed",
        "totalVariations": len(plan.configs),
        "completedRuns": completed,
        "failedRuns": failed,
        "spec_snapshot": spec.model_dump() if hasattr(spec, "model_dump") else {},
        "child_runs": [
            {
                "label": r.label,
                "status": "Succeeded" if r.success else "Failed",
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
    )
    # Build children manifest by walking variations x trials in the same order
    # the orchestrator emits results. plan.trials is the static upper bound;
    # adaptive strategies may emit fewer or more, so we walk len(results) and
    # roll variation_index forward as trials saturate.
    children: list[dict[str, Any]] = []
    trials_per_variation = max(int(getattr(plan, "trials", 1) or 1), 1)
    for idx, r in enumerate(results):
        var_idx = min(idx // trials_per_variation, len(plan.variations) - 1)
        trial_idx = idx % trials_per_variation
        variation = plan.variations[var_idx]
        trial_for_name = trial_idx if with_trial_suffix else None
        child_name = build_child_name(
            sweep_name=name,
            sweep_run_epoch=sweep_run_epoch,
            variation_index=var_idx,
            trial_index=trial_for_name,
        )
        children.append(
            {
                "namespace": namespace,
                "name": child_name,
                "variation_index": var_idx,
                "variation_label": getattr(variation, "label", ""),
                "trial_index": trial_idx if with_trial_suffix else None,
                "child_run_epoch": sweep_run_epoch,
                "label": r.label,
                "status": "Succeeded" if r.success else "Failed",
            }
        )
    write_children_manifest(
        base_dir=base_dir,
        namespace=namespace,
        sweep_name=name,
        sweep_run_epoch=sweep_run_epoch,
        children=children,
    )


async def _idle_until_terminated() -> None:
    """Sleep forever; SIGTERM from K8s ends us cleanly."""
    while True:
        await asyncio.sleep(3600)


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

    from aiperf._cli_runner_helpers import aggregate_and_export, build_strategy
    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.kubernetes.client import k8s_client
    from aiperf.kubernetes.sweep_models import AIPerfSweepSpec
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
        status_writer = SweepStatusWriter(
            api, name=sweep_name, namespace=sweep_namespace
        )
        executor = K8sChildJobExecutor(
            api=api,
            sweep=sweep_cr,
            with_trial_suffix=needs_trial_suffix(
                multi_run_trials=(spec.multi_run.trials if spec.multi_run else None),
                has_convergence=spec.convergence is not None,
            ),
            base_dir=RESULTS_DIR,
            status_writer=status_writer,
            cancel_check=lambda: cancel_flag["requested"],
            sweep_run_epoch=sweep_run_epoch,
        )

        orchestrator = MultiRunOrchestrator(base_dir=RESULTS_DIR)
        try:
            all_results = await orchestrator.execute(
                plan,
                executor,
                cancel_check=lambda: cancel_flag["requested"],
            )
        finally:
            cancel_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cancel_task

        if not aggregate_marker_exists(RESULTS_DIR):
            await status_writer.aggregation_running()
            try:
                # Top-level strategy mirrors cli_runner.py — only used for
                # aggregate-path resolution; per-cell strategies were rebuilt
                # inside the orchestrator.
                strategy = build_strategy(plan, aiperf_logger)
                aggregate_dir = strategy.get_aggregate_path(RESULTS_DIR)
                aggregate_dir.mkdir(parents=True, exist_ok=True)
                await aggregate_and_export(
                    all_results,
                    plan,
                    strategy=strategy,
                    base_dir=RESULTS_DIR,
                    logger=aiperf_logger,
                )
                _write_aggregate_manifest(aggregate_dir, sweep_cr, all_results, plan)
                _write_sweep_parent_aggregate(
                    base_dir=RESULTS_DIR,
                    sweep_cr=sweep_cr,
                    spec=spec,
                    results=all_results,
                    plan=plan,
                    sweep_run_epoch=sweep_run_epoch,
                    with_trial_suffix=needs_trial_suffix(
                        multi_run_trials=(
                            spec.multi_run.trials if spec.multi_run else None
                        ),
                        has_convergence=spec.convergence is not None,
                    ),
                )
                write_aggregate_marker(RESULTS_DIR)
                controller_host = os.environ.get("HOSTNAME", "")
                await status_writer.aggregation_complete(
                    aggregate_path=(
                        f"/api/v1/results/{sweep_namespace}/{sweep_name}/aggregate"
                    ),
                    controller_host=controller_host,
                    port=RESULTS_SERVER_PORT,
                )
            except Exception as e:  # noqa: BLE001
                logger.exception("aggregation failed")
                await status_writer.aggregation_failed(error=str(e))
                return 1
        else:
            logger.info("aggregation already complete (marker present)")

        await _idle_until_terminated()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
