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
            status_writer=status_writer,
            cancel_check=lambda: cancel_flag["requested"],
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
                aggregate_and_export(
                    all_results,
                    plan,
                    strategy=strategy,
                    base_dir=RESULTS_DIR,
                    logger=aiperf_logger,
                )
                _write_aggregate_manifest(aggregate_dir, sweep_cr, all_results, plan)
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
