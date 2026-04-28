# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K8sChildJobExecutor: creates AIPerfJob children, watches them, collects results.

The single substantive seam between the shared MultiRunOrchestrator and the
K8s sweep flow. Task 13 (separate) implements the execute()/watch/result-pull
body; this module provides the helpers, identity check, and child-spec/metadata
construction.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson
from kubernetes_asyncio.client import ApiException, CustomObjectsApi

from aiperf.operator.environment import OperatorEnvironment
from aiperf.orchestrator.executor import RunExecutor
from aiperf.orchestrator.models import RunResult
from aiperf.sweep_controller._naming import (
    build_child_name,
    derive_child_name,
    needs_trial_suffix,
)
from aiperf.sweep_controller._naming import (
    sanitize_for_label as _sanitize_for_label,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from aiperf.config.benchmark import BenchmarkPlan, BenchmarkRun


logger = logging.getLogger(__name__)


SWEEP_LABEL = "aiperf.nvidia.com/sweep"
SWEEP_UID_LABEL = "aiperf.nvidia.com/sweep-uid"
SWEEP_RUN_EPOCH_LABEL = "aiperf.nvidia.com/sweep-run-epoch"
VARIATION_INDEX_LABEL = "aiperf.nvidia.com/variation-index"
VARIATION_LABEL_LABEL = "aiperf.nvidia.com/variation-label"
TRIAL_INDEX_LABEL = "aiperf.nvidia.com/trial-index"

TERMINAL_PHASES = frozenset(
    {"Completed", "Succeeded", "Failed", "Cancelled", "PartiallyFailed"}
)
RESULTS_SERVER_PORT = 19090
DEFAULT_POLL_INTERVAL_SECONDS = 5.0


__all__ = [
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "RESULTS_SERVER_PORT",
    "SWEEP_LABEL",
    "SWEEP_RUN_EPOCH_LABEL",
    "SWEEP_UID_LABEL",
    "TERMINAL_PHASES",
    "TRIAL_INDEX_LABEL",
    "VARIATION_INDEX_LABEL",
    "VARIATION_LABEL_LABEL",
    "ApiException",
    "ChildNameConflictError",
    "CustomObjectsApi",
    "K8sChildJobExecutor",
    "build_child_name",
    "derive_child_name",
    "is_my_child",
    "needs_trial_suffix",
    "write_child_sweep_marker",
]


class ChildNameConflictError(Exception):
    """Raised when a child-name slot is occupied by an AIPerfJob this sweep does not own."""


def is_my_child(child: dict[str, Any], *, sweep_uid: str, sweep_name: str) -> bool:
    """True if `child` is owned by the sweep (uid AND sweep label both match)."""
    meta = child.get("metadata", {})
    refs = meta.get("ownerReferences") or []
    owner_match = any(ref.get("uid") == sweep_uid for ref in refs)
    label_match = (meta.get("labels") or {}).get(SWEEP_LABEL) == sweep_name
    return owner_match and label_match


class K8sChildJobExecutor(RunExecutor):
    """RunExecutor that creates child AIPerfJob CRs and awaits their terminal phase.

    Designed to run inside the sweep-controller pod with a kubernetes_asyncio
    ApiClient connected via in-cluster service-account credentials.
    """

    def __init__(
        self,
        api: Any,
        sweep: dict[str, Any],
        *,
        with_trial_suffix: bool,
        base_dir: Path | None = None,
        status_writer: Any | None = None,
        cancel_check: Callable[[], bool] | None = None,
        sweep_run_epoch: str | None = None,
    ) -> None:
        self._api = api
        self.sweep = sweep
        self.sweep_name: str = sweep["metadata"]["name"]
        self.sweep_namespace: str = sweep["metadata"]["namespace"]
        self.sweep_uid: str = sweep["metadata"]["uid"]
        self.with_trial_suffix = with_trial_suffix
        self.base_dir = Path(base_dir) if base_dir is not None else None
        self._status_writer = status_writer
        self._cancel_check = cancel_check
        # Sweep-run epoch is stamped on each child as the
        # ``aiperf.nvidia.com/sweep-run-epoch`` label and written into the
        # per-child sweep marker file. It is **not** in the child name —
        # collisions with cascade-deleting prior-run children are handled by
        # ``_wait_for_stale_child`` instead. Optional only because in-process
        # unit tests construct executors without epoch wiring.
        self.sweep_run_epoch = sweep_run_epoch

    def derive_id(self, plan: BenchmarkPlan | None, var_idx: int, trial: int) -> str:
        return derive_child_name(
            self.sweep_name,
            var_idx,
            trial,
            with_trial_suffix=self.with_trial_suffix,
        )

    def _build_child_spec(self, run: BenchmarkRun) -> dict[str, Any]:
        """Produce the child AIPerfJob spec from the sweep template + this run."""
        template_spec = copy.deepcopy(self.sweep["spec"]["template"]["spec"])
        benchmark_dump = run.cfg.model_dump(
            mode="json", by_alias=True, exclude_none=True
        )
        # The orchestrator validates each variant as BenchmarkConfig, which fills in
        # runtime.service_run_type=multiprocessing (and similar k8s-context fields).
        # The child AIPerfJob operator validates as AIPerfConfig, which rejects
        # service_run_type as extra_forbidden — apply_k8s_runtime_config sets it
        # itself on the child side. Strip these fields so the child re-resolves them.
        runtime = benchmark_dump.get("runtime") or {}
        for k8s_resolved in (
            "serviceRunType",
            "service_run_type",
            "apiHost",
            "api_host",
            "apiPort",
            "api_port",
            "datasetApiBaseUrl",
            "dataset_api_base_url",
            "communication",
        ):
            runtime.pop(k8s_resolved, None)
        benchmark_dump["runtime"] = runtime
        template_spec["benchmark"] = benchmark_dump
        return template_spec

    def _build_child_metadata(
        self, run: BenchmarkRun, child_name: str
    ) -> dict[str, Any]:
        """Produce child metadata: name, namespace, labels, ownerReferences."""
        template_meta = copy.deepcopy(
            self.sweep["spec"]["template"].get("metadata") or {}
        )
        labels = dict(template_meta.get("labels") or {})
        labels[SWEEP_LABEL] = self.sweep_name
        labels[SWEEP_UID_LABEL] = self.sweep_uid
        if self.sweep_run_epoch is not None:
            labels[SWEEP_RUN_EPOCH_LABEL] = self.sweep_run_epoch
        if run.variation is not None:
            labels[VARIATION_INDEX_LABEL] = f"{run.variation.index:02d}"
            labels[VARIATION_LABEL_LABEL] = _sanitize_for_label(run.variation.label)
        if self.with_trial_suffix:
            labels[TRIAL_INDEX_LABEL] = f"{run.trial:01d}"
        return {
            "name": child_name,
            "namespace": self.sweep_namespace,
            "labels": labels,
            "annotations": template_meta.get("annotations") or {},
            "ownerReferences": [
                {
                    "apiVersion": "aiperf.nvidia.com/v1alpha1",
                    "kind": "AIPerfSweep",
                    "name": self.sweep_name,
                    "uid": self.sweep_uid,
                    "controller": True,
                    "blockOwnerDeletion": True,
                }
            ],
        }

    async def execute(self, run: BenchmarkRun) -> RunResult:
        """Get-or-create the child, await terminal phase, then collect a RunResult."""
        var_idx = run.variation.index if run.variation else 0
        child_name = self.derive_id(plan=None, var_idx=var_idx, trial=run.trial)
        if self._cancel_check is not None and self._cancel_check():
            logger.info(f"cancel requested before starting child {child_name}")
            return RunResult(
                label=run.label,
                success=False,
                error="sweep cancelled before child started",
                artifacts_path=run.artifact_dir,
            )
        if self._status_writer is not None:
            try:
                await self._status_writer.current_cell(
                    variation_index=var_idx,
                    label=run.label,
                    trial=run.trial,
                )
            except Exception as e:  # noqa: BLE001 - status update is best-effort
                logger.warning(f"current_cell status write failed: {e}")
        await self._get_or_create(child_name, run)
        await self._wait_until_terminal(child_name, cancel_check=self._cancel_check)
        terminal = await self._try_read_child(child_name)
        if terminal is None:
            return RunResult(
                label=run.label,
                success=False,
                error=f"child {child_name} disappeared before terminal phase",
                artifacts_path=run.artifact_dir,
            )
        return await self._collect_run_result(terminal, run)

    async def _try_read_child(self, name: str) -> dict[str, Any] | None:
        """Read an AIPerfJob by name; return None on 404."""
        custom = CustomObjectsApi(self._api)
        try:
            return await custom.get_namespaced_custom_object(
                group="aiperf.nvidia.com",
                version="v1alpha1",
                namespace=self.sweep_namespace,
                plural="aiperfjobs",
                name=name,
            )
        except ApiException as e:
            if getattr(e, "status", None) == 404:
                return None
            raise

    async def _wait_for_stale_child(self, name: str) -> dict[str, Any] | None:
        """If a same-named AIPerfJob from a prior sweep run is mid-deletion,
        wait for cascade-delete to complete before our caller creates a new one.

        Triggered when a user deletes a sweep CR and re-creates one with the
        same name while old children are still terminating: the new
        sweep-controller's child-creates would otherwise race with the kube
        garbage collector.

        Returns:
          - the existing AIPerfJob if it is owned by *us* (resumable), or
          - ``None`` if no AIPerfJob with this name exists (free slot, caller may create).

        Raises ChildNameConflictError when:
          - the existing AIPerfJob is foreign and not deleting (real conflict), or
          - the existing AIPerfJob is still mid-deletion past
            ``OperatorEnvironment.SWEEP_CONTROLLER.STALE_CHILD_DELETION_TIMEOUT_SECONDS``
            (likely a stuck finalizer on the prior sweep).
        """
        deadline = (
            asyncio.get_event_loop().time()
            + OperatorEnvironment.SWEEP_CONTROLLER.STALE_CHILD_DELETION_TIMEOUT_SECONDS
        )
        poll = OperatorEnvironment.SWEEP_CONTROLLER.STALE_CHILD_POLL_INTERVAL_SECONDS
        while True:
            existing = await self._try_read_child(name)
            if existing is None:
                return None
            if is_my_child(
                existing, sweep_uid=self.sweep_uid, sweep_name=self.sweep_name
            ):
                return existing
            if (existing.get("metadata") or {}).get("deletionTimestamp") is None:
                raise ChildNameConflictError(
                    f"child name {name!r} exists and is not owned by this sweep "
                    f"(uid={self.sweep_uid})"
                )
            if asyncio.get_event_loop().time() > deadline:
                raise ChildNameConflictError(
                    f"child name {name!r} still mid-deletion after "
                    f"{OperatorEnvironment.SWEEP_CONTROLLER.STALE_CHILD_DELETION_TIMEOUT_SECONDS}s "
                    f"— prior sweep may have a stuck finalizer"
                )
            logger.info(f"waiting for prior child {name!r} to finish cascade-deletion")
            await asyncio.sleep(poll)

    async def _get_or_create(self, name: str, run: BenchmarkRun) -> dict[str, Any]:
        """Read the child if it exists; otherwise create it from the sweep template."""
        existing = await self._wait_for_stale_child(name)
        if existing is not None:
            logger.info(f"resuming existing child {name}")
            return existing
        body = {
            "apiVersion": "aiperf.nvidia.com/v1alpha1",
            "kind": "AIPerfJob",
            "metadata": self._build_child_metadata(run, name),
            "spec": self._build_child_spec(run),
        }
        if (
            self.base_dir is not None
            and run.variation is not None
            and self.sweep_run_epoch is not None
        ):
            try:
                write_child_sweep_marker(
                    base_dir=self.base_dir,
                    namespace=self.sweep_namespace,
                    child_name=name,
                    sweep_name=self.sweep_name,
                    variation_index=run.variation.index,
                    variation_label=run.variation.label,
                    trial_index=run.trial if self.with_trial_suffix else None,
                    sweep_run_epoch=self.sweep_run_epoch,
                    # Fresh child: child epoch == sweep epoch. On a child rerun
                    # (Task 6+), the controller derives a new child epoch and
                    # passes it here so the back-link points at the right run.
                    child_run_epoch=self.sweep_run_epoch,
                )
            except OSError as e:
                logger.warning(
                    f"failed to write child sweep marker for {name}: {e}; continuing"
                )
        custom = CustomObjectsApi(self._api)
        logger.info(f"creating child {name}")
        return await custom.create_namespaced_custom_object(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            namespace=self.sweep_namespace,
            plural="aiperfjobs",
            body=body,
        )

    async def _wait_until_terminal(
        self,
        child_name: str,
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        """Poll the child until status.phase reaches a terminal value.

        Periodic list-fallback rather than long-lived Watch: simpler under
        partial network failures, and AIPerfJob phase transitions are rare
        enough that a 5s poll is fine.
        """
        while True:
            child = await self._try_read_child(child_name)
            phase = (child or {}).get("status", {}).get("phase")
            if phase in TERMINAL_PHASES:
                return
            if cancel_check is not None and cancel_check():
                logger.info(f"cancel requested while waiting on {child_name}")
                await self._patch_child_cancel(child_name)
                # Continue polling; the operator will eventually mark Cancelled.
            await asyncio.sleep(poll_interval)

    async def _patch_child_cancel(self, child_name: str) -> None:
        """Patch the child's spec.cancel = true to propagate the cancel signal."""
        custom = CustomObjectsApi(self._api)
        await custom.patch_namespaced_custom_object(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            namespace=self.sweep_namespace,
            plural="aiperfjobs",
            name=child_name,
            body={"spec": {"cancel": True}},
        )

    async def _collect_run_result(
        self, child: dict[str, Any], run: BenchmarkRun
    ) -> RunResult:
        """Translate a terminal child + summary metrics into a RunResult."""
        status = child.get("status") or {}
        phase = status.get("phase")
        if phase not in {"Completed", "Succeeded"}:
            return RunResult(
                label=run.label,
                success=False,
                error=status.get("message") or f"child terminal phase={phase}",
                artifacts_path=run.artifact_dir,
            )
        metrics = await self._pull_summary_metrics(child)
        # A terminal-success child with empty summary is still a success — the
        # child finished without error, the operator just hasn't written
        # status.summary yet (or wrote an empty one). Surface this as success
        # with empty metrics so failure_policy doesn't trip on a write race;
        # the warning is logged inside _pull_summary_metrics.
        return RunResult(
            label=run.label,
            success=True,
            summary_metrics=metrics,
            artifacts_path=run.artifact_dir,
        )

    async def _pull_summary_metrics(self, child: dict[str, Any]) -> dict[str, Any]:
        """Read per-cell summary metrics directly from AIPerfJob.status.summary.

        The AIPerfJob operator writes the summary dict (latency_avg_ms,
        throughput_rps, ttft_p99_ms, etc.) into status.summary at completion
        time — no HTTP fetch needed.

        ``status.summary`` mixes JsonMetricResult-shaped per-tag dicts with
        bolted-on top-level scalars (``total_requests``, ``error_rate``); the
        scalars and any per-tag extras (``count``, ``header``, ``sum``) are
        filtered out by ``JsonMetricResult.project_summary_dict`` so the
        downstream ``RunResult.summary_metrics: dict[str, JsonMetricResult]``
        Pydantic validation accepts the result.
        """
        from aiperf.common.models.export_models import JsonMetricResult

        status = child.get("status") or {}
        summary = status.get("summary") or {}
        name = child["metadata"]["name"]
        if not summary:
            logger.warning(f"child {name}: status.summary is empty")
            return {}
        return JsonMetricResult.project_summary_dict(summary)


def write_child_sweep_marker(
    *,
    base_dir: Path,
    namespace: str,
    child_name: str,
    sweep_name: str,
    variation_index: int,
    variation_label: str,
    trial_index: int | None,
    sweep_run_epoch: str,
    child_run_epoch: str,
) -> None:
    """Drop the per-child ``sweep.json`` marker into the child's results directory.

    Called by the sweep-controller before each child AIPerfJob CR is created;
    the marker survives parent-CR TTL reap so the operator's job_union can
    populate the back-link on archived children. Atomic write via ``os.replace``.

    ``sweep_run_epoch`` and ``child_run_epoch`` are read by job_union and the
    dual-backed jobs API for back-link rendering on archived children. For a
    fresh child, the two epochs are equal; on a child rerun (Task 6+) the child
    epoch advances independently of the sweep epoch.

    Idempotent: overwriting an existing marker is fine, since deterministic
    child names anchor identity to the apiserver, not to the marker.
    """
    target_dir = Path(base_dir) / namespace / child_name
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sweep_name": sweep_name,
        "variation_index": variation_index,
        "variation_label": variation_label,
        "trial_index": trial_index,
        "sweep_run_epoch": sweep_run_epoch,
        "child_run_epoch": child_run_epoch,
    }
    fd, tmp_path = tempfile.mkstemp(prefix=".sweep.", suffix=".json", dir=target_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        os.replace(tmp_path, target_dir / "sweep.json")
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
