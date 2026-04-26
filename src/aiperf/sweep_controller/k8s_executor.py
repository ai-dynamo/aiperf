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
import copy
import logging
import re
from typing import TYPE_CHECKING, Any

from kubernetes_asyncio.client import ApiException, CustomObjectsApi

from aiperf.orchestrator.executor import RunExecutor
from aiperf.orchestrator.models import RunResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from aiperf.config.benchmark import BenchmarkPlan, BenchmarkRun


logger = logging.getLogger(__name__)


SWEEP_LABEL = "aiperf.nvidia.com/sweep"
SWEEP_UID_LABEL = "aiperf.nvidia.com/sweep-uid"
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
    "SWEEP_UID_LABEL",
    "TERMINAL_PHASES",
    "TRIAL_INDEX_LABEL",
    "VARIATION_INDEX_LABEL",
    "VARIATION_LABEL_LABEL",
    "ApiException",
    "ChildNameConflictError",
    "CustomObjectsApi",
    "K8sChildJobExecutor",
    "derive_child_name",
    "is_my_child",
    "needs_trial_suffix",
]


class ChildNameConflictError(Exception):
    """Raised when a child-name slot is occupied by an AIPerfJob this sweep does not own."""


def needs_trial_suffix(
    multi_run_trials: int | None,
    has_convergence: bool,
) -> bool:
    """Whether child names should include a `-tNN` trial suffix."""
    if has_convergence:
        return True
    return (multi_run_trials or 1) > 1


def derive_child_name(
    sweep_name: str,
    var_idx: int,
    trial: int,
    *,
    with_trial_suffix: bool,
) -> str:
    """Deterministic DNS-safe child name from (sweep, var_idx, trial)."""
    base = f"{sweep_name}-v{var_idx:04d}"
    if with_trial_suffix:
        return f"{base}-t{trial:02d}"
    return base


def is_my_child(child: dict[str, Any], *, sweep_uid: str, sweep_name: str) -> bool:
    """True if `child` is owned by the sweep (uid AND sweep label both match)."""
    meta = child.get("metadata", {})
    refs = meta.get("ownerReferences") or []
    owner_match = any(ref.get("uid") == sweep_uid for ref in refs)
    label_match = (meta.get("labels") or {}).get(SWEEP_LABEL) == sweep_name
    return owner_match and label_match


def _sanitize_for_label(value: str) -> str:
    """Reduce a free-form string to a valid k8s label value.

    Label values must match ``(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?`` and
    be at most 63 characters. We:

    1. Lowercase + replace runs of disallowed chars with a single ``-``.
    2. Strip leading/trailing non-alnum.
    3. Truncate to 63.
    4. Re-strip leading/trailing non-alnum (the truncation may have left a
       trailing ``.``/``_``/``-``, which would re-fail validation).
    5. Fall back to ``"v"`` when sanitization eats every character.
    """
    sanitized = re.sub(r"[^a-z0-9._-]+", "-", value.lower())
    sanitized = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", sanitized)
    sanitized = sanitized[:63]
    sanitized = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", sanitized)
    return sanitized or "v"


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
        status_writer: Any | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        self._api = api
        self.sweep = sweep
        self.sweep_name: str = sweep["metadata"]["name"]
        self.sweep_namespace: str = sweep["metadata"]["namespace"]
        self.sweep_uid: str = sweep["metadata"]["uid"]
        self.with_trial_suffix = with_trial_suffix
        self._status_writer = status_writer
        self._cancel_check = cancel_check

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
        if run.variation is not None:
            labels[VARIATION_INDEX_LABEL] = f"{run.variation.index:04d}"
            labels[VARIATION_LABEL_LABEL] = _sanitize_for_label(run.variation.label)
        if self.with_trial_suffix:
            labels[TRIAL_INDEX_LABEL] = f"{run.trial:02d}"
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

    async def _get_or_create(self, name: str, run: BenchmarkRun) -> dict[str, Any]:
        """Read the child if it exists; otherwise create it from the sweep template."""
        existing = await self._try_read_child(name)
        if existing is not None:
            if is_my_child(
                existing, sweep_uid=self.sweep_uid, sweep_name=self.sweep_name
            ):
                logger.info(f"resuming existing child {name}")
                return existing
            raise ChildNameConflictError(
                f"child name {name!r} exists but is not owned by this sweep "
                f"(uid={self.sweep_uid})"
            )
        body = {
            "apiVersion": "aiperf.nvidia.com/v1alpha1",
            "kind": "AIPerfJob",
            "metadata": self._build_child_metadata(run, name),
            "spec": self._build_child_spec(run),
        }
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
        """
        status = child.get("status") or {}
        summary = status.get("summary") or {}
        name = child["metadata"]["name"]
        if not summary:
            logger.warning(f"child {name}: status.summary is empty")
            return {}
        return dict(summary)
