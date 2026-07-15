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

import aiohttp
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

    from aiperf.common.models.export_models import JsonMetricResult
    from aiperf.config.resolution.plan import BenchmarkPlan, BenchmarkRun


logger = logging.getLogger(__name__)


SWEEP_LABEL = "aiperf.nvidia.com/sweep"
SWEEP_UID_LABEL = "aiperf.nvidia.com/sweep-uid"
SWEEP_RUN_EPOCH_LABEL = "aiperf.nvidia.com/sweep-run-epoch"
VARIATION_INDEX_LABEL = "aiperf.nvidia.com/variation-index"
VARIATION_LABEL_LABEL = "aiperf.nvidia.com/variation-label"
VARIATION_VALUES_ANNOTATION = "aiperf.nvidia.com/variation-values"
VARIATION_VALUES_MAX_ANNOTATION_BYTES = 2048
TRIAL_INDEX_LABEL = "aiperf.nvidia.com/trial-index"

TERMINAL_PHASES = frozenset(
    {"Completed", "Succeeded", "Failed", "Cancelled", "PartiallyFailed"}
)
# Port the sweep-controller pod's own results-sidecar listens on (in-pod
# service to expose its parent-aggregate over HTTP). DISTINCT from the
# OPERATOR'S results-server port (`aiperf.kubernetes.results_operator.
# RESULTS_SERVER_PORT`, default 8081) — different containers, different
# pods, different default values. The shared name was a footgun: the two
# constants live in different modules but are easy to misimport.
SWEEP_CONTROLLER_RESULTS_SIDECAR_PORT = 19090
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
# When ``_pull_summary_metrics`` reads a terminal-Completed child but neither
# ``status.summary`` nor ``status.runEpoch`` is populated yet, refresh the CR
# this many times with this delay before falling back. The operator's monitor
# tick / completion handler can land AFTER ``phase=Completed`` is observed on
# the sweep-controller side — fast adaptive probes (e.g. concurrency=1, low
# request count) finish in seconds and routinely race the operator reconcile.
# Without this, the bracket collapses to ``observed: null`` because both the
# primary read and the operator-API fallback (which needs ``runEpoch``) come
# up empty.
SUMMARY_RACE_REFRESH_ATTEMPTS = 6
SUMMARY_RACE_REFRESH_SECONDS = 2.0


__all__ = [
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "SUMMARY_RACE_REFRESH_ATTEMPTS",
    "SUMMARY_RACE_REFRESH_SECONDS",
    "SWEEP_CONTROLLER_RESULTS_SIDECAR_PORT",
    "SWEEP_LABEL",
    "SWEEP_RUN_EPOCH_LABEL",
    "SWEEP_UID_LABEL",
    "TERMINAL_PHASES",
    "TRIAL_INDEX_LABEL",
    "VARIATION_INDEX_LABEL",
    "VARIATION_LABEL_LABEL",
    "VARIATION_VALUES_ANNOTATION",
    "VARIATION_VALUES_MAX_ANNOTATION_BYTES",
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


def _variation_values_truncated_payload(original_bytes: int) -> dict[str, Any]:
    return {
        "__aiperf_truncated__": True,
        "reason": "variation values exceeded metadata byte limit",
        "limitBytes": VARIATION_VALUES_MAX_ANNOTATION_BYTES,
        "originalBytes": original_bytes,
    }


def _bounded_variation_values_json(values: Any) -> str:
    encoded = orjson.dumps(values)
    if len(encoded) <= VARIATION_VALUES_MAX_ANNOTATION_BYTES:
        return encoded.decode()
    return orjson.dumps(_variation_values_truncated_payload(len(encoded))).decode()


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
        # Accumulated terminal-child manifest entries — appended in
        # _record_terminal_child after each cell completes, snapshotted
        # onto the parent CR via status_writer.partial_children. Lives on
        # the executor (one per sweep) so survives the orchestrator's
        # variation/trial loop without external state.
        self._terminal_children: list[dict[str, Any]] = []
        # The child object ``_pull_summary_metrics`` last resolved — possibly a
        # refreshed read whose ``status.runEpoch`` was stamped AFTER the
        # terminal-phase read in ``execute``. Both the RunResult's
        # ``child_run_epoch`` and the children-manifest back-link must derive
        # the epoch from THIS object, not the stale terminal read, or the
        # variation silently drops out of the runs index (the result dir is
        # written under the child's real epoch).
        self._last_resolved_child: dict[str, Any] | None = None

    def derive_id(self, plan: BenchmarkPlan | None, var_idx: int, trial: int) -> str:
        return derive_child_name(
            self.sweep_name,
            var_idx,
            trial,
            with_trial_suffix=self.with_trial_suffix,
        )

    def _build_child_spec(self, run: BenchmarkRun) -> dict[str, Any]:
        """Build a child AIPerfJob spec from the parent AIPerfSweep + this run.

        The parent AIPerfSweep CR carries the flat envelope shape (no
        `template` wrapping). The child AIPerfJob spec gets:
          - All deployment fields (image, podTemplate, resources, ...) and
            inheritable envelope fields (multi_run, variables, random_seed)
            inherited verbatim from the parent.
          - benchmark = the rendered per-variation BenchmarkConfig.
          - sweep = None (single variation, no further fanout).
          - Stripped: AIPerfSweep-only orchestration metadata that must not
            propagate to children. The parent's failurePolicy governs
            sweep-level abort behavior, and its ttlSecondsAfterFinished would
            delete children (and their results) out from under the sweep
            controller before the aggregate harvest.
        """
        parent_spec = self.sweep["spec"]
        # The apiserver stores camelCase (declared CRD property names); the
        # snake_case spellings cover hand-built CRs and tests — strip both,
        # mirroring how _build_child_metadata reads childMetadata.
        child_spec: dict[str, Any] = {
            k: copy.deepcopy(v)
            for k, v in parent_spec.items()
            if k
            not in {
                "sweep",
                "failurePolicy",
                "failure_policy",
                "cancel",
                "ttlSecondsAfterFinished",
                "ttl_seconds_after_finished",
                "childMetadata",
                "child_metadata",
            }
        }
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
        child_spec["benchmark"] = benchmark_dump
        child_spec["sweep"] = None
        return child_spec

    def _build_child_metadata(
        self, run: BenchmarkRun, child_name: str
    ) -> dict[str, Any]:
        """Produce child metadata: name, namespace, labels, ownerReferences.

        User-supplied labels/annotations come from the optional
        ``spec.childMetadata`` (snake_case ``child_metadata``) field on
        AIPerfSweep. They are merged first, then sweep-tracking entries
        (sweep, sweep-uid, sweep-run-epoch, variation-*, trial-index) are
        applied last so they ALWAYS win against any user-supplied key with
        the same name. Sweep-tracking labels drive the label-selector queries
        that find children for status rollup; allowing user override would
        silently break ``is_my_child``/list-children logic.
        """
        parent_spec = self.sweep["spec"]
        # Read camelCase first (CRD storage normalizes to declared property
        # names) and fall back to snake_case for tests / hand-built CRs.
        child_meta_input = (
            parent_spec.get("childMetadata") or parent_spec.get("child_metadata") or {}
        )
        user_labels = dict(child_meta_input.get("labels") or {})
        user_annotations = dict(child_meta_input.get("annotations") or {})

        labels: dict[str, str] = {**user_labels}
        labels[SWEEP_LABEL] = self.sweep_name
        labels[SWEEP_UID_LABEL] = self.sweep_uid
        if self.sweep_run_epoch is not None:
            labels[SWEEP_RUN_EPOCH_LABEL] = self.sweep_run_epoch
        if run.variation is not None:
            labels[VARIATION_INDEX_LABEL] = f"{run.variation.index:02d}"
            labels[VARIATION_LABEL_LABEL] = _sanitize_for_label(run.variation.label)
            user_annotations[VARIATION_VALUES_ANNOTATION] = (
                _bounded_variation_values_json(run.variation.values)
            )
        if self.with_trial_suffix:
            labels[TRIAL_INDEX_LABEL] = f"{run.trial:01d}"
        return {
            "name": child_name,
            "namespace": self.sweep_namespace,
            "labels": labels,
            "annotations": user_annotations,
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
        # Honor the orchestrator's authoritative child name rather than
        # re-deriving here: ``BenchmarkRun.benchmark_id`` is set by
        # ``orchestrator.derive_id(plan, var_idx, trial)`` at construction and
        # is the single source of truth for the child AIPerfJob name. Re-running
        # ``derive_id`` from ``run.variation.index`` is equivalent today but
        # would silently diverge if the orchestrator ever maps variation index
        # to a dense child slot (e.g. to fit the 0..199 child-name budget under
        # adaptive search where ``variation.index`` is the iteration counter).
        child_name = run.benchmark_id or self.derive_id(
            plan=None, var_idx=var_idx, trial=run.trial
        )
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
        cancelled = await self._wait_until_terminal(
            child_name, run, cancel_check=self._cancel_check
        )
        if cancelled is not None:
            return cancelled
        terminal = await self._try_read_child(child_name)
        if terminal is None:
            return RunResult(
                label=run.label,
                success=False,
                error=f"child {child_name} disappeared before terminal phase",
                artifacts_path=run.artifact_dir,
            )
        self._last_resolved_child = None
        result = await self._collect_run_result(terminal, run)
        # Record against the child object ``_collect_run_result`` actually
        # resolved (a race-grace refresh may have stamped runEpoch after the
        # terminal read), falling back to ``terminal`` when no refresh ran.
        await self._record_terminal_child(
            child_name, run, self._last_resolved_child or terminal
        )
        return result

    async def _record_terminal_child(
        self, child_name: str, run: BenchmarkRun, child: dict[str, Any]
    ) -> None:
        """Append a terminal child to ``status.aggregate.children`` incrementally.

        Without this, ``status.aggregate.children`` only appears after
        ``aggregation_complete`` patches the full doc — a multi-minute
        delay during which any consumer reading the manifest (SweepDetail's
        live-variations rollup, watch loops, ``aiperf kube list``) sees
        an empty list. After each cell terminates, snapshot the running
        ``self._terminal_children`` list onto the parent CR. The terminal
        writer overwrites the same path with the full post-aggregation
        manifest, so partial snapshots are never load-bearing downstream.
        """
        var_idx = run.variation.index if run.variation else 0
        var_label = run.variation.label if run.variation else ""
        child_run_epoch = str((child.get("status") or {}).get("runEpoch") or "")
        self._terminal_children.append(
            {
                "namespace": self.sweep_namespace,
                "name": child_name,
                "variation_index": var_idx,
                "variation_label": var_label,
                "trial_index": run.trial,
                "child_run_epoch": child_run_epoch,
            }
        )
        if self._status_writer is None:
            return
        try:
            await self._status_writer.partial_children(
                sweep_run_epoch=self.sweep_run_epoch,
                children=list(self._terminal_children),
            )
        except Exception as e:  # noqa: BLE001 — partial-manifest patch is best-effort
            logger.warning(f"partial_children status write failed: {e}")

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
        """Read the child if it exists; otherwise create it from the parent AIPerfSweep."""
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
                    # Provisional back-link written at create time, before the
                    # operator stamps the child's own ``status.runEpoch`` (which
                    # it derives from the child AIPerfJob's creationTimestamp/uid
                    # via epoch_key_from_body — NOT equal to the sweep epoch).
                    # The authoritative epoch flows into children.json from
                    # ``_record_terminal_child`` once the child reaches terminal
                    # phase; this marker is the best-effort value available pre-
                    # terminal for job_union's archived-child back-link.
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
        run: BenchmarkRun,
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
        cancel_check: Callable[[], bool] | None = None,
    ) -> RunResult | None:
        """Poll the child until status.phase reaches a terminal value.

        Periodic list-fallback rather than long-lived Watch: simpler under
        partial network failures, and AIPerfJob phase transitions are rare
        enough that a 5s poll is fine.

        Returns ``None`` once the child reaches a terminal phase (caller
        proceeds to collect the result). On cancellation, the cancel
        merge-patch is issued exactly once, then the wait is bounded by
        ``CANCEL_GRACE_SECONDS``: if the child has not reached a terminal
        phase by the deadline (operator cancel path stalled, wedged pod,
        repeatedly-failing JobSet delete), a cancelled ``RunResult`` is
        returned so the orchestrator advances instead of blocking forever.

        Independently of cancel, a child that goes missing (404) before its
        terminal phase — deleted out-of-band by a user or the kube garbage
        collector — arms a ``CHILD_MISSING_TIMEOUT_SECONDS`` deadline. Once
        the child has been continuously absent past that bound, a cancelled
        ``RunResult`` is returned so the sequential sweep advances instead of
        polling a deleted variation forever. A reappearing child (the missing
        read was transient) clears the deadline.
        """
        cancel_patched = False
        cancel_deadline: float | None = None
        missing_deadline: float | None = None
        while True:
            child = await self._try_read_child(child_name)
            phase = (child or {}).get("status", {}).get("phase")
            if phase in TERMINAL_PHASES:
                return None
            if child is None:
                if missing_deadline is None:
                    missing_deadline = (
                        asyncio.get_event_loop().time()
                        + OperatorEnvironment.SWEEP_CONTROLLER.CHILD_MISSING_TIMEOUT_SECONDS
                    )
                elif asyncio.get_event_loop().time() > missing_deadline:
                    logger.warning(
                        f"child {child_name} missing (404) for more than "
                        f"{OperatorEnvironment.SWEEP_CONTROLLER.CHILD_MISSING_TIMEOUT_SECONDS}s "
                        f"before reaching a terminal phase; advancing sweep"
                    )
                    return RunResult(
                        label=run.label,
                        success=False,
                        error=f"child {child_name} disappeared before terminal "
                        f"phase; phase=Cancelled",
                        artifacts_path=run.artifact_dir,
                    )
            else:
                missing_deadline = None
            if cancel_check is not None and cancel_check():
                if not cancel_patched:
                    logger.info(f"cancel requested while waiting on {child_name}")
                    await self._patch_child_cancel(child_name)
                    cancel_patched = True
                    cancel_deadline = (
                        asyncio.get_event_loop().time()
                        + OperatorEnvironment.SWEEP_CONTROLLER.CANCEL_GRACE_SECONDS
                    )
                elif (
                    cancel_deadline is not None
                    and asyncio.get_event_loop().time() > cancel_deadline
                ):
                    logger.warning(
                        f"child {child_name} did not reach terminal phase within "
                        f"{OperatorEnvironment.SWEEP_CONTROLLER.CANCEL_GRACE_SECONDS}s "
                        f"cancel grace; advancing sweep"
                    )
                    return RunResult(
                        label=run.label,
                        success=False,
                        error=f"child {child_name} did not reach terminal phase "
                        f"within cancel grace; phase=Cancelled",
                        artifacts_path=run.artifact_dir,
                    )
                # Otherwise keep polling; the operator will eventually mark Cancelled.
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
            _content_type="application/merge-patch+json",
        )

    async def _collect_run_result(
        self, child: dict[str, Any], run: BenchmarkRun
    ) -> RunResult:
        """Translate a terminal child + summary metrics into a RunResult."""
        status = child.get("status") or {}
        phase = status.get("phase")
        if phase not in {"Completed", "Succeeded"}:
            # A child cancelled out of band keeps phase=Cancelled, but the
            # operator's on_cancel does NOT clear a stale status.error stamped
            # by an earlier monitor tick. Anchor the error on the terminal
            # phase so ``main._is_cancelled_result`` (which discriminates on the
            # ``phase=Cancelled`` suffix) buckets it as cancelled, not failed —
            # otherwise the leftover error trips ``failed >= max_failures`` and
            # mis-resolves an externally cancelled sweep to Failed.
            if phase == "Cancelled":
                error = f"child terminal phase={phase}"
            else:
                error = (
                    status.get("error")
                    or status.get("message")
                    or f"child terminal phase={phase}"
                )
            return RunResult(
                label=run.label,
                success=False,
                error=error,
                artifacts_path=run.artifact_dir,
                child_run_epoch=str(status.get("runEpoch") or ""),
            )
        metrics = await self._pull_summary_metrics(child)
        # A terminal-success child with empty summary is still a success — the
        # child finished without error, the operator just hasn't written
        # status.summary yet (or wrote an empty one). Surface this as success
        # with empty metrics so failure_policy doesn't trip on a write race;
        # the warning is logged inside _pull_summary_metrics.
        #
        # Re-derive status from the child _pull_summary_metrics last resolved:
        # the race-grace refresh may have stamped runEpoch AFTER the terminal
        # read that produced ``child`` here. Using the stale epoch back-links
        # the manifest at the wrong (empty) epoch and drops the variation from
        # the runs index, even though the metrics themselves were recovered.
        resolved = self._last_resolved_child or child
        resolved_status = resolved.get("status") or status
        return RunResult(
            label=run.label,
            success=True,
            summary_metrics=metrics,
            artifacts_path=run.artifact_dir,
            child_run_epoch=str(resolved_status.get("runEpoch") or ""),
        )

    async def _pull_summary_metrics(self, child: dict[str, Any]) -> dict[str, Any]:
        """Read per-cell summary metrics from the child AIPerfJob.

        Primary path: AIPerfJob.status.summary, written by the operator's
        monitor tick at completion time — no HTTP fetch needed.

        Fallback path: when ``status.summary`` is empty (the
        ``CompletedBeforeMonitor`` race, or a completion-handler bug that
        skips the summary write), fetch ``profile_export_aiperf.json`` from
        the operator's PVC-backed results API. The PVC survives JobSet
        deletion, so this is robust against the controller-pod-already-gone
        race that breaks any per-child sidecar fetch.

        Race-aware refresh: when both ``status.summary`` AND ``status.runEpoch``
        are unset on the child, the operator's reconcile may simply not have
        run yet — ``_wait_until_terminal`` returns as soon as ``status.phase``
        is in ``TERMINAL_PHASES``, but ``set_summary`` / ``set_run_epoch`` fire
        from a separate code path that isn't atomic with the phase write.
        Without this re-read, fast adaptive probes (concurrency=1, few
        requests) collapse the SLA bracket to ``observed: null`` because both
        primary AND fallback see empty state. Six refreshes × 2s = 12s grace
        is enough to absorb a missed monitor tick (5s default) without
        meaningfully slowing the orchestrator.

        ``status.summary`` mixes JsonMetricResult-shaped per-tag dicts with
        bolted-on top-level scalars (``total_requests``, ``error_rate``); the
        scalars and any per-tag extras (``count``, ``header``, ``sum``) are
        filtered out by ``JsonMetricResult.project_summary_dict`` so the
        downstream ``RunResult.summary_metrics: dict[str, JsonMetricResult]``
        Pydantic validation accepts the result.
        """
        from aiperf.common.models.export_models import JsonMetricResult

        # Track the child this method last resolved so the caller derives
        # ``child_run_epoch`` from the (possibly refreshed) object below,
        # not the stale terminal read. Updated on every rebind of ``child``.
        self._last_resolved_child = child
        status = child.get("status") or {}
        summary = status.get("summary") or {}
        name = child["metadata"]["name"]
        if summary:
            return JsonMetricResult.project_summary_dict(summary)

        # Race grace: the child reached terminal phase but the operator's
        # next reconcile has not yet stamped status.summary or runEpoch. Both
        # the primary read AND the operator-API fallback need at least one of
        # those, so re-read the CR a few times before giving up. We exit the
        # loop the moment either field is populated; the first hit short-
        # circuits the worst-case 12s wait.
        if not status.get("runEpoch"):
            for attempt in range(SUMMARY_RACE_REFRESH_ATTEMPTS):
                await asyncio.sleep(SUMMARY_RACE_REFRESH_SECONDS)
                refreshed = await self._try_read_child(name)
                if refreshed is None:
                    break
                child = refreshed
                self._last_resolved_child = child
                status = refreshed.get("status") or {}
                summary = status.get("summary") or {}
                if summary:
                    logger.info(
                        f"child {name}: status.summary populated after "
                        f"{(attempt + 1) * SUMMARY_RACE_REFRESH_SECONDS:.0f}s grace"
                    )
                    return JsonMetricResult.project_summary_dict(summary)
                if status.get("runEpoch"):
                    logger.info(
                        f"child {name}: runEpoch populated after "
                        f"{(attempt + 1) * SUMMARY_RACE_REFRESH_SECONDS:.0f}s grace; "
                        f"attempting operator-API fallback"
                    )
                    break

        recovered = await self._fetch_summary_from_operator(child)
        if recovered:
            # Log the actual tag set so an SLA-filter bracket collapse with
            # ``observed: null`` can be diagnosed against the recovered
            # payload, not against the disk file. The SLA filter is keyed on
            # plain metric tags (``time_to_first_token``); naming the keys here
            # makes a missing-tag mismatch obvious in operator logs.
            tags = sorted(recovered.keys())
            sample = ", ".join(tags[:8])
            if len(tags) > 8:
                sample += f", ... (+{len(tags) - 8} more)"
            logger.info(
                f"child {name}: recovered summary via operator API "
                f"({len(recovered)} metrics): [{sample}]"
            )
            return recovered
        logger.warning(
            f"child {name}: status.summary is empty and operator API fetch failed"
        )
        return {}

    async def _fetch_summary_from_operator(
        self, child: dict[str, Any]
    ) -> dict[str, JsonMetricResult]:
        """Fetch ``profile_export_aiperf.json`` from the operator's results API.

        Hits ``{AIPERF_OPERATOR_BASE_URL}/api/v1/results/{ns}/{name}/runs/{epoch}/profile_export_aiperf.json``.
        BASE_URL points at the results-server container (port 8081 in the
        chart) — that's the only container in the operator Pod that hosts
        ``/api/v1/*`` routers; the operator container on port 8080 has only
        kopf health/metrics. Skips the call when the child has no
        ``status.runEpoch`` yet.

        Note on ``runEpoch`` semantics: ``set_run_epoch`` is invoked from
        ``_record_results_on_status`` only when ``has_files=True`` — so
        Failed / Cancelled children (which by definition have no results
        files) NEVER carry ``runEpoch``, and that's correct: the fallback
        has nothing to fetch. The other case ``runEpoch`` may be unset is
        the transient ``phase=Completed`` window before the operator's
        next reconcile stamps the epoch label; the orchestrator's outer
        retry loop covers that race naturally. Without an epoch the URL
        would be 422-rejected by the operator's epoch allowlist (regex
        ``^\\d{9,11}$``), so short-circuiting is also safer than
        synthesizing ``latest``.

        Returns the projected ``dict[str, JsonMetricResult]`` shape on
        success (same shape as ``_pull_summary_metrics``), or ``{}`` on any
        failure (operator unreachable, file 404, parse error). Failure is
        non-fatal: callers treat empty as "metrics unrecoverable" and fall
        through.

        Why not the child's results-sidecar? The operator deletes the
        child JobSet on success (``_maybe_delete_jobset_after_success``),
        which tears down the controller pod and its sidecar. Any in-flight
        fallback then hits ``Connect failed`` or ``Name or service not known``
        and loses the metrics. The operator's PVC-backed API is the durable
        alternative — same JSON, no race.

        Example:
            >>> child = {
            ...     "metadata": {"namespace": "aiperf-benchmarks",
            ...                  "name": "sweep-conc-demo-v00-t0"},
            ...     "status": {"runEpoch": 1778027130},
            ... }
            >>> # builds: http://aiperf-operator.aiperf-system:8081/api/v1/
            >>> #         results/aiperf-benchmarks/sweep-conc-demo-v00-t0/
            >>> #         runs/1778027130/profile_export_aiperf.json
        """
        from aiperf.common.models.export_models import JsonMetricResult
        from aiperf.operator.environment import OperatorEnvironment

        status = child.get("status") or {}
        meta = child.get("metadata") or {}
        namespace = meta.get("namespace")
        name = meta.get("name")
        epoch = status.get("runEpoch")
        if not namespace or not name or not epoch:
            return {}

        base_url = OperatorEnvironment.SERVICE.BASE_URL.rstrip("/")
        url = (
            f"{base_url}/api/v1/results/{namespace}/{name}"
            f"/runs/{epoch}/profile_export_aiperf.json"
        )

        # Bounded retry on transient 5xx / connection errors. The operator
        # restarts during sweep finalize (e.g. helm upgrade mid-sweep) drop
        # individual children silently without this — the caller treats {}
        # as "metrics unrecoverable" and the variation falls out of the
        # parent aggregate. 4xx (404, 422 epoch allowlist) is permanent
        # and short-circuits.
        max_attempts = 3
        backoff = 1.0
        last_status: int | None = None
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                async with (
                    aiohttp.ClientSession() as session,
                    session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp,
                ):
                    last_status = resp.status
                    if resp.status == 200:
                        raw = await resp.read()
                        break
                    if 400 <= resp.status < 500:
                        # Permanent — no retry.
                        logger.debug(
                            f"operator API fetch for {namespace}/{name}: "
                            f"HTTP {resp.status} from {url}"
                        )
                        return {}
                    # 5xx → transient, retry below.
            except (aiohttp.ClientError, ConnectionError, TimeoutError) as e:
                last_exc = e
            if attempt < max_attempts:
                await asyncio.sleep(backoff)
                backoff *= 2
        else:
            if last_exc is not None:
                logger.debug(
                    f"operator API transport error for {namespace}/{name} "
                    f"({url}) after {max_attempts} attempts: "
                    f"{type(last_exc).__name__}: {last_exc}"
                )
            else:
                logger.warning(
                    f"operator API fetch for {namespace}/{name}: "
                    f"HTTP {last_status} from {url} "
                    f"(persistent after {max_attempts} attempts)"
                )
            return {}
        try:
            payload = orjson.loads(raw)
        except orjson.JSONDecodeError as e:
            logger.debug(
                f"operator API parse error for {namespace}/{name} "
                f"({url}): {type(e).__name__}: {e}"
            )
            return {}
        if not isinstance(payload, dict):
            return {}
        return JsonMetricResult.project_summary_dict(payload)


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
