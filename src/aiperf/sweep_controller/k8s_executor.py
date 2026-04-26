# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K8sChildJobExecutor: creates AIPerfJob children, watches them, collects results.

The single substantive seam between the shared MultiRunOrchestrator and the
K8s sweep flow. Task 13 (separate) implements the execute()/watch/result-pull
body; this module provides the helpers, identity check, and child-spec/metadata
construction.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import TYPE_CHECKING, Any

from aiperf.orchestrator.executor import RunExecutor

if TYPE_CHECKING:
    from aiperf.config.benchmark import BenchmarkPlan, BenchmarkRun
    from aiperf.orchestrator.models import RunResult


logger = logging.getLogger(__name__)


SWEEP_LABEL = "aiperf.nvidia.com/sweep"
SWEEP_UID_LABEL = "aiperf.nvidia.com/sweep-uid"
VARIATION_INDEX_LABEL = "aiperf.nvidia.com/variation-index"
VARIATION_LABEL_LABEL = "aiperf.nvidia.com/variation-label"
TRIAL_INDEX_LABEL = "aiperf.nvidia.com/trial-index"


__all__ = [
    "SWEEP_LABEL",
    "SWEEP_UID_LABEL",
    "TRIAL_INDEX_LABEL",
    "VARIATION_INDEX_LABEL",
    "VARIATION_LABEL_LABEL",
    "ChildNameConflictError",
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
    """Reduce a free-form string to a valid k8s label value (RFC 1123, <=63 chars)."""
    sanitized = re.sub(r"[^a-z0-9._-]+", "-", value.lower())
    sanitized = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", sanitized)
    return sanitized[:63] or "v"


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
    ) -> None:
        self._api = api
        self.sweep = sweep
        self.sweep_name: str = sweep["metadata"]["name"]
        self.sweep_namespace: str = sweep["metadata"]["namespace"]
        self.sweep_uid: str = sweep["metadata"]["uid"]
        self.with_trial_suffix = with_trial_suffix

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
        template_spec["benchmark"] = run.cfg.model_dump(
            by_alias=True, exclude_none=True
        )
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
                    "apiVersion": "aiperf.nvidia.com/v1",
                    "kind": "AIPerfSweep",
                    "name": self.sweep_name,
                    "uid": self.sweep_uid,
                    "controller": True,
                    "blockOwnerDeletion": True,
                }
            ],
        }

    async def execute(self, run: BenchmarkRun) -> RunResult:
        """Body lands in Task 13 (apiserver create/watch/result-pull)."""
        raise NotImplementedError("execute() body landed in Task 13")
