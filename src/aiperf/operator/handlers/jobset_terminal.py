# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Watch-driven JobSet completion-detection fast path.

When a JobSet flips to ``type=Completed status=True``, set the
``aiperf.nvidia.com/benchmark-complete`` annotation on the parent
AIPerfJob CR. That makes kopf dispatch ``on_benchmark_complete``
immediately, dropping completion latency from MONITOR.INTERVAL down
to a single apiserver round-trip.

Failed JobSets stay on the monitor-timer path because the recovery
logic there (classify controller vs workers, salvage from disk) has
no equivalent shortcut.

The kopf decorator binding lives in ``operator/main.py``; this module
is decorator-free so it can be unit-tested without kopf.
"""

from __future__ import annotations

import logging
from typing import Any

from aiperf.kubernetes.constants import AIPerfLabels, Annotations
from aiperf.kubernetes.cr_refs import AIPERF_JOB_API_VERSION

logger = logging.getLogger(__name__)


def _has_completed_condition(conditions: list[dict[str, Any]] | None) -> bool:
    """Return True if any condition is ``type=Completed status=True``.

    Defensive against non-dict entries (None / strings / numbers) that can
    appear if a malformed JobSet status leaks through the apiserver — kopf
    delivers the conditions list as-is, so we cannot assume well-formedness.
    """
    for cond in conditions or []:
        if not isinstance(cond, dict):
            continue
        if cond.get("type") == "Completed" and cond.get("status") == "True":
            return True
    return False


async def _lookup_aiperfjob_body(
    namespace: str, jobset_name: str
) -> dict[str, Any] | None:
    """Fetch the parent AIPerfJob CR body. JobSet name pattern: ``aiperf-<aiperfjob-name>``.

    Sweep-owned JobSets resolve to a non-existent AIPerfJob CR (the parent
    there is an AIPerfSweep) and return None -- the handler then silently
    skips. A 404 from the apiserver returns None too.
    """
    from kubernetes_asyncio.client import CustomObjectsApi
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.kubernetes.client import k8s_client
    from aiperf.kubernetes.cr_refs import (
        AIPERF_GROUP,
        AIPERF_PLURAL,
        AIPERF_VERSION,
    )

    if not jobset_name.startswith("aiperf-"):
        return None
    ajob_name = jobset_name.removeprefix("aiperf-")
    try:
        async with k8s_client() as api:
            custom = CustomObjectsApi(api)
            return await custom.get_namespaced_custom_object(
                group=AIPERF_GROUP,
                version=AIPERF_VERSION,
                namespace=namespace,
                plural=AIPERF_PLURAL,
                name=ajob_name,
            )
    except ApiException as e:
        if e.status == 404:
            return None
        raise
    except Exception:  # noqa: BLE001 - best-effort lookup; absent CR means sweep-owned or already deleted
        return None


def _is_trusted_aiperf_jobset(
    *,
    jobset_body: dict[str, Any] | None,
    parent_body: dict[str, Any],
    jobset_name: str,
) -> bool:
    """Return True when the JobSet body proves AIPerfJob ownership."""
    metadata = (jobset_body or {}).get("metadata") or {}
    parent_metadata = parent_body.get("metadata") or {}
    parent_name = parent_metadata.get("name")
    parent_uid = parent_metadata.get("uid")
    if not isinstance(parent_name, str) or not isinstance(parent_uid, str):
        return False
    labels = metadata.get("labels") or {}
    if metadata.get("name") != jobset_name:
        return False
    if labels.get(AIPerfLabels.APP_KEY) != AIPerfLabels.APP_VALUE:
        return False
    if labels.get(AIPerfLabels.JOB_ID) != parent_name:
        return False
    owner_refs = metadata.get("ownerReferences") or []
    return any(
        isinstance(ref, dict)
        and ref.get("apiVersion") == AIPERF_JOB_API_VERSION
        and ref.get("kind") == "AIPerfJob"
        and ref.get("name") == parent_name
        and ref.get("uid") == parent_uid
        for ref in owner_refs
    )


async def _set_benchmark_complete_annotation(
    namespace: str, aiperfjob_name: str
) -> None:
    """Patch ``metadata.annotations[BENCHMARK_COMPLETE] = "true"`` on the AIPerfJob.

    Setting the annotation makes kopf dispatch ``on_benchmark_complete``,
    which is idempotent (it short-circuits if status.phase is terminal and
    ``try_claim_completion`` returns False if already claimed). Racing the
    controller pod (which also sets this annotation when done) is therefore
    safe -- whichever fires first wins.
    """
    from kubernetes_asyncio.client import CustomObjectsApi
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.kubernetes.client import k8s_client
    from aiperf.kubernetes.cr_refs import (
        AIPERF_GROUP,
        AIPERF_PLURAL,
        AIPERF_VERSION,
    )

    try:
        async with k8s_client() as api:
            custom = CustomObjectsApi(api)
            await custom.patch_namespaced_custom_object(
                group=AIPERF_GROUP,
                version=AIPERF_VERSION,
                namespace=namespace,
                plural=AIPERF_PLURAL,
                name=aiperfjob_name,
                body={
                    "metadata": {
                        "annotations": {
                            Annotations.BENCHMARK_COMPLETE: "true",
                        },
                    },
                },
            )
    except ApiException as e:
        logger.warning(
            "Failed to set benchmark-complete annotation on AIPerfJob %s/%s: %s",
            namespace,
            aiperfjob_name,
            e,
        )


async def handle_jobset_conditions(
    *,
    old: list[dict[str, Any]] | None,
    new: list[dict[str, Any]] | None,
    namespace: str,
    jobset_name: str,
    jobset_body: dict[str, Any] | None = None,
) -> None:
    """React to a JobSet conditions transition; success annotates the parent AIPerfJob.

    Failed JobSets are intentionally a no-op here -- the existing monitor-tick
    path (`_handle_jobset_failed_condition`) owns recovery (classify controller
    vs workers, salvage from disk on the failure branch).
    """
    if _has_completed_condition(old):
        return  # already saw it; idempotent re-fires no-op
    if not _has_completed_condition(new):
        return  # not (yet) terminal-success
    body = await _lookup_aiperfjob_body(namespace, jobset_name)
    if body is None:
        return  # sweep-owned or missing
    if not _is_trusted_aiperf_jobset(
        jobset_body=jobset_body,
        parent_body=body,
        jobset_name=jobset_name,
    ):
        return  # name collision or non-AIPerf JobSet
    existing = (body.get("metadata") or {}).get("annotations") or {}
    if existing.get(Annotations.BENCHMARK_COMPLETE) == "true":
        return  # controller pod already annotated; on_benchmark_complete is in flight
    ajob_name = jobset_name.removeprefix("aiperf-")
    await _set_benchmark_complete_annotation(namespace, ajob_name)
