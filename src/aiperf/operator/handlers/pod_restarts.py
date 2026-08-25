# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Watch-driven pod-restart event emitter.

Replaces the per-monitor-tick polling that lived in ``monitor.py:_check_pod_restarts``.
The kopf decorator binding lives in ``operator/main.py``; this module is decorator-free
so it can be unit-tested without kopf.

Sweep-owned JobSets (``aiperf-<sweep-name>``) won't have a matching AIPerfJob CR;
``_lookup_aiperfjob_body`` returns ``None`` in that case and the handler silently
skips, which is the right behavior — sweep-controller pod restarts belong on the
AIPerfSweep CR, not the AIPerfJob CR.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import kopf
from kubernetes_asyncio import client
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.cr_refs import (
    AIPERF_JOB_API_VERSION,
    AIPERF_JOB_GROUP,
    AIPERF_JOB_PLURAL,
    AIPERF_JOB_VERSION,
    JOBSET_API_VERSION,
    JOBSET_GROUP,
    JOBSET_PLURAL,
    JOBSET_VERSION,
)
from aiperf.operator import events
from aiperf.operator.client_cache import _warned_pod_restarts, job_key

logger = logging.getLogger(__name__)


def _owner_lookup_failure(
    exc: Exception,
    *,
    namespace: str,
    jobset_name: str,
) -> None:
    """Return for absence; raise a bounded retry for every other failure."""
    if isinstance(exc, ApiException) and exc.status == 404:
        return None
    raise kopf.TemporaryError(
        f"Pod owner lookup for {namespace}/{jobset_name} failed: {exc}; retrying",
        delay=5,
    ) from exc


async def _lookup_aiperfjob_body(
    namespace: str, jobset_name: str, pod_body: dict[str, Any]
) -> dict[str, Any] | None:
    """Resolve Pod -> batch Job -> JobSet -> exact AIPerfJob ownership.

    Labels only select a candidate JobSet. Immutable controller-owner UIDs prove
    every hop so a delayed Pod event cannot target a same-named replacement.
    """
    from aiperf.kubernetes.client import k8s_client

    if not jobset_name.startswith("aiperf-"):
        return None
    pod_owner = _controller_owner(
        pod_body,
        api_version="batch/v1",
        kind="Job",
    )
    if pod_owner is None:
        return None
    job_name, job_uid = pod_owner
    try:
        async with k8s_client() as api:
            job = await client.BatchV1Api(api).read_namespaced_job(
                namespace=namespace,
                name=job_name,
            )
            if not _has_resource_identity(job, name=job_name, uid=job_uid):
                return None
            jobset_owner = _controller_owner(
                job,
                api_version=JOBSET_API_VERSION,
                kind="JobSet",
            )
            if jobset_owner is None or jobset_owner[0] != jobset_name:
                return None
            _, jobset_uid = jobset_owner

            custom = client.CustomObjectsApi(api)
            jobset = await custom.get_namespaced_custom_object(
                group=JOBSET_GROUP,
                version=JOBSET_VERSION,
                namespace=namespace,
                plural=JOBSET_PLURAL,
                name=jobset_name,
            )
            if not _has_resource_identity(jobset, name=jobset_name, uid=jobset_uid):
                return None
            parent_owner = _controller_owner(
                jobset,
                api_version=AIPERF_JOB_API_VERSION,
                kind="AIPerfJob",
            )
            if parent_owner is None:
                return None
            parent_name, parent_uid = parent_owner

            parent = await custom.get_namespaced_custom_object(
                group=AIPERF_JOB_GROUP,
                version=AIPERF_JOB_VERSION,
                namespace=namespace,
                plural=AIPERF_JOB_PLURAL,
                name=parent_name,
            )
            if not _has_resource_identity(parent, name=parent_name, uid=parent_uid):
                return None
            return parent
    except Exception as exc:  # noqa: BLE001 - one-shot watch events must be retried rather than dropped
        return _owner_lookup_failure(
            exc,
            namespace=namespace,
            jobset_name=jobset_name,
        )


def _metadata(resource: Any) -> Any:
    """Return Kubernetes metadata from dict or generated client models."""
    if isinstance(resource, Mapping):
        return resource.get("metadata") or {}
    return getattr(resource, "metadata", None)


def _metadata_value(metadata: Any, camel: str, snake: str) -> Any:
    """Read a metadata field across dict and generated client model shapes."""
    if isinstance(metadata, Mapping):
        return metadata.get(camel) or metadata.get(snake)
    return getattr(metadata, snake, None)


def _has_resource_identity(resource: Any, *, name: str, uid: str) -> bool:
    """Return whether a live Kubernetes object has the expected immutable ID."""
    metadata = _metadata(resource)
    return (
        _metadata_value(metadata, "name", "name") == name
        and _metadata_value(metadata, "uid", "uid") == uid
    )


def _controller_owner(
    resource: Any,
    *,
    api_version: str,
    kind: str,
) -> tuple[str, str] | None:
    """Return the exact controller-owner name and UID for one resource."""
    metadata = _metadata(resource)
    refs = _metadata_value(metadata, "ownerReferences", "owner_references") or []
    for ref in refs:
        if isinstance(ref, Mapping):
            ref_api_version = ref.get("apiVersion") or ref.get("api_version")
            ref_kind = ref.get("kind")
            ref_name = ref.get("name")
            ref_uid = ref.get("uid")
            controller = ref.get("controller")
        else:
            ref_api_version = getattr(ref, "api_version", None)
            ref_kind = getattr(ref, "kind", None)
            ref_name = getattr(ref, "name", None)
            ref_uid = getattr(ref, "uid", None)
            controller = getattr(ref, "controller", None)
        if (
            ref_api_version == api_version
            and ref_kind == kind
            and isinstance(ref_name, str)
            and isinstance(ref_uid, str)
            and controller is True
        ):
            return ref_name, ref_uid
    return None


def _extract_reason(cs: dict[str, Any]) -> str:
    """Pull the human-readable restart reason from a containerStatus.

    Prefers ``state.waiting.reason`` (current state) over
    ``lastState.terminated.reason`` (previous-cycle state); falls back to
    ``"Unknown"`` if neither is set or both are empty.
    """
    reason = "Unknown"
    last_state = cs.get("lastState") or {}
    if last_state.get("terminated"):
        reason = last_state["terminated"].get("reason") or reason
    state = cs.get("state") or {}
    if state.get("waiting"):
        reason = state["waiting"].get("reason") or reason
    return reason


def _claim_dedup_candidates(
    new: list[dict[str, Any]] | None,
    *,
    name: str,
    threshold: int,
    pre_warned: set[tuple[str, int]],
) -> list[tuple[dict[str, Any], int]]:
    """Pre-claim ``(name, restart_count)`` dedup keys for not-yet-warned
    statuses at-or-above threshold. Atomic under asyncio (no await between
    membership-check and add)."""
    candidates: list[tuple[dict[str, Any], int]] = []
    for cs in new or []:
        restart_count = int(cs.get("restartCount") or 0)
        if restart_count < threshold:
            continue
        dedup_key = (name, restart_count)
        if dedup_key in pre_warned:
            continue
        pre_warned.add(dedup_key)
        candidates.append((cs, restart_count))
    return candidates


async def handle_pod_restart(
    *,
    old: list[dict[str, Any]],
    new: list[dict[str, Any]],
    body: dict[str, Any],
    meta: dict[str, Any],
    namespace: str,
    name: str,
    threshold: int,
) -> None:
    """Inspect a Pod containerStatuses transition and emit a single event per (pod, restart-count).

    Lookup-first ordering: we resolve the parent AIPerfJob CR BEFORE
    pre-claiming dedup state. This avoids two leaks the previous order had:
      1. Sweep-owned JobSets (lookup returns None) would still leave a
         pre-claim entry under the jobset-name-keyed dict that no eviction
         path ever cleaned up (sweep JobSets have no AIPerfJob, so
         ``client_cache._close_unlocked`` never sees the matching job_id key).
      2. Successful lookups migrated dedup state to the canonical job-id
         key but left the original jobset-name-keyed entry orphaned, since
         eviction is keyed by job_id.
    Pre-claim atomicity (the round-1 dedup race fix) is preserved because
    ``_claim_dedup_candidates`` does the in/add under a single coroutine
    step with no await between membership-check and add.
    """
    jobset_name = (meta.get("labels") or {}).get("jobset.sigs.k8s.io/jobset-name")
    if not jobset_name:
        return

    # Quick early-out: nothing in the new statuses is at-or-above threshold,
    # so don't pay the apiserver round-trip for the AIPerfJob lookup.
    if not _has_above_threshold(new, threshold=threshold):
        return

    aiperfjob_body = await _lookup_aiperfjob_body(namespace, jobset_name, body)
    if aiperfjob_body is None:
        return  # sweep-owned or already deleted; no pre-claim leaked

    job_id = (aiperfjob_body.get("status") or {}).get("jobId") or jobset_name
    parent_uid = (aiperfjob_body.get("metadata") or {}).get("uid")
    real_key = job_key(
        namespace,
        job_id,
        str(parent_uid) if parent_uid is not None else None,
    )
    pre_warned = _warned_pod_restarts.setdefault(real_key, set())
    candidates = _claim_dedup_candidates(
        new, name=name, threshold=threshold, pre_warned=pre_warned
    )
    if not candidates:
        return

    for cs, restart_count in candidates:
        events.pod_restarts(aiperfjob_body, name, restart_count, _extract_reason(cs))


def _has_above_threshold(
    statuses: list[dict[str, Any]] | None, *, threshold: int
) -> bool:
    """Return True if any containerStatus restartCount is at-or-above threshold."""
    return any(int(cs.get("restartCount") or 0) >= threshold for cs in statuses or [])
