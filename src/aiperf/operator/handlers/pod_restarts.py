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
from typing import Any

from aiperf.operator import events
from aiperf.operator.client_cache import _warned_pod_restarts, job_key

logger = logging.getLogger(__name__)


async def _lookup_aiperfjob_body(
    namespace: str, jobset_name: str
) -> dict[str, Any] | None:
    """Walk a Pod's owner JobSet name to its AIPerfJob CR body, for events.pod_restarts.

    The JobSet name pattern is ``aiperf-<aiperfjob-name>`` (set in
    ``spec_converter.py`` and ``handlers/sweep/create.py``). For a sweep-owned
    JobSet, the AIPerfJob lookup returns 404 and we return None so the caller skips.
    """
    from kubernetes_asyncio.client import CustomObjectsApi

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
    except Exception:  # noqa: BLE001 - best-effort lookup; absent CR means sweep-owned or already deleted
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
    """Inspect a Pod containerStatuses transition and emit a single event per (pod, restart-count)."""
    jobset_name = (meta.get("labels") or {}).get("jobset.sigs.k8s.io/jobset-name")
    if not jobset_name:
        return

    # Pre-claim dedup keys BEFORE the apiserver lookup. Two concurrent fires
    # for the same (pod, count) would both reach the await if the check
    # happened after — both would emit. Pre-claim is atomic under asyncio
    # because there is no await between `in` and `add`.
    cache_key_for_jobset = job_key(namespace, jobset_name)
    pre_warned = _warned_pod_restarts.setdefault(cache_key_for_jobset, set())
    candidates = _claim_dedup_candidates(
        new, name=name, threshold=threshold, pre_warned=pre_warned
    )
    if not candidates:
        return

    aiperfjob_body = await _lookup_aiperfjob_body(namespace, jobset_name)
    if aiperfjob_body is None:
        return

    job_id = (aiperfjob_body.get("status") or {}).get("jobId") or jobset_name
    real_key = job_key(namespace, job_id)
    if real_key != cache_key_for_jobset:
        # Migrate dedup state to the canonical job-id key so future fires
        # share dedup entries.
        warned = _warned_pod_restarts.setdefault(real_key, set())
        for _cs, rc in candidates:
            warned.add((name, rc))

    for cs, restart_count in candidates:
        events.pod_restarts(aiperfjob_body, name, restart_count, _extract_reason(cs))
