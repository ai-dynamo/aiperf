# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""kopf handlers for AIPerfSweep lifecycle: cancel mirroring, delete, TTL reap.

The sweep-controller pod observes spec.cancel via its own poll; the
operator's job is to mirror the cancel signal into status.conditions
for kubectl observability and to handle parent-CR deletion / TTL.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import kopf

from aiperf.operator.handlers.sweep._child_phase_buckets import _is_owned_child

__all__ = ["acknowledge_ttl_update", "cancel", "maybe_reap_finished", "on_delete"]

logger = logging.getLogger(__name__)

TERMINAL_PHASES = frozenset({"Succeeded", "Failed", "Cancelled", "PartiallyFailed"})
SWEEP_LABEL = "aiperf.nvidia.com/sweep"
SWEEP_UID_LABEL = "aiperf.nvidia.com/sweep-uid"
SWEEP_RUN_EPOCH_LABEL = "aiperf.nvidia.com/sweep-run-epoch"


def _normalized_run_epoch(body: dict[str, Any]) -> str | None:
    value = (body.get("status") or {}).get("runEpoch")
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    normalized = str(value)
    return normalized if normalized.isdecimal() else None


async def cancel(
    *,
    body: dict[str, Any],
    spec: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Mirror spec.cancel into status.conditions[Cancelling].

    On cancel=true: append (or replace) Cancelling=True condition.
    On cancel=false: clear any existing Cancelling condition (sticky-flag fix).
    Skips when the sweep has already reached a terminal phase — cancelling a
    finished sweep is a no-op visually.
    """
    cancelling = bool(spec.get("cancel"))
    status_block = body.get("status") or {}
    parent_phase = status_block.get("phase") or ""
    if parent_phase not in TERMINAL_PHASES:
        existing = status_block.get("conditions") or []
        new_conditions = [c for c in existing if c.get("type") != "Cancelling"]
        if cancelling:
            now = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            new_conditions.append(
                {
                    "type": "Cancelling",
                    "status": "True",
                    "reason": "UserRequested",
                    "message": "spec.cancel set to true",
                    "lastTransitionTime": now,
                }
            )
            patch.status["conditions"] = new_conditions
        elif len(new_conditions) != len(existing):
            patch.status["conditions"] = new_conditions

    generation = (body.get("metadata") or {}).get("generation")
    if generation is not None:
        patch.status["observedGeneration"] = int(generation)


async def acknowledge_ttl_update(
    *,
    body: dict[str, Any],
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Acknowledge a parent-sweep TTL update consumed by the reaper timer."""
    generation = (body.get("metadata") or {}).get("generation")
    if generation is not None:
        patch.status["observedGeneration"] = int(generation)


async def on_delete(
    *,
    body: dict[str, Any],
    uid: str,
    name: str,
    namespace: str,
    **_: Any,
) -> None:
    """Cooperative cancel child jobs before cascade GC reaps them.

    OwnerReferences will SIGKILL the sweep-controller pod and child
    AIPerfJobs anyway, but flipping each child's spec.cancel=true gives
    them a brief window to write partial results and shut workers down
    cleanly. Labels narrow discovery, but only an ownerReference matching the
    deleting sweep's kind, name, and immutable UID authorizes a patch. Listing
    failures are best-effort because namespace teardown can make child
    enumeration unreliable; non-race child patch failures are retried by kopf
    so a transient apiserver error does not immediately cede to cascade
    deletion.
    """
    import aiohttp
    from kubernetes_asyncio import client as k8s
    from kubernetes_asyncio.client import ApiException

    from aiperf.kubernetes.client import k8s_client

    selector = f"{SWEEP_LABEL}={name},{SWEEP_UID_LABEL}={uid}"
    run_epoch = _normalized_run_epoch(body)
    if run_epoch is None:
        return
    selector += f",{SWEEP_RUN_EPOCH_LABEL}={run_epoch}"

    try:
        async with k8s_client() as api:
            custom = k8s.CustomObjectsApi(api)
            try:
                resp = await custom.list_namespaced_custom_object(
                    group="aiperf.nvidia.com",
                    version="v1alpha1",
                    namespace=namespace,
                    plural="aiperfjobs",
                    label_selector=selector,
                )
            except (
                ApiException,
                aiohttp.ClientError,
                ConnectionError,
                TimeoutError,
            ) as e:
                logger.warning(
                    "AIPerfSweep on_delete: cooperative-cancel best-effort failed for %s/%s: %s",
                    namespace,
                    name,
                    e,
                )
                return
            for child in resp.get("items", []):
                if not _is_owned_child(
                    child,
                    sweep_name=name,
                    sweep_uid=uid,
                    run_epoch=run_epoch,
                ):
                    continue
                child_metadata = child.get("metadata") or {}
                child_name = child_metadata.get("name")
                child_uid = child_metadata.get("uid")
                if not child_name or not isinstance(child_uid, str):
                    continue
                try:
                    await custom.patch_namespaced_custom_object(
                        group="aiperf.nvidia.com",
                        version="v1alpha1",
                        namespace=namespace,
                        plural="aiperfjobs",
                        name=child_name,
                        body=[
                            {
                                "op": "test",
                                "path": "/metadata/uid",
                                "value": child_uid,
                            },
                            {"op": "add", "path": "/spec/cancel", "value": True},
                        ],
                        _content_type="application/json-patch+json",
                    )
                except ApiException as e:
                    if e.status in (404, 409, 422):
                        continue
                    raise kopf.TemporaryError(
                        "apiserver rejected cooperative-cancel patch for "
                        f"{namespace}/{child_name} ({e.status}): {e.reason}",
                        delay=15,
                    ) from e
                except (aiohttp.ClientError, ConnectionError, TimeoutError) as e:
                    raise kopf.TemporaryError(
                        "apiserver unreachable during cooperative-cancel patch for "
                        f"{namespace}/{child_name}: {e}",
                        delay=15,
                    ) from e
    except (ApiException, aiohttp.ClientError, ConnectionError, TimeoutError) as e:
        logger.warning(
            "AIPerfSweep on_delete: cooperative-cancel best-effort failed for %s/%s: %s",
            namespace,
            name,
            e,
        )


def _expired_sweep_ttl(
    body: dict[str, Any],
    status: dict[str, Any],
) -> tuple[int, float] | None:
    """Return the configured TTL and current age for a sweep ready to reap."""
    spec = body.get("spec") or {}
    ttl = spec.get("ttlSecondsAfterFinished")
    if ttl is None or ttl < 0:
        return None
    phase = status.get("phase") or ""
    if phase not in TERMINAL_PHASES:
        return None
    aggregation = status.get("aggregation") or {}
    aggregate_ref = status.get("aggregateRef") or {}
    if aggregation.get("phase") == "Complete" and not (
        status.get("resultsAvailable") is True
        and isinstance(aggregate_ref, dict)
        and bool(aggregate_ref.get("url"))
    ):
        # A terminal phase is published while the only complete aggregate is
        # still in the sweep-controller's emptyDir. Reaping now can race the
        # operator's sidecar harvest and permanently lose the result. The
        # durable operator-backed ref carries ``url``; the live controller ref
        # deliberately does not.
        return None
    completed_at = status.get("completionTime") or (body.get("metadata") or {}).get(
        "creationTimestamp"
    )
    if not completed_at:
        return None
    try:
        # ``fromisoformat`` accepts both whole-second (K8s convention) and
        # sub-second RFC3339 timestamps; ``strptime("%Y-%m-%dT%H:%M:%SZ")``
        # rejects the sub-second form and would silently disable the TTL
        # reaper for any controller-written ``completionTime`` carrying
        # fractional seconds.
        finished = datetime.fromisoformat(completed_at.rstrip("Z") + "+00:00")
    except ValueError:
        return None
    age_seconds = (datetime.now(tz=UTC) - finished).total_seconds()
    if age_seconds < ttl:
        return None
    return int(ttl), age_seconds


async def maybe_reap_finished(
    *,
    body: dict[str, Any],
    status: dict[str, Any],
    name: str,
    namespace: str,
    **_: Any,
) -> None:
    """Delete an expired terminal sweep using its immutable UID precondition.

    The TTL is computed from the most recent transition into a terminal
    phase (status.completionTime if present, else metadata.creationTimestamp
    as a conservative fallback). ``completionTime`` is the CRD-declared
    field name and is written by the sweep-controller's
    ``aggregation_complete`` / ``aggregation_failed`` writers. Children's
    own ttlSecondsAfterFinished governs their cleanup; the parent only
    reaps itself.
    """
    expired = _expired_sweep_ttl(body, status)
    if expired is None:
        return
    ttl, age_seconds = expired
    sweep_uid = str((body.get("metadata") or {}).get("uid") or "")
    if not sweep_uid:
        logger.warning(
            "AIPerfSweep TTL reaper skipped %s/%s because metadata.uid is missing",
            namespace,
            name,
        )
        return

    import aiohttp
    from kubernetes_asyncio import client as k8s
    from kubernetes_asyncio.client import ApiException

    from aiperf.kubernetes.client import k8s_client

    try:
        async with k8s_client() as api:
            custom = k8s.CustomObjectsApi(api)
            await custom.delete_namespaced_custom_object(
                group="aiperf.nvidia.com",
                version="v1alpha1",
                namespace=namespace,
                plural="aiperfsweeps",
                name=name,
                body=k8s.V1DeleteOptions(
                    preconditions=k8s.V1Preconditions(uid=sweep_uid)
                ),
            )
        logger.info(
            "reaped AIPerfSweep %s/%s after TTL=%ss (age=%.0fs)",
            namespace,
            name,
            ttl,
            age_seconds,
        )
    except ApiException as e:
        if e.status in (404, 409):
            return
        raise kopf.TemporaryError(
            f"apiserver rejected TTL delete ({e.status}): {e.reason}", delay=60
        ) from e
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as e:
        raise kopf.TemporaryError(
            f"apiserver unreachable during TTL delete: {e}", delay=60
        ) from e
