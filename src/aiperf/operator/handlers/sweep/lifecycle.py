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

__all__ = ["cancel", "maybe_reap_finished", "on_delete"]

logger = logging.getLogger(__name__)

TERMINAL_PHASES = frozenset({"Succeeded", "Failed", "Cancelled", "PartiallyFailed"})


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
    if parent_phase in TERMINAL_PHASES:
        return
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
    elif len(new_conditions) == len(existing):
        # spec.cancel=false and no prior Cancelling condition: nothing to do.
        return
    patch.status["conditions"] = new_conditions


async def on_delete(*, name: str, namespace: str, **_: Any) -> None:
    """Cooperative cancel child jobs before cascade GC reaps them.

    OwnerReferences will SIGKILL the sweep-controller pod and child
    AIPerfJobs anyway, but flipping each child's spec.cancel=true gives
    them a brief window to write partial results and shut workers down
    cleanly. Listing failures are best-effort because namespace teardown can
    make child enumeration unreliable; non-race child patch failures are
    retried by kopf so a transient apiserver error does not immediately cede
    to cascade deletion.
    """
    import aiohttp
    from kubernetes_asyncio import client as k8s
    from kubernetes_asyncio.client import ApiException

    from aiperf.kubernetes.client import k8s_client

    try:
        async with k8s_client() as api:
            custom = k8s.CustomObjectsApi(api)
            try:
                resp = await custom.list_namespaced_custom_object(
                    group="aiperf.nvidia.com",
                    version="v1alpha1",
                    namespace=namespace,
                    plural="aiperfjobs",
                    label_selector=f"aiperf.nvidia.com/sweep={name}",
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
                child_name = (child.get("metadata") or {}).get("name")
                if not child_name:
                    continue
                try:
                    await custom.patch_namespaced_custom_object(
                        group="aiperf.nvidia.com",
                        version="v1alpha1",
                        namespace=namespace,
                        plural="aiperfjobs",
                        name=child_name,
                        body={"spec": {"cancel": True}},
                        _content_type="application/merge-patch+json",
                    )
                except ApiException as e:
                    if e.status in (404, 409):
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


async def maybe_reap_finished(
    *,
    body: dict[str, Any],
    status: dict[str, Any],
    name: str,
    namespace: str,
    **_: Any,
) -> None:
    """If the sweep is terminal AND ttlSecondsAfterFinished has elapsed, delete the CR.

    The TTL is computed from the most recent transition into a terminal
    phase (status.completionTime if present, else metadata.creationTimestamp
    as a conservative fallback). ``completionTime`` is the CRD-declared
    field name and is written by the sweep-controller's
    ``aggregation_complete`` / ``aggregation_failed`` writers. Children's
    own ttlSecondsAfterFinished governs their cleanup; the parent only
    reaps itself.
    """
    spec = body.get("spec") or {}
    ttl = spec.get("ttlSecondsAfterFinished")
    if ttl is None or ttl < 0:
        return
    phase = status.get("phase") or ""
    if phase not in TERMINAL_PHASES:
        return
    completed_at = status.get("completionTime") or (body.get("metadata") or {}).get(
        "creationTimestamp"
    )
    if not completed_at:
        return
    try:
        # ``fromisoformat`` accepts both whole-second (K8s convention) and
        # sub-second RFC3339 timestamps; ``strptime("%Y-%m-%dT%H:%M:%SZ")``
        # rejects the sub-second form and would silently disable the TTL
        # reaper for any controller-written ``completionTime`` carrying
        # fractional seconds.
        finished = datetime.fromisoformat(completed_at.rstrip("Z") + "+00:00")
    except ValueError:
        return
    age_seconds = (datetime.now(tz=UTC) - finished).total_seconds()
    if age_seconds < ttl:
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
            )
        logger.info(
            "reaped AIPerfSweep %s/%s after TTL=%ss (age=%.0fs)",
            namespace,
            name,
            ttl,
            age_seconds,
        )
    except ApiException as e:
        if e.status == 404:
            return
        raise kopf.TemporaryError(
            f"apiserver rejected TTL delete ({e.status}): {e.reason}", delay=60
        ) from e
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as e:
        raise kopf.TemporaryError(
            f"apiserver unreachable during TTL delete: {e}", delay=60
        ) from e
