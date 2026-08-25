# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostics helpers for the sweeps router.

Owns the sweep-controller pod resolver plus the events/logs endpoint
implementations. Mirrors the job-side handlers in ``jobs.py`` /
``jobs_logs.py`` but rooted at the AIPerfSweep CR + its single
sweep-controller pod under JobSet ``aiperf-<name>``.

Extracted out of ``sweeps.py`` to keep that file under the 500-line
ergonomics ceiling.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.client import get_pods, list_events_for_object
from aiperf.operator.routers._path_params import validate_results_path_params
from aiperf.operator.routers.jobs import (
    MAX_EVENTS_RETURNED,
    _event_to_entry,
    _is_noise_event,
    _pod_summary,
    event_sort_key,
)
from aiperf.operator.routers.jobs_logs import (
    _default_container,
    _read_pod_log_text,
    _stream_pod_log,
    _validate_log_args,
)
from aiperf.operator.routers.jobs_models import JobEventsResponse, JobPodSummary

logger = logging.getLogger("aiperf.operator.ui")


async def list_sweep_controller_pods(
    api: ApiClient,
    namespace: str,
    name: str,
) -> list[Any]:
    """Return the sweep-controller pod(s) for an AIPerfSweep.

    The sweep-controller runs as a single replica under JobSet
    ``aiperf-<name>``; JobSet auto-stamps the
    ``jobset.sigs.k8s.io/jobset-name=<jobset>`` label on every pod under
    it, so that selector reliably picks up exactly the controller pod
    (and any transient restarts) without requiring custom AIPerf labels
    on the pod template.
    """
    selector = f"jobset.sigs.k8s.io/jobset-name=aiperf-{name}"
    return await get_pods(api, namespace, selector)


async def fetch_sweep_pod_summaries(
    api: ApiClient,
    namespace: str,
    name: str,
    source: str,
) -> list[JobPodSummary]:
    """Return ``JobPodSummary`` rows for the sweep-controller pod(s).

    Returns an empty list when the sweep is archived (no live CR / pod)
    or when the apiserver query fails — the panel degrades gracefully
    rather than 500ing the whole sweep-detail response.
    """
    if source == "archived":
        return []
    try:
        pod_objs = await list_sweep_controller_pods(api, namespace, name)
    except Exception as e:  # noqa: BLE001 - degrade to empty pod list
        logger.warning(
            "failed to list sweep-controller pods for %s/%s: %s",
            namespace,
            name,
            e,
        )
        return []
    return [_pod_summary(p) for p in pod_objs]


async def list_sweep_events_impl(
    api: ApiClient,
    namespace: str,
    name: str,
) -> JobEventsResponse:
    """Body of GET /api/v1/sweeps/{namespace}/{name}/events.

    Mirror of :func:`_list_events_impl` from ``jobs.py`` but rooted at
    the AIPerfSweep CR: collects events for (1) the parent CR and
    (2) the sweep-controller pod(s) under the JobSet ``aiperf-<name>``.
    Owned intermediate resources (the JobSet, the controller's k8s Job)
    are intentionally omitted — pod-level events surface the interesting
    failures (ImagePull, OOMKilled, FailedScheduling, ...) and CR-level
    events surface kopf reconcile transitions.

    Archived (PVC-only) sweeps whose CR is gone return an empty list.

    Raises:
        ApiException: Non-404 ``kubernetes_asyncio.client.ApiException``
            errors propagate via the app-level handler.
    """
    from aiperf.kubernetes.client import get_raw_aiperfsweep

    validate_results_path_params(namespace, name)

    cr = await get_raw_aiperfsweep(api, namespace, name)
    if cr is None:
        return JobEventsResponse(events=[])

    cr_events = await list_events_for_object(api, namespace, name)

    pods = await list_sweep_controller_pods(api, namespace, name)
    pod_names = [p.metadata.name for p in pods if p.metadata and p.metadata.name]

    pod_event_lists: list[list[Any]] = []
    for pod_name in pod_names:
        # Best-effort per pod, matching ``jobs._list_events_impl``: a
        # sweep-controller pod garbage-collected between the pod listing and
        # this call (404), or one RBAC denial, must not sink the whole
        # response — the CR events fetched above are still worth returning.
        try:
            pod_event_lists.append(
                await list_events_for_object(api, namespace, pod_name)
            )
        except ApiException as e:
            logger.warning(
                "Failed to list events for sweep pod %s/%s (apiserver %s): %s",
                namespace,
                pod_name,
                e.status,
                e,
            )

    raw_events: list[Any] = [*cr_events]
    for lst in pod_event_lists:
        raw_events.extend(lst)

    raw_events = [e for e in raw_events if not _is_noise_event(e)]

    entries = [_event_to_entry(e) for e in raw_events]
    entries.sort(key=event_sort_key, reverse=True)
    return JobEventsResponse(events=entries[:MAX_EVENTS_RETURNED])


async def get_sweep_logs_impl(
    api: ApiClient,
    namespace: str,
    name: str,
    *,
    pod: str | None,
    follow: bool,
    tail_lines: int,
    container: str | None,
) -> Any:
    """Body of GET /api/v1/sweeps/{namespace}/{name}/logs.

    Resolves the sweep-controller pod via the JobSet selector (or honors
    an explicit ``pod=`` query when more than one transient pod exists)
    and reads logs through the same Core V1 ``read_namespaced_pod_log``
    helpers used by the job-side endpoint. Container defaults to the
    pod's ``kubectl.kubernetes.io/default-container`` annotation if set,
    else the first non-sidecar container — matching
    ``jobs_logs._default_container``.

    Raises:
        HTTPException: 404 if no sweep-controller pod can be found, or
            if the pod query parameter does not match a pod under the
            sweep's JobSet; 400 on out-of-range ``tail_lines`` or
            malformed names.
    """
    from kubernetes_asyncio import client as k8s

    # Validate before ``name`` reaches the JobSet label selector built by
    # ``list_sweep_controller_pods`` — an unvalidated name can inject selector
    # syntax and match pods outside this sweep.
    validate_results_path_params(namespace, name)

    pods = await list_sweep_controller_pods(api, namespace, name)
    if not pods:
        raise HTTPException(
            404,
            f"No sweep-controller pod found for sweep {namespace}/{name} "
            f"(JobSet aiperf-{name} has no pods)",
        )

    if pod is not None:
        match = next((p for p in pods if p.metadata and p.metadata.name == pod), None)
        if match is None:
            raise HTTPException(
                404,
                f"Pod {pod!r} is not part of sweep {namespace}/{name}",
            )
        target = match
    else:
        # Prefer a Running pod when multiple exist (e.g. during a restart).
        running = next(
            (p for p in pods if (p.status and (p.status.phase or "")) == "Running"),
            None,
        )
        target = running or pods[0]

    target_pod = target.metadata.name
    effective_container = container or _default_container(target)
    _validate_log_args(target_pod, effective_container, tail_lines)

    core = k8s.CoreV1Api(api)
    if not follow:
        return await _read_pod_log_text(
            core,
            namespace=namespace,
            pod=target_pod,
            container=effective_container,
            tail_lines=tail_lines,
        )
    return await _stream_pod_log(
        core,
        namespace=namespace,
        pod=target_pod,
        container=effective_container,
        tail_lines=tail_lines,
    )


def register_diagnostics_routes(
    router: APIRouter,
    require_api: Callable[[], ApiClient],
) -> None:
    """Attach the events and logs endpoints to the sweeps router.

    Wires:
      - ``GET /sweeps/{namespace}/{name}/events`` -> :func:`list_sweep_events_impl`
      - ``GET /sweeps/{namespace}/{name}/logs``   -> :func:`get_sweep_logs_impl`

    ``require_api`` is the same ``_require_api`` factory used by the
    other sweep routes — it lazily resolves a live ``ApiClient`` once the
    operator has finished startup.
    """

    @router.get(
        "/sweeps/{namespace}/{name}/events",
        response_model=JobEventsResponse,
    )
    async def list_sweep_events(namespace: str, name: str) -> JobEventsResponse:
        return await list_sweep_events_impl(require_api(), namespace, name)

    @router.get("/sweeps/{namespace}/{name}/logs")
    async def get_sweep_logs(
        namespace: str,
        name: str,
        *,
        pod: str | None = None,
        follow: int = 0,
        tail_lines: int = 200,
        container: str | None = None,
    ) -> Any:
        return await get_sweep_logs_impl(
            require_api(),
            namespace,
            name,
            pod=pod,
            follow=bool(follow),
            tail_lines=tail_lines,
            container=container,
        )
