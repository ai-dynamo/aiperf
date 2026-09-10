# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pod log streaming implementation for GET /api/v1/jobs/{ns}/{name}/logs."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse
from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.client import get_pods
from aiperf.kubernetes.constants import Containers

logger = logging.getLogger("aiperf.operator.ui")

_K8S_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,252}$")
MIN_TAIL_LINES = 1
MAX_TAIL_LINES = 10_000
_SIDECAR_CONTAINERS: frozenset[str] = frozenset(
    {
        Containers.EVENT_BUS_PROXY,
        Containers.RESULTS_SIDECAR,
        "istio-proxy",
        "linkerd-proxy",
    }
)


def _validate_k8s_name(value: str, kind: str) -> str:
    """Reject pod/container names that don't match DNS-1123-ish shape.

    Guards the upstream apiserver from obviously malformed / injection-y input
    before we hand the string to ``read_namespaced_pod_log``.
    """
    if not _K8S_NAME_RE.match(value):
        raise HTTPException(400, f"Invalid {kind} name: {value!r}")
    return value


async def _resolve_owned_pod(
    api: ApiClient,
    namespace: str,
    name: str,
    pod: str,
) -> Any:
    """Return the V1Pod iff it exists and is labelled for this AIPerfJob.

    The label ``aiperf.nvidia.com/job-id=<name>`` is the same selector the jobs
    detail / events endpoints use, so a pod that is not in that set is treated
    as "not part of this run" and produces a 404.
    """
    pods = await get_pods(api, namespace, f"aiperf.nvidia.com/job-id={name}")
    for p in pods:
        if p.metadata and p.metadata.name == pod:
            return p
    raise HTTPException(404, f"Pod {pod!r} is not part of job {namespace}/{name}")


def _default_container(pod: Any) -> str | None:
    """Pick a sensible default container for logs when one is not specified.

    Prefers (in order): the ``kubectl.kubernetes.io/default-container``
    annotation, the control plane, the first non-sidecar container, then the
    first spec container. Returns None if the spec has no containers.
    """
    meta = getattr(pod, "metadata", None)
    spec = getattr(pod, "spec", None)
    annotations = (getattr(meta, "annotations", None) or {}) if meta else {}
    annotated = annotations.get("kubectl.kubernetes.io/default-container")
    if annotated:
        return annotated
    containers = list(getattr(spec, "containers", None) or [])
    if not containers:
        return None
    names = [getattr(container, "name", "") or "" for container in containers]
    if Containers.CONTROL_PLANE in names:
        return Containers.CONTROL_PLANE
    for cname in names:
        if cname not in _SIDECAR_CONTAINERS:
            return cname
    return names[0]


def _validate_log_args(pod: str, container: str | None, tail_lines: int) -> None:
    """Reject malformed pod/container names and out-of-range tail_lines."""
    _validate_k8s_name(pod, "pod")
    if container is not None:
        _validate_k8s_name(container, "container")
    if not (MIN_TAIL_LINES <= tail_lines <= MAX_TAIL_LINES):
        raise HTTPException(
            400,
            f"tail_lines must be in [{MIN_TAIL_LINES}, {MAX_TAIL_LINES}]",
        )


async def _read_pod_log_text(
    core: client.CoreV1Api,
    *,
    namespace: str,
    pod: str,
    container: str | None,
    tail_lines: int,
) -> Response:
    """Non-follow path: fetch the full tail as a single text/plain response."""
    try:
        text = await core.read_namespaced_pod_log(
            name=pod,
            namespace=namespace,
            tail_lines=tail_lines,
            container=container,
            _preload_content=True,
        )
    except ApiException as e:
        detail = e.body or e.reason or "Kubernetes API error"
        raise HTTPException(e.status or 500, detail) from e
    return Response(content=text or "", media_type="text/plain")


async def _stream_pod_log(
    core: client.CoreV1Api,
    *,
    namespace: str,
    pod: str,
    container: str | None,
    tail_lines: int,
) -> StreamingResponse:
    """Follow path: keep the aiohttp response open and stream chunks to the client."""
    try:
        resp = await core.read_namespaced_pod_log(
            name=pod,
            namespace=namespace,
            tail_lines=tail_lines,
            container=container,
            follow=True,
            _preload_content=False,
        )
    except ApiException as e:
        detail = e.body or e.reason or "Kubernetes API error"
        raise HTTPException(e.status or 500, detail) from e

    async def _iter_log_chunks() -> Any:
        # Explicit close on client disconnect keeps the upstream apiserver
        # watch from leaking — _preload_content=False hands back the raw
        # aiohttp response, which owns the TCP connection.
        try:
            async for chunk in resp.content.iter_any():
                if chunk:
                    yield chunk
        except asyncio.CancelledError:
            raise
        finally:
            try:
                resp.close()
            except Exception as e:  # noqa: BLE001 - best-effort cleanup
                logger.debug(f"Error closing pod-log stream for {pod}: {e}")

    # GZipMiddleware buffers chunks until its zlib window fills, which can
    # stall live log streaming for tens of seconds. Setting Content-Encoding:
    # identity tells GZipResponder to pass this response through untouched
    # (Starlette checks content_encoding_set before compressing).
    return StreamingResponse(
        _iter_log_chunks(),
        media_type="text/plain",
        headers={"Content-Encoding": "identity"},
    )


async def get_pod_logs_impl(
    api: ApiClient,
    namespace: str,
    name: str,
    *,
    pod: str,
    follow: bool,
    tail_lines: int,
    container: str | None,
) -> Response:
    """Body of GET /api/v1/jobs/{namespace}/{name}/logs.

    Pod ownership is enforced via the ``aiperf.nvidia.com/job-id=<name>`` label
    selector — arbitrary pod names in the same namespace cannot be tailed
    through this endpoint. See :func:`_read_pod_log_text` and
    :func:`_stream_pod_log` for the per-mode response bodies.

    Raises:
        HTTPException: 400 on malformed pod/container names or out-of-range
            ``tail_lines``; 404 if the pod is not owned by this AIPerfJob;
            other apiserver errors propagate with their status codes.
    """
    _validate_log_args(pod, container, tail_lines)
    pod_obj = await _resolve_owned_pod(api, namespace, name, pod)
    effective_container = container or _default_container(pod_obj)
    # The default container can come from the pod's
    # ``kubectl.kubernetes.io/default-container`` annotation, which is
    # attacker-controllable on a hostile manifest. Validate the resolved name
    # too — not just the query param — so a path-traversal-shaped annotation
    # can never reach ``read_namespaced_pod_log``.
    if effective_container is not None:
        _validate_k8s_name(effective_container, "container")
    core = client.CoreV1Api(api)
    if not follow:
        return await _read_pod_log_text(
            core,
            namespace=namespace,
            pod=pod,
            container=effective_container,
            tail_lines=tail_lines,
        )
    return await _stream_pod_log(
        core,
        namespace=namespace,
        pod=pod,
        container=effective_container,
        tail_lines=tail_lines,
    )
