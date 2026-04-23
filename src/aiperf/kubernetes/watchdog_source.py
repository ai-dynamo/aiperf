# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""kubernetes_asyncio-backed implementation of ``WatchdogDataSource``."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import aiohttp
from kubernetes_asyncio import client
from kubernetes_asyncio.client import ApiClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.watchdog_models import (
    ContainerInfo,
    EventInfo,
    NodeResources,
    PodMetrics,
    WatchdogPodSnapshot,
    _metrics_item_to_pod_metrics,
    _state_from_container_status,
)

if TYPE_CHECKING:
    from kubernetes_asyncio.client.models import V1Pod


class K8sWatchdogSource:
    """WatchdogDataSource backed by kubernetes_asyncio.

    Implements the seven ``get_*`` methods required by the
    ``WatchdogDataSource`` protocol:

    - ``get_pods`` -- list pods in a namespace
    - ``get_events`` -- list recent events in a namespace
    - ``get_node_resources`` -- list node allocatable CPU/memory/GPU
    - ``get_namespaces`` -- list namespace names, optionally filtered
    - ``get_pod_logs`` -- fetch tail of pod logs (best-effort)
    - ``get_pod_metrics`` -- fetch pod CPU/memory via metrics.k8s.io

    Contract: these methods swallow transient/configuration errors (API
    server unreachable, pod deleted mid-read, metrics-server not
    installed, etc.) and return an empty value (``[]`` or ``""``) rather
    than raising. The watchdog loop treats missing data as "not
    available yet" and retries on the next tick. Callers needing strict
    error surfaces should wrap the ``kubernetes_asyncio`` client
    directly.

    See the ``aiperf.kubernetes.watchdog`` module docstring for an
    end-to-end usage example.
    """

    def __init__(self, api: ApiClient) -> None:
        self._api = api

    @classmethod
    async def create(
        cls,
        *,
        kubeconfig: str | None = None,
        kube_context: str | None = None,
    ) -> K8sWatchdogSource:
        """Create a source with explicit context/kubeconfig.

        Loads k8s config (in-cluster first, kubeconfig fallback) and
        constructs an ``ApiClient``. Caller owns the ApiClient lifecycle;
        the watchdog itself does not close it.
        """
        from kubernetes_asyncio import config

        try:
            config.load_incluster_config()
        except config.ConfigException:
            await config.load_kube_config(config_file=kubeconfig, context=kube_context)
        return cls(ApiClient())

    async def get_pods(self, namespace: str) -> list[WatchdogPodSnapshot]:
        """List pods in a namespace via CoreV1Api."""
        core = client.CoreV1Api(self._api)
        pod_list = await core.list_namespaced_pod(namespace)
        return [self._pod_to_info(p) for p in pod_list.items]

    async def get_events(self, namespace: str, limit: int = 20) -> list[EventInfo]:
        """List recent events in a namespace via CoreV1Api."""
        core = client.CoreV1Api(self._api)
        ev_list = await core.list_namespaced_event(namespace)
        result: list[EventInfo] = []
        for ev in ev_list.items:
            involved = ev.involved_object
            obj_name = involved.name if involved and involved.name else ""
            result.append(
                EventInfo(
                    type=ev.type or "Normal",
                    reason=ev.reason or "",
                    message=ev.message or "",
                    involved_object=obj_name,
                    last_timestamp=ev.last_timestamp,
                )
            )
        result.sort(
            key=lambda e: e.last_timestamp or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return result[:limit]

    async def get_node_resources(self) -> list[NodeResources]:
        """List node allocatable resources via CoreV1Api."""
        core = client.CoreV1Api(self._api)
        node_list = await core.list_node()
        result: list[NodeResources] = []
        for node in node_list.items:
            allocatable = (
                node.status.allocatable
                if node.status and node.status.allocatable
                else {}
            )
            gpu_str = allocatable.get("nvidia.com/gpu", "0")
            try:
                gpu_count = int(gpu_str)
            except (ValueError, TypeError):
                gpu_count = 0

            name = node.metadata.name if node.metadata and node.metadata.name else ""
            result.append(
                NodeResources(
                    name=name,
                    allocatable_cpu=allocatable.get("cpu", "0"),
                    allocatable_memory=allocatable.get("memory", "0"),
                    allocatable_gpu=gpu_count,
                )
            )
        return result

    async def get_namespaces(self, label_selector: str | None = None) -> list[str]:
        """List namespace names via CoreV1Api."""
        core = client.CoreV1Api(self._api)
        kwargs: dict[str, Any] = {}
        if label_selector:
            kwargs["label_selector"] = label_selector
        ns_list = await core.list_namespace(**kwargs)
        return [
            ns.metadata.name if ns.metadata and ns.metadata.name else ""
            for ns in ns_list.items
        ]

    async def get_pod_logs(self, name: str, namespace: str, tail: int = 50) -> str:
        """Fetch pod logs via CoreV1Api (best-effort).

        The pod may have been deleted mid-read, or the API server may be
        transiently unreachable; in those cases this returns ``""``
        instead of raising, so the watchdog loop can continue.
        """
        core = client.CoreV1Api(self._api)
        try:
            return await core.read_namespaced_pod_log(
                name=name,
                namespace=namespace,
                tail_lines=tail,
            )
        except (ApiException, aiohttp.ClientError, asyncio.TimeoutError, OSError):
            # Best-effort log fetch: the pod may have been deleted mid-read, or
            # the API server may be unreachable; return empty rather than fail.
            return ""

    async def get_pod_metrics(self, namespace: str) -> list[PodMetrics]:
        """Fetch pod metrics via metrics.k8s.io API (best-effort).

        The ``metrics.k8s.io`` aggregated API may not be installed on the
        cluster, and freshly-created pods may not have usage samples
        yet. In either case this returns ``[]`` instead of raising.
        """
        try:
            resp = await self._api.call_api(
                f"/apis/metrics.k8s.io/v1beta1/namespaces/{namespace}/pods",
                "GET",
                response_type="object",
                auth_settings=["BearerToken"],
                _return_http_data_only=True,
            )
            data = resp if isinstance(resp, dict) else {}
            return [
                _metrics_item_to_pod_metrics(item) for item in data.get("items", [])
            ]
        except (
            ApiException,
            aiohttp.ClientError,
            asyncio.TimeoutError,
            KeyError,
            TypeError,
            ValueError,
            OSError,
        ):
            # metrics.k8s.io may not be installed, or pods may lack metrics yet.
            return []

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _pod_to_info(pod: V1Pod) -> WatchdogPodSnapshot:
        """Convert a V1Pod into a WatchdogPodSnapshot dataclass."""
        status = pod.status
        phase = status.phase if status and status.phase else "Unknown"

        container_statuses: list[ContainerInfo] = []
        all_ready = True
        total_restarts = 0

        primary = (status.container_statuses if status else None) or []
        for cs in primary:
            total_restarts += cs.restart_count or 0
            c_ready = bool(cs.ready)
            if not c_ready:
                all_ready = False

            c_state, c_reason, c_message, c_exit = _state_from_container_status(cs)

            container_statuses.append(
                ContainerInfo(
                    name=cs.name or "",
                    ready=c_ready,
                    state=c_state,
                    reason=c_reason,
                    message=c_message,
                    exit_code=c_exit,
                )
            )

        init_statuses = (status.init_container_statuses if status else None) or []
        for cs in init_statuses:
            total_restarts += cs.restart_count or 0
            c_ready = bool(cs.ready)
            if not c_ready:
                all_ready = False

            c_state, c_reason, c_message, c_exit = _state_from_container_status(cs)

            container_statuses.append(
                ContainerInfo(
                    name=cs.name or "",
                    ready=c_ready,
                    state=c_state,
                    reason=c_reason,
                    message=c_message,
                    exit_code=c_exit,
                )
            )

        if not container_statuses:
            all_ready = False

        metadata = pod.metadata
        parsed_creation: datetime | None = None
        if metadata and metadata.creation_timestamp:
            ts = metadata.creation_timestamp
            if isinstance(ts, datetime):
                parsed_creation = ts
            else:
                with contextlib.suppress(ValueError, TypeError):
                    parsed_creation = datetime.fromisoformat(
                        str(ts).replace("Z", "+00:00")
                    )

        return WatchdogPodSnapshot(
            name=(metadata.name if metadata and metadata.name else ""),
            namespace=(metadata.namespace if metadata and metadata.namespace else ""),
            phase=phase,
            ready=all_ready and phase == "Running",
            restarts=total_restarts,
            container_statuses=container_statuses,
            creation_timestamp=parsed_creation,
        )
