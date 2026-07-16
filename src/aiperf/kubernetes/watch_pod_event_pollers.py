# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pod and event pollers for the watch command."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

import aiohttp
from kubernetes_asyncio import client
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.kubernetes.client import job_selector
from aiperf.kubernetes.constants import AIPerfLabels
from aiperf.kubernetes.watch_models import EventSnapshot, PodSnapshot

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient

logger = logging.getLogger(__name__)


class PodPoller:
    """Polls the K8s Pod API for this job's pod/container states."""

    def __init__(
        self,
        api: ApiClient,
        job_id: str,
        namespace: str,
        *,
        label_selector: str | None = None,
        job_ids_provider: Callable[[], Sequence[str]] | None = None,
    ) -> None:
        self._api = api
        self._job_id = job_id
        self._namespace = namespace
        self._label_selector = label_selector
        self._job_ids_provider = job_ids_provider
        self.pods: list[PodSnapshot] = []

    async def poll(self) -> None:
        """Fetch latest pod status."""
        core = client.CoreV1Api(self._api)
        try:
            pod_list = await core.list_namespaced_pod(
                self._namespace,
                label_selector=self._current_label_selector(),
            )
        except (TimeoutError, ApiException, aiohttp.ClientError, OSError):
            logger.debug(f"Failed to list pods for {self._job_id}", exc_info=True)
            return
        self.pods = [PodSnapshot.from_raw(_pod_to_raw(p)) for p in pod_list.items]

    def _current_label_selector(self) -> str:
        if self._label_selector is not None:
            return self._label_selector
        if self._job_ids_provider is None:
            return job_selector(self._job_id)
        job_ids = [job_id for job_id in self._job_ids_provider() if job_id]
        if not job_ids:
            return f"{AIPerfLabels.SELECTOR},{AIPerfLabels.JOB_ID}=__pending_sweep_children__"
        if len(job_ids) == 1:
            return job_selector(job_ids[0])
        return f"{AIPerfLabels.SELECTOR},{AIPerfLabels.JOB_ID} in ({','.join(job_ids)})"


class EventPoller:
    """Polls the K8s Event API filtered to this job's resources."""

    def __init__(self, api: ApiClient, job_id: str, namespace: str) -> None:
        self._api = api
        self._job_id = job_id
        self._namespace = namespace
        self.events: list[EventSnapshot] = []

    async def poll(self) -> None:
        """Fetch latest events."""
        core = client.CoreV1Api(self._api)
        try:
            ev_list = await core.list_namespaced_event(self._namespace)
        except (TimeoutError, ApiException, aiohttp.ClientError, OSError):
            logger.debug(f"Failed to list events for {self._job_id}", exc_info=True)
            return

        filtered = []
        for ev in ev_list.items:
            involved = ev.involved_object
            involved_name = involved.name if involved and involved.name else ""
            if self._job_id not in involved_name:
                continue
            ts = ev.last_timestamp
            filtered.append(
                EventSnapshot(
                    timestamp=ts.isoformat() if ts else "",
                    type=ev.type or "",
                    reason=ev.reason or "",
                    object=involved_name,
                    message=ev.message or "",
                    count=1,
                )
            )

        self.events = sorted(filtered, key=lambda e: e.timestamp)[-20:]


def _pod_metadata_raw(metadata: Any) -> dict[str, Any]:
    if not metadata:
        return {}
    raw: dict[str, Any] = {
        "name": metadata.name or "",
        "namespace": metadata.namespace or "",
    }
    ts = metadata.creation_timestamp
    if ts:
        raw["creationTimestamp"] = (
            ts.isoformat() if isinstance(ts, datetime) else str(ts)
        )
    if metadata.labels:
        raw["labels"] = dict(metadata.labels)
    return raw


def _container_state_raw(state: Any) -> dict[str, Any]:
    if state is None:
        return {}
    if state.running is not None:
        return {"running": {}}
    if state.waiting is not None:
        w = state.waiting
        return {
            "waiting": {
                k: v
                for k, v in {"reason": w.reason, "message": w.message}.items()
                if v is not None
            }
        }
    if state.terminated is not None:
        t = state.terminated
        return {
            "terminated": {
                k: v
                for k, v in {
                    "reason": t.reason,
                    "message": t.message,
                    "exitCode": t.exit_code,
                }.items()
                if v is not None
            }
        }
    return {}


def _container_status_raw(cs: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": cs.name or "",
        "ready": bool(cs.ready),
        "restartCount": cs.restart_count or 0,
    }
    state_dict = _container_state_raw(cs.state)
    if state_dict:
        entry["state"] = state_dict
    return entry


def _pod_status_raw(status: Any) -> dict[str, Any]:
    if not status:
        return {}
    raw: dict[str, Any] = {}
    if status.phase:
        raw["phase"] = status.phase
    containers_raw = [
        _container_status_raw(cs) for cs in status.container_statuses or []
    ]
    if containers_raw:
        raw["containerStatuses"] = containers_raw
    return raw


def _pod_to_raw(pod: Any) -> dict[str, Any]:
    return {
        "metadata": _pod_metadata_raw(pod.metadata),
        "status": _pod_status_raw(pod.status),
    }
