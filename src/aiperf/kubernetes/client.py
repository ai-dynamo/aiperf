# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf Kubernetes client.

Provides AIPerfKubeClient, the central interface for all AIPerf-specific
Kubernetes operations (JobSet lookup, pod management, label construction, etc.).
Lower-level kr8s API access is available via the .api property.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import kr8s
from kr8s.asyncio.objects import ConfigMap, Role, RoleBinding

from aiperf.common.noisy_loggers import suppress_noisy_http_loggers
from aiperf.kubernetes.console import print_success, print_warning
from aiperf.kubernetes.constants import JobSetLabels, Labels
from aiperf.kubernetes.enums import PodPhase
from aiperf.kubernetes.kr8s_resources import AsyncAIPerfJob, AsyncJobSet
from aiperf.kubernetes.models import AIPerfJobCR, AIPerfJobInfo, JobSetInfo, PodSummary

logger = logging.getLogger(__name__)


def _kr8s_kwargs(kubeconfig: str | None, kube_context: str | None) -> dict[str, str]:
    """Build kwargs for kr8s api(), working around kr8s#737.

    kr8s crashes with KeyError when kubeconfig has no current-context field,
    even when context= is explicitly passed. Passing namespace= prevents
    kr8s from reading current-context to resolve the namespace.
    See https://github.com/kr8s-org/kr8s/issues/737
    """
    kwargs: dict[str, str] = {}
    if kubeconfig is not None:
        kwargs["kubeconfig"] = kubeconfig
    if kube_context is not None:
        kwargs["context"] = kube_context
        kwargs.setdefault("namespace", "default")
    return kwargs


async def get_api(
    *,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> kr8s.Api:
    """Get a kr8s async API client.

    Uses kr8s built-in config loading with client caching.
    Same arguments return the same cached client instance.
    kr8s auto-detects in-cluster vs kubeconfig environments.

    Args:
        kubeconfig: Path to kubeconfig file. If not specified,
                    uses KUBECONFIG env var or ~/.kube/config.
        kube_context: Kubernetes context name. If not specified,
                uses the current context in kubeconfig.

    Returns:
        kr8s async API client.
    """
    import kr8s.asyncio

    suppress_noisy_http_loggers()
    return await kr8s.asyncio.api(**_kr8s_kwargs(kubeconfig, kube_context))


class AIPerfKubeClient:
    """Async Kubernetes client for AIPerf operations.

    Wraps a kr8s API client and provides AIPerf-specific helpers for
    JobSet management, pod lookup, label construction, and resource cleanup.

    Create via the async classmethod::

        client = await AIPerfKubeClient.create(kubeconfig=..., kube_context=...)

    Or wrap an existing kr8s API::

        client = AIPerfKubeClient(api)
    """

    def __init__(self, api: kr8s.Api) -> None:
        self._api = api

    @classmethod
    async def create(
        cls,
        *,
        kubeconfig: str | None = None,
        kube_context: str | None = None,
    ) -> AIPerfKubeClient:
        """Create a new client connected to the cluster.

        Args:
            kubeconfig: Path to kubeconfig file.
            kube_context: Kubernetes context name.

        Returns:
            Connected AIPerfKubeClient instance.
        """
        api = await get_api(kubeconfig=kubeconfig, kube_context=kube_context)
        return cls(api)

    @property
    def api(self) -> kr8s.Api:
        """Underlying kr8s API client for advanced operations."""
        return self._api

    # -- Label selectors ------------------------------------------------

    @staticmethod
    def job_selector(job_id: str) -> str:
        """Label selector for all AIPerf resources belonging to a job.

        Args:
            job_id: AIPerf job ID.

        Returns:
            Label selector string for Kubernetes API.
        """
        return f"{Labels.SELECTOR},{Labels.JOB_ID}={job_id}"

    @staticmethod
    def controller_selector(job_id: str) -> str:
        """Label selector for the controller pod of a job.

        Args:
            job_id: AIPerf job ID.

        Returns:
            Label selector string for Kubernetes API.
        """
        return (
            f"{Labels.SELECTOR},{Labels.JOB_ID}={job_id},"
            f"{JobSetLabels.REPLICATED_JOB_NAME}=controller"
        )

    # -- AIPerfJob CR operations -----------------------------------------

    async def list_jobs(
        self,
        namespace: str | None = None,
        all_namespaces: bool = False,
        status_filter: str | None = None,
    ) -> list[AIPerfJobInfo]:
        """List AIPerfJob custom resources.

        Args:
            namespace: Namespace to search in (default: "default").
            all_namespaces: If True, search all namespaces.
            status_filter: Optional phase filter (e.g. "Running", "Completed", "Failed").

        Returns:
            List of AIPerfJobInfo, sorted by creation time (newest first).
        """
        ns = kr8s.ALL if all_namespaces else (namespace or "default")
        try:
            jobs = [j async for j in self._api.async_get(AsyncAIPerfJob, namespace=ns)]
        except kr8s.ServerError as e:
            if e.response and e.response.status_code == 404:
                return []
            raise

        infos = [AIPerfJobCR.model_validate(j.raw).to_info() for j in jobs]

        if status_filter:
            infos = [i for i in infos if i.phase == status_filter]

        infos.sort(key=lambda x: x.created, reverse=True)
        return infos

    async def find_job(
        self,
        name: str,
        namespace: str | None = None,
    ) -> AIPerfJobInfo | None:
        """Find an AIPerfJob CR by name or jobId.

        Searches by resource name first. If namespace is None, searches
        all namespaces.

        Args:
            name: The AIPerfJob name or job ID to search for.
            namespace: Optional namespace to search in.

        Returns:
            AIPerfJobInfo if found, None otherwise.
        """
        ns = namespace or kr8s.ALL
        try:
            jobs = [
                j
                async for j in self._api.async_get(
                    AsyncAIPerfJob,
                    namespace=ns,
                    field_selector=f"metadata.name={name}",
                )
            ]
            if jobs:
                return AIPerfJobCR.model_validate(jobs[0].raw).to_info()
        except kr8s.ServerError as e:
            if e.response and e.response.status_code == 404:
                pass
            else:
                raise

        # Fallback: scan all jobs looking for matching jobId in status
        try:
            all_jobs = [
                j async for j in self._api.async_get(AsyncAIPerfJob, namespace=ns)
            ]
        except kr8s.ServerError as e:
            if e.response and e.response.status_code == 404:
                return None
            raise

        for j in all_jobs:
            raw = j.raw
            cr = AIPerfJobCR.model_validate(raw)
            if cr.status.job_id == name:
                return cr.to_info()

        return None

    async def get_raw_status(self, name: str, namespace: str) -> dict[str, Any]:
        """Get the raw status dict for an AIPerfJob by name.

        Useful for accessing sweep runDetails and other fields not in the
        typed model.

        Returns:
            Raw status dict, or empty dict if not found.
        """
        try:
            jobs = [
                j
                async for j in self._api.async_get(
                    AsyncAIPerfJob,
                    namespace=namespace,
                    field_selector=f"metadata.name={name}",
                )
            ]
            if jobs:
                return jobs[0].raw.get("status", {})
        except Exception:
            pass
        return {}

    async def cancel_job(self, name: str, namespace: str) -> None:
        """Cancel an AIPerfJob by setting spec.cancel=true.

        Args:
            name: Job name.
            namespace: Namespace containing the job.
        """
        jobs = [
            j
            async for j in self._api.async_get(
                AsyncAIPerfJob,
                namespace=namespace,
                field_selector=f"metadata.name={name}",
            )
        ]
        if not jobs:
            return
        await jobs[0].patch({"spec": {"cancel": True}}, type="merge")

    # -- JobSet operations -----------------------------------------------

    async def _list_jobsets(
        self,
        label_selector: str,
        namespace: str | None = None,
        field_selector: str | None = None,
    ) -> list[JobSetInfo]:
        """List JobSet objects from Kubernetes API.

        Args:
            label_selector: Label selector string.
            namespace: Namespace to search in. If None, searches all namespaces.
            field_selector: Optional field selector (e.g. 'metadata.name=foo').

        Returns:
            List of JobSetInfo parsed from API response.
        """
        ns = namespace or kr8s.ALL
        kwargs: dict[str, Any] = {"namespace": ns, "label_selector": label_selector}
        if field_selector:
            kwargs["field_selector"] = field_selector

        jobsets = [js async for js in self._api.async_get(AsyncJobSet, **kwargs)]
        return [JobSetInfo.from_raw(js.raw) for js in jobsets]

    async def find_jobset(
        self,
        job_id: str,
        namespace: str | None = None,
    ) -> JobSetInfo | None:
        """Find a JobSet by job ID or JobSet name.

        First searches by the AIPerf job ID label. If not found, falls back
        to matching the JobSet name directly (e.g. 'aiperf-abc123').

        Args:
            job_id: The AIPerf job ID or JobSet name to search for.
            namespace: Optional namespace to search in. If None, searches all.

        Returns:
            JobSetInfo if found, None otherwise.
        """
        label_selector = self.job_selector(job_id)
        items = await self._list_jobsets(label_selector, namespace)
        if items:
            return items[0]

        items = await self._list_jobsets(
            Labels.SELECTOR, namespace, field_selector=f"metadata.name={job_id}"
        )
        if items:
            return items[0]

        return None

    async def list_jobsets(
        self,
        namespace: str | None = None,
        all_namespaces: bool = False,
        job_id: str | None = None,
        status_filter: str | None = None,
    ) -> list[JobSetInfo]:
        """List AIPerf JobSets.

        Args:
            namespace: Namespace to search in (default: "default").
            all_namespaces: If True, search all namespaces.
            job_id: Optional job ID to filter by.
            status_filter: Optional status filter ("Running", "Completed", "Failed").

        Returns:
            List of JobSetInfo, sorted by creation time (newest first).
        """
        label_selector = Labels.SELECTOR
        if job_id:
            label_selector += f",{Labels.JOB_ID}={job_id}"

        ns = None if all_namespaces else (namespace or "default")
        try:
            infos = await self._list_jobsets(label_selector, ns)
        except kr8s.ServerError as e:
            if e.response and e.response.status_code == 404:
                return []
            raise

        if status_filter:
            infos = [info for info in infos if info.status == status_filter]

        infos.sort(key=lambda x: x.created, reverse=True)

        return infos

    # -- Pod operations --------------------------------------------------

    async def get_pod_summary(self, jobset_name: str, namespace: str) -> PodSummary:
        """Get pod readiness summary for a JobSet.

        Args:
            jobset_name: Name of the JobSet.
            namespace: Namespace of the JobSet.

        Returns:
            PodSummary with ready count, total count, and restart count.
        """
        try:
            label_selector = f"{JobSetLabels.JOBSET_NAME}={jobset_name}"
            pods = [
                p
                async for p in self._api.async_get(
                    "pods", namespace=namespace, label_selector=label_selector
                )
            ]
        except kr8s.ServerError:
            return PodSummary(ready=0, total=0, restarts=0)

        total = len(pods)
        ready = 0
        restarts = 0
        for pod in pods:
            raw = pod.raw
            statuses = raw.get("status", {}).get("containerStatuses", [])
            pod_ready = all(s.get("ready") for s in statuses) and statuses
            if pod_ready and raw.get("status", {}).get("phase") == PodPhase.RUNNING:
                ready += 1
            restarts += sum(s.get("restartCount", 0) for s in statuses)

        return PodSummary(ready=ready, total=total, restarts=restarts)

    async def find_operator_pod(
        self,
        namespace: str = "aiperf-system",
        label_selector: str = "app.kubernetes.io/name=aiperf-operator",
    ) -> tuple[str, PodPhase] | None:
        """Find the operator pod and return (pod_name, pod_phase).

        Args:
            namespace: Namespace where the operator is deployed.
            label_selector: Label selector for the operator pod.

        Returns:
            Tuple of (pod_name, pod_phase) if found, None otherwise.
        """
        pods = [
            p
            async for p in self._api.async_get(
                "pods",
                namespace=namespace,
                label_selector=label_selector,
            )
        ]
        if not pods:
            return None
        pod = pods[0]
        raw_phase = pod.raw.get("status", {}).get("phase", "Unknown")
        return (pod.name, PodPhase(raw_phase))

    async def find_controller_pod(
        self,
        namespace: str,
        job_id: str,
    ) -> tuple[str, PodPhase] | None:
        """Find controller pod and return (pod_name, pod_phase).

        Args:
            namespace: Kubernetes namespace.
            job_id: AIPerf job ID.

        Returns:
            Tuple of (pod_name, pod_phase) if found, None otherwise.
        """
        pods = [
            p
            async for p in self._api.async_get(
                "pods",
                namespace=namespace,
                label_selector=self.controller_selector(job_id),
            )
        ]
        if not pods:
            return None
        pod = pods[0]
        raw_phase = pod.raw.get("status", {}).get("phase", "Unknown")
        return (pod.name, PodPhase(raw_phase))

    async def find_retrievable_pod(
        self,
        namespace: str,
        job_id: str,
        *,
        require_running: bool = False,
    ) -> tuple[str, PodPhase] | None:
        """Find controller pod and validate its phase for retrieval.

        Args:
            namespace: Kubernetes namespace.
            job_id: AIPerf job ID.
            require_running: If True, only accept Running phase (not Succeeded).

        Returns:
            (pod_name, pod_phase) if pod is in a retrievable state, None otherwise.
        """
        pod_info = await self.find_controller_pod(namespace, job_id)
        if not pod_info:
            return None

        pod_name, pod_phase = pod_info
        if require_running:
            if pod_phase != PodPhase.RUNNING:
                return None
        elif not pod_phase.is_retrievable:
            return None

        return pod_name, pod_phase

    async def wait_for_controller_pod_ready(
        self,
        namespace: str,
        job_id: str,
        timeout: int = 300,
    ) -> str:
        """Wait for controller pod to be in Running state.

        Args:
            namespace: Kubernetes namespace.
            job_id: AIPerf job ID.
            timeout: Maximum time to wait in seconds.

        Returns:
            Controller pod name when ready.

        Raises:
            TimeoutError: If pod not ready within timeout.
        """
        start_time = asyncio.get_running_loop().time()
        last_log = 0.0

        while True:
            result = await self.find_controller_pod(namespace, job_id)
            elapsed = asyncio.get_running_loop().time() - start_time

            if result:
                pod_name, phase = result
                if phase == PodPhase.RUNNING:
                    return pod_name
                if elapsed - last_log >= 10:
                    logger.info(
                        "Controller pod %s: %s (%.0fs)", pod_name, phase, elapsed
                    )
                    last_log = elapsed
            elif elapsed - last_log >= 10:
                logger.info("No controller pod found yet (%.0fs)", elapsed)
                last_log = elapsed

            if elapsed > timeout:
                raise TimeoutError(
                    f"Controller pod not ready after {timeout}s. "
                    f"Check pod status with: kubectl get pods -n {namespace}"
                )

            await asyncio.sleep(2)

    async def get_pods(self, namespace: str, label_selector: str) -> list[Any]:
        """Get pods matching a label selector.

        Args:
            namespace: Kubernetes namespace.
            label_selector: Label selector string.

        Returns:
            List of kr8s Pod objects.
        """
        return [
            p
            async for p in self._api.async_get(
                "pods", namespace=namespace, label_selector=label_selector
            )
        ]

    # -- Resource management ---------------------------------------------

    async def delete_jobset(self, name: str, namespace: str) -> None:
        """Delete a JobSet and its associated resources.

        Args:
            name: JobSet name.
            namespace: Namespace containing the JobSet.
        """
        try:
            jobset = await AsyncJobSet.get(name, namespace=namespace, api=self._api)
            await jobset.delete()
            print_success(f"Deleted JobSet/{name}")
        except kr8s.NotFoundError:
            print_warning(f"JobSet/{name} not found")

        for cls, resource_name, kind_name in [
            (ConfigMap, f"{name}-config", "ConfigMap"),
            (Role, f"{name}-role", "Role"),
            (RoleBinding, f"{name}-binding", "RoleBinding"),
        ]:
            try:
                obj = await cls.get(resource_name, namespace=namespace, api=self._api)
                await obj.delete()
                print_success(f"Deleted {kind_name}/{resource_name}")
            except kr8s.NotFoundError:
                pass
            except kr8s.ServerError as e:
                status = e.response.status_code if e.response else 0
                if status != 404:
                    print_warning(f"Failed to delete {kind_name}: {e}")

    async def delete_namespace(self, namespace: str) -> None:
        """Delete a Kubernetes namespace.

        Args:
            namespace: Namespace to delete.
        """
        from kr8s.asyncio.objects import Namespace

        from aiperf.kubernetes.console import print_info

        try:
            ns = await Namespace.get(namespace, api=self._api)
            await ns.delete()
            print_success(f"Deleted Namespace/{namespace}")
        except kr8s.NotFoundError:
            print_info(f"Namespace {namespace} not found (may already be deleted)")
        except kr8s.ServerError as e:
            print_warning(f"Failed to delete namespace: {e}")

    # -- Cluster info ----------------------------------------------------

    async def version(self) -> dict[str, Any]:
        """Get Kubernetes cluster version info.

        Returns:
            Version info dict from the Kubernetes API.
        """
        return await self._api.async_version()
