# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kueue installation and resource management for Kind cluster E2E tests."""

from __future__ import annotations

import asyncio
from typing import Any

import yaml

from aiperf.common.aiperf_logger import AIPerfLogger
from dev.versions import KUEUE_MANIFEST_URL_TEMPLATE, KUEUE_VERSION
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)


class KueueManager:
    """Manages Kueue controller installation and queue resources."""

    def __init__(self, kubectl: KubectlClient) -> None:
        self.kubectl = kubectl
        self._created_resources: list[tuple[str, str, str | None]] = []

    async def install(self) -> None:
        """Install Kueue controller and wait for it to become available."""
        url = KUEUE_MANIFEST_URL_TEMPLATE.format(version=KUEUE_VERSION)
        logger.info(f"Installing Kueue {KUEUE_VERSION}")
        await self.kubectl.apply_server_side(url)

        success = await self.kubectl.wait_for_condition(
            "deployment",
            "kueue-controller-manager",
            "available",
            namespace="kueue-system",
            timeout=120,
        )
        if not success:
            raise RuntimeError("Kueue controller-manager did not become available")
        logger.info("Kueue controller-manager is available")

    async def uninstall(self) -> None:
        """Delete the Kueue installation."""
        url = KUEUE_MANIFEST_URL_TEMPLATE.format(version=KUEUE_VERSION)
        logger.info(f"Uninstalling Kueue {KUEUE_VERSION}")
        result = await self.kubectl.run(
            "delete", "-f", url, "--ignore-not-found", check=False
        )
        if result.returncode != 0:
            logger.warning(f"Kueue uninstall returned non-zero: {result.stderr}")

    async def create_resource_flavor(self, name: str = "default-flavor") -> None:
        """Create a ResourceFlavor CR.

        Args:
            name: ResourceFlavor name.
        """
        manifest = yaml.dump(
            {
                "apiVersion": "kueue.x-k8s.io/v1beta1",
                "kind": "ResourceFlavor",
                "metadata": {"name": name},
            }
        )
        await self.kubectl.apply(manifest)
        self._created_resources.append(("resourceflavor", name, None))
        logger.info(f"Created ResourceFlavor {name}")

    async def create_cluster_queue(
        self,
        name: str = "cluster-queue",
        resource_flavor: str = "default-flavor",
        cpu: str = "100",
        memory: str = "100Gi",
    ) -> None:
        """Create a ClusterQueue with a single resource group.

        Args:
            name: ClusterQueue name.
            resource_flavor: ResourceFlavor to reference.
            cpu: CPU nominal quota.
            memory: Memory nominal quota.
        """
        manifest = yaml.dump(
            {
                "apiVersion": "kueue.x-k8s.io/v1beta1",
                "kind": "ClusterQueue",
                "metadata": {"name": name},
                "spec": {
                    "namespaceSelector": {},
                    "resourceGroups": [
                        {
                            "coveredResources": ["cpu", "memory"],
                            "flavors": [
                                {
                                    "name": resource_flavor,
                                    "resources": [
                                        {
                                            "name": "cpu",
                                            "nominalQuota": int(cpu)
                                            if cpu.isdigit()
                                            else cpu,
                                        },
                                        {"name": "memory", "nominalQuota": memory},
                                    ],
                                }
                            ],
                        }
                    ],
                },
            }
        )
        await self.kubectl.apply(manifest)
        self._created_resources.append(("clusterqueue", name, None))
        logger.info(f"Created ClusterQueue {name}")

    async def create_local_queue(
        self,
        name: str = "local-queue",
        namespace: str = "default",
        cluster_queue: str = "cluster-queue",
    ) -> None:
        """Create a LocalQueue pointing to a ClusterQueue.

        Args:
            name: LocalQueue name.
            namespace: Target namespace.
            cluster_queue: ClusterQueue to reference.
        """
        manifest = yaml.dump(
            {
                "apiVersion": "kueue.x-k8s.io/v1beta1",
                "kind": "LocalQueue",
                "metadata": {"name": name, "namespace": namespace},
                "spec": {"clusterQueue": cluster_queue},
            }
        )
        await self.kubectl.apply(manifest, namespace=namespace)
        self._created_resources.append(("localqueue", name, namespace))
        logger.info(f"Created LocalQueue {name} in namespace {namespace}")

    async def setup_default_queues(self, namespace: str = "default") -> str:
        """Create ResourceFlavor, ClusterQueue, and LocalQueue with defaults.

        Args:
            namespace: Namespace for the LocalQueue.

        Returns:
            The LocalQueue name.
        """
        await self.create_resource_flavor()
        await self.create_cluster_queue()
        await self.create_local_queue(namespace=namespace)
        return "local-queue"

    async def cleanup_queues(self) -> None:
        """Delete all created queue resources in reverse order."""
        for resource_type, name, namespace in reversed(self._created_resources):
            try:
                await self.kubectl.delete(
                    resource_type, name, namespace=namespace, ignore_not_found=True
                )
                logger.info(f"Deleted {resource_type} {name}")
            except Exception as e:
                logger.warning(f"Failed to delete {resource_type} {name}: {e}")
        self._created_resources.clear()

    async def get_workloads(self, namespace: str) -> list[dict[str, Any]]:
        """List Kueue Workload objects in a namespace.

        Args:
            namespace: Namespace to query.

        Returns:
            List of Workload resource dicts.
        """
        data = await self.kubectl.get_json("workload", namespace=namespace)
        if isinstance(data, dict):
            return data.get("items", [])
        return data

    async def wait_for_workload_admitted(
        self, namespace: str, timeout: int = 60
    ) -> dict[str, Any]:
        """Wait until a Workload in the namespace has Admitted=True.

        Args:
            namespace: Namespace to watch.
            timeout: Timeout in seconds.

        Returns:
            The admitted Workload resource dict.

        Raises:
            TimeoutError: If no workload is admitted within the timeout.
        """
        start = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start
            if elapsed > timeout:
                raise TimeoutError(
                    f"No workload admitted in namespace {namespace} after {timeout}s"
                )

            try:
                workloads = await self.get_workloads(namespace)
                for wl in workloads:
                    conditions = wl.get("status", {}).get("conditions", [])
                    for cond in conditions:
                        if (
                            cond.get("type") == "Admitted"
                            and cond.get("status") == "True"
                        ):
                            name = wl.get("metadata", {}).get("name", "unknown")
                            logger.info(f"Workload {name} admitted")
                            return wl
            except Exception:
                pass

            logger.debug(
                lambda elapsed=elapsed: f"Waiting for workload admission ({elapsed:.0f}s)"
            )
            await asyncio.sleep(2)
