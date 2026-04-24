# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helm deployment helper for Kubernetes E2E tests."""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.helpers.cluster import _run_streaming
from tests.kubernetes.helpers.kubectl import KubectlClient, background_status
from tests.kubernetes.helpers.log_streamer import PodLogStreamer
from tests.kubernetes.helpers.operator import (
    AIPerfJobConfig,
    AIPerfJobStatus,
    OperatorJobResult,
)

logger = AIPerfLogger(__name__)


@dataclass
class HelmValues:
    """Helm values configuration for aiperf-operator chart."""

    image_repository: str = "aiperf"
    """Container image repository for the operator."""

    image_tag: str = "local"
    """Container image tag for the operator."""

    image_pull_policy: str = "Never"
    """Image pull policy for the operator pod."""

    operator_replicas: int = 1
    """Number of operator replicas."""

    monitor_interval: str = "10.0"
    """Operator monitor polling interval in seconds."""

    monitor_initial_delay: str = "5.0"
    """Delay before first monitor poll in seconds."""

    storage_enabled: bool = False
    """Whether to enable persistent storage for results."""

    storage_size: str = "1Gi"
    """PVC size when storage is enabled."""

    default_image: str = "aiperf:local"
    """Default benchmark job container image."""

    default_image_pull_policy: str = "Never"
    """Default image pull policy for benchmark jobs."""

    resources_requests_cpu: str = "250m"
    """CPU request for the operator pod."""

    resources_requests_memory: str = "256Mi"
    """Memory request for the operator pod."""

    resources_limits_cpu: str = "500m"
    """CPU limit for the operator pod."""

    resources_limits_memory: str = "512Mi"
    """Memory limit for the operator pod."""

    def to_set_args(self) -> list[str]:
        """Convert to helm --set arguments."""
        args = [
            f"image.repository={self.image_repository}",
            f"image.tag={self.image_tag}",
            f"image.pullPolicy={self.image_pull_policy}",
            f"operator.replicas={self.operator_replicas}",
            f"operator.env.monitorInterval={self.monitor_interval}",
            f"operator.env.monitorInitialDelay={self.monitor_initial_delay}",
            f"storage.enabled={str(self.storage_enabled).lower()}",
            f"storage.size={self.storage_size}",
            f"defaults.image={self.default_image}",
            f"defaults.imagePullPolicy={self.default_image_pull_policy}",
            f"operator.resources.requests.cpu={self.resources_requests_cpu}",
            f"operator.resources.requests.memory={self.resources_requests_memory}",
            f"operator.resources.limits.cpu={self.resources_limits_cpu}",
            f"operator.resources.limits.memory={self.resources_limits_memory}",
        ]
        return args


@dataclass
class HelmRelease:
    """Information about a Helm release."""

    name: str
    """Release name."""

    namespace: str
    """Namespace the release is deployed in."""

    chart_path: Path
    """Path to the Helm chart directory."""

    values: HelmValues
    """Helm values used for this release."""

    status: str = "unknown"
    """Current release status."""

    revision: int = 0
    """Release revision number."""


class HelmClient:
    """Client for interacting with Helm CLI."""

    def __init__(self, kubecontext: str | None = None) -> None:
        """Initialize Helm client.

        Args:
            kubecontext: Kubernetes context to use.
        """
        self.kubecontext = kubecontext

    async def _run(
        self,
        *args: str,
        check: bool = True,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run a helm command.

        Args:
            *args: Helm command arguments.
            check: Raise on non-zero exit code.
            capture_output: Capture stdout/stderr.

        Returns:
            Completed process result.
        """
        cmd = ["helm"]
        if self.kubecontext:
            cmd.extend(["--kube-context", self.kubecontext])
        cmd.extend(args)

        logger.debug(lambda cmd=cmd: f"Running: {' '.join(cmd)}")

        if capture_output:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
            proc = await asyncio.create_subprocess_exec(*cmd)

        stdout_bytes, stderr_bytes = await proc.communicate()

        stdout = stdout_bytes.decode() if stdout_bytes else ""
        stderr = stderr_bytes.decode() if stderr_bytes else ""

        result = subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode or 0,
            stdout=stdout,
            stderr=stderr,
        )

        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, stdout, stderr)

        return result

    async def install(
        self,
        release_name: str,
        chart_path: Path,
        namespace: str,
        values: HelmValues | None = None,
        wait: bool = True,
        timeout: str = "5m",
        create_namespace: bool = True,
    ) -> None:
        """Install a Helm chart.

        Args:
            release_name: Name of the release.
            chart_path: Path to the chart directory.
            namespace: Target namespace.
            values: Helm values to set.
            wait: Wait for resources to be ready.
            timeout: Timeout for wait.
            create_namespace: Create namespace if it doesn't exist.
        """
        args = [
            "install",
            release_name,
            str(chart_path),
            "--namespace",
            namespace,
        ]

        if create_namespace:
            args.append("--create-namespace")

        if wait:
            args.extend(["--wait", "--timeout", timeout])

        if values:
            for set_arg in values.to_set_args():
                args.extend(["--set", set_arg])

        logger.info(f"Installing Helm release: {release_name} in {namespace}")
        cmd = ["helm"]
        if self.kubecontext:
            cmd.extend(["--kube-context", self.kubecontext])
        cmd.extend(args)
        await _run_streaming(cmd, "HELM", f"Failed to install {release_name}")

    async def upgrade(
        self,
        release_name: str,
        chart_path: Path,
        namespace: str,
        values: HelmValues | None = None,
        wait: bool = True,
        timeout: str = "5m",
    ) -> None:
        """Upgrade a Helm release.

        Args:
            release_name: Name of the release.
            chart_path: Path to the chart directory.
            namespace: Target namespace.
            values: Helm values to set.
            wait: Wait for resources to be ready.
            timeout: Timeout for wait.
        """
        args = [
            "upgrade",
            release_name,
            str(chart_path),
            "--namespace",
            namespace,
        ]

        if wait:
            args.extend(["--wait", "--timeout", timeout])

        if values:
            for set_arg in values.to_set_args():
                args.extend(["--set", set_arg])

        logger.info(f"Upgrading Helm release: {release_name}")
        cmd = ["helm"]
        if self.kubecontext:
            cmd.extend(["--kube-context", self.kubecontext])
        cmd.extend(args)
        await _run_streaming(cmd, "HELM", f"Failed to upgrade {release_name}")

    async def uninstall(
        self,
        release_name: str,
        namespace: str,
        wait: bool = True,
        ignore_not_found: bool = True,
    ) -> None:
        """Uninstall a Helm release.

        Args:
            release_name: Name of the release.
            namespace: Namespace of the release.
            wait: Wait for resources to be deleted.
            ignore_not_found: Don't error if release doesn't exist.
        """
        args = ["uninstall", release_name, "--namespace", namespace]

        if wait:
            args.append("--wait")

        logger.info(f"Uninstalling Helm release: {release_name}")
        result = await self._run(*args, check=not ignore_not_found)

        if (
            result.returncode != 0
            and ignore_not_found
            and "not found" not in result.stderr.lower()
        ):
            raise subprocess.CalledProcessError(
                result.returncode, result.args, result.stdout, result.stderr
            )

    async def get_release_status(self, release_name: str, namespace: str) -> str:
        """Get the status of a Helm release.

        Args:
            release_name: Name of the release.
            namespace: Namespace of the release.

        Returns:
            Release status string.
        """
        result = await self._run(
            "status",
            release_name,
            "--namespace",
            namespace,
            "-o",
            "json",
            check=False,
        )
        if result.returncode != 0:
            return "not-found"

        import json

        try:
            data = json.loads(result.stdout)
            return data.get("info", {}).get("status", "unknown")
        except json.JSONDecodeError:
            return "unknown"

    async def list_releases(self, namespace: str | None = None) -> list[dict[str, Any]]:
        """List Helm releases.

        Args:
            namespace: Filter by namespace, or all if None.

        Returns:
            List of release info dicts.
        """
        args = ["list", "-o", "json"]
        if namespace:
            args.extend(["--namespace", namespace])
        else:
            args.append("--all-namespaces")

        result = await self._run(*args)

        import json

        try:
            return json.loads(result.stdout) or []
        except json.JSONDecodeError:
            return []

    async def template(
        self,
        release_name: str,
        chart_path: Path,
        namespace: str,
        values: HelmValues | None = None,
    ) -> str:
        """Render chart templates locally.

        Args:
            release_name: Name for the release.
            chart_path: Path to the chart directory.
            namespace: Target namespace.
            values: Helm values to set.

        Returns:
            Rendered YAML manifests.
        """
        args = [
            "template",
            release_name,
            str(chart_path),
            "--namespace",
            namespace,
        ]

        if values:
            for set_arg in values.to_set_args():
                args.extend(["--set", set_arg])

        result = await self._run(*args)
        return result.stdout


class HelmDeployer:
    """Manages Helm-based operator deployment and AIPerfJob lifecycle."""

    # Use a distinct release name from the kubectl-deployed operator so the
    # generated cluster-scoped resources (ClusterRole, ClusterRoleBinding)
    # have different names. Otherwise a helm test's CRB would clobber the
    # kubectl deploy's CRB and break cross-test RBAC checks.
    RELEASE_NAME = "aiperf-operator-helm"
    OPERATOR_NAMESPACE = "aiperf-system"
    CRD_NAME = "aiperfjobs.aiperf.nvidia.com"

    def __init__(
        self,
        kubectl: KubectlClient,
        helm: HelmClient,
        project_root: Path,
        values: HelmValues | None = None,
        operator_namespace: str | None = None,
        default_job_namespace: str = "default",
    ) -> None:
        """Initialize Helm deployer.

        Args:
            kubectl: Kubectl client.
            helm: Helm client.
            project_root: Path to project root.
            values: Helm values for deployment.
            operator_namespace: Namespace for operator deployment. Defaults to
                OPERATOR_NAMESPACE. Use a unique namespace to avoid conflicts
                with the package-scoped OperatorDeployer.
        """
        self.kubectl = kubectl
        self.helm = helm
        self.project_root = project_root
        self.chart_path = project_root / "deploy" / "helm" / "aiperf-operator"
        self.values = values or HelmValues()
        self._deployed_jobs: list[OperatorJobResult] = []
        self._release_installed = False
        self.default_job_namespace = default_job_namespace
        if operator_namespace:
            self.OPERATOR_NAMESPACE = operator_namespace

    async def install_chart(self, wait: bool = True, timeout: str = "5m") -> None:
        """Install the aiperf-operator Helm chart.

        Idempotent: if a release already exists, it is uninstalled first.
        Also cleans up any non-helm operator deployment to avoid conflicts.

        Args:
            wait: Wait for deployment to be ready.
            timeout: Timeout for wait.
        """
        logger.info(f"Installing Helm chart from {self.chart_path}")

        # Clean up any existing release for idempotent install
        if await self.is_installed():
            logger.info("Existing helm release found, uninstalling for clean install")
            await self.uninstall_chart(wait=False)
            await asyncio.sleep(3)

        # Clean up any non-helm resources that could conflict (PVCs, deployments, etc.)
        # Namespace-scoped resources
        await self.kubectl.run(
            "delete",
            "all,pvc,sa,roles,rolebindings",
            "--all",
            "-n",
            self.OPERATOR_NAMESPACE,
            check=False,
        )
        # Cluster-scoped resources from non-helm operator deployments
        for resource in ["clusterrole", "clusterrolebinding"]:
            await self.kubectl.run(
                "delete",
                resource,
                "aiperf-operator",
                check=False,
            )
        # Add Helm ownership metadata to ALL pre-existing resources that the
        # chart manages. The OperatorDeployer creates these via kubectl apply
        # (not helm), which leaves them without Helm metadata.
        # Rather than annotating each resource individually, render the chart
        # and annotate everything it would create.
        import subprocess as sp

        rendered = sp.run(
            [
                "helm",
                "template",
                self.RELEASE_NAME,
                str(self.chart_path),
                "-n",
                self.OPERATOR_NAMESPACE,
                *[f"--set={a}" for a in self.values.to_set_args()],
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if rendered.returncode == 0:
            for doc in yaml.safe_load_all(rendered.stdout):
                if not doc or "kind" not in doc:
                    continue
                kind = doc["kind"]
                name = doc.get("metadata", {}).get("name", "")
                ns = doc.get("metadata", {}).get("namespace")
                if not name:
                    continue
                ns_args = ["-n", ns] if ns else []
                await self.kubectl.run(
                    "annotate",
                    kind.lower(),
                    name,
                    *ns_args,
                    "--overwrite",
                    f"meta.helm.sh/release-name={self.RELEASE_NAME}",
                    f"meta.helm.sh/release-namespace={self.OPERATOR_NAMESPACE}",
                    check=False,
                )
                await self.kubectl.run(
                    "label",
                    kind.lower(),
                    name,
                    *ns_args,
                    "--overwrite",
                    "app.kubernetes.io/managed-by=Helm",
                    check=False,
                )
        await asyncio.sleep(3)

        await self.helm.install(
            self.RELEASE_NAME,
            self.chart_path,
            self.OPERATOR_NAMESPACE,
            values=self.values,
            wait=wait,
            timeout=timeout,
        )
        self._release_installed = True

        await self._wait_for_crd_established()

    async def _wait_for_crd_established(self, timeout: int = 60) -> None:
        """Wait for CRD to be established."""
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            result = await self.kubectl.run(
                "get",
                "crd",
                self.CRD_NAME,
                "-o",
                "jsonpath={.status.conditions[?(@.type=='Established')].status}",
                check=False,
            )
            if result.stdout.strip() == "True":
                logger.info("CRD established")
                return
            await asyncio.sleep(1)
        raise TimeoutError(f"CRD {self.CRD_NAME} not established within {timeout}s")

    async def upgrade_chart(
        self,
        values: HelmValues | None = None,
        wait: bool = True,
        timeout: str = "5m",
    ) -> None:
        """Upgrade the Helm release with new values.

        Args:
            values: New Helm values.
            wait: Wait for deployment to be ready.
            timeout: Timeout for wait.
        """
        use_values = values or self.values
        await self.helm.upgrade(
            self.RELEASE_NAME,
            self.chart_path,
            self.OPERATOR_NAMESPACE,
            values=use_values,
            wait=wait,
            timeout=timeout,
        )
        if values:
            self.values = values

    async def uninstall_chart(self, wait: bool = True) -> None:
        """Uninstall the Helm release."""
        logger.info("Uninstalling Helm release")
        await self.helm.uninstall(
            self.RELEASE_NAME,
            self.OPERATOR_NAMESPACE,
            wait=wait,
        )
        self._release_installed = False

    async def get_release_status(self) -> str:
        """Get the status of the Helm release."""
        return await self.helm.get_release_status(
            self.RELEASE_NAME, self.OPERATOR_NAMESPACE
        )

    async def is_installed(self) -> bool:
        """Check if the release exists in any state (deployed, failed, pending, etc.)."""
        status = await self.get_release_status()
        return status != "not-found"

    async def get_operator_logs(self, tail: int = 100) -> str:
        """Get operator logs.

        Args:
            tail: Number of lines to tail.

        Returns:
            Log content.
        """
        return await self.kubectl.get_logs(
            f"deployment/{self.RELEASE_NAME}",
            namespace=self.OPERATOR_NAMESPACE,
            tail=tail,
        )

    async def create_job(
        self,
        config: AIPerfJobConfig,
        name: str | None = None,
        namespace: str | None = None,
    ) -> OperatorJobResult:
        """Create an AIPerfJob CR.

        Args:
            config: Job configuration.
            name: Job name (auto-generated if not provided).
            namespace: Target namespace. Defaults to the deployer's
                ``default_job_namespace`` so xdist workers stay isolated.

        Returns:
            OperatorJobResult with initial state.
        """
        import uuid

        if name is None:
            name = f"benchmark-{uuid.uuid4().hex[:8]}"
        if namespace is None:
            namespace = self.default_job_namespace

        manifest = config.to_cr_manifest(name, namespace)
        logger.info(f"Creating AIPerfJob {namespace}/{name}")

        await self.kubectl.apply(manifest)

        result = OperatorJobResult(
            namespace=namespace,
            job_name=name,
            config=config,
        )
        self._deployed_jobs.append(result)

        return result

    async def get_job_status(self, name: str, namespace: str) -> AIPerfJobStatus:
        """Get current status of an AIPerfJob.

        Args:
            name: Job name.
            namespace: Namespace.

        Returns:
            AIPerfJobStatus with current state.
        """
        data = await self.kubectl.get_json("aiperfjob", name, namespace=namespace)
        return AIPerfJobStatus.from_json(data)

    async def wait_for_job_completion(
        self,
        name: str,
        namespace: str,
        timeout: int = 300,
        poll_interval: int = 5,
    ) -> AIPerfJobStatus:
        """Wait for an AIPerfJob to reach terminal state.

        Args:
            name: Job name.
            namespace: Namespace.
            timeout: Timeout in seconds.
            poll_interval: Polling interval in seconds.

        Returns:
            Final job status.

        Raises:
            TimeoutError: If timeout exceeded.
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time

            if elapsed > timeout:
                status = await self.get_job_status(name, namespace)
                raise TimeoutError(
                    f"Timeout waiting for AIPerfJob {name} completion. "
                    f"Current phase: {status.phase}"
                )

            status = await self.get_job_status(name, namespace)

            if status.is_terminal:
                logger.info(f"AIPerfJob {name} reached terminal state: {status.phase}")
                return status

            logger.info(
                f"AIPerfJob {name}: phase={status.phase}, "
                f"workers={status.workers_ready}/{status.workers_total}, "
                f"elapsed={elapsed:.0f}s"
            )
            await asyncio.sleep(poll_interval)

    async def delete_job(self, name: str, namespace: str) -> None:
        """Delete an AIPerfJob CR.

        Args:
            name: Job name.
            namespace: Namespace.
        """
        logger.info(f"Deleting AIPerfJob {namespace}/{name}")
        await self.kubectl.delete("aiperfjob", name, namespace=namespace)

    async def run_job(
        self,
        config: AIPerfJobConfig,
        name: str | None = None,
        namespace: str | None = None,
        timeout: int = 300,
        stream_logs: bool = False,
    ) -> OperatorJobResult:
        """Create a job and wait for completion.

        Args:
            config: Job configuration.
            name: Job name (auto-generated if not provided).
            namespace: Target namespace.
            timeout: Timeout in seconds.
            stream_logs: If True, stream pod logs in the background.

        Returns:
            OperatorJobResult with final state.
        """
        start_time = asyncio.get_event_loop().time()

        result = await self.create_job(config, name, namespace)
        name = result.job_name
        namespace = result.namespace

        async with (
            PodLogStreamer(self.kubectl, namespace, prefix="HELM") as streamer,
            background_status(self.kubectl, namespace, label="HELM", interval=15),
        ):
            if stream_logs:
                streamer.watch()

            try:
                status = await self.wait_for_job_completion(name, namespace, timeout)
                result.status = status
                result.success = status.is_completed

                if status.is_failed:
                    result.error_message = status.error

            except TimeoutError as e:
                result.success = False
                result.error_message = str(e)
                result.status = await self.get_job_status(name, namespace)

        if result.status and result.status.jobset_name:
            with contextlib.suppress(Exception):
                result.jobset_status = await self.kubectl.get_jobset(
                    result.status.jobset_name, namespace
                )

            result.pods = await self.kubectl.get_pods(namespace)

        result.duration_seconds = asyncio.get_event_loop().time() - start_time

        return result

    async def cleanup_job(self, result: OperatorJobResult) -> None:
        """Clean up a job and its resources.

        Removes the AIPerfJob finalizer, the JobSet, and force-deletes any
        zombie benchmark pods so the next session starts on a clean cluster.
        """
        ns = result.namespace
        name = result.job_name
        try:
            await self.kubectl.run(
                "patch",
                "aiperfjob",
                name,
                "-n",
                ns,
                "--type=json",
                '-p=[{"op":"remove","path":"/metadata/finalizers"}]',
                check=False,
            )
            await self.kubectl.run(
                "delete",
                "aiperfjob",
                name,
                "-n",
                ns,
                "--ignore-not-found",
                check=False,
            )
            await self.kubectl.run(
                "delete",
                "jobsets",
                "--all",
                "-n",
                ns,
                "--ignore-not-found",
                check=False,
            )
            pods = await self.kubectl.run(
                "get",
                "pods",
                "-n",
                ns,
                "-l",
                "jobset.sigs.k8s.io/jobset-name",
                "-o",
                "jsonpath={.items[*].metadata.name}",
                check=False,
            )
            if pods.returncode == 0 and pods.stdout.strip():
                for pod in pods.stdout.strip().split():
                    await self.kubectl.run(
                        "patch",
                        "pod",
                        pod,
                        "-n",
                        ns,
                        "--type=json",
                        '-p=[{"op":"remove","path":"/metadata/finalizers"}]',
                        check=False,
                    )
                await self.kubectl.run(
                    "delete",
                    "pods",
                    "-l",
                    "jobset.sigs.k8s.io/jobset-name",
                    "-n",
                    ns,
                    "--force",
                    "--grace-period=0",
                    "--ignore-not-found",
                    check=False,
                )
        except Exception as e:
            logger.warning(f"Failed to delete job {name}: {e}")

    async def cleanup_all(self) -> None:
        """Clean up all deployed jobs in parallel."""
        if self._deployed_jobs:
            await asyncio.gather(*[self.cleanup_job(r) for r in self._deployed_jobs])
        self._deployed_jobs.clear()
