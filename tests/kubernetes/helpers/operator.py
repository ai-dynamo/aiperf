# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator deployment and AIPerfJob CR management for Kubernetes E2E tests."""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.helpers.kubectl import (
    JobSetStatus,
    KubectlClient,
    PodStatus,
    background_status,
)
from tests.kubernetes.helpers.log_streamer import PodLogStreamer
from tests.kubernetes.helpers.watchdog import BenchmarkWatchdog, make_watchdog_source

logger = AIPerfLogger(__name__)


@dataclass
class AIPerfJobConfig:
    """Configuration for an AIPerfJob CR."""

    # Required fields
    endpoint_url: str = "http://aiperf-mock-server.default.svc.cluster.local:8000/v1"
    model_name: str = "mock-model"

    # Optional endpoint config
    endpoint_type: str = "chat"

    # Load generation
    concurrency: int = 5
    request_count: int | None = 50
    warmup_request_count: int = 5
    benchmark_duration: float | None = None  # Run for this many seconds

    # Tokenizer
    tokenizer_name: str = "gpt2"

    # Container
    image: str = "aiperf:local"
    image_pull_policy: str = "Never"

    # Operator-specific
    connections_per_worker: int | None = None

    # Kueue / gang-scheduling
    queue_name: str | None = None
    priority_class: str | None = None

    def to_flat_spec(self) -> dict[str, Any]:
        """Generate flat CRD spec (config v3 format, no userConfig wrapper)."""
        load: dict[str, Any] = {
            "profiling": {
                "type": "concurrency",
                "concurrency": self.concurrency,
            },
        }
        if self.request_count is not None:
            load["profiling"]["requests"] = self.request_count
        if self.benchmark_duration is not None:
            load["profiling"]["duration"] = self.benchmark_duration
        if self.warmup_request_count:
            load["warmup"] = {
                "type": "concurrency",
                "concurrency": self.concurrency,
                "requests": self.warmup_request_count,
                "exclude_from_results": True,
            }

        return {
            "models": {"items": [{"name": self.model_name}]},
            "endpoint": {"urls": [self.endpoint_url]},
            "datasets": {
                "main": {
                    "type": "synthetic",
                    "entries": max(self.request_count or 100, 10),
                    "prompts": {"isl": {"mean": 550}},
                },
            },
            "phases": load,
            "tokenizer": {"name": self.tokenizer_name},
            "runtime": {"ui": "none"},
        }

    def to_cr_manifest(self, name: str, namespace: str) -> str:
        """Generate AIPerfJob CR manifest (flat spec, no userConfig wrapper).

        Args:
            name: CR name.
            namespace: Namespace for the CR.

        Returns:
            YAML manifest string.
        """
        spec: dict[str, Any] = {
            "image": self.image,
            "imagePullPolicy": self.image_pull_policy,
            "benchmark": self.to_flat_spec(),
        }

        if self.connections_per_worker is not None:
            spec["connectionsPerWorker"] = self.connections_per_worker

        if self.queue_name is not None or self.priority_class is not None:
            scheduling: dict[str, str] = {}
            if self.queue_name is not None:
                scheduling["queueName"] = self.queue_name
            if self.priority_class is not None:
                scheduling["priorityClass"] = self.priority_class
            spec["scheduling"] = scheduling

        cr = {
            "apiVersion": "aiperf.nvidia.com/v1alpha1",
            "kind": "AIPerfJob",
            "metadata": {
                "name": name,
                "namespace": namespace,
            },
            "spec": spec,
        }

        return yaml.dump(cr, default_flow_style=False)


@dataclass
class AIPerfJobStatus:
    """Status of an AIPerfJob CR."""

    name: str
    namespace: str
    phase: str | None = None
    current_phase: str | None = None
    job_id: str | None = None
    jobset_name: str | None = None
    error: str | None = None
    workers: dict[str, int] | None = None
    conditions: list[dict[str, Any]] = field(default_factory=list)
    phases: dict[str, dict[str, Any]] = field(default_factory=dict)
    results: dict[str, Any] | None = None
    results_path: str | None = None
    live_metrics: dict[str, Any] | None = None
    raw_status: dict[str, Any] = field(default_factory=dict)

    @property
    def is_pending(self) -> bool:
        """Check if job is pending."""
        return self.phase == "Pending"

    @property
    def is_running(self) -> bool:
        """Check if job is running."""
        return self.phase == "Running"

    @property
    def is_completed(self) -> bool:
        """Check if job completed successfully."""
        return self.phase == "Completed"

    @property
    def is_failed(self) -> bool:
        """Check if job failed."""
        return self.phase == "Failed"

    @property
    def is_cancelled(self) -> bool:
        """Check if job was cancelled."""
        return self.phase == "Cancelled"

    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.phase in ("Completed", "Failed", "Cancelled")

    @property
    def workers_ready(self) -> int:
        """Get number of ready workers."""
        if self.workers:
            return self.workers.get("ready", 0)
        return 0

    @property
    def workers_total(self) -> int:
        """Get total number of workers."""
        if self.workers:
            return self.workers.get("total", 0)
        return 0

    def get_condition(self, condition_type: str) -> dict[str, Any] | None:
        """Get a specific condition by type."""
        for cond in self.conditions:
            if cond.get("type") == condition_type:
                return cond
        return None

    def is_condition_true(self, condition_type: str) -> bool:
        """Check if a condition is True."""
        cond = self.get_condition(condition_type)
        return cond is not None and cond.get("status") == "True"

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> AIPerfJobStatus:
        """Create from kubectl JSON output."""
        metadata = data.get("metadata", {})
        status = data.get("status", {})

        return cls(
            name=metadata.get("name", ""),
            namespace=metadata.get("namespace", ""),
            phase=status.get("phase"),
            current_phase=status.get("currentPhase"),
            job_id=status.get("jobId"),
            jobset_name=status.get("jobSetName"),
            error=status.get("error"),
            workers=status.get("workers"),
            conditions=status.get("conditions", []),
            phases=status.get("phases", {}),
            results=status.get("results"),
            results_path=status.get("resultsPath"),
            live_metrics=status.get("liveMetrics"),
            raw_status=status,
        )


@dataclass
class OperatorJobResult:
    """Result of an operator-managed benchmark job."""

    namespace: str
    job_name: str
    config: AIPerfJobConfig
    status: AIPerfJobStatus | None = None
    jobset_status: JobSetStatus | None = None
    pods: list[PodStatus] = field(default_factory=list)
    duration_seconds: float = 0.0
    success: bool = False
    error_message: str | None = None
    events: list[str] = field(default_factory=list)

    @property
    def controller_pod(self) -> PodStatus | None:
        """Get the controller pod."""
        for pod in self.pods:
            if "controller" in pod.name:
                return pod
        return None


class OperatorDeployer:
    """Manages operator deployment and AIPerfJob lifecycle."""

    OPERATOR_NAMESPACE = "aiperf-system"
    CRD_NAME = "aiperfjobs.aiperf.nvidia.com"

    def __init__(
        self,
        kubectl: KubectlClient,
        project_root: Path,
        operator_image: str = "aiperf:local",
    ) -> None:
        """Initialize operator deployer.

        Args:
            kubectl: Kubectl client.
            project_root: Path to project root.
            operator_image: Operator image name.
        """
        self.kubectl = kubectl
        self.project_root = project_root
        self.operator_image = operator_image
        self._deployed_jobs: list[OperatorJobResult] = []

    async def install_crd(self) -> None:
        """Install the AIPerfJob CRD by rendering it from the Helm chart."""
        chart_path = self.project_root / "deploy" / "helm" / "aiperf-operator"
        logger.info(f"Installing CRD from chart {chart_path}")

        result = subprocess.run(
            [
                "helm",
                "template",
                "aiperf-operator",
                str(chart_path),
                "--show-only",
                "templates/crd.yaml",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        await self.kubectl.apply(result.stdout)

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

    async def deploy_operator(self) -> None:
        """Deploy the operator to the cluster.

        Renders the Helm chart via `helm template` and applies it with kubectl.
        Handles idempotency: if an existing deployment has incompatible labels
        (e.g. from a helm install), it is deleted first. Any existing helm
        release is also uninstalled to avoid conflicts.
        """
        chart_path = self.project_root / "deploy" / "helm" / "aiperf-operator"
        logger.info(f"Deploying operator from chart {chart_path}")

        # Clean up any existing helm release that could conflict
        await self._cleanup_existing_operator()

        # Ensure the operator namespace exists
        await self.kubectl.run(
            "create",
            "namespace",
            self.OPERATOR_NAMESPACE,
            check=False,
        )

        # Strip Helm ownership annotations from the namespace to avoid
        # conflicts when re-deploying after a helm uninstall test.
        await self.kubectl.run(
            "annotate",
            "namespace",
            self.OPERATOR_NAMESPACE,
            "meta.helm.sh/release-name-",
            "meta.helm.sh/release-namespace-",
            check=False,
        )
        await self.kubectl.run(
            "label",
            "namespace",
            self.OPERATOR_NAMESPACE,
            "app.kubernetes.io/managed-by-",
            check=False,
        )

        # Render the Helm chart to a manifest
        result = subprocess.run(
            [
                "helm",
                "template",
                "aiperf-operator",
                str(chart_path),
                "-n",
                self.OPERATOR_NAMESPACE,
                "--set",
                f"image.repository={self.operator_image.rsplit(':', 1)[0]}",
                "--set",
                f"image.tag={self.operator_image.rsplit(':', 1)[-1]}",
                "--set",
                "image.pullPolicy=Never",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        manifest = result.stdout

        await self.kubectl.apply(manifest)

        # Use defaults that match production; Kind node has enough memory
        await self.kubectl.run(
            "set",
            "env",
            "deployment/aiperf-operator",
            "AIPERF_K8S_WORKER_POD_CPU=3350m",
            "AIPERF_K8S_WORKER_POD_MEMORY=6144Mi",
            "AIPERF_K8S_SYSTEM_CONTROLLER_CPU=250m",
            "AIPERF_K8S_SYSTEM_CONTROLLER_MEMORY=256Mi",
            "AIPERF_K8S_WORKER_MANAGER_CPU=250m",
            "AIPERF_K8S_WORKER_MANAGER_MEMORY=256Mi",
            "AIPERF_K8S_TIMING_MANAGER_CPU=1000m",
            "AIPERF_K8S_TIMING_MANAGER_MEMORY=512Mi",
            "AIPERF_K8S_DATASET_MANAGER_CPU=500m",
            "AIPERF_K8S_DATASET_MANAGER_MEMORY=512Mi",
            "AIPERF_K8S_RECORDS_MANAGER_CPU=500m",
            "AIPERF_K8S_RECORDS_MANAGER_MEMORY=512Mi",
            "AIPERF_K8S_API_CPU=250m",
            "AIPERF_K8S_API_MEMORY=256Mi",
            "AIPERF_K8S_GPU_TELEMETRY_MANAGER_CPU=125m",
            "AIPERF_K8S_GPU_TELEMETRY_MANAGER_MEMORY=128Mi",
            "AIPERF_K8S_SERVER_METRICS_MANAGER_CPU=125m",
            "AIPERF_K8S_SERVER_METRICS_MANAGER_MEMORY=128Mi",
            "AIPERF_K8S_RESULTS_SIDECAR_CPU=100m",
            "AIPERF_K8S_RESULTS_SIDECAR_MEMORY=128Mi",
            "-n",
            self.OPERATOR_NAMESPACE,
            check=True,
        )

        success = await self.kubectl.wait_for_rollout(
            "deployment",
            "aiperf-operator",
            namespace=self.OPERATOR_NAMESPACE,
            timeout=300,
        )

        if not success:
            logs = await self.kubectl.get_logs(
                "deployment/aiperf-operator",
                namespace=self.OPERATOR_NAMESPACE,
            )
            raise RuntimeError(f"Operator deployment failed. Logs:\n{logs}")

        logger.info("Operator deployed and ready")

    async def _cleanup_existing_operator(self) -> None:
        """Remove any existing operator deployment that could conflict.

        Handles both helm-installed and directly-applied operators. This ensures
        idempotent deployment regardless of prior cluster state.
        """
        ctx_args = []
        if self.kubectl.context:
            ctx_args = ["--kube-context", self.kubectl.context]

        # Uninstall any helm release first
        helm_list = await self._run_cmd(
            "helm",
            *ctx_args,
            "list",
            "-n",
            self.OPERATOR_NAMESPACE,
            "-q",
            "--filter",
            "aiperf",
        )
        if helm_list.returncode == 0 and helm_list.stdout.strip():
            for release in helm_list.stdout.strip().split("\n"):
                release = release.strip()
                if release:
                    logger.info(f"Uninstalling existing helm release: {release}")
                    await self._run_cmd(
                        "helm",
                        *ctx_args,
                        "uninstall",
                        release,
                        "-n",
                        self.OPERATOR_NAMESPACE,
                    )
                    await asyncio.sleep(5)

        # Delete all resources in the operator namespace for a clean slate
        await self.kubectl.run(
            "delete",
            "all,sa,roles,rolebindings",
            "--all",
            "-n",
            self.OPERATOR_NAMESPACE,
            check=False,
        )
        # Force-delete PVCs (may have finalizers that block normal deletion)
        await self.kubectl.run(
            "delete",
            "pvc",
            "--all",
            "-n",
            self.OPERATOR_NAMESPACE,
            "--force",
            "--grace-period=0",
            check=False,
        )
        # Clean up cluster-scoped resources too
        for resource in ["clusterrole", "clusterrolebinding"]:
            await self.kubectl.run(
                "delete",
                resource,
                "aiperf-operator",
                check=False,
            )

        # Delete existing deployment if present (handles label selector conflicts)
        existing = await self.kubectl.run(
            "get",
            "deployment",
            "aiperf-operator",
            "-n",
            self.OPERATOR_NAMESPACE,
            "-o",
            "name",
            check=False,
        )
        if existing.returncode == 0 and existing.stdout.strip():
            logger.info("Deleting existing operator deployment for clean redeploy")
            await self.kubectl.delete(
                "deployment",
                "aiperf-operator",
                namespace=self.OPERATOR_NAMESPACE,
                ignore_not_found=True,
            )
            await asyncio.sleep(3)

    @staticmethod
    async def _run_cmd(*args: str) -> subprocess.CompletedProcess:
        """Run an arbitrary command and return CompletedProcess."""
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return subprocess.CompletedProcess(
            args,
            proc.returncode,
            stdout.decode() if stdout else "",
            stderr.decode() if stderr else "",
        )

    async def uninstall_operator(self, timeout: int = 90) -> None:
        """Uninstall the operator with a timeout to prevent fixture hangs."""
        logger.info("Uninstalling operator")
        try:
            await asyncio.wait_for(self._do_uninstall(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Operator uninstall timed out after {timeout}s")

    async def _do_uninstall(self) -> None:
        """Inner uninstall logic."""
        await self._cleanup_existing_operator()

        await self.kubectl.delete(
            "deployment",
            "aiperf-operator",
            namespace=self.OPERATOR_NAMESPACE,
            ignore_not_found=True,
        )
        await self.kubectl.delete(
            "namespace", self.OPERATOR_NAMESPACE, ignore_not_found=True
        )

    async def create_job(
        self,
        config: AIPerfJobConfig,
        name: str | None = None,
        namespace: str = "default",
    ) -> OperatorJobResult:
        """Create an AIPerfJob CR.

        Args:
            config: Job configuration.
            name: Job name (auto-generated if not provided).
            namespace: Target namespace.

        Returns:
            OperatorJobResult with initial state.
        """
        import uuid

        if name is None:
            name = f"benchmark-{uuid.uuid4().hex[:8]}"

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
                f"current_phase={status.current_phase}, "
                f"workers={status.workers_ready}/{status.workers_total}, "
                f"elapsed={elapsed:.0f}s"
            )
            await asyncio.sleep(poll_interval)

    async def wait_for_phase(
        self,
        name: str,
        namespace: str,
        target_phase: str,
        timeout: int = 120,
        poll_interval: int = 2,
    ) -> AIPerfJobStatus:
        """Wait for an AIPerfJob to reach a specific phase.

        Args:
            name: Job name.
            namespace: Namespace.
            target_phase: Target phase to wait for.
            timeout: Timeout in seconds.
            poll_interval: Polling interval in seconds.

        Returns:
            Job status when phase is reached.

        Raises:
            TimeoutError: If timeout exceeded.
            RuntimeError: If job fails before reaching target phase.
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time

            if elapsed > timeout:
                status = await self.get_job_status(name, namespace)
                raise TimeoutError(
                    f"Timeout waiting for AIPerfJob {name} to reach {target_phase}. "
                    f"Current phase: {status.phase}"
                )

            status = await self.get_job_status(name, namespace)

            if status.phase == target_phase:
                return status

            if status.is_failed:
                raise RuntimeError(
                    f"AIPerfJob {name} failed before reaching {target_phase}: "
                    f"{status.error}"
                )

            await asyncio.sleep(poll_interval)

    async def cancel_job(self, name: str, namespace: str) -> None:
        """Cancel an AIPerfJob by setting spec.cancel=true.

        Args:
            name: Job name.
            namespace: Namespace.
        """
        logger.info(f"Cancelling AIPerfJob {namespace}/{name}")
        await self.kubectl.run(
            "patch",
            "aiperfjob",
            name,
            "--type=merge",
            "-p",
            '{"spec":{"cancel":true}}',
            namespace=namespace,
        )

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
        namespace: str = "default",
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

        async with (
            BenchmarkWatchdog(
                await make_watchdog_source(self.kubectl),
                namespace,
                timeout=timeout,
                poll_interval=5.0,
                pending_threshold=30.0,
            ) as _watchdog,
            PodLogStreamer(self.kubectl, namespace, prefix="OPERATOR") as streamer,
            background_status(self.kubectl, namespace, label="OPERATOR", interval=15),
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

        with contextlib.suppress(Exception):
            result.events = await self._get_job_events(name, namespace)

        result.duration_seconds = asyncio.get_event_loop().time() - start_time

        return result

    async def _get_job_events(self, name: str, namespace: str) -> list[str]:
        """Get events related to an AIPerfJob."""
        output = await self.kubectl.get_events(namespace)
        lines = []
        for line in output.splitlines():
            if name in line or "aiperfjob" in line.lower():
                lines.append(line)
        return lines

    async def get_operator_logs(self, tail: int = 100) -> str:
        """Get operator logs.

        Args:
            tail: Number of lines to tail.

        Returns:
            Log content.
        """
        return await self.kubectl.get_logs(
            "deployment/aiperf-operator",
            namespace=self.OPERATOR_NAMESPACE,
            tail=tail,
        )

    async def cleanup_job(self, result: OperatorJobResult) -> None:
        """Clean up a job and its resources.

        Args:
            result: Job result to clean up.
        """
        try:
            await self.delete_job(result.job_name, result.namespace)
        except Exception as e:
            logger.warning(f"Failed to delete job {result.job_name}: {e}")

    async def cleanup_all(self) -> None:
        """Clean up all deployed jobs in parallel."""
        if self._deployed_jobs:
            await asyncio.gather(*[self.cleanup_job(r) for r in self._deployed_jobs])
        self._deployed_jobs.clear()
