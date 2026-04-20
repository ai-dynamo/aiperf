# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kubectl wrapper utilities for Kubernetes E2E tests."""

from __future__ import annotations

import asyncio
import json
import subprocess
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from aiperf.common.aiperf_logger import AIPerfLogger

logger = AIPerfLogger(__name__)


@asynccontextmanager
async def background_status(
    kubectl: KubectlClient,
    namespace: str,
    label: str = "STATUS",
    interval: int = 15,
):
    """Background task that periodically logs pod status and K8s events.

    Use as an async context manager around long-running waits to provide
    continuous visibility into cluster state without blocking the main flow.

    Args:
        kubectl: Kubectl client.
        namespace: Namespace to monitor.
        label: Log prefix label.
        interval: Seconds between status reports.
    """

    seen_events: set[str] = set()

    async def _report() -> None:
        try:
            while True:
                await asyncio.sleep(interval)
                try:
                    pods = await kubectl.get_pods(namespace)
                    if pods:
                        lines = [
                            f"  {p.name:<55} {p.phase:<12} "
                            f"ready={p.ready:<5} restarts={p.restarts}"
                            for p in pods
                        ]
                        logger.info(
                            f"[{label}] Pods in {namespace}:\n" + "\n".join(lines)
                        )
                    else:
                        logger.info(f"[{label}] No pods in {namespace} yet")
                except Exception:
                    pass

                try:
                    events = await kubectl.get_events(namespace, limit=8)
                    if events.strip():
                        raw_lines = events.rstrip().split("\n")
                        header = raw_lines[0] if raw_lines else ""
                        new_lines = []
                        for line in raw_lines[1:]:
                            stripped = line.strip()
                            if not stripped:
                                continue
                            # Strip the relative timestamp column (e.g. "15s", "2m30s")
                            # to avoid treating the same event as new when age changes.
                            parts = stripped.split(None, 1)
                            fingerprint = parts[1] if len(parts) > 1 else stripped
                            if fingerprint not in seen_events:
                                seen_events.add(fingerprint)
                                new_lines.append(line)
                        if new_lines:
                            logger.info(
                                f"[{label}] New events in {namespace}:\n"
                                + header
                                + "\n"
                                + "\n".join(new_lines)
                            )
                except Exception:
                    pass
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(_report(), name=f"status-reporter-{namespace}")
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@dataclass
class PodStatus:
    """Status of a Kubernetes pod."""

    name: str
    """Pod name."""

    namespace: str
    """Namespace the pod belongs to."""

    phase: str
    """Current pod phase (Running, Pending, Succeeded, Failed)."""

    ready: str
    """Ready container ratio string (e.g. '2/3')."""

    restarts: int
    """Total container restart count."""

    containers: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per-container status keyed by container name."""

    @property
    def is_ready(self) -> bool:
        """Check if pod is ready."""
        return (
            self.phase == "Running"
            and "/" in self.ready
            and self.ready.split("/")[0] == self.ready.split("/")[1]
        )

    @property
    def is_completed(self) -> bool:
        """Check if pod completed successfully."""
        return self.phase == "Succeeded" or self.phase == "Completed"

    @property
    def is_failed(self) -> bool:
        """Check if pod failed."""
        return self.phase == "Failed"


@dataclass
class JobSetStatus:
    """Status of a JobSet."""

    name: str
    """JobSet name."""

    namespace: str
    """Namespace the JobSet belongs to."""

    terminal_state: str | None
    """Terminal state (Completed, Failed) or None if still running."""

    completed: bool
    """Whether the JobSet has completed."""

    restarts: int
    """Number of JobSet restarts."""

    @property
    def is_completed(self) -> bool:
        """Check if JobSet completed."""
        return self.terminal_state == "Completed" or self.completed

    @property
    def is_failed(self) -> bool:
        """Check if JobSet failed."""
        return self.terminal_state == "Failed"


class KubectlClient:
    """Client for kubectl operations."""

    def __init__(
        self, context: str | None = None, kubeconfig: str | None = None
    ) -> None:
        """Initialize kubectl client.

        Args:
            context: Kubernetes context to use.
            kubeconfig: Path to kubeconfig file.
        """
        self.context = context
        self.kubeconfig = kubeconfig

    def _build_cmd(self, *args: str, namespace: str | None = None) -> list[str]:
        """Build kubectl command with common options."""
        cmd = ["kubectl"]

        if self.context:
            cmd.extend(["--context", self.context])

        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])

        if namespace:
            cmd.extend(["-n", namespace])

        cmd.extend(args)
        return cmd

    async def run(
        self,
        *args: str,
        namespace: str | None = None,
        check: bool = True,
        capture_output: bool = True,
        timeout: int = 120,
    ) -> subprocess.CompletedProcess:
        """Run a kubectl command.

        Args:
            *args: Command arguments.
            namespace: Kubernetes namespace.
            check: Raise exception on failure.
            capture_output: Capture stdout/stderr.
            timeout: Timeout in seconds (prevents hangs on finalizers).

        Returns:
            Completed process result.
        """
        cmd = self._build_cmd(*args, namespace=namespace)
        logger.debug(lambda cmd=cmd: f"Running: {' '.join(cmd)}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE if capture_output else None,
            stderr=asyncio.subprocess.PIPE if capture_output else None,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.warning(f"kubectl timed out after {timeout}s: {' '.join(cmd)}")
            stdout = b""
            stderr = f"kubectl timed out after {timeout}s".encode()

        result = subprocess.CompletedProcess(
            cmd,
            proc.returncode,
            stdout.decode() if stdout else "",
            stderr.decode() if stderr else "",
        )

        if check and result.returncode != 0:
            raise RuntimeError(f"kubectl command failed: {result.stderr}")

        return result

    async def apply(self, manifest: str, namespace: str | None = None) -> str:
        """Apply a YAML manifest.

        Args:
            manifest: YAML manifest content.
            namespace: Target namespace.

        Returns:
            Command output.
        """
        cmd = self._build_cmd("apply", "-f", "-", namespace=namespace)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=manifest.encode())

        if proc.returncode != 0:
            raise RuntimeError(f"Failed to apply manifest: {stderr.decode()}")

        return stdout.decode()

    async def apply_server_side(self, url: str) -> str:
        """Apply a manifest from URL with server-side apply.

        Args:
            url: URL to manifest.

        Returns:
            Command output.
        """
        result = await self.run("apply", "--server-side", "-f", url, check=True)
        return result.stdout

    async def delete(
        self,
        resource: str,
        name: str | None = None,
        namespace: str | None = None,
        ignore_not_found: bool = True,
    ) -> None:
        """Delete a resource.

        Args:
            resource: Resource type (e.g., "deployment", "namespace").
            name: Resource name (optional).
            namespace: Namespace.
            ignore_not_found: Don't error if resource doesn't exist.
        """
        args = ["delete", resource]
        if name:
            args.append(name)
        if ignore_not_found:
            args.append("--ignore-not-found")

        await self.run(*args, namespace=namespace, check=True)

    async def get_json(
        self,
        resource: str,
        name: str | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Get resource as JSON.

        Args:
            resource: Resource type.
            name: Resource name (optional).
            namespace: Namespace.

        Returns:
            Resource data as dict or list.
        """
        args = ["get", resource, "-o", "json"]
        if name:
            args.append(name)

        result = await self.run(*args, namespace=namespace, check=True)
        return json.loads(result.stdout)

    async def get_pods(self, namespace: str) -> list[PodStatus]:
        """Get all pods in a namespace.

        Args:
            namespace: Kubernetes namespace.

        Returns:
            List of pod statuses.
        """
        data = await self.get_json("pods", namespace=namespace)
        pods = []

        for item in data.get("items", []):
            metadata = item.get("metadata", {})
            status = item.get("status", {})

            containers = {}
            for cs in status.get("containerStatuses", []):
                containers[cs["name"]] = {
                    "ready": cs.get("ready", False),
                    "state": cs.get("state", {}),
                    "restartCount": cs.get("restartCount", 0),
                }

            ready_count = sum(1 for c in containers.values() if c["ready"])
            total_count = len(containers)
            ready_str = f"{ready_count}/{total_count}"

            total_restarts = sum(c["restartCount"] for c in containers.values())

            pods.append(
                PodStatus(
                    name=metadata.get("name", ""),
                    namespace=metadata.get("namespace", ""),
                    phase=status.get("phase", "Unknown"),
                    ready=ready_str,
                    restarts=total_restarts,
                    containers=containers,
                )
            )

        return pods

    async def get_jobset(self, name: str, namespace: str) -> JobSetStatus:
        """Get JobSet status.

        Args:
            name: JobSet name.
            namespace: Kubernetes namespace.

        Returns:
            JobSet status.
        """
        data = await self.get_json("jobset", name, namespace=namespace)
        status = data.get("status", {})

        return JobSetStatus(
            name=data.get("metadata", {}).get("name", ""),
            namespace=namespace,
            terminal_state=status.get("terminalState"),
            completed=status.get("completed", False),
            restarts=status.get("restarts", 0),
        )

    async def get_jobsets(self, namespace: str) -> list[JobSetStatus]:
        """Get all JobSets in a namespace.

        Args:
            namespace: Kubernetes namespace.

        Returns:
            List of JobSet statuses.
        """
        data = await self.get_json("jobset", namespace=namespace)
        jobsets = []

        for item in data.get("items", []):
            status = item.get("status", {})
            jobsets.append(
                JobSetStatus(
                    name=item.get("metadata", {}).get("name", ""),
                    namespace=namespace,
                    terminal_state=status.get("terminalState"),
                    completed=status.get("completed", False),
                    restarts=status.get("restarts", 0),
                )
            )

        return jobsets

    async def get_logs(
        self,
        pod: str,
        container: str | None = None,
        namespace: str | None = None,
        tail: int | None = None,
    ) -> str:
        """Get pod logs.

        Args:
            pod: Pod name.
            container: Container name (optional).
            namespace: Namespace.
            tail: Number of lines to tail.

        Returns:
            Log content.
        """
        args = ["logs", pod]
        if container:
            args.extend(["-c", container])
        if tail:
            args.extend(["--tail", str(tail)])

        result = await self.run(*args, namespace=namespace, check=False)
        return result.stdout

    async def wait_for_rollout(
        self,
        resource: str,
        name: str,
        namespace: str | None = None,
        timeout: int = 120,
    ) -> bool:
        """Wait for a deployment rollout to complete.

        Args:
            resource: Resource type (e.g., "deployment").
            name: Resource name.
            namespace: Namespace.
            timeout: Timeout in seconds.

        Returns:
            True if rollout completed successfully.
        """
        result = await self.run(
            "rollout",
            "status",
            f"{resource}/{name}",
            f"--timeout={timeout}s",
            namespace=namespace,
            check=False,
        )
        return result.returncode == 0

    async def wait_for_condition(
        self,
        resource: str,
        name: str,
        condition: str,
        namespace: str | None = None,
        timeout: int = 60,
    ) -> bool:
        """Wait for a resource condition.

        Args:
            resource: Resource type.
            name: Resource name.
            condition: Condition to wait for.
            namespace: Namespace.
            timeout: Timeout in seconds.

        Returns:
            True if condition met.
        """
        result = await self.run(
            "wait",
            f"--for=condition={condition}",
            f"--timeout={timeout}s",
            f"{resource}/{name}",
            namespace=namespace,
            check=False,
        )
        return result.returncode == 0

    async def wait_for_jobset_completion(
        self,
        name: str,
        namespace: str,
        timeout: int = 300,
        poll_interval: int = 5,
    ) -> JobSetStatus:
        """Wait for a JobSet to complete.

        Args:
            name: JobSet name.
            namespace: Namespace.
            timeout: Timeout in seconds.
            poll_interval: Polling interval in seconds.

        Returns:
            Final JobSet status.

        Raises:
            TimeoutError: If timeout exceeded.
            RuntimeError: If JobSet failed.
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time

            if elapsed > timeout:
                raise TimeoutError(f"Timeout waiting for JobSet {name} completion")

            status = await self.get_jobset(name, namespace)

            if status.is_completed:
                return status

            if status.is_failed:
                raise RuntimeError(f"JobSet {name} failed")

            logger.info(
                f"JobSet {name}: state={status.terminal_state}, "
                f"restarts={status.restarts}, elapsed={elapsed:.0f}s"
            )
            await asyncio.sleep(poll_interval)

    async def namespace_exists(self, namespace: str) -> bool:
        """Check if a namespace exists.

        Args:
            namespace: Namespace name.

        Returns:
            True if namespace exists.
        """
        result = await self.run("get", "namespace", namespace, check=False)
        return result.returncode == 0

    async def create_namespace(self, namespace: str) -> None:
        """Create a namespace (idempotent - ignores AlreadyExists).

        Args:
            namespace: Namespace name.
        """
        result = await self.run("create", "namespace", namespace, check=False)
        if result.returncode != 0 and "AlreadyExists" not in result.stderr:
            raise RuntimeError(f"Failed to create namespace: {result.stderr}")

    async def delete_namespace(self, namespace: str, wait: bool = True) -> None:
        """Delete a namespace.

        Args:
            namespace: Namespace name.
            wait: Wait for deletion to complete.
        """
        logger.info(f"Deleting namespace {namespace} (wait={wait})")
        await self.delete("namespace", namespace, ignore_not_found=True)

        if wait:
            start = asyncio.get_event_loop().time()
            delay = 0.1
            finalizers_stripped = False
            while True:
                if not await self.namespace_exists(namespace):
                    elapsed = asyncio.get_event_loop().time() - start
                    logger.info(f"Namespace {namespace} deleted ({elapsed:.0f}s)")
                    return
                elapsed = asyncio.get_event_loop().time() - start
                if elapsed >= 30 and not finalizers_stripped:
                    finalizers_stripped = True
                    logger.warning(
                        f"Namespace {namespace} stuck after {elapsed:.0f}s, "
                        "stripping finalizers from remaining resources"
                    )
                    await self._force_remove_finalizers(namespace)
                if elapsed >= 90:
                    logger.warning(f"Namespace {namespace} still exists after 90s")
                    break
                if elapsed >= 10 and int(elapsed) % 10 < delay + 0.5:
                    logger.info(
                        f"Waiting for namespace {namespace} deletion ({elapsed:.0f}s)..."
                    )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 2.0)

    async def force_cleanup_namespace_pods(self, namespace: str) -> None:
        """Force-delete all pods in a namespace with finalizers stripped.

        Pods created by Jobs carry a `batch.kubernetes.io/job-tracking`
        finalizer held by the Job controller until the pod terminates
        gracefully. In tests we don't care about graceful shutdown, so we
        strip all pod finalizers and force-delete to unblock namespace
        termination immediately.
        """
        result = await self.run(
            "get",
            "pods",
            "-n",
            namespace,
            "-o",
            "jsonpath={.items[*].metadata.name}",
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return
        pod_names = result.stdout.strip().split()
        for name in pod_names:
            await self.run(
                "patch",
                "pod",
                name,
                "-n",
                namespace,
                "--type=json",
                '-p=[{"op":"remove","path":"/metadata/finalizers"}]',
                check=False,
            )
        await self.run(
            "delete",
            "pods",
            "--all",
            "-n",
            namespace,
            "--force",
            "--grace-period=0",
            "--ignore-not-found",
            check=False,
        )

    async def _force_remove_finalizers(self, namespace: str) -> None:
        """Remove finalizers from all resources in a namespace to unblock deletion."""
        for resource_type in (
            "aiperfjobs.aiperf.nvidia.com",
            "jobsets.jobset.x-k8s.io",
        ):
            result = await self.run(
                "get",
                resource_type,
                "-n",
                namespace,
                "-o",
                "jsonpath={.items[*].metadata.name}",
                check=False,
            )
            if result.returncode != 0 or not result.stdout.strip():
                continue
            for name in result.stdout.strip().split():
                patch_result = await self.run(
                    "patch",
                    resource_type.split(".")[0],
                    name,
                    "-n",
                    namespace,
                    "--type=json",
                    '-p=[{"op":"remove","path":"/metadata/finalizers"}]',
                    check=False,
                )
                if patch_result.returncode == 0:
                    logger.info(f"Removed finalizers from {resource_type}/{name}")

    async def get_configmap(
        self,
        name: str,
        namespace: str,
    ) -> dict[str, Any] | None:
        """Get ConfigMap data.

        Args:
            name: ConfigMap name.
            namespace: Namespace.

        Returns:
            ConfigMap data dict or None if not found.
        """
        result = await self.run(
            "get",
            "configmap",
            name,
            "-o",
            "json",
            namespace=namespace,
            check=False,
        )

        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        return data.get("data", {})

    async def list_configmaps(self, namespace: str) -> list[str]:
        """List ConfigMap names in a namespace.

        Args:
            namespace: Namespace.

        Returns:
            List of ConfigMap names.
        """
        data = await self.get_json("configmap", namespace=namespace)
        return [
            item.get("metadata", {}).get("name", "") for item in data.get("items", [])
        ]

    async def get_events(
        self,
        namespace: str,
        sort_by: str = ".lastTimestamp",
        limit: int = 20,
    ) -> str:
        """Get namespace events.

        Args:
            namespace: Namespace.
            sort_by: Field to sort by.
            limit: Max events to return.

        Returns:
            Event output.
        """
        result = await self.run(
            "get",
            "events",
            f"--sort-by={sort_by}",
            namespace=namespace,
            check=False,
        )

        lines = result.stdout.strip().split("\n")
        if len(lines) > limit + 1:  # +1 for header
            lines = [lines[0]] + lines[-(limit):]

        return "\n".join(lines)

    @asynccontextmanager
    async def port_forward(
        self,
        pod: str,
        remote_port: int,
        local_port: int = 0,
        namespace: str | None = None,
    ):
        """Async context manager for kubectl port-forward to a pod.

        Args:
            pod: Pod name.
            remote_port: Port on the pod to forward.
            local_port: Local port to bind (0 = auto-select).
            namespace: Namespace.

        Yields:
            The local port that was bound.
        """
        import socket

        if local_port == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                local_port = s.getsockname()[1]

        cmd = self._build_cmd(
            "port-forward",
            pod,
            f"{local_port}:{remote_port}",
            namespace=namespace,
        )

        logger.info(f"Starting port-forward: {' '.join(cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        await asyncio.sleep(2)
        if proc.returncode is not None:
            stderr = await proc.stderr.read() if proc.stderr else b""
            raise RuntimeError(f"Port-forward failed to start: {stderr.decode()}")

        try:
            yield local_port
        finally:
            if proc.returncode is None:
                proc.terminate()
                await proc.wait()
            else:
                logger.debug(f"Port-forward already exited (rc={proc.returncode})")

    async def wait_for_benchmark_api(
        self,
        pod: str,
        namespace: str,
        api_port: int = 9090,
        timeout: int = 300,
        poll_interval: int = 5,
    ) -> dict[str, Any]:
        """Wait for the benchmark API to report completion, fetch results, and shut down.

        Port-forwards to the controller pod's API, polls /api/results until the
        benchmark is complete, then calls POST /api/shutdown to let the pod exit.

        While polling, logs live progress (completed requests, throughput, errors)
        so operators can monitor the benchmark in real time.

        Args:
            pod: Controller pod name.
            namespace: Namespace.
            api_port: API port on the controller pod.
            timeout: Timeout in seconds.
            poll_interval: Polling interval in seconds.

        Returns:
            Results dict from /api/results.

        Raises:
            TimeoutError: If timeout exceeded waiting for benchmark completion.
        """
        from aiperf.transports.aiohttp_client import create_tcp_connector

        start_time = asyncio.get_event_loop().time()
        results: dict[str, Any] = {}
        consecutive_failures = 0
        max_consecutive_failures = 5

        pf_ctx = self.port_forward(pod, api_port, namespace=namespace)
        local_port = await pf_ctx.__aenter__()
        base_url = f"http://127.0.0.1:{local_port}"

        try:
            connector = create_tcp_connector()
            async with aiohttp.ClientSession(connector=connector) as session:
                while True:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > timeout:
                        raise TimeoutError(
                            f"Timeout waiting for benchmark completion via API "
                            f"(pod={pod}, elapsed={elapsed:.0f}s)"
                        )

                    api_status = None
                    try:
                        async with session.get(
                            f"{base_url}/api/results",
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                api_status = data.get("status", "")
                                consecutive_failures = 0
                                if api_status in ("complete", "cancelled"):
                                    logger.info(
                                        f"Benchmark API reports status={api_status}"
                                    )
                                    results = data
                                    break
                                self._log_benchmark_progress(data, elapsed)
                    except aiohttp.ClientConnectorError:
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            logger.warning(
                                f"Port-forward dead after {consecutive_failures} "
                                f"failures, restarting..."
                            )
                            await pf_ctx.__aexit__(None, None, None)
                            pf_ctx = self.port_forward(
                                pod, api_port, namespace=namespace
                            )
                            local_port = await pf_ctx.__aenter__()
                            base_url = f"http://127.0.0.1:{local_port}"
                            consecutive_failures = 0
                        else:
                            logger.debug(lambda: "API not yet reachable, retrying...")
                    except Exception as e:
                        logger.debug(lambda e=e: f"API poll error: {e}")

                    # Fetch live progress from /api/progress for richer metrics
                    if api_status == "running":
                        await self._log_live_progress(session, base_url, elapsed)
                    elif api_status != "connecting":
                        logger.info(
                            f"Benchmark in progress: elapsed={int(elapsed)}s, "
                            f"api={api_status or 'unknown'}"
                        )
                    await asyncio.sleep(poll_interval)

                try:
                    async with session.post(
                        f"{base_url}/api/shutdown",
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status == 200:
                            logger.info("API shutdown requested successfully")
                        else:
                            body = await resp.text()
                            logger.warning(
                                f"API shutdown returned {resp.status}: {body}"
                            )
                except Exception as e:
                    logger.warning(f"Failed to call /api/shutdown: {e}")
        finally:
            await pf_ctx.__aexit__(None, None, None)

        return results

    @staticmethod
    def _log_benchmark_progress(data: dict[str, Any], elapsed: float) -> None:
        """Extract and log live progress from an /api/results response.

        Args:
            data: Response JSON from /api/results.
            elapsed: Seconds since benchmark started.
        """
        status = data.get("status", "unknown")
        inner = data.get("results", {})
        if isinstance(inner, dict):
            inner = inner.get("results", inner)

        if not isinstance(inner, dict):
            logger.info(f"[PROGRESS] elapsed={int(elapsed)}s status={status}")
            return

        completed = inner.get("completed")
        total = inner.get("total_expected")
        errors = inner.get("error_count", 0)

        # Extract throughput from records if available
        throughput = None
        records = inner.get("records", [])
        if isinstance(records, list):
            for rec in records:
                tag = rec.get("header", rec.get("tag", ""))
                if "request throughput" in tag.lower():
                    throughput = rec.get("avg")
                    break

        parts = [f"elapsed={int(elapsed)}s", f"status={status}"]
        if completed is not None and total is not None:
            parts.append(f"requests={completed}/{total}")
        elif completed is not None:
            parts.append(f"completed={completed}")
        if throughput is not None:
            parts.append(f"throughput={throughput:.1f} req/s")
        if errors:
            parts.append(f"errors={errors}")

        logger.info(f"[PROGRESS] {', '.join(parts)}")

    @staticmethod
    async def _log_live_progress(
        session: aiohttp.ClientSession,
        base_url: str,
        elapsed: float,
    ) -> None:
        """Fetch and log live progress from /api/progress endpoint.

        Args:
            session: Active aiohttp session.
            base_url: Base URL for the benchmark API.
            elapsed: Seconds since benchmark started.
        """
        try:
            async with session.get(
                f"{base_url}/api/progress",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()

            phases = data.get("phases", {})
            if not phases:
                return

            parts: list[str] = [f"elapsed={int(elapsed)}s"]
            for phase_name, stats in phases.items():
                completed = stats.get("requests_completed", 0)
                total = stats.get("total_expected_requests")
                pct = stats.get("requests_progress_percent")
                avg_lat = stats.get("avg_request_latency_ms")

                phase_parts = [f"phase={phase_name}"]
                if total:
                    phase_parts.append(f"requests={completed}/{total}")
                elif completed:
                    phase_parts.append(f"completed={completed}")
                if pct is not None:
                    phase_parts.append(f"progress={pct:.1f}%")
                if avg_lat is not None:
                    phase_parts.append(f"avg_latency={avg_lat:.0f}ms")

                parts.extend(phase_parts)

            logger.info(f"[LIVE] {', '.join(parts)}")
        except Exception:
            pass
