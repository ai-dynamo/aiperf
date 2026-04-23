# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chaos injection helper.

Wraps the raw kubectl surface with a narrow, intent-revealing API for
the fault scenarios exercised in this suite.

Usage::

    async def test_example(chaos_injector: ChaosInjector) -> None:
        await chaos_injector.delete_cr_no_wait("aiperf-jobs-master", "foo")
        await chaos_injector.wait_for_cr_gone("aiperf-jobs-master", "foo", timeout=30)
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import time
from dataclasses import dataclass
from typing import Any

import orjson

from tests.kubernetes.helpers.kubectl import KubectlClient

logger = logging.getLogger(__name__)


OPERATOR_NAMESPACE = "aiperf-system"
OPERATOR_SELECTOR = "app.kubernetes.io/name=aiperf-operator"
AIPERF_CLAIM_ANNOTATION = "aiperf.nvidia.com/completion-claimed"
AIPERF_BENCHMARK_COMPLETE_ANNOTATION = "aiperf.nvidia.com/benchmark-complete"


@dataclass(frozen=True)
class ChaosTimings:
    """Shared timeouts used by chaos scenarios."""

    cr_cleanup_seconds: float = 60.0
    """How long we wait for a deleted CR + JobSet + pods to vanish."""

    pod_termination_grace: float = 45.0
    """Pods can hold for ~30 s after JobSet delete (graceful SIGTERM)."""

    operator_recovery_seconds: float = 30.0
    """How long a new operator pod has to become Ready after a kill."""

    completion_wait_seconds: float = 180.0
    """Max wait for an AIPerfJob to reach a terminal phase."""


class ChaosInjector:
    """Inject faults against a running AIPerfJob deployment.

    Every method is async and delegates to ``KubectlClient``; no direct
    subprocess calls so the helper composes cleanly with the existing
    test harness.
    """

    def __init__(self, kubectl: KubectlClient) -> None:
        """Initialize the injector.

        Args:
            kubectl: Async kubectl wrapper pinned to the chaos cluster.
        """
        self.kubectl = kubectl
        self.timings = ChaosTimings()

    async def delete_cr_no_wait(self, namespace: str, name: str) -> float:
        """Delete an AIPerfJob CR without blocking on finalizer removal.

        Returns the monotonic timestamp at which the delete call was
        issued so tests can compute cleanup latency.
        """
        ts = time.monotonic()
        await self.kubectl.run(
            "delete",
            "aiperfjob",
            name,
            "-n",
            namespace,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )
        return ts

    async def delete_cr_twice(self, namespace: str, name: str) -> tuple[int, int]:
        """Issue two rapid delete calls; return (first_rc, second_rc).

        Second call is expected to hit NotFound (404), which is success
        for the idempotence test.
        """
        first = await self.kubectl.run(
            "delete",
            "aiperfjob",
            name,
            "-n",
            namespace,
            "--wait=false",
            check=False,
        )
        await asyncio.sleep(0.05)
        second = await self.kubectl.run(
            "delete",
            "aiperfjob",
            name,
            "-n",
            namespace,
            "--wait=false",
            check=False,
        )
        return first.returncode, second.returncode

    async def kill_operator_pod(self, force: bool = True) -> None:
        """Force-delete the operator pod (ReplicaSet will spawn a new one)."""
        args = [
            "delete",
            "pod",
            "-l",
            OPERATOR_SELECTOR,
            "-n",
            OPERATOR_NAMESPACE,
            "--ignore-not-found",
        ]
        if force:
            args.extend(["--grace-period=0", "--force"])
        await self.kubectl.run(*args, check=False)

    async def stamp_completion_claim(
        self, namespace: str, name: str, timestamp_iso: str | None = None
    ) -> None:
        """Manually set the `completion-claimed` annotation on a CR.

        Simulates "operator crashed after claiming but before finishing
        handle_completion". Used to exercise the recovery path that
        new-process monitor ticks must take when the claim annotation
        is already present.
        """
        ts = timestamp_iso or datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.000000Z"
        )
        await self.kubectl.run(
            "annotate",
            "aiperfjob",
            name,
            "-n",
            namespace,
            f"{AIPERF_CLAIM_ANNOTATION}={ts}",
            "--overwrite",
            check=True,
        )

    async def kill_container_in_pod(
        self, namespace: str, pod: str, container: str
    ) -> None:
        """Kill a specific container inside a multi-container pod.

        Uses ``kubectl exec`` with ``kill -KILL 1`` against the target
        container. The JobSet Job spec decides whether kubelet restarts
        it; for AIPerf's controller pod none of the service containers
        restart by default, so this is effectively an unrecoverable fault.
        """
        await self.kubectl.run(
            "exec",
            pod,
            "-c",
            container,
            "-n",
            namespace,
            "--",
            "sh",
            "-c",
            "kill -KILL 1",
            check=False,
        )

    async def wait_for_cr_gone(
        self, namespace: str, name: str, timeout: float | None = None
    ) -> float:
        """Block until the CR is removed from the apiserver.

        Returns elapsed wall-clock seconds from call to disappearance,
        or raises ``TimeoutError`` after the timeout.
        """
        deadline = time.monotonic() + (timeout or self.timings.cr_cleanup_seconds)
        start = time.monotonic()
        while time.monotonic() < deadline:
            res = await self.kubectl.run(
                "get",
                "aiperfjob",
                name,
                "-n",
                namespace,
                "--ignore-not-found",
                "-o",
                "name",
                check=False,
            )
            if not res.stdout.strip():
                return time.monotonic() - start
            await asyncio.sleep(0.5)
        raise TimeoutError(
            f"AIPerfJob {namespace}/{name} still present after "
            f"{timeout or self.timings.cr_cleanup_seconds} s"
        )

    async def wait_for_pods_gone(
        self, namespace: str, timeout: float | None = None
    ) -> float:
        """Block until all pods in the namespace are reaped."""
        deadline = time.monotonic() + (timeout or self.timings.pod_termination_grace)
        start = time.monotonic()
        while time.monotonic() < deadline:
            res = await self.kubectl.run(
                "get",
                "pods",
                "-n",
                namespace,
                "-o",
                "name",
                check=False,
            )
            if not res.stdout.strip():
                return time.monotonic() - start
            await asyncio.sleep(0.5)
        raise TimeoutError(
            f"Pods in namespace {namespace} still present after "
            f"{timeout or self.timings.pod_termination_grace} s"
        )

    async def wait_for_operator_ready(self, timeout: float | None = None) -> float:
        """Block until an operator pod is Ready (2/2)."""
        deadline = time.monotonic() + (
            timeout or self.timings.operator_recovery_seconds
        )
        start = time.monotonic()
        while time.monotonic() < deadline:
            res = await self.kubectl.run(
                "get",
                "pods",
                "-l",
                OPERATOR_SELECTOR,
                "-n",
                OPERATOR_NAMESPACE,
                "-o",
                "jsonpath={.items[*].status.containerStatuses[*].ready}",
                check=False,
            )
            readys = res.stdout.strip().split()
            if readys and all(r == "true" for r in readys):
                return time.monotonic() - start
            await asyncio.sleep(0.5)
        raise TimeoutError(
            f"Operator pod did not reach Ready within "
            f"{timeout or self.timings.operator_recovery_seconds} s"
        )

    async def wait_for_phase(
        self,
        namespace: str,
        name: str,
        phases: tuple[str, ...],
        timeout: float | None = None,
        *,
        current_phase: str | None = None,
    ) -> str:
        """Block until CR ``.status.phase`` is one of ``phases``.

        When ``current_phase`` is set, also require ``.status.currentPhase``
        to match. Useful for ``wait_for_phase(..., ("Running",),
        current_phase="profiling")`` to catch actively-benchmarking state.
        Returns the phase that was observed.
        """
        deadline = time.monotonic() + (timeout or self.timings.completion_wait_seconds)
        while time.monotonic() < deadline:
            res = await self.kubectl.run(
                "get",
                "aiperfjob",
                name,
                "-n",
                namespace,
                "-o",
                "jsonpath={.status.phase}|{.status.currentPhase}",
                check=False,
            )
            phase, _, curr = res.stdout.strip().partition("|")
            if phase in phases and (current_phase is None or curr == current_phase):
                return phase
            await asyncio.sleep(1.0)
        raise TimeoutError(
            f"AIPerfJob {namespace}/{name} did not reach phase "
            f"{phases} (currentPhase={current_phase!r}) within "
            f"{timeout or self.timings.completion_wait_seconds} s"
        )

    async def read_claim_annotation(self, namespace: str, name: str) -> str | None:
        """Return the current `completion-claimed` annotation value, or None.

        Uses `-o yaml` then greps for the key because kubectl's jsonpath
        does not handle annotation keys containing `/` cleanly.
        """
        res = await self.kubectl.run(
            "get",
            "aiperfjob",
            name,
            "-n",
            namespace,
            "-o",
            "yaml",
            check=False,
        )
        for line in res.stdout.splitlines():
            stripped = line.strip()
            prefix = f"{AIPERF_CLAIM_ANNOTATION}:"
            if stripped.startswith(prefix):
                value = stripped[len(prefix) :].strip()
                return value.strip('"').strip("'") or None
        return None

    async def get_controller_pod_name(self, namespace: str, job_name: str) -> str:
        """Return the controller pod name for an AIPerfJob.

        The JobSet spawns a single controller replica named
        ``aiperf-<job>-controller-0-0-...``; we match the standard
        ``jobset.sigs.k8s.io/replicatedjob-name=controller`` label.

        Raises:
            RuntimeError: When no controller pod is present (e.g. job still
                Pending or already reaped).
        """
        res = await self.kubectl.run(
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            f"jobset.sigs.k8s.io/jobset-name=aiperf-{job_name},"
            "jobset.sigs.k8s.io/replicatedjob-name=controller",
            "-o",
            "jsonpath={.items[0].metadata.name}",
            check=False,
        )
        name = res.stdout.strip()
        if not name:
            raise RuntimeError(
                f"no controller pod found for AIPerfJob {namespace}/{job_name} "
                f"(label jobset.sigs.k8s.io/jobset-name=aiperf-{job_name}); "
                "is the job still Pending?"
            )
        return name

    async def get_worker_pod_names(self, namespace: str, job_name: str) -> list[str]:
        """Return every worker pod name for an AIPerfJob.

        Matches the ``replicatedjob-name=workers`` label that the JobSet
        applies to every worker pod. Returns an empty list when the job
        has not yet created workers.
        """
        res = await self.kubectl.run(
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            f"jobset.sigs.k8s.io/jobset-name=aiperf-{job_name},"
            "jobset.sigs.k8s.io/replicatedjob-name=workers",
            "-o",
            "jsonpath={.items[*].metadata.name}",
            check=False,
        )
        names = res.stdout.strip().split()
        return [n for n in names if n]

    async def get_mock_server_pod_name(
        self, namespace: str = "default", deployment: str = "aiperf-mock-server"
    ) -> str:
        """Return the mock-server pod name that serves benchmark traffic.

        The k8s test harness runs a single-replica ``aiperf-mock-server``
        Deployment in the ``default`` namespace (see ``tests/kubernetes/conftest.py``).

        Raises:
            RuntimeError: When no pod matches (deployment missing or
                scaled to zero).
        """
        res = await self.kubectl.run(
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            f"app={deployment}",
            "-o",
            "jsonpath={.items[0].metadata.name}",
            check=False,
        )
        name = res.stdout.strip()
        if not name:
            raise RuntimeError(
                f"no pod found for mock-server deployment {namespace}/{deployment} "
                "(expected label app=aiperf-mock-server); has the fixture been deployed?"
            )
        return name

    async def kill_container_by_pid(
        self,
        pod: str,
        container_pid: int,
        namespace: str,
        exec_container: str,
    ) -> None:
        """Kill a sibling container by PID via shared-PID-namespace exec.

        Requires the pod to have been rendered with
        ``spec.shareProcessNamespace: true`` (see
        ``AIPERF_K8S_SHARE_PROCESS_NAMESPACE``). ``kubectl exec`` into
        ``exec_container`` and issue ``kill -9 <pid>`` against the target
        PID, which the kernel resolves to a sibling container because the
        PID namespace is shared.

        Args:
            pod: Pod name hosting both containers.
            container_pid: PID of the target process (obtain via
                ``kubectl exec <pod> -c <any> -- pgrep -n <name>`` upstream).
            namespace: Namespace of the pod.
            exec_container: Container to exec into to issue the kill
                (must have ``sh`` on PATH).
        """
        await self.kubectl.run(
            "exec",
            pod,
            "-c",
            exec_container,
            "-n",
            namespace,
            "--",
            "sh",
            "-c",
            f"kill -9 {container_pid}",
            check=False,
        )

    async def wait_for_container_restart(
        self,
        pod: str,
        container: str,
        namespace: str,
        *,
        since_count: int,
        timeout: float = 60.0,
    ) -> int:
        """Poll ``containerStatuses[].restartCount`` until it exceeds ``since_count``.

        Args:
            pod: Pod name.
            container: Container name within the pod.
            namespace: Namespace of the pod.
            since_count: Baseline restartCount captured before the fault
                injection (tests must snapshot this first).
            timeout: Max seconds to wait for a new restart to be observed.

        Returns:
            The observed restartCount (strictly greater than ``since_count``).

        Raises:
            TimeoutError: When ``restartCount`` does not advance within
                ``timeout`` seconds.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            res = await self.kubectl.run(
                "get",
                "pod",
                pod,
                "-n",
                namespace,
                "-o",
                "jsonpath={.status.containerStatuses[?(@.name=='"
                + container
                + "')].restartCount}",
                check=False,
            )
            raw = res.stdout.strip()
            if raw.isdigit():
                count = int(raw)
                if count > since_count:
                    return count
            await asyncio.sleep(0.5)
        raise TimeoutError(
            f"container {namespace}/{pod}:{container} did not restart "
            f"(restartCount still <= {since_count}) within {timeout} s"
        )

    async def create_invalid_cr(
        self,
        namespace: str,
        name: str,
        spec_patch: dict[str, Any],
    ) -> None:
        """Apply an AIPerfJob CR with a deliberately malformed spec patch.

        Builds a minimal benchmark spec, overlays ``spec_patch`` on top, and
        applies via ``kubectl apply -f -``. Used by chaos tests that assert
        the operator surfaces a validation error (Failed phase + status
        condition). The CR is intentionally accepted by the CRD OpenAPI
        schema — validation happens in the operator spec-converter.

        Args:
            namespace: Target namespace.
            name: CR name.
            spec_patch: Dict merged on top of a minimal valid spec. For
                example ``{"benchmark": {"endpoint": {"urls": ["not a url"]}}}``
                exercises the endpoint URL validator.
        """
        base_spec: dict[str, Any] = {
            "image": "aiperf:local",
            "imagePullPolicy": "Never",
            "benchmark": {
                "models": {"items": [{"name": "mock-model"}]},
                "endpoint": {
                    "urls": [
                        "http://aiperf-mock-server.default.svc.cluster.local:8000/v1"
                    ]
                },
                "phases": {
                    "type": "concurrency",
                    "concurrency": 1,
                    "requests": 1,
                },
                "tokenizer": {"name": "gpt2"},
                "runtime": {"ui": "none"},
            },
        }
        for key, value in spec_patch.items():
            base_spec[key] = value
        manifest = (
            "apiVersion: aiperf.nvidia.com/v1alpha1\n"
            "kind: AIPerfJob\n"
            "metadata:\n"
            f"  name: {name}\n"
            f"  namespace: {namespace}\n"
            "spec: " + orjson.dumps(base_spec).decode() + "\n"
        )
        await self.kubectl.apply(manifest, namespace=namespace)
