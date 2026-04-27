# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bare-pod deployer: runs ``aiperf profile`` in a single ``batch/v1.Job``.

This is the "oracle" side of the audit. No operator, no JobSet, no controller,
no workers - just one pod running the local CLI against the in-cluster mock
server. Results are extracted via ``kubectl cp`` before the Job is deleted.
"""

from __future__ import annotations

import asyncio
import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.audit.cases import AuditCase
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)


@dataclass
class BarePodConfig:
    """Resolved settings for one bare-pod run."""

    image: str = "aiperf:local"
    image_pull_policy: str = "Never"
    endpoint_url: str = "http://aiperf-mock-server.default.svc.cluster.local:8000/v1"
    model_name: str = "mock-model"
    tokenizer_name: str = "gpt2"


class BarePodDeployer:
    """Submits a raw Job, waits for the aiperf sentinel, copies artifacts out."""

    def __init__(
        self,
        kubectl: KubectlClient,
        config: BarePodConfig | None = None,
    ) -> None:
        self.kubectl = kubectl
        self.config = config or BarePodConfig()

    def _build_args(
        self, case: AuditCase, *, swept_value: object | None = None
    ) -> list[str]:
        """Translate AuditCase -> ``aiperf profile`` argv (excluding the binary)."""
        concurrency = case.concurrency
        if case.sweep and "concurrency" in case.sweep and swept_value is not None:
            concurrency = int(swept_value)

        args: list[str] = [
            "profile",
            "--model",
            self.config.model_name,
            "--url",
            self.config.endpoint_url,
            "--endpoint-type",
            case.endpoint_type,
            "--tokenizer",
            self.config.tokenizer_name,
            "--concurrency",
            str(concurrency),
            "--request-count",
            str(case.request_count),
            "--random-seed",
            str(case.seed),
            "--ui",
            "none",
            "--artifact-dir",
            "/aiperf-output",
        ]
        if case.num_conversations is not None:
            args += ["--num-conversations", str(case.num_conversations)]
        return args

    def _build_job_manifest(
        self,
        *,
        name: str,
        namespace: str,
        argv: list[str],
    ) -> str:
        """Build the batch/v1.Job manifest as a YAML string.

        The container runs aiperf, writes the exit code to a sentinel file,
        then sleeps so ``kubectl cp`` (which uses ``kubectl exec`` under the
        hood) can still exec into a Running container. The Job is killed
        explicitly via ``kubectl delete job`` at the end of ``run()``.
        """
        wrapped_argv = " ".join(shlex.quote(a) for a in argv)
        shell_cmd = (
            f"aiperf {wrapped_argv}; echo $? > /tmp/.aiperf_exit_code; sleep 3600"
        )
        body = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {"app.kubernetes.io/name": "aiperf-bare-audit"},
            },
            "spec": {
                "ttlSecondsAfterFinished": 3600,
                "backoffLimit": 0,
                "template": {
                    "metadata": {
                        "labels": {"app.kubernetes.io/name": "aiperf-bare-audit"}
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": "aiperf",
                                "image": self.config.image,
                                "imagePullPolicy": self.config.image_pull_policy,
                                "command": ["/bin/bash", "-c"],
                                "args": [shell_cmd],
                                "volumeMounts": [
                                    {"name": "output", "mountPath": "/aiperf-output"},
                                ],
                            },
                        ],
                        "volumes": [{"name": "output", "emptyDir": {}}],
                    },
                },
            },
        }
        return yaml.safe_dump(body, sort_keys=False)

    async def _wait_for_pod_running(
        self, job_name: str, namespace: str, timeout: int = 120
    ) -> str:
        """Poll until the Job's pod exists and is Running. Return pod name."""
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            name_result = await self.kubectl.run(
                "get",
                "pod",
                "-n",
                namespace,
                "-l",
                f"job-name={job_name}",
                "-o",
                "jsonpath={.items[0].metadata.name}",
                check=False,
            )
            pod = name_result.stdout.strip() if name_result.returncode == 0 else ""
            if pod:
                phase_result = await self.kubectl.run(
                    "get",
                    "pod",
                    pod,
                    "-n",
                    namespace,
                    "-o",
                    "jsonpath={.status.phase}",
                    check=False,
                )
                if (
                    phase_result.returncode == 0
                    and phase_result.stdout.strip() == "Running"
                ):
                    return pod
            await asyncio.sleep(2)
        raise TimeoutError(
            f"bare-pod {namespace} job {job_name}: pod did not reach Running within {timeout}s"
        )

    async def _wait_for_sentinel(self, pod: str, namespace: str, timeout: int) -> int:
        """Poll until /tmp/.aiperf_exit_code exists. Return the exit code."""
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            result = await self.kubectl.run(
                "exec",
                "-n",
                namespace,
                pod,
                "-c",
                "aiperf",
                "--",
                "cat",
                "/tmp/.aiperf_exit_code",
                check=False,
            )
            if result.returncode == 0:
                try:
                    return int(result.stdout.strip())
                except ValueError:
                    pass  # sentinel half-written; retry
            await asyncio.sleep(3)
        raise TimeoutError(
            f"bare-pod {namespace}/{pod}: aiperf did not finish "
            f"(no /tmp/.aiperf_exit_code) within {timeout}s"
        )

    async def _kubectl_cp(self, pod: str, namespace: str, dest_dir: Path) -> None:
        """Copy /aiperf-output from the (still-running) pod to dest_dir."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        await self.kubectl.run(
            "cp",
            f"{namespace}/{pod}:/aiperf-output/.",
            str(dest_dir),
            "-c",
            "aiperf",
            check=True,
        )

    async def run(
        self,
        *,
        case: AuditCase,
        namespace: str,
        dest_dir: Path,
        swept_value: object | None = None,
        timeout: int = 600,
    ) -> Path:
        """Run one bare-pod invocation; return ``dest_dir`` with artifacts copied in."""
        suffix = uuid.uuid4().hex[:6]
        name = f"bare-{case.case_id}-{suffix}"

        await self.kubectl.create_namespace(namespace)

        argv = self._build_args(case, swept_value=swept_value)
        manifest = self._build_job_manifest(name=name, namespace=namespace, argv=argv)
        await self.kubectl.apply(manifest, namespace=namespace)

        try:
            pod = await self._wait_for_pod_running(name, namespace, timeout=120)
            exit_code = await self._wait_for_sentinel(pod, namespace, timeout=timeout)
            await self._kubectl_cp(pod, namespace, dest_dir)
            if exit_code != 0:
                logger.warning(
                    f"bare-pod aiperf exited with code {exit_code} (job {name}); "
                    f"artifacts copied for diff to surface partial state"
                )
        finally:
            await self.kubectl.run(
                "delete",
                "job",
                name,
                "-n",
                namespace,
                "--wait=false",
                check=False,
            )

        return dest_dir
