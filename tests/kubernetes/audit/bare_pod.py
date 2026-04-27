# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bare-pod deployer: runs ``aiperf profile`` in a single ``batch/v1.Job``.

This is the "oracle" side of the audit. No operator, no JobSet, no controller,
no workers - just one pod running the local CLI against the in-cluster mock
server. Results are extracted via ``kubectl cp`` before the Job is deleted.
"""

from __future__ import annotations

import asyncio
import json
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
    """Submits a raw Job, waits for completion, copies artifacts out."""

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
        """Build the batch/v1.Job manifest as a YAML string."""
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
                                "command": ["aiperf"],
                                "args": argv,
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

    async def _wait_for_terminal(self, name: str, namespace: str, timeout: int) -> str:
        """Poll the Job until it reports Complete or Failed. Returns final phase."""
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            result = await self.kubectl.run(
                "get",
                "job",
                name,
                "-n",
                namespace,
                "-o",
                "json",
                check=False,
            )
            if result.returncode == 0:
                payload = json.loads(result.stdout)
                conditions = payload.get("status", {}).get("conditions", []) or []
                for c in conditions:
                    if c.get("type") == "Complete" and c.get("status") == "True":
                        return "Complete"
                    if c.get("type") == "Failed" and c.get("status") == "True":
                        return "Failed"
            await asyncio.sleep(3)
        raise TimeoutError(
            f"bare-pod job {namespace}/{name} did not reach terminal state in {timeout}s"
        )

    async def _pod_for_job(self, name: str, namespace: str) -> str:
        result = await self.kubectl.run(
            "get",
            "pod",
            "-n",
            namespace,
            "-l",
            f"job-name={name}",
            "-o",
            "jsonpath={.items[0].metadata.name}",
            check=True,
        )
        pod = result.stdout.strip()
        if not pod:
            raise RuntimeError(f"no pod found for job {namespace}/{name}")
        return pod

    async def _kubectl_cp(self, pod: str, namespace: str, dest_dir: Path) -> None:
        """Copy /aiperf-output from the (terminal) pod to dest_dir."""
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
        name = f"audit-bare-{case.case_id}-{suffix}"

        await self.kubectl.create_namespace(namespace)

        argv = self._build_args(case, swept_value=swept_value)
        manifest = self._build_job_manifest(name=name, namespace=namespace, argv=argv)
        await self.kubectl.apply(manifest, namespace=namespace)

        try:
            phase = await self._wait_for_terminal(name, namespace, timeout)
            pod = await self._pod_for_job(name, namespace)
            await self._kubectl_cp(pod, namespace, dest_dir)
            if phase != "Complete":
                logger.warning(f"bare-pod job {name} terminal phase = {phase}")
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
