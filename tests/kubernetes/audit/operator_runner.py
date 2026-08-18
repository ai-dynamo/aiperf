# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator-side runner: deploy ``AIPerfJob``, then ``aiperf kube results``.

This wraps the existing ``OperatorDeployer`` to submit a CR that mirrors the
``AuditCase``'s profile args, waits for completion, then shells out to the
user-facing ``aiperf kube results <id> --output <dir>`` CLI to download
artifacts. Exercising the download CLI is part of the audit's purpose: if the
operator-managed run produces correct artifacts internally but ships them
incorrectly to the user, the audit catches that.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.audit import aiperf_cli
from tests.kubernetes.audit.cases import AuditCase
from tests.kubernetes.helpers.operator import (
    AIPerfJobConfig,
    OperatorDeployer,
)

logger = AIPerfLogger(__name__)


@dataclass
class OperatorAuditConfig:
    image: str = "aiperf:local"
    image_pull_policy: str = "Never"
    endpoint_url: str = "http://aiperf-mock-server.default.svc.cluster.local:8000/v1"
    model_name: str = "mock-model"
    tokenizer_name: str = "gpt2"


class OperatorAuditRunner:
    """Submits an AIPerfJob and downloads its artifacts via ``aiperf kube results``."""

    def __init__(
        self,
        deployer: OperatorDeployer,
        config: OperatorAuditConfig | None = None,
    ) -> None:
        self.deployer = deployer
        self.config = config or OperatorAuditConfig()

    def _build_job_config(
        self, case: AuditCase, *, swept_value: object | None = None
    ) -> AIPerfJobConfig:
        concurrency = case.concurrency
        if case.sweep and "concurrency" in case.sweep and swept_value is not None:
            concurrency = int(swept_value)

        return AIPerfJobConfig(
            endpoint_url=self.config.endpoint_url,
            model_name=self.config.model_name,
            endpoint_type=case.endpoint_type,
            concurrency=concurrency,
            request_count=case.request_count,
            warmup_request_count=0,
            num_conversations=case.num_conversations,
            tokenizer_name=self.config.tokenizer_name,
            image=self.config.image,
            image_pull_policy=self.config.image_pull_policy,
            random_seed=case.seed,
        )

    async def _download_results(
        self,
        *,
        namespace: str,
        job_name: str,
        dest_dir: Path,
        kubeconfig: str | None,
    ) -> None:
        """Shell out to ``aiperf kube results`` against the test cluster.

        The audit suite already talks to Kubernetes through a ``KubectlClient``
        pinned to an explicit kind/minikube context. Reuse those credentials
        here; otherwise the shell-out falls back to the user's ambient
        kubectl context (often DGX), and the lookup fails with
        "No AIPerfJob ... found" even though the CR exists on the test cluster.
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            aiperf_cli(),
            "kube",
            "results",
            job_name,
            "--namespace",
            namespace,
            "--output",
            str(dest_dir),
            "--all",
        ]
        if self.deployer.kubectl.context:
            cmd.extend(["--kube-context", self.deployer.kubectl.context])
        env = dict(os.environ)
        if kubeconfig:
            env["KUBECONFIG"] = kubeconfig
        elif self.deployer.kubectl.kubeconfig:
            env["KUBECONFIG"] = self.deployer.kubectl.kubeconfig

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"aiperf kube results failed (rc={proc.returncode})\n"
                f"stdout:\n{stdout.decode(errors='replace')}\n"
                f"stderr:\n{stderr.decode(errors='replace')}"
            )
        # `aiperf kube results` exits 0 even when retrieval internally fails
        # (e.g. operator pod not found). Verify dest_dir is non-empty so the
        # underlying error surfaces instead of an empty-bucket diff failure.
        files = [p for p in dest_dir.rglob("*") if p.is_file()]
        if not files:
            raise RuntimeError(
                f"aiperf kube results {job_name} exited 0 but produced no files in {dest_dir}\n"
                f"stdout:\n{stdout.decode(errors='replace')}\n"
                f"stderr:\n{stderr.decode(errors='replace')}"
            )

    async def run(
        self,
        *,
        case: AuditCase,
        namespace: str,
        dest_dir: Path,
        kubeconfig: str | None = None,
        swept_value: object | None = None,
        timeout: int = 600,
    ) -> Path:
        suffix = uuid.uuid4().hex[:6]
        job_name = f"op-{case.case_id}-{suffix}"
        cfg = self._build_job_config(case, swept_value=swept_value)

        await self.deployer.kubectl.run("create", "namespace", namespace, check=False)

        result = await self.deployer.run_job(
            config=cfg,
            name=job_name,
            namespace=namespace,
            timeout=timeout,
        )
        if not result.success:
            raise RuntimeError(
                f"operator job {namespace}/{job_name} did not succeed: "
                f"{result.error_message or 'unknown'}"
            )

        await self._download_results(
            namespace=namespace,
            job_name=job_name,
            dest_dir=dest_dir,
            kubeconfig=kubeconfig,
        )
        return dest_dir
