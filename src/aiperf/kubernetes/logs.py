# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Log retrieval from Kubernetes benchmark pods."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from aiperf.kubernetes.subproc import run_command

if TYPE_CHECKING:
    from aiperf.kubernetes.client import AIPerfKubeClient


async def save_pod_logs(
    job_id: str,
    namespace: str,
    output_dir: Path,
    client: AIPerfKubeClient,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> None:
    """Save logs from all benchmark pods to the output directory.

    Creates a ``logs/`` subdirectory and writes one file per pod
    (e.g. ``logs/controller-pod.log``).

    Args:
        job_id: AIPerf job ID.
        namespace: Kubernetes namespace.
        output_dir: Base output directory for artifacts.
        client: Kubernetes client.
        kubeconfig: Path to kubeconfig file.
        kube_context: Kubernetes context name.
    """
    pods = await client.get_pods(namespace, client.job_selector(job_id))
    if not pods:
        return

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    kube_args: list[str] = []
    if kubeconfig:
        kube_args.extend(["--kubeconfig", kubeconfig])
    if kube_context:
        kube_args.extend(["--context", kube_context])

    for pod in pods:
        cmd = [
            "kubectl",
            "logs",
            "-n",
            namespace,
            pod.name,
            *kube_args,
        ]
        result = await run_command(cmd)
        if result.returncode == 0 and result.stdout:
            log_file = logs_dir / f"{pod.name}.log"
            await asyncio.to_thread(log_file.write_text, result.stdout)
