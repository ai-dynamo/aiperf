# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Log retrieval from Kubernetes benchmark pods."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from aiperf.kubernetes.client import get_pods, job_selector
from aiperf.kubernetes.subproc import run_command

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient


@dataclass(slots=True)
class SavedPodLogs:
    """What a bulk pod-log dump actually wrote, so callers can narrate it.

    Every "nothing was written" path is representable: an unmatched label
    selector leaves ``pods_matched`` at zero, and a pod whose ``kubectl logs``
    failed or came back empty contributes to ``failures`` instead of
    ``files_written``. Without this, callers can only assume success.
    """

    logs_dir: Path
    """``<output_dir>/logs``. Not created when no pod matched."""

    pods_matched: int = 0
    """Pods the job's label selector matched."""

    files_written: list[str] = field(default_factory=list)
    """Basenames of the ``.log`` files that landed on disk."""

    failures: list[str] = field(default_factory=list)
    """``<pod>: <reason>`` for every matched pod that produced no file."""

    @property
    def wrote_anything(self) -> bool:
        """Whether at least one log file landed on disk."""
        return bool(self.files_written)


async def save_pod_logs(
    job_id: str,
    namespace: str,
    output_dir: Path,
    api: ApiClient,
    *,
    kubeconfig: str | None = None,
    kube_context: str | None = None,
) -> SavedPodLogs:
    """Save logs from all benchmark pods to the output directory.

    Creates a ``logs/`` subdirectory and writes one file per pod
    (e.g. ``logs/controller-pod.log``).

    Args:
        job_id: AIPerf job ID.
        namespace: Kubernetes namespace.
        output_dir: Base output directory for artifacts.
        api: Connected kubernetes_asyncio ApiClient.
        kubeconfig: Path to kubeconfig file.
        kube_context: Kubernetes context name.

    Returns:
        A :class:`SavedPodLogs` describing which pods matched, which files were
        written, and why any matched pod produced nothing.
    """
    saved = SavedPodLogs(logs_dir=output_dir / "logs")
    pods = await get_pods(api, namespace, job_selector(job_id))
    saved.pods_matched = len(pods)
    if not pods:
        return saved

    saved.logs_dir.mkdir(parents=True, exist_ok=True)

    kube_args: list[str] = []
    if kubeconfig:
        kube_args.extend(["--kubeconfig", kubeconfig])
    if kube_context:
        kube_args.extend(["--context", kube_context])

    for pod in pods:
        pod_name = pod.metadata.name if pod.metadata and pod.metadata.name else ""
        if not pod_name:
            saved.failures.append("<unnamed pod>: no metadata.name to fetch logs by")
            continue
        # Controller pods carry 5+ service containers + sidecars; default
        # ``kubectl logs`` only emits the first container, so always pass
        # ``--all-containers`` and ``--prefix`` to interleave per-container
        # output prefixed by ``[pod/<container>]``.
        cmd = [
            "kubectl",
            "logs",
            "-n",
            namespace,
            pod_name,
            "--all-containers=true",
            "--prefix",
            *kube_args,
        ]
        result = await run_command(cmd)
        if result.returncode != 0:
            reason = (result.stderr or "").strip() or "no stderr"
            saved.failures.append(
                f"{pod_name}: kubectl logs exited {result.returncode}: {reason}"
            )
            continue
        if not result.stdout:
            saved.failures.append(f"{pod_name}: kubectl logs returned no output")
            continue
        log_file = saved.logs_dir / f"{pod_name}.log"
        await asyncio.to_thread(log_file.write_text, result.stdout)
        saved.files_written.append(log_file.name)

    return saved
