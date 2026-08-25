# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the k8s-audit OperatorAuditRunner shell-out path."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.kubernetes.audit.operator_runner import OperatorAuditRunner
from tests.kubernetes.helpers.kubectl import KubectlClient


class _Proc:
    """Minimal subprocess stub for OperatorAuditRunner._download_results tests."""

    def __init__(self, *, returncode: int = 0) -> None:
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"ok", b""


@pytest.mark.asyncio
async def test_download_results_passes_explicit_kube_context_and_kubeconfig(
    tmp_path: Path,
) -> None:
    """Audit downloads must target the test cluster, not ambient kubectl context.

    Why: the audit runner shells out to the user-facing ``aiperf kube results``
    CLI. Without explicit ``--kube-context`` / ``KUBECONFIG``, the shell-out
    inherits the user's ambient context (often DGX), and the lookup fails with
    "No AIPerfJob ... found" even though the CR exists on the kind test cluster.
    """
    dest_dir = tmp_path / "operator"
    deployer = SimpleNamespace(
        kubectl=KubectlClient(context="kind-aiperf-test", kubeconfig="/tmp/kc")
    )
    runner = OperatorAuditRunner(deployer=deployer)

    async def _fake_exec(*cmd: str, **kwargs: object) -> _Proc:
        assert "--kube-context" in cmd
        assert "kind-aiperf-test" in cmd
        assert kwargs["env"]["KUBECONFIG"] == "/tmp/kc"
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "profile_export_aiperf.json").write_text("{}", encoding="utf-8")
        return _Proc()

    with patch.object(asyncio, "create_subprocess_exec", side_effect=_fake_exec):
        await runner._download_results(
            namespace="audit-ns",
            job_name="job-1",
            dest_dir=dest_dir,
            kubeconfig=None,
        )


@pytest.mark.asyncio
async def test_download_results_prefers_explicit_kubeconfig_over_deployer_default(
    tmp_path: Path,
) -> None:
    """Caller-provided kubeconfig wins over the deployer's default."""
    dest_dir = tmp_path / "operator"
    deployer = SimpleNamespace(
        kubectl=KubectlClient(context="kind-aiperf-test", kubeconfig="/tmp/default-kc")
    )
    runner = OperatorAuditRunner(deployer=deployer)

    async def _fake_exec(*cmd: str, **kwargs: object) -> _Proc:
        assert "--kube-context" in cmd
        assert kwargs["env"]["KUBECONFIG"] == "/tmp/explicit-kc"
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "profile_export_aiperf.json").write_text("{}", encoding="utf-8")
        return _Proc()

    with patch.object(asyncio, "create_subprocess_exec", side_effect=_fake_exec):
        await runner._download_results(
            namespace="audit-ns",
            job_name="job-1",
            dest_dir=dest_dir,
            kubeconfig="/tmp/explicit-kc",
        )
