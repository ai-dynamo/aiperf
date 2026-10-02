# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the reused-cluster purge helper in tests/kubernetes/conftest.py.

These exercise ``_purge_reused_cluster_resources`` directly against a mocked
kubectl client so the isolation contract (never touch namespaces that could be
owned by other concurrently-running pytest sessions) can be verified without a
real cluster.
"""

import subprocess
from unittest.mock import AsyncMock

import pytest

from tests.kubernetes.conftest import _purge_reused_cluster_resources


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    """Build a fake kubectl CompletedProcess result."""
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )


@pytest.mark.asyncio
async def test_purge_reused_cluster_resources_shared_default_namespace_untouched():
    """The shared 'default' namespace must never be force-purged.

    Other concurrently-running pytest sessions can own resources in
    'default'; the function's own docstring says it exists to avoid touching
    namespaces owned by other sessions, so no kubectl call it issues may
    target 'default'.
    """
    kubectl = AsyncMock()
    kubectl.run.return_value = _completed()

    await _purge_reused_cluster_resources(
        kubectl,
        worker_namespace_suffix="gw0",
        operator_job_namespace="aiperf-jobs-gw0-deadbeef",
        benchmark_namespace="aiperf-bench-gw0-deadbeef",
    )

    offending_calls = [
        call.args for call in kubectl.run.await_args_list if "default" in call.args
    ]
    assert not offending_calls, (
        f"kubectl.run was called against the shared 'default' namespace: {offending_calls}"
    )
