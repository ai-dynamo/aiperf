# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for :py:class:`PodInjector`.

Uses a mocked :py:class:`KubectlClient` -- no cluster required. The legacy
:py:class:`ChaosInjector` is patched so we can assert that ``pod.kill_container``
and ``pod.kill_pid`` dispatch to the right legacy method with the right
arguments; the actual ``kubectl exec`` shape is covered by the legacy suite.
"""

from __future__ import annotations

import subprocess
from unittest.mock import AsyncMock, patch

import pytest

from tests.kubernetes.chaos_common.base import (
    FaultMechanismError,
    FaultPreconditionError,
    FaultSpec,
)
from tests.kubernetes.chaos_common.injectors.pod import PodInjector


def _make_kubectl_mock() -> AsyncMock:
    """Return an :py:class:`AsyncMock` shaped like :py:class:`KubectlClient`.

    The single method tests touch is ``run``, which returns a
    :py:class:`subprocess.CompletedProcess`. Returning rc=0 lets ``check=True``
    callers pass; tests that need failure paths reconfigure ``side_effect``.
    """
    kubectl = AsyncMock()
    kubectl.run = AsyncMock(
        return_value=subprocess.CompletedProcess(
            args=["kubectl"], returncode=0, stdout="", stderr=""
        )
    )
    return kubectl


@pytest.mark.asyncio
async def test_pod_kill_calls_kubectl_delete_with_force() -> None:
    kubectl = _make_kubectl_mock()
    injector = PodInjector(kubectl)

    spec = FaultSpec(
        fault_id="pod.kill",
        target={"pod": "controller-0", "ns": "aiperf-test-abc"},
    )
    applied = await injector.inject(spec)

    kubectl.run.assert_awaited_once_with(
        "delete",
        "pod",
        "controller-0",
        "-n",
        "aiperf-test-abc",
        "--force",
        "--grace-period=0",
        check=True,
    )
    assert applied.metadata["pod"] == "controller-0"
    assert applied.metadata["namespace"] == "aiperf-test-abc"


@pytest.mark.asyncio
async def test_pod_kill_container_routes_to_chaos_injector() -> None:
    kubectl = _make_kubectl_mock()
    injector = PodInjector(kubectl)

    spec = FaultSpec(
        fault_id="pod.kill_container",
        target={
            "pod": "controller-0",
            "ns": "aiperf-test-abc",
            "container": "records-manager",
        },
    )

    with patch(
        "tests.kubernetes.chaos_common.injectors.pod.ChaosInjector"
    ) as mock_chaos_cls:
        mock_chaos = mock_chaos_cls.return_value
        mock_chaos.kill_container_in_pod = AsyncMock(return_value=None)

        applied = await injector.inject(spec)

    mock_chaos_cls.assert_called_once_with(kubectl)
    mock_chaos.kill_container_in_pod.assert_awaited_once_with(
        namespace="aiperf-test-abc",
        pod="controller-0",
        container="records-manager",
    )
    assert applied.metadata["container"] == "records-manager"


@pytest.mark.asyncio
async def test_pod_kill_pid_requires_container_pid_param() -> None:
    kubectl = _make_kubectl_mock()
    injector = PodInjector(kubectl)

    spec = FaultSpec(
        fault_id="pod.kill_pid",
        target={
            "pod": "controller-0",
            "ns": "aiperf-test-abc",
            "exec_container": "system-controller",
        },
        params={},  # missing container_pid
    )

    with pytest.raises(FaultPreconditionError, match="container_pid"):
        await injector.inject(spec)


@pytest.mark.asyncio
async def test_pod_kill_pid_routes_to_chaos_injector() -> None:
    """Happy-path coverage for the third fault id (rounds out kill_pid)."""
    kubectl = _make_kubectl_mock()
    injector = PodInjector(kubectl)

    spec = FaultSpec(
        fault_id="pod.kill_pid",
        target={
            "pod": "controller-0",
            "ns": "aiperf-test-abc",
            "exec_container": "system-controller",
        },
        params={"container_pid": 1234},
    )

    with patch(
        "tests.kubernetes.chaos_common.injectors.pod.ChaosInjector"
    ) as mock_chaos_cls:
        mock_chaos = mock_chaos_cls.return_value
        mock_chaos.kill_container_by_pid = AsyncMock(return_value=None)

        applied = await injector.inject(spec)

    mock_chaos.kill_container_by_pid.assert_awaited_once_with(
        pod="controller-0",
        container_pid=1234,
        namespace="aiperf-test-abc",
        exec_container="system-controller",
    )
    assert applied.metadata["container_pid"] == 1234


def test_handles_prefix_match_pod() -> None:
    assert PodInjector.handles("pod") is True
    assert PodInjector.handles("pod.kill") is True
    assert PodInjector.handles("pod.kill_container") is True
    assert PodInjector.handles("pod.kill_pid") is True
    assert PodInjector.handles("network.latency") is False
    # Prefix safety: "podx.kill" must NOT match the "pod" namespace.
    assert PodInjector.handles("podx.kill") is False


@pytest.mark.asyncio
async def test_pod_fault_restore_is_noop() -> None:
    kubectl = _make_kubectl_mock()
    injector = PodInjector(kubectl)

    spec = FaultSpec(
        fault_id="pod.kill",
        target={"pod": "controller-0", "ns": "aiperf-test-abc"},
    )
    applied = await injector.inject(spec)

    # Restore must complete cleanly and mark metadata; second call is idempotent.
    await applied.restore()
    assert applied.metadata.get("restored") is True
    await applied.restore()  # idempotent
    assert applied.metadata.get("restored") is True


@pytest.mark.asyncio
async def test_pod_kill_missing_target_raises_precondition_error() -> None:
    kubectl = _make_kubectl_mock()
    injector = PodInjector(kubectl)

    spec = FaultSpec(fault_id="pod.kill", target={"pod": "controller-0"})  # no ns
    with pytest.raises(FaultPreconditionError, match="ns"):
        await injector.inject(spec)


@pytest.mark.asyncio
async def test_pod_kill_kubectl_failure_raises_mechanism_error() -> None:
    kubectl = _make_kubectl_mock()
    kubectl.run = AsyncMock(side_effect=RuntimeError("apiserver 500"))
    injector = PodInjector(kubectl)

    spec = FaultSpec(
        fault_id="pod.kill",
        target={"pod": "controller-0", "ns": "aiperf-test-abc"},
    )
    with pytest.raises(FaultMechanismError, match="kubectl delete pod"):
        await injector.inject(spec)


@pytest.mark.asyncio
async def test_unknown_fault_id_raises_precondition_error() -> None:
    kubectl = _make_kubectl_mock()
    injector = PodInjector(kubectl)

    spec = FaultSpec(fault_id="pod.detonate", target={"pod": "x", "ns": "y"})
    with pytest.raises(FaultPreconditionError, match="does not recognize"):
        await injector.inject(spec)
