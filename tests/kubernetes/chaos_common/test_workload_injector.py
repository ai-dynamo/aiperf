# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for :py:class:`WorkloadInjector`.

These tests exercise dispatch + LIFO restore through a mocked
:py:class:`KubectlClient` so no cluster is required. They cover every
``fault_id`` the injector handles, the precondition contract, and the
prefix-match guarantee that the registry relies on.
"""

from __future__ import annotations

import subprocess
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tests.kubernetes.chaos_common.base import (
    FaultPreconditionError,
    FaultSpec,
)
from tests.kubernetes.chaos_common.injectors.workload import WorkloadInjector
from tests.kubernetes.helpers.kubectl import KubectlClient


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    """Build the shape :py:meth:`KubectlClient.run` returns to its callers."""
    return subprocess.CompletedProcess(
        args=["kubectl"], returncode=returncode, stdout=stdout, stderr=""
    )


def _make_kubectl(
    *, get_replicas: str = "3"
) -> tuple[KubectlClient, AsyncMock, list[tuple[tuple[str, ...], dict[str, Any]]]]:
    """Return a KubectlClient whose ``.run`` is a scripted AsyncMock.

    The mock returns ``stdout=get_replicas`` on the ``get deployment ...`` call
    used by :py:meth:`MockServerInjector.scale` to capture the prior count, and
    a generic success for everything else. ``call_log`` records (args, kwargs)
    in order so tests can assert exact kubectl argv shapes.
    """
    kubectl = KubectlClient()
    call_log: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    async def fake_run(*args: str, **kwargs: Any) -> subprocess.CompletedProcess:
        call_log.append((args, kwargs))
        if args[:2] == ("get", "deployment"):
            return _completed(stdout=get_replicas)
        return _completed()

    run = AsyncMock(side_effect=fake_run)
    kubectl.run = run  # type: ignore[method-assign]
    return kubectl, run, call_log


@pytest.mark.asyncio
async def test_workload_restart_calls_kubectl_rollout() -> None:
    kubectl, _run, calls = _make_kubectl()
    injector = WorkloadInjector(kubectl)

    spec = FaultSpec(
        fault_id="workload.restart",
        target={"ns": "default", "deployment": "aiperf-mock-server"},
    )
    applied = await injector.inject(spec)

    assert any(
        args[:3] == ("rollout", "restart", "deployment/aiperf-mock-server")
        and kwargs.get("check") is True
        for args, kwargs in calls
    ), f"expected rollout restart call; got {calls!r}"

    # Restore strips the restartedAt annotation via strategic patch.
    await applied.restore()
    assert any(
        args[:3] == ("patch", "deployment", "aiperf-mock-server")
        for args, _kwargs in calls
    ), f"expected restore patch; got {calls!r}"


@pytest.mark.asyncio
async def test_workload_scale_records_prior_replicas_for_restore() -> None:
    kubectl, _run, calls = _make_kubectl(get_replicas="3")
    injector = WorkloadInjector(kubectl)

    spec = FaultSpec(
        fault_id="workload.scale",
        params={"replicas": 1},
        target={"ns": "default", "deployment": "aiperf-mock-server"},
    )
    applied = await injector.inject(spec)

    scale_calls = [args for args, _ in calls if args[:1] == ("scale",)]
    assert scale_calls, f"expected at least one scale call; got {calls!r}"
    # First scale: down to the spec-requested replica count.
    assert scale_calls[0] == (
        "scale",
        "deployment/aiperf-mock-server",
        "--replicas=1",
        "-n",
        "default",
    )

    await applied.restore()
    scale_calls = [args for args, _ in calls if args[:1] == ("scale",)]
    # Second scale: restore to the prior replica count read from `get`.
    assert scale_calls[-1] == (
        "scale",
        "deployment/aiperf-mock-server",
        "--replicas=3",
        "-n",
        "default",
    )


@pytest.mark.asyncio
async def test_workload_set_env_then_restore_removes_var() -> None:
    kubectl, _run, calls = _make_kubectl()
    injector = WorkloadInjector(kubectl)

    spec = FaultSpec(
        fault_id="workload.set_env",
        params={"env_var": "FOO", "value": "bar"},
        target={"ns": "default", "deployment": "aiperf-mock-server"},
    )
    applied = await injector.inject(spec)

    set_env_calls = [args for args, _ in calls if args[:2] == ("set", "env")]
    assert set_env_calls, f"expected set env call; got {calls!r}"
    assert "FOO=bar" in set_env_calls[0]

    await applied.restore()
    set_env_calls = [args for args, _ in calls if args[:2] == ("set", "env")]
    # `kubectl set env ... FOO-` is the canonical "unset" form.
    assert any("FOO-" in args for args in set_env_calls), (
        f"expected FOO- unset on restore; got {set_env_calls!r}"
    )


@pytest.mark.asyncio
async def test_workload_rolling_upgrade_uses_same_mechanism_as_restart() -> None:
    kubectl_a, _, calls_restart = _make_kubectl()
    kubectl_b, _, calls_upgrade = _make_kubectl()

    target = {"ns": "default", "deployment": "aiperf-mock-server"}
    await WorkloadInjector(kubectl_a).inject(
        FaultSpec(fault_id="workload.restart", target=target)
    )
    await WorkloadInjector(kubectl_b).inject(
        FaultSpec(fault_id="workload.rolling_upgrade", target=target)
    )

    def rollout_args(
        calls: list[tuple[tuple[str, ...], dict[str, Any]]],
    ) -> list[tuple[str, ...]]:
        return [args for args, _ in calls if args[:1] == ("rollout",)]

    assert rollout_args(calls_restart) == rollout_args(calls_upgrade), (
        f"restart vs rolling_upgrade kubectl argv diverged: "
        f"{calls_restart!r} vs {calls_upgrade!r}"
    )


@pytest.mark.asyncio
async def test_missing_target_raises_precondition() -> None:
    kubectl, _, _ = _make_kubectl()
    injector = WorkloadInjector(kubectl)

    with pytest.raises(FaultPreconditionError, match="spec.target"):
        await injector.inject(FaultSpec(fault_id="workload.restart"))


def test_handles_prefix_match_workload() -> None:
    assert WorkloadInjector.handles("workload")
    assert WorkloadInjector.handles("workload.restart")
    assert WorkloadInjector.handles("workload.rolling_upgrade")
    assert WorkloadInjector.handles("workload.scale")
    assert WorkloadInjector.handles("workload.set_env")
    assert not WorkloadInjector.handles("pod")
    assert not WorkloadInjector.handles("pod.delete")
    assert not WorkloadInjector.handles("network.latency")
    # Prefix match must not over-match a sibling that just starts with "workload".
    assert not WorkloadInjector.handles("workloads")


@pytest.mark.asyncio
async def test_scale_missing_replicas_raises_precondition() -> None:
    kubectl, _, _ = _make_kubectl()
    injector = WorkloadInjector(kubectl)

    spec = FaultSpec(
        fault_id="workload.scale",
        target={"ns": "default", "deployment": "aiperf-mock-server"},
    )
    with pytest.raises(FaultPreconditionError, match="replicas"):
        await injector.inject(spec)


@pytest.mark.asyncio
async def test_set_env_missing_env_var_raises_precondition() -> None:
    kubectl, _, _ = _make_kubectl()
    injector = WorkloadInjector(kubectl)

    spec = FaultSpec(
        fault_id="workload.set_env",
        params={"value": "bar"},
        target={"ns": "default", "deployment": "aiperf-mock-server"},
    )
    with pytest.raises(FaultPreconditionError, match="env_var"):
        await injector.inject(spec)
