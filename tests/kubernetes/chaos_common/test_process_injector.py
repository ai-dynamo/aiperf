# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for :py:class:`ProcessInjector`.

Uses a mocked :py:class:`KubectlClient` -- no cluster required. Tests assert
the exact ``kubectl exec ... -- kill -<SIG> <pid>`` argv we send, plus restore
semantics for each ``_RESTORE_SIGNAL_MAP`` branch (SIGSTOP→SIGCONT, SIGKILL
no-op).
"""

from __future__ import annotations

import subprocess
from unittest.mock import AsyncMock

import pytest

from tests.kubernetes.chaos_common.base import (
    FaultPreconditionError,
    FaultSpec,
)
from tests.kubernetes.chaos_common.injectors.process import ProcessInjector


def _make_kubectl_mock() -> AsyncMock:
    """Return an :py:class:`AsyncMock` shaped like :py:class:`KubectlClient`."""
    kubectl = AsyncMock()
    kubectl.run = AsyncMock(
        return_value=subprocess.CompletedProcess(
            args=["kubectl"], returncode=0, stdout="", stderr=""
        )
    )
    return kubectl


def _make_spec(**overrides: object) -> FaultSpec:
    """Build a baseline ``process.signal`` spec; per-test overrides via kwargs."""
    target: dict[str, object] = {
        "kind": "pod",
        "ns": "aiperf-test-abc",
        "pod": "worker-0",
        "container": "engine",
        "pid": 12345,
    }
    params: dict[str, object] = {"signal": "SIGSTOP"}
    if "target" in overrides:
        target = {**target, **overrides.pop("target")}  # type: ignore[arg-type]
    if "params" in overrides:
        params = {**params, **overrides.pop("params")}  # type: ignore[arg-type]
    return FaultSpec(fault_id="process.signal", target=target, params=params)


@pytest.mark.asyncio
async def test_process_signal_sigstop_runs_kubectl_exec_kill_stop() -> None:
    kubectl = _make_kubectl_mock()
    injector = ProcessInjector(kubectl)

    applied = await injector.inject(_make_spec())

    kubectl.run.assert_awaited_once_with(
        "exec",
        "worker-0",
        "-c",
        "engine",
        "-n",
        "aiperf-test-abc",
        "--",
        "kill",
        "-SIGSTOP",
        "12345",
        check=True,
    )
    assert applied.metadata["pid"] == 12345
    assert applied.metadata["signal"] == "SIGSTOP"


@pytest.mark.asyncio
async def test_process_signal_sigstop_restore_sends_sigcont() -> None:
    kubectl = _make_kubectl_mock()
    injector = ProcessInjector(kubectl)

    applied = await injector.inject(_make_spec())
    kubectl.run.reset_mock()

    await applied.restore()

    kubectl.run.assert_awaited_once_with(
        "exec",
        "worker-0",
        "-c",
        "engine",
        "-n",
        "aiperf-test-abc",
        "--",
        "kill",
        "-SIGCONT",
        "12345",
        check=True,
    )
    # Idempotent: second restore is a no-op (no additional kubectl call).
    await applied.restore()
    kubectl.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_signal_sigkill_restore_is_noop() -> None:
    kubectl = _make_kubectl_mock()
    injector = ProcessInjector(kubectl)

    applied = await injector.inject(_make_spec(params={"signal": "SIGKILL"}))
    kubectl.run.reset_mock()

    await applied.restore()

    kubectl.run.assert_not_awaited()
    assert applied.metadata["restored"] is True


@pytest.mark.asyncio
async def test_process_signal_missing_pid_raises_precondition() -> None:
    kubectl = _make_kubectl_mock()
    injector = ProcessInjector(kubectl)

    spec = FaultSpec(
        fault_id="process.signal",
        target={
            "kind": "pod",
            "ns": "aiperf-test-abc",
            "pod": "worker-0",
            "container": "engine",
            # pid intentionally omitted
        },
        params={"signal": "SIGSTOP"},
    )
    with pytest.raises(FaultPreconditionError, match="pid"):
        await injector.inject(spec)


@pytest.mark.asyncio
async def test_process_signal_unsupported_kind_raises_with_phase_reference() -> None:
    kubectl = _make_kubectl_mock()
    injector = ProcessInjector(kubectl)

    spec = _make_spec(target={"kind": "managed_process"})

    with pytest.raises(FaultPreconditionError) as exc_info:
        await injector.inject(spec)

    msg = str(exc_info.value)
    assert "managed_process" in msg
    assert "Phase" in msg
    assert "deferred" in msg


@pytest.mark.asyncio
async def test_process_signal_malformed_signal_name_raises_precondition() -> None:
    kubectl = _make_kubectl_mock()
    injector = ProcessInjector(kubectl)

    spec = _make_spec(params={"signal": "usr1"})

    with pytest.raises(FaultPreconditionError, match="signal"):
        await injector.inject(spec)


def test_handles_prefix_match_process() -> None:
    assert ProcessInjector.handles("process") is True
    assert ProcessInjector.handles("process.signal") is True
    assert ProcessInjector.handles("pod") is False
    assert ProcessInjector.handles("network") is False
    # Prefix safety: "processx.signal" must NOT match the "process" namespace.
    assert ProcessInjector.handles("processx.signal") is False
