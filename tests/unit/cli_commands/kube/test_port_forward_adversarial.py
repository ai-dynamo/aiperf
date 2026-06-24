# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for Kubernetes port-forward helpers and CLI wiring.

Focuses on:
- kubectl argv construction across kubeconfig/context and hostile identifiers.
- timeout and stderr diagnostics for port-forward startup failures.
- API readiness verification restart boundaries and cleanup.
- process termination cleanup for context-manager and dashboard reconnect paths.
- liveness monitor context propagation for auto-close behavior.

Out of scope: live Kubernetes port binding and browser behavior; those require an
integration cluster or the existing dashboard smoke tests.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest
from pytest import param

from aiperf.cli_commands.kube.dashboard import _serve_dashboard
from aiperf.config.kube import KubeManageOptions
from aiperf.kubernetes import port_forward as pf
from aiperf.kubernetes.subproc import CommandResult

# ============================================================
# Helpers
# ============================================================


class _ReadableStream:
    """Async byte stream exposing the subprocess stdout/stderr surface used here."""

    def __init__(
        self, *, lines: list[bytes] | None = None, read_bytes: bytes = b""
    ) -> None:
        self._lines = list(lines or [])
        self._read_bytes = read_bytes

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""

    async def read(self) -> bytes:
        return self._read_bytes


class _HangingStdout:
    """Stdout stream that never emits kubectl's ready marker."""

    async def readline(self) -> bytes:
        await asyncio.Future()
        return b""


@dataclass(slots=True)
class _FakeProcess:
    """Minimal asyncio subprocess surface consumed by port_forward helpers."""

    stdout: _ReadableStream | _HangingStdout | None = None
    stderr: _ReadableStream | None = None
    returncode: int | None = None
    wait_result: int | BaseException = 0
    terminate_count: int = 0
    kill_count: int = 0

    def terminate(self) -> None:
        self.terminate_count += 1
        self.returncode = -15

    def kill(self) -> None:
        self.kill_count += 1
        self.returncode = -9

    async def wait(self) -> int:
        if isinstance(self.wait_result, BaseException):
            raise self.wait_result
        self.returncode = self.wait_result
        return self.wait_result


def _ready_process(port: int = 31817) -> _FakeProcess:
    return _FakeProcess(
        stdout=_ReadableStream(
            lines=[f"Forwarding from 127.0.0.1:{port} -> 8081\n".encode()]
        ),
        stderr=_ReadableStream(read_bytes=b""),
    )


@pytest.fixture
def fast_port_forward_retries() -> Iterator[None]:
    """Shrink retry counts without changing production defaults globally."""
    with (
        patch.object(pf, "_API_MAX_RETRIES", 2),
        patch.object(pf, "_API_RETRY_DELAY", 0.0),
        patch.object(pf, "_API_INITIAL_DELAY", 0.0),
    ):
        yield


# ============================================================
# kubectl argv construction and startup diagnostics
# ============================================================


class TestPortForwardProcessStartup:
    """Port-forward startup treats Kubernetes values as argv cells, not shell text."""

    @pytest.mark.asyncio
    async def test_start_port_forward_process_hostile_identifiers_remain_argv_cells(
        self,
    ) -> None:
        proc = _ready_process(port=31991)
        with patch(
            "aiperf.kubernetes.subproc.start_streaming_process",
            new=AsyncMock(return_value=proc),
        ) as mock_start:
            _, actual_port = await pf._start_port_forward_process(
                "tenant-a; kubectl delete ns prod",
                "aiperf-operator-7f2a && curl attacker",
                0,
                8081,
                timeout=1.0,
                kubeconfig="/secure/kubeconfigs/dgx prod.yaml; rm -rf /",
                kube_context="dgx-prod-admin$(id)",
            )

        assert actual_port == 31991
        assert mock_start.await_args.args == (
            [
                "kubectl",
                "port-forward",
                "-n",
                "tenant-a; kubectl delete ns prod",
                "pod/aiperf-operator-7f2a && curl attacker",
                "0:8081",
                "--kubeconfig",
                "/secure/kubeconfigs/dgx prod.yaml; rm -rf /",
                "--context",
                "dgx-prod-admin$(id)",
            ],
        )

    @pytest.mark.asyncio
    async def test_start_port_forward_process_timeout_reports_stderr_and_port_hint(
        self,
    ) -> None:
        proc = _FakeProcess(
            stdout=_HangingStdout(),
            stderr=_ReadableStream(
                read_bytes=b"error: unable to listen on any requested ports: address already in use"
            ),
        )
        with (
            patch(
                "aiperf.kubernetes.subproc.start_streaming_process",
                new=AsyncMock(return_value=proc),
            ),
            patch(
                "aiperf.kubernetes.subproc.terminate_process", new=AsyncMock()
            ) as mock_terminate,
            pytest.raises(
                RuntimeError,
                match=r"(?s)Port-forward did not become ready within 0\.0s.*address already in use.*port 19090",
            ),
        ):
            await pf._start_port_forward_process(
                "bench-prod",
                "aiperf-operator-7f2a",
                19090,
                8081,
                timeout=0.0,
            )

        mock_terminate.assert_awaited_once_with(proc)

    @pytest.mark.asyncio
    async def test_start_port_forward_process_early_exit_reports_kubectl_stderr(
        self,
    ) -> None:
        proc = _FakeProcess(
            stdout=_ReadableStream(lines=[]),
            stderr=_ReadableStream(read_bytes=b'pods "aiperf-operator-7f2a" not found'),
            returncode=1,
        )
        with (
            patch(
                "aiperf.kubernetes.subproc.start_streaming_process",
                new=AsyncMock(return_value=proc),
            ),
            pytest.raises(
                RuntimeError,
                match=r"Port-forward exited unexpectedly: pods .*aiperf-operator-7f2a.* not found",
            ),
        ):
            await pf._start_port_forward_process(
                "bench-prod",
                "aiperf-operator-7f2a",
                0,
                8081,
                timeout=1.0,
            )


# ============================================================
# API readiness verification and cleanup boundaries
# ============================================================


class TestApiReadinessVerification:
    """API verification restarts stale port-forwards and preserves cluster selection."""

    @pytest.mark.asyncio
    async def test_start_port_forward_api_failure_restarts_with_kubeconfig_and_context(
        self, fast_port_forward_retries: None
    ) -> None:
        first_proc = _ready_process(port=32001)
        second_proc = _ready_process(port=32002)
        with (
            patch.object(
                pf,
                "_start_port_forward_process",
                new=AsyncMock(side_effect=[(first_proc, 32001), (second_proc, 32002)]),
            ) as mock_start,
            patch.object(
                pf,
                "_wait_for_api_ready",
                new=AsyncMock(side_effect=[RuntimeError("connection refused"), None]),
            ) as mock_wait,
            patch.object(pf, "cleanup_port_forward", new=AsyncMock()) as mock_cleanup,
        ):
            proc, actual_port = await pf.start_port_forward(
                "bench-prod",
                "aiperf-operator-7f2a",
                0,
                8081,
                timeout=30.0,
                verify_api=True,
                kubeconfig="/secure/kubeconfigs/dgx-prod.yaml",
                kube_context="dgx-prod-admin",
            )

        assert (proc, actual_port) == (second_proc, 32002)
        assert mock_wait.await_args_list[0].args == (32001, first_proc)
        assert mock_wait.await_args_list[1].args == (32002, second_proc)
        assert mock_cleanup.await_args_list[0].args == (first_proc,)
        assert mock_start.await_args_list[1].kwargs["timeout"] > 0.0
        assert mock_start.await_args_list[1].kwargs["timeout"] <= 30.0
        assert (
            mock_start.await_args_list[1].kwargs["kubeconfig"]
            == "/secure/kubeconfigs/dgx-prod.yaml"
        )
        assert mock_start.await_args_list[1].kwargs["kube_context"] == "dgx-prod-admin"
        assert mock_start.await_args_list[1].args == (
            "bench-prod",
            "aiperf-operator-7f2a",
            0,
            8081,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status", "should_return"),
        [
            (200, True),
            (404, True),
            param(503, False, id="server-error-keeps-polling-until-budget"),
        ],
    )  # fmt: skip
    async def test_wait_for_api_ready_status_boundary_returns_only_for_ready_api(
        self, status: int, should_return: bool
    ) -> None:
        class _Response:
            def __init__(self, response_status: int) -> None:
                self.status = response_status

            async def __aenter__(self) -> _Response:
                return self

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: object | None,
            ) -> None:
                return None

        class _Session:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self._calls = 0

            async def __aenter__(self) -> _Session:
                return self

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: object | None,
            ) -> None:
                return None

            def get(self, url: str) -> _Response:
                self._calls += 1
                if self._calls > 1:
                    proc.returncode = 0
                return _Response(status)

        proc = _FakeProcess(stderr=_ReadableStream(read_bytes=b"api container exited"))
        with (
            patch("aiohttp.ClientSession", new=_Session),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=None,
            ),
            patch.object(pf, "_API_INITIAL_DELAY", 0.0),
        ):
            if should_return:
                await pf._wait_for_api_ready(32080, proc, check_interval=0.0)
            else:
                with pytest.raises(
                    RuntimeError,
                    match=r"Port-forward process exited \(code 0\).*api container exited",
                ):
                    await pf._wait_for_api_ready(32080, proc, check_interval=0.0)

    @pytest.mark.asyncio
    async def test_start_port_forward_verify_api_false_skips_http_readiness_probe(
        self,
    ) -> None:
        proc = _ready_process(port=32003)
        with (
            patch.object(
                pf,
                "_start_port_forward_process",
                new=AsyncMock(return_value=(proc, 32003)),
            ),
            patch.object(pf, "_wait_for_api_ready", new=AsyncMock()) as mock_wait,
        ):
            returned_proc, actual_port = await pf.start_port_forward(
                "bench-prod",
                "aiperf-operator-7f2a",
                verify_api=False,
            )

        assert (returned_proc, actual_port) == (proc, 32003)
        mock_wait.assert_not_awaited()


# ============================================================
# Cleanup, liveness monitor, and dashboard reconnect CLI behavior
# ============================================================


class TestPortForwardCleanupAndReconnect:
    """Cleanup paths terminate stale kubectl processes before reconnecting."""

    @pytest.mark.asyncio
    async def test_monitor_pod_liveness_missing_pod_terminates_with_context_args(
        self,
    ) -> None:
        proc = _FakeProcess()
        with patch(
            "aiperf.kubernetes.subproc.run_command",
            new=AsyncMock(return_value=CommandResult(1, "", "not found")),
        ) as mock_run:
            await pf._monitor_pod_liveness(
                "bench-prod",
                "aiperf-operator-7f2a",
                proc,
                check_interval=0.0,
                kubeconfig="/secure/kubeconfigs/dgx-prod.yaml",
                kube_context="dgx-prod-admin",
            )

        assert proc.terminate_count == 1
        assert mock_run.await_args.args == (
            [
                "kubectl",
                "get",
                "pod",
                "aiperf-operator-7f2a",
                "-n",
                "bench-prod",
                "-o",
                "name",
                "--kubeconfig",
                "/secure/kubeconfigs/dgx-prod.yaml",
                "--context",
                "dgx-prod-admin",
            ],
        )
        assert mock_run.await_args.kwargs == {"timeout": 0.0}

    @pytest.mark.asyncio
    async def test_port_forward_to_controller_body_exception_still_cleans_process(
        self,
    ) -> None:
        proc = _ready_process(port=32100)
        with (
            patch.object(
                pf, "start_port_forward", new=AsyncMock(return_value=(proc, 32100))
            ),
            patch.object(pf, "cleanup_port_forward", new=AsyncMock()) as mock_cleanup,
            pytest.raises(
                RuntimeError, match="consumer failed while reading dashboard"
            ),
        ):
            async with pf.port_forward_to_controller(
                "bench-prod",
                "aiperf-operator-7f2a",
                kubeconfig="/secure/kubeconfigs/dgx-prod.yaml",
                kube_context="dgx-prod-admin",
            ):
                raise RuntimeError("consumer failed while reading dashboard")

        mock_cleanup.assert_awaited_once_with(proc)

    @pytest.mark.asyncio
    async def test_dashboard_reconnect_pins_port_and_propagates_context_each_attempt(
        self,
    ) -> None:
        first_proc = _FakeProcess(returncode=0, wait_result=0)
        second_proc = _FakeProcess(wait_result=asyncio.CancelledError())
        opts = KubeManageOptions(
            namespace="tenant-a",
            kubeconfig="/secure/kubeconfigs/dgx-prod.yaml",
            kube_context="dgx-prod-admin",
        )
        with (
            patch(
                "aiperf.kubernetes.port_forward.start_port_forward",
                new=AsyncMock(side_effect=[(first_proc, 32111), (second_proc, 32111)]),
            ) as mock_start,
            patch(
                "aiperf.kubernetes.port_forward.cleanup_port_forward",
                new=AsyncMock(),
            ) as mock_cleanup,
            patch(
                "aiperf.cli_commands.kube.dashboard._refresh_operator_pod",
                new=AsyncMock(return_value="aiperf-operator-9b6c"),
            ) as mock_refresh,
            patch("webbrowser.open"),
        ):
            await _serve_dashboard(
                opts,
                "aiperf-system",
                "aiperf-operator-7f2a",
                port=0,
                no_browser=True,
            )

        assert mock_start.await_args_list[0].args[:4] == (
            "aiperf-system",
            "aiperf-operator-7f2a",
            0,
            8081,
        )
        assert mock_start.await_args_list[1].args[:4] == (
            "aiperf-system",
            "aiperf-operator-9b6c",
            32111,
            8081,
        )
        for call in mock_start.await_args_list:
            assert call.kwargs == {
                "verify_api": True,
                "kubeconfig": "/secure/kubeconfigs/dgx-prod.yaml",
                "kube_context": "dgx-prod-admin",
            }
        assert mock_cleanup.await_count == 2
        assert mock_refresh.await_args.args == (opts, "aiperf-system")
        assert mock_refresh.await_args.kwargs == {"fallback": "aiperf-operator-7f2a"}
