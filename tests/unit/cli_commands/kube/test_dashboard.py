# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf kube dashboard command."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.cli_commands.kube.dashboard import dashboard
from aiperf.config.kube import KubeManageOptions
from aiperf.kubernetes.enums import PodPhase
from aiperf.kubernetes.environment import K8sEnvironment
from aiperf.kubernetes.results import RESULTS_SERVER_PORT


@asynccontextmanager
async def _fake_k8s_client(api: Any):
    """Drop-in replacement for ``k8s_client`` yielding the provided api object."""
    yield api


def _make_proc(returncode: int = 0) -> MagicMock:
    """Build a mock kubectl port-forward subprocess that exits via CancelledError."""
    proc = MagicMock()
    proc.wait = AsyncMock(side_effect=asyncio.CancelledError)
    proc.returncode = returncode
    return proc


@pytest.fixture
def manage_options():
    return KubeManageOptions(kubeconfig=None, namespace=None)


@pytest.fixture
def patched_k8s():
    """Patch k8s_client and find_operator_pod; yields the find_operator_pod mock."""
    api = MagicMock()
    mock_find = AsyncMock(return_value=("aiperf-operator-abc", PodPhase.RUNNING))

    async def _fake_resolve(_api, *, explicit, default="aiperf-system"):
        return explicit if explicit is not None else default

    with (
        patch(
            "aiperf.kubernetes.client.k8s_client",
            side_effect=lambda *_a, **_k: _fake_k8s_client(api),
        ),
        patch(
            "aiperf.kubernetes.client.find_operator_pod",
            new=mock_find,
        ),
        patch(
            "aiperf.kubernetes.client.resolve_operator_namespace",
            new=_fake_resolve,
        ),
    ):
        yield mock_find


class TestDashboardCommand:
    """Tests for the kube dashboard command."""

    async def test_dashboard_opens_browser(
        self, patched_k8s: AsyncMock, manage_options: KubeManageOptions
    ) -> None:
        """Test dashboard port-forwards and opens browser."""
        mock_start = AsyncMock(return_value=(_make_proc(), 54321))
        mock_cleanup = AsyncMock()
        with (
            patch(
                "aiperf.kubernetes.port_forward.start_port_forward",
                new=mock_start,
            ),
            patch(
                "aiperf.kubernetes.port_forward.cleanup_port_forward",
                new=mock_cleanup,
            ),
            patch("webbrowser.open") as mock_browser,
        ):
            await dashboard(manage_options=manage_options)

        mock_start.assert_called_once()
        args = mock_start.call_args[0]
        assert args[0] == "aiperf-system"
        assert args[1] == "aiperf-operator-abc"
        assert args[3] == RESULTS_SERVER_PORT
        # Bug 4: verify_api=True so the browser only opens once the operator
        # API is actually serving traffic.
        assert mock_start.call_args.kwargs["verify_api"] is True

        mock_browser.assert_called_once_with("http://localhost:54321")

    async def test_dashboard_no_browser_flag(
        self, patched_k8s: AsyncMock, manage_options: KubeManageOptions
    ) -> None:
        """Test --no-browser skips opening the browser."""
        mock_start = AsyncMock(return_value=(_make_proc(), 54321))
        mock_cleanup = AsyncMock()
        with (
            patch(
                "aiperf.kubernetes.port_forward.start_port_forward",
                new=mock_start,
            ),
            patch(
                "aiperf.kubernetes.port_forward.cleanup_port_forward",
                new=mock_cleanup,
            ),
            patch("webbrowser.open") as mock_browser,
        ):
            await dashboard(manage_options=manage_options, no_browser=True)

        mock_browser.assert_not_called()

    async def test_dashboard_operator_not_found(
        self, patched_k8s: AsyncMock, manage_options: KubeManageOptions
    ) -> None:
        """Test dashboard exits gracefully when operator pod not found."""
        patched_k8s.return_value = None

        with patch("webbrowser.open") as mock_browser:
            await dashboard(manage_options=manage_options)

        mock_browser.assert_not_called()

    async def test_dashboard_custom_port(
        self, patched_k8s: AsyncMock, manage_options: KubeManageOptions
    ) -> None:
        """Test dashboard with custom local port."""
        mock_start = AsyncMock(return_value=(_make_proc(), 8081))
        mock_cleanup = AsyncMock()
        with (
            patch(
                "aiperf.kubernetes.port_forward.start_port_forward",
                new=mock_start,
            ),
            patch(
                "aiperf.kubernetes.port_forward.cleanup_port_forward",
                new=mock_cleanup,
            ),
            patch("webbrowser.open"),
        ):
            await dashboard(manage_options=manage_options, port=8081)

        assert mock_start.call_args[0][2] == 8081

    async def test_dashboard_custom_operator_namespace(
        self, patched_k8s: AsyncMock, manage_options: KubeManageOptions
    ) -> None:
        """Test dashboard with custom operator namespace."""
        mock_start = AsyncMock(return_value=(_make_proc(), 54321))
        mock_cleanup = AsyncMock()
        with (
            patch(
                "aiperf.kubernetes.port_forward.start_port_forward",
                new=mock_start,
            ),
            patch(
                "aiperf.kubernetes.port_forward.cleanup_port_forward",
                new=mock_cleanup,
            ),
            patch("webbrowser.open"),
        ):
            await dashboard(
                manage_options=manage_options,
                operator_namespace="custom-ns",
            )

        assert mock_start.call_args[0][0] == "custom-ns"
        patched_k8s.assert_called_once()
        call_kwargs = patched_k8s.call_args
        assert call_kwargs.kwargs["namespace"] == "custom-ns"

    async def test_dashboard_reconnects_with_pinned_port(
        self, patched_k8s: AsyncMock, manage_options: KubeManageOptions
    ) -> None:
        """After a disconnect, second start_port_forward call binds the same port."""
        first_proc = MagicMock()
        first_proc.wait = AsyncMock(return_value=0)  # disconnects normally
        first_proc.returncode = 0
        second_proc = _make_proc()  # exits the loop via CancelledError

        mock_start = AsyncMock(side_effect=[(first_proc, 54321), (second_proc, 54321)])
        mock_cleanup = AsyncMock()
        with (
            patch(
                "aiperf.kubernetes.port_forward.start_port_forward",
                new=mock_start,
            ),
            patch(
                "aiperf.kubernetes.port_forward.cleanup_port_forward",
                new=mock_cleanup,
            ),
            patch("webbrowser.open"),
            patch("asyncio.sleep", new=AsyncMock()),
        ):
            await dashboard(manage_options=manage_options)

        assert mock_start.call_count == 2
        # First call uses the user-supplied port (0 = ephemeral).
        assert mock_start.call_args_list[0][0][2] == 0
        # Second call pins the local port that kubectl handed back (54321).
        assert mock_start.call_args_list[1][0][2] == 54321

    async def test_dashboard_retries_with_configured_backoff(
        self,
        patched_k8s: AsyncMock,
        manage_options: KubeManageOptions,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Port-forward retries use the configured multiplier and cap."""
        settings = K8sEnvironment.PORT_FORWARD
        monkeypatch.setattr(settings, "RECONNECT_INITIAL_BACKOFF_SECONDS", 4.0)
        monkeypatch.setattr(settings, "RECONNECT_BACKOFF_MULTIPLIER", 3.0)
        monkeypatch.setattr(settings, "RECONNECT_MAX_BACKOFF_SECONDS", 10.0)
        success_proc = _make_proc()
        mock_start = AsyncMock(
            side_effect=[
                RuntimeError("kubectl refused"),
                RuntimeError("kubectl still refused"),
                (success_proc, 54321),
            ]
        )
        mock_cleanup = AsyncMock()
        sleep = AsyncMock()
        with (
            patch(
                "aiperf.kubernetes.port_forward.start_port_forward",
                new=mock_start,
            ),
            patch(
                "aiperf.kubernetes.port_forward.cleanup_port_forward",
                new=mock_cleanup,
            ),
            patch("webbrowser.open") as mock_browser,
            patch("asyncio.sleep", new=sleep),
        ):
            await dashboard(manage_options=manage_options)

        assert mock_start.call_count == 3
        assert [await_call.args[0] for await_call in sleep.await_args_list] == [4.0, 10.0]
        mock_browser.assert_called_once_with("http://localhost:54321")
