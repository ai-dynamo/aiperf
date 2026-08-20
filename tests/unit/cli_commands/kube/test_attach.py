# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for `aiperf kube attach` cyclopts subcommand wiring.

Focus is on:
- module exposes `app` cyclopts.App; subcommand registered in `aiperf kube`
- `attach` callable signature accepts the documented flags
- end-to-end: monkeypatched cli_helpers.resolve_job + kube_attach.attach_to_benchmark
  receive the right kwargs and propagation of manage_options/port works
- when resolve_job returns None, attach_to_benchmark is never called
- underlying errors surface through cli_utils.exit_on_error as SystemExit
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.config.kube import KubeManageOptions


def _make_resolved(
    job_id: str = "job-abc",
    namespace: str = "ns-1",
    api: str = "http://localhost:38465",
    phase: str = "Running",
) -> Any:
    """Build a minimal resolved-job object matching ResolvedJob's surface."""
    job_info = MagicMock()
    job_info.phase = phase
    resolved = MagicMock()
    resolved.job_id = job_id
    resolved.namespace = namespace
    resolved.api = api
    resolved.job_info = job_info
    resolved.aclose = AsyncMock()
    return resolved


def test_attach_module_importable() -> None:
    """The attach module must be importable and expose an `app` attribute."""
    from aiperf.cli_commands.kube import attach as attach_mod

    assert hasattr(attach_mod, "app"), "attach.app (cyclopts App) must be defined"


def test_attach_registered_in_kube_app() -> None:
    """The `attach` subcommand must be wired into `aiperf kube`."""
    from aiperf.cli_commands.kube._app import app

    assert "attach" in set(app)


class TestAttachCallableSignature:
    """`attach` must accept the documented CLI flags as kwargs."""

    @pytest.mark.parametrize(
        "param_name",
        [
            "job_id",
            "manage_options",
            "port",
        ],
    )  # fmt: skip
    def test_signature_has_param(self, param_name: str) -> None:
        from aiperf.cli_commands.kube.attach import attach

        sig = inspect.signature(attach)
        assert param_name in sig.parameters

    def test_signature_defaults(self) -> None:
        from aiperf.cli_commands.kube.attach import attach

        sig = inspect.signature(attach)
        assert sig.parameters["job_id"].default is None
        assert sig.parameters["manage_options"].default is None
        assert sig.parameters["port"].default == 0


class TestAttachDispatch:
    """End-to-end dispatch: resolve_job -> attach_to_benchmark."""

    @pytest.mark.asyncio
    async def test_happy_path_calls_attach_to_benchmark_with_resolved_fields(
        self,
    ) -> None:
        """All resolved fields propagate to attach_to_benchmark, plus port + manage opts."""
        from aiperf.cli_commands.kube.attach import attach

        resolved = _make_resolved(
            job_id="real-job",
            namespace="real-ns",
            api="http://op:38465",
            phase="Profiling",
        )
        opts = KubeManageOptions(
            kubeconfig="/tmp/kc",
            kube_context="ctx-1",
            namespace="passthru-ns",
        )

        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_job",
                new=AsyncMock(return_value=resolved),
            ) as mock_resolve,
            patch(
                "aiperf.kubernetes.attach.attach_to_benchmark",
                new=AsyncMock(),
            ) as mock_attach,
        ):
            await attach(job_id="real-job", manage_options=opts, port=9091)

        # resolve_job receives the raw inputs from the CLI flags
        mock_resolve.assert_awaited_once()
        call_args = mock_resolve.await_args
        # job_id, namespace are positional in resolve_job(job_id, namespace, ...)
        assert call_args.args[0] == "real-job"
        assert call_args.args[1] == "passthru-ns"
        assert call_args.kwargs.get("kubeconfig") == "/tmp/kc"
        assert call_args.kwargs.get("kube_context") == "ctx-1"

        # attach_to_benchmark receives the resolved fields, port and kubeconfig/context
        mock_attach.assert_awaited_once()
        attach_args = mock_attach.await_args
        assert attach_args.args[0] == "real-job"
        assert attach_args.args[1] == "real-ns"
        assert attach_args.args[2] == 9091
        assert attach_args.args[3] == "http://op:38465"
        assert attach_args.kwargs.get("phase") == "Profiling"
        assert attach_args.kwargs.get("kubeconfig") == "/tmp/kc"
        assert attach_args.kwargs.get("kube_context") == "ctx-1"
        resolved.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_manage_options_uses_defaults(self) -> None:
        """When manage_options is None, KubeManageOptions() defaults are passed through."""
        from aiperf.cli_commands.kube.attach import attach

        resolved = _make_resolved()
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_job",
                new=AsyncMock(return_value=resolved),
            ) as mock_resolve,
            patch(
                "aiperf.kubernetes.attach.attach_to_benchmark",
                new=AsyncMock(),
            ),
        ):
            await attach()

        kwargs = mock_resolve.await_args.kwargs
        assert kwargs.get("kubeconfig") is None
        assert kwargs.get("kube_context") is None
        # namespace positional default = None when manage_options unset
        assert mock_resolve.await_args.args[1] is None

    @pytest.mark.asyncio
    async def test_unresolved_job_skips_attach(self) -> None:
        """resolve_job returning None must short-circuit before attach_to_benchmark."""
        from aiperf.cli_commands.kube.attach import attach

        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_job",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "aiperf.kubernetes.attach.attach_to_benchmark",
                new=AsyncMock(),
            ) as mock_attach,
        ):
            await attach(job_id="missing")

        mock_attach.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_default_port_zero_propagates(self) -> None:
        """Default port=0 (ephemeral) is passed through to attach_to_benchmark."""
        from aiperf.cli_commands.kube.attach import attach

        resolved = _make_resolved()
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_job",
                new=AsyncMock(return_value=resolved),
            ),
            patch(
                "aiperf.kubernetes.attach.attach_to_benchmark",
                new=AsyncMock(),
            ) as mock_attach,
        ):
            await attach()

        # third positional is the port
        assert mock_attach.await_args.args[2] == 0


class TestAttachErrorWrapping:
    """`exit_on_error` must catch underlying exceptions and exit cleanly."""

    @pytest.mark.asyncio
    async def test_resolve_error_becomes_system_exit(self) -> None:
        """A RuntimeError from resolve_job -> SystemExit with code 1."""
        from aiperf.cli_commands.kube.attach import attach

        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_job",
                new=AsyncMock(side_effect=RuntimeError("forbidden")),
            ),
            patch("aiperf.cli_utils.console"),
            pytest.raises(SystemExit) as exc_info,
        ):
            await attach()

        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_attach_error_becomes_system_exit(self) -> None:
        """A RuntimeError from attach_to_benchmark -> SystemExit, not raw traceback."""
        from aiperf.cli_commands.kube.attach import attach

        resolved = _make_resolved()
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_job",
                new=AsyncMock(return_value=resolved),
            ),
            patch(
                "aiperf.kubernetes.attach.attach_to_benchmark",
                new=AsyncMock(side_effect=RuntimeError("dropped")),
            ),
            patch("aiperf.cli_utils.console"),
            pytest.raises(SystemExit) as exc_info,
        ):
            await attach()

        assert exc_info.value.code == 1
