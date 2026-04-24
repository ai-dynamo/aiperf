# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the `aiperf kube watch` CLI command.

Covers the renderer factory (rich/text/json) and orchestrator wiring:
the watch command must build the right renderer class from `--output`,
then pass through `manage_options` / `all_jobs` / `interval` /
`follow_logs` into `WatchOrchestrator`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.cli_commands.kube.watch import _build_renderer, watch
from aiperf.config.kube import KubeManageOptions
from aiperf.kubernetes.watch_render_json import JsonRenderer
from aiperf.kubernetes.watch_render_rich import RichRenderer
from aiperf.kubernetes.watch_render_text import TextRenderer


class TestBuildRenderer:
    """Tests for the `_build_renderer` output->renderer factory."""

    @pytest.mark.parametrize(
        "output, expected_cls",
        [
            param("rich", RichRenderer, id="rich"),
            param("text", TextRenderer, id="text"),
            param("json", JsonRenderer, id="json"),
            param("anything-else", RichRenderer, id="unknown-defaults-to-rich"),
            param("", RichRenderer, id="empty-defaults-to-rich"),
        ],
    )  # fmt: skip
    def test_factory_maps_output_to_renderer(
        self, output: str, expected_cls: type
    ) -> None:
        """_build_renderer returns the right concrete renderer by --output."""
        renderer = _build_renderer(output)
        assert isinstance(renderer, expected_cls)


class TestWatchCommandWiring:
    """Tests for the top-level `watch` CLI entry point."""

    async def test_watch_builds_orchestrator_with_rich_renderer_default(
        self,
    ) -> None:
        """Default output=rich builds a RichRenderer and runs the orchestrator."""
        instance = MagicMock()
        instance.run = AsyncMock()
        with patch(
            "aiperf.kubernetes.watch_orchestrator.WatchOrchestrator",
            return_value=instance,
        ) as mock_ctor:
            await watch(manage_options=KubeManageOptions())

        kwargs = mock_ctor.call_args.kwargs
        assert isinstance(kwargs["renderer"], RichRenderer)
        assert kwargs["all_jobs"] is False
        assert kwargs["interval"] == 2.0
        assert kwargs["follow_logs"] is False
        instance.run.assert_awaited_once()

    async def test_watch_json_output_uses_json_renderer(self) -> None:
        """`--output json` constructs JsonRenderer."""
        instance = MagicMock()
        instance.run = AsyncMock()
        with patch(
            "aiperf.kubernetes.watch_orchestrator.WatchOrchestrator",
            return_value=instance,
        ) as mock_ctor:
            await watch(manage_options=KubeManageOptions(), output="json")

        assert isinstance(mock_ctor.call_args.kwargs["renderer"], JsonRenderer)

    async def test_watch_text_output_uses_text_renderer(self) -> None:
        """`--output text` constructs TextRenderer."""
        instance = MagicMock()
        instance.run = AsyncMock()
        with patch(
            "aiperf.kubernetes.watch_orchestrator.WatchOrchestrator",
            return_value=instance,
        ) as mock_ctor:
            await watch(manage_options=KubeManageOptions(), output="text")

        assert isinstance(mock_ctor.call_args.kwargs["renderer"], TextRenderer)

    async def test_watch_forwards_job_id_namespace_and_flags(self) -> None:
        """job_id + manage_options + flags pass through to the orchestrator ctor."""
        instance = MagicMock()
        instance.run = AsyncMock()
        opts = KubeManageOptions(
            kubeconfig="/tmp/kc.yaml",
            kube_context="dev",
            namespace="my-ns",
        )
        with patch(
            "aiperf.kubernetes.watch_orchestrator.WatchOrchestrator",
            return_value=instance,
        ) as mock_ctor:
            await watch(
                job_id="abc123",
                manage_options=opts,
                all_jobs=True,
                interval=0.5,
                follow_logs=True,
            )

        kwargs = mock_ctor.call_args.kwargs
        assert kwargs["job_id"] == "abc123"
        assert kwargs["namespace"] == "my-ns"
        assert kwargs["kubeconfig"] == "/tmp/kc.yaml"
        assert kwargs["kube_context"] == "dev"
        assert kwargs["all_jobs"] is True
        assert kwargs["interval"] == 0.5
        assert kwargs["follow_logs"] is True

    async def test_watch_default_manage_options_when_omitted(self) -> None:
        """Omitting manage_options yields default KubeManageOptions()."""
        instance = MagicMock()
        instance.run = AsyncMock()
        with patch(
            "aiperf.kubernetes.watch_orchestrator.WatchOrchestrator",
            return_value=instance,
        ) as mock_ctor:
            await watch(output="text")

        kwargs = mock_ctor.call_args.kwargs
        assert kwargs["kubeconfig"] is None
        assert kwargs["kube_context"] is None
        assert kwargs["namespace"] is None


class TestWatchCommandErrorSurface:
    """Errors raised by the orchestrator surface via `exit_on_error`."""

    async def test_orchestrator_run_error_is_caught(self, capsys: Any) -> None:
        """A RuntimeError from orchestrator.run() is swallowed into SystemExit."""
        instance = MagicMock()
        instance.run = AsyncMock(side_effect=RuntimeError("boom"))
        with (
            patch(
                "aiperf.kubernetes.watch_orchestrator.WatchOrchestrator",
                return_value=instance,
            ),
            pytest.raises(SystemExit),
        ):
            await watch(manage_options=KubeManageOptions(), output="text")
