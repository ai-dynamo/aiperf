# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `aiperf kube logs` helpers.

The top-level `logs()` command is exercised in `test_kube.py`; this file
targets the small pure helpers (`_collect_log_targets`) and the
`--output <dir>` save-to-directory path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.cli_commands.kube.logs import _collect_log_targets, logs
from aiperf.config.kube import KubeManageOptions


def _pod(name: str, containers: list[str]) -> MagicMock:
    """Build a mock pod object with .spec.containers[*].name and .metadata.name."""
    pod = MagicMock()
    pod.metadata.name = name
    pod.spec.containers = [MagicMock(name=c) for c in containers]
    # mock.name is special; assign explicitly
    for mock_c, real_name in zip(pod.spec.containers, containers, strict=False):
        mock_c.name = real_name
    return pod


class TestCollectLogTargets:
    """Tests for the pure `_collect_log_targets` helper."""

    def test_no_container_filter_returns_all_containers(self) -> None:
        """Omitting `container` yields every container on every pod."""
        pods = [_pod("a", ["c1", "c2"]), _pod("b", ["c3"])]
        targets = _collect_log_targets(pods, container=None)
        labels = [(p.metadata.name, c) for p, c in targets]
        assert labels == [("a", "c1"), ("a", "c2"), ("b", "c3")]

    def test_container_filter_matches_only_named(self) -> None:
        """A container filter keeps only pods that have that container."""
        pods = [_pod("a", ["c1", "c2"]), _pod("b", ["other"])]
        targets = _collect_log_targets(pods, container="c1")
        labels = [(p.metadata.name, c) for p, c in targets]
        assert labels == [("a", "c1")]

    def test_container_filter_unknown_yields_empty(self) -> None:
        """Unknown container name yields no targets."""
        pods = [_pod("a", ["c1"])]
        assert _collect_log_targets(pods, container="missing") == []

    def test_no_pods_returns_empty(self) -> None:
        """Empty pod list trivially yields empty targets."""
        assert _collect_log_targets([], container=None) == []

    @pytest.mark.parametrize(
        "containers",
        [
            param([], id="no-containers"),
            param(["only"], id="single-container"),
        ],
    )  # fmt: skip
    def test_pod_without_matching_container_contributes_nothing(
        self, containers: list[str]
    ) -> None:
        """Pods without the filter name contribute nothing to the targets list."""
        pods = [_pod("a", containers)]
        assert _collect_log_targets(pods, container="nope") == []


class TestLogsOutputDirectory:
    """Tests for the `--output` save-to-directory code path."""

    async def test_output_flag_invokes_save_helper(self, tmp_path: Path) -> None:
        """When --output is set, logs delegates to `save_pod_logs`."""
        out_dir = tmp_path / "saved-logs"
        opts = KubeManageOptions(namespace="ns-1")

        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_job_id_and_namespace",
                return_value=("jid", "ns-1"),
            ),
            patch(
                "aiperf.cli_commands.kube.logs._save_logs_to_directory",
                new=AsyncMock(),
            ) as mock_save,
            patch(
                "aiperf.cli_commands.kube.logs._print_pod_logs",
                new=AsyncMock(),
            ) as mock_print_pod,
        ):
            await logs(job_id="jid", manage_options=opts, output=out_dir)

        mock_save.assert_awaited_once()
        mock_print_pod.assert_not_awaited()
        # First positional arg is job_id
        assert mock_save.await_args.args[0] == "jid"

    async def test_no_output_flag_streams_to_stdout(self) -> None:
        """Without --output, logs delegates to `_print_pod_logs`."""
        opts = KubeManageOptions(namespace="ns-1")

        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_job_id_and_namespace",
                return_value=("jid", "ns-1"),
            ),
            patch(
                "aiperf.cli_commands.kube.logs._save_logs_to_directory",
                new=AsyncMock(),
            ) as mock_save,
            patch(
                "aiperf.cli_commands.kube.logs._print_pod_logs",
                new=AsyncMock(),
            ) as mock_print_pod,
        ):
            await logs(job_id="jid", manage_options=opts)

        mock_save.assert_not_awaited()
        mock_print_pod.assert_awaited_once()

    async def test_unresolved_job_id_returns_without_fetching(self) -> None:
        """If resolve_job_id_and_namespace returns None, logs exits cleanly."""
        opts = KubeManageOptions()

        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_job_id_and_namespace",
                return_value=None,
            ),
            patch(
                "aiperf.cli_commands.kube.logs._save_logs_to_directory",
                new=AsyncMock(),
            ) as mock_save,
            patch(
                "aiperf.cli_commands.kube.logs._print_pod_logs",
                new=AsyncMock(),
            ) as mock_print_pod,
        ):
            await logs(manage_options=opts)

        mock_save.assert_not_awaited()
        mock_print_pod.assert_not_awaited()


class TestLogsSaveHelper:
    """Tests for `_save_logs_to_directory`."""

    async def test_creates_output_dir_and_invokes_save_pod_logs(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """Output dir is created and `save_pod_logs` is awaited with creds."""
        from aiperf.cli_commands.kube.logs import _save_logs_to_directory

        out_dir = tmp_path / "new-dir" / "nested"
        opts = KubeManageOptions(kubeconfig="/tmp/kc", kube_context="dev")

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _fake_client(**_kw):
            yield MagicMock()

        with (
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_client),
            patch(
                "aiperf.kubernetes.logs.save_pod_logs",
                new=AsyncMock(),
            ) as mock_save,
        ):
            await _save_logs_to_directory("jid", "ns", out_dir, opts)

        assert out_dir.is_dir()
        mock_save.assert_awaited_once()
        # success line goes through kube_console
        assert "Logs saved to" in capsys.readouterr().out
