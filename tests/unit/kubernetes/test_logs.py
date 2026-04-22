# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.logs module.

Focuses on:
- Pod log retrieval and file writing behavior
- Handling of empty pod lists, failed commands, and empty stdout
- Kubeconfig/context argument forwarding to kubectl
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client import ApiClient
from pytest import param

from aiperf.kubernetes.logs import save_pod_logs
from aiperf.kubernetes.subproc import CommandResult

# =============================================================================
# Fixtures
# =============================================================================


def _make_pod(name: str, namespace: str = "default") -> MagicMock:
    """Build an object that looks like a V1Pod with metadata.name."""
    pod = MagicMock()
    pod.metadata = MagicMock()
    pod.metadata.name = name
    pod.metadata.namespace = namespace
    return pod


@pytest.fixture
def mock_api() -> MagicMock:
    """Mock kubernetes_asyncio ApiClient."""
    api = MagicMock(spec=ApiClient)
    api.close = AsyncMock()
    return api


@pytest.fixture
def pods() -> list[MagicMock]:
    """Create a list of fake V1Pod objects with names."""
    return [
        _make_pod("aiperf-job-controller-0-0"),
        _make_pod("aiperf-job-worker-0-0"),
    ]


# =============================================================================
# Happy Path
# =============================================================================


class TestSavePodLogsHappyPath:
    """Verify normal log saving operations."""

    @pytest.mark.asyncio
    async def test_save_pod_logs_writes_log_files(
        self, mock_api: MagicMock, pods: list[MagicMock], tmp_path: Path
    ) -> None:
        results = {
            pods[0].metadata.name: CommandResult(
                returncode=0, stdout="controller logs\n", stderr=""
            ),
            pods[1].metadata.name: CommandResult(
                returncode=0, stdout="worker logs\n", stderr=""
            ),
        }

        async def _fake_run(cmd: list[str]) -> CommandResult:
            pod_name = cmd[4]
            return results[pod_name]

        with (
            patch(
                "aiperf.kubernetes.logs.get_pods",
                new_callable=AsyncMock,
                return_value=pods,
            ),
            patch("aiperf.kubernetes.logs.run_command", side_effect=_fake_run),
        ):
            await save_pod_logs("job-123", "default", tmp_path, mock_api)

        logs_dir = tmp_path / "logs"
        assert logs_dir.is_dir()
        assert (
            logs_dir / "aiperf-job-controller-0-0.log"
        ).read_text() == "controller logs\n"
        assert (logs_dir / "aiperf-job-worker-0-0.log").read_text() == "worker logs\n"

    @pytest.mark.asyncio
    async def test_save_pod_logs_calls_kubectl_with_correct_args(
        self, mock_api: MagicMock, pods: list[MagicMock], tmp_path: Path
    ) -> None:
        mock_run = AsyncMock(
            return_value=CommandResult(returncode=0, stdout="logs", stderr="")
        )

        with (
            patch(
                "aiperf.kubernetes.logs.get_pods",
                new_callable=AsyncMock,
                return_value=[pods[0]],
            ),
            patch("aiperf.kubernetes.logs.run_command", mock_run),
        ):
            await save_pod_logs("job-123", "my-ns", tmp_path, mock_api)

        mock_run.assert_called_once_with(
            ["kubectl", "logs", "-n", "my-ns", "aiperf-job-controller-0-0"]
        )

    @pytest.mark.asyncio
    async def test_save_pod_logs_creates_logs_subdirectory(
        self, mock_api: MagicMock, pods: list[MagicMock], tmp_path: Path
    ) -> None:
        mock_run = AsyncMock(
            return_value=CommandResult(returncode=0, stdout="x", stderr="")
        )

        with (
            patch(
                "aiperf.kubernetes.logs.get_pods",
                new_callable=AsyncMock,
                return_value=[pods[0]],
            ),
            patch("aiperf.kubernetes.logs.run_command", mock_run),
        ):
            await save_pod_logs("job-123", "default", tmp_path, mock_api)

        assert (tmp_path / "logs").is_dir()

    @pytest.mark.asyncio
    async def test_save_pod_logs_passes_job_selector(
        self, mock_api: MagicMock, tmp_path: Path
    ) -> None:
        with (
            patch(
                "aiperf.kubernetes.logs.get_pods",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_get,
            patch(
                "aiperf.kubernetes.logs.job_selector",
                return_value="app=aiperf,aiperf.nvidia.com/job-id=my-job",
            ) as mock_js,
            patch("aiperf.kubernetes.logs.run_command", AsyncMock()),
        ):
            await save_pod_logs("my-job", "ns", tmp_path, mock_api)

        mock_js.assert_called_once_with("my-job")
        mock_get.assert_called_once_with(
            mock_api, "ns", "app=aiperf,aiperf.nvidia.com/job-id=my-job"
        )


# =============================================================================
# Kubeconfig / Context Forwarding
# =============================================================================


class TestSavePodLogsKubeArgs:
    """Verify kubeconfig and context arguments are forwarded to kubectl."""

    @pytest.mark.parametrize(
        "kubeconfig,kube_context,expected_extra_args",
        [
            (None, None, []),
            ("/path/to/config", None, ["--kubeconfig", "/path/to/config"]),
            (None, "my-ctx", ["--context", "my-ctx"]),
            param(
                "/cfg", "ctx",
                ["--kubeconfig", "/cfg", "--context", "ctx"],
                id="both-kubeconfig-and-context",
            ),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_save_pod_logs_forwards_kube_args(
        self,
        mock_api: MagicMock,
        pods: list[MagicMock],
        tmp_path: Path,
        kubeconfig: str | None,
        kube_context: str | None,
        expected_extra_args: list[str],
    ) -> None:
        mock_run = AsyncMock(
            return_value=CommandResult(returncode=0, stdout="log", stderr="")
        )

        with (
            patch(
                "aiperf.kubernetes.logs.get_pods",
                new_callable=AsyncMock,
                return_value=[pods[0]],
            ),
            patch("aiperf.kubernetes.logs.run_command", mock_run),
        ):
            await save_pod_logs(
                "job-123",
                "default",
                tmp_path,
                mock_api,
                kubeconfig=kubeconfig,
                kube_context=kube_context,
            )

        expected_cmd = [
            "kubectl",
            "logs",
            "-n",
            "default",
            pods[0].metadata.name,
            *expected_extra_args,
        ]
        mock_run.assert_called_once_with(expected_cmd)


# =============================================================================
# Edge Cases
# =============================================================================


class TestSavePodLogsEdgeCases:
    """Verify boundary conditions and special cases."""

    @pytest.mark.asyncio
    async def test_save_pod_logs_no_pods_returns_early(
        self, mock_api: MagicMock, tmp_path: Path
    ) -> None:
        with (
            patch(
                "aiperf.kubernetes.logs.get_pods",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("aiperf.kubernetes.logs.run_command") as mock_run,
        ):
            await save_pod_logs("job-123", "default", tmp_path, mock_api)

        mock_run.assert_not_called()
        assert not (tmp_path / "logs").exists()

    @pytest.mark.asyncio
    async def test_save_pod_logs_nonzero_returncode_skips_file(
        self, mock_api: MagicMock, pods: list[MagicMock], tmp_path: Path
    ) -> None:
        failed_result = CommandResult(returncode=1, stdout="", stderr="error")
        mock_run = AsyncMock(return_value=failed_result)

        with (
            patch(
                "aiperf.kubernetes.logs.get_pods",
                new_callable=AsyncMock,
                return_value=[pods[0]],
            ),
            patch("aiperf.kubernetes.logs.run_command", mock_run),
        ):
            await save_pod_logs("job-123", "default", tmp_path, mock_api)

        logs_dir = tmp_path / "logs"
        assert logs_dir.is_dir()
        assert not (logs_dir / "aiperf-job-controller-0-0.log").exists()

    @pytest.mark.asyncio
    async def test_save_pod_logs_empty_stdout_skips_file(
        self, mock_api: MagicMock, pods: list[MagicMock], tmp_path: Path
    ) -> None:
        empty_result = CommandResult(returncode=0, stdout="", stderr="")
        mock_run = AsyncMock(return_value=empty_result)

        with (
            patch(
                "aiperf.kubernetes.logs.get_pods",
                new_callable=AsyncMock,
                return_value=[pods[0]],
            ),
            patch("aiperf.kubernetes.logs.run_command", mock_run),
        ):
            await save_pod_logs("job-123", "default", tmp_path, mock_api)

        logs_dir = tmp_path / "logs"
        assert not (logs_dir / "aiperf-job-controller-0-0.log").exists()

    @pytest.mark.asyncio
    async def test_save_pod_logs_nonzero_with_stdout_skips_file(
        self, mock_api: MagicMock, pods: list[MagicMock], tmp_path: Path
    ) -> None:
        """When returncode is nonzero, stdout content is not written even if present."""
        result = CommandResult(returncode=1, stdout="partial output", stderr="error")
        mock_run = AsyncMock(return_value=result)

        with (
            patch(
                "aiperf.kubernetes.logs.get_pods",
                new_callable=AsyncMock,
                return_value=[pods[0]],
            ),
            patch("aiperf.kubernetes.logs.run_command", mock_run),
        ):
            await save_pod_logs("job-123", "default", tmp_path, mock_api)

        assert not (tmp_path / "logs" / "aiperf-job-controller-0-0.log").exists()

    @pytest.mark.asyncio
    async def test_save_pod_logs_mixed_success_and_failure(
        self, mock_api: MagicMock, pods: list[MagicMock], tmp_path: Path
    ) -> None:
        """First pod succeeds, second pod fails -- only first log is written."""
        results = [
            CommandResult(returncode=0, stdout="good logs", stderr=""),
            CommandResult(returncode=1, stdout="", stderr="failed"),
        ]
        mock_run = AsyncMock(side_effect=results)

        with (
            patch(
                "aiperf.kubernetes.logs.get_pods",
                new_callable=AsyncMock,
                return_value=pods,
            ),
            patch("aiperf.kubernetes.logs.run_command", mock_run),
        ):
            await save_pod_logs("job-123", "default", tmp_path, mock_api)

        logs_dir = tmp_path / "logs"
        assert (logs_dir / "aiperf-job-controller-0-0.log").read_text() == "good logs"
        assert not (logs_dir / "aiperf-job-worker-0-0.log").exists()

    @pytest.mark.asyncio
    async def test_save_pod_logs_nested_output_dir(
        self, mock_api: MagicMock, pods: list[MagicMock], tmp_path: Path
    ) -> None:
        """logs_dir.mkdir(parents=True) creates intermediate directories."""
        mock_run = AsyncMock(
            return_value=CommandResult(returncode=0, stdout="data", stderr="")
        )
        deep_path = tmp_path / "a" / "b" / "c"

        with (
            patch(
                "aiperf.kubernetes.logs.get_pods",
                new_callable=AsyncMock,
                return_value=[pods[0]],
            ),
            patch("aiperf.kubernetes.logs.run_command", mock_run),
        ):
            await save_pod_logs("job-123", "default", deep_path, mock_api)

        assert (
            deep_path / "logs" / "aiperf-job-controller-0-0.log"
        ).read_text() == "data"
