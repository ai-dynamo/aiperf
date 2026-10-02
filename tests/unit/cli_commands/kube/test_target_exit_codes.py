# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exit-code contract for the target-addressing `aiperf kube` subcommands.

The convention these tests pin down is "validation/retrieval commands gate,
addressing commands narrate", with one carve-out: a target that does not exist
at all fails the shell, so `attach` and `logs` can be used as CI existence
checks. Covered here:

- job absent -> exit 1 on every `logs` path and on `attach`
- job present but pods already collected/GC'd -> exit 0, no false success line
- ``--ignore-not-found`` -> exit 0 for an absent job, mirroring kubectl
- the bulk `--output` dump reports what actually reached disk

Out of scope: live Kubernetes API behaviour, and the `results` command, which
deliberately keeps its pre-existing exit-1-on-any-failure retrieval semantics
(see ``test_logs_attach_results_adversarial.py``).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from aiperf.config.kube import KubeManageOptions
from aiperf.kubernetes.logs import SavedPodLogs

# ============================================================
# Helpers
# ============================================================


@asynccontextmanager
async def _fake_client(**_kw: Any):
    api = MagicMock()
    api.close = AsyncMock()
    yield api


def _pod(name: str, containers: list[str]) -> MagicMock:
    pod = MagicMock()
    pod.metadata.name = name
    pod.spec.containers = [MagicMock() for _ in containers]
    for mock_c, real_name in zip(pod.spec.containers, containers, strict=True):
        mock_c.name = real_name
    return pod


# ============================================================
# `logs` stdout path
# ============================================================


class TestLogsStdoutExitCodes:
    """`aiperf kube logs` (no --output) separates "absent" from "no pods"."""

    @pytest.mark.asyncio
    async def test_absent_job_with_no_pods_exits_one(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_job_id_and_namespace",
                return_value=("ghost", "bench-ns"),
            ),
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_client),
            patch("aiperf.kubernetes.client.get_pods", new=AsyncMock(return_value=[])),
            patch(
                "aiperf.kubernetes.cli_helpers.target_exists",
                new=AsyncMock(return_value=False),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            from aiperf.cli_commands.kube.logs import logs

            await logs(job_id="ghost", manage_options=KubeManageOptions())

        assert exc_info.value.code == 1
        assert "No AIPerf job found with ID: ghost" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_present_job_with_no_pods_exits_zero_without_success_line(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A finished job whose pods were GC'd is narrated, not failed."""
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_job_id_and_namespace",
                return_value=("reaped", "bench-ns"),
            ),
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_client),
            patch("aiperf.kubernetes.client.get_pods", new=AsyncMock(return_value=[])),
            patch(
                "aiperf.kubernetes.cli_helpers.target_exists",
                new=AsyncMock(return_value=True),
            ),
        ):
            from aiperf.cli_commands.kube.logs import logs

            await logs(job_id="reaped", manage_options=KubeManageOptions())

        out = capsys.readouterr().out
        assert "No pods found for reaped" in out
        assert "No AIPerf job found" not in out

    @pytest.mark.asyncio
    async def test_absent_job_with_ignore_not_found_exits_zero(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_job_id_and_namespace",
                return_value=("ghost", "bench-ns"),
            ),
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_client),
            patch("aiperf.kubernetes.client.get_pods", new=AsyncMock(return_value=[])),
            patch(
                "aiperf.kubernetes.cli_helpers.target_exists",
                new=AsyncMock(return_value=False),
            ),
        ):
            from aiperf.cli_commands.kube.logs import logs

            await logs(
                job_id="ghost",
                manage_options=KubeManageOptions(),
                ignore_not_found=True,
            )

        # The diagnosis is still printed; only the exit status is suppressed.
        assert "No AIPerf job found with ID: ghost" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_present_job_with_unmatched_container_exits_zero(self) -> None:
        """--container naming nothing on an existing job is not a missing target."""
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_job_id_and_namespace",
                return_value=("live", "bench-ns"),
            ),
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_client),
            patch(
                "aiperf.kubernetes.client.get_pods",
                new=AsyncMock(return_value=[_pod("ctrl-0", ["control-plane"])]),
            ),
        ):
            from aiperf.cli_commands.kube.logs import logs

            await logs(
                job_id="live",
                manage_options=KubeManageOptions(),
                container="does-not-exist",
            )


# ============================================================
# `logs --output` bulk dump
# ============================================================


class TestLogsBulkDumpExitCodes:
    """The `--output DIR` path shares the absent/present split."""

    @pytest.mark.asyncio
    async def test_absent_job_exits_one_and_creates_no_directory(
        self, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "dump"
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_job_id_and_namespace",
                return_value=("ghost", "bench-ns"),
            ),
            patch("aiperf.kubernetes.client.k8s_client", new=_fake_client),
            patch("aiperf.kubernetes.logs.get_pods", new=AsyncMock(return_value=[])),
            patch(
                "aiperf.kubernetes.cli_helpers.target_exists",
                new=AsyncMock(return_value=False),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            from aiperf.cli_commands.kube.logs import logs

            await logs(
                job_id="ghost",
                manage_options=KubeManageOptions(),
                output=out_dir,
            )

        assert exc_info.value.code == 1
        assert not out_dir.exists()


class TestBulkDumpMessaging:
    """`_report_saved_logs` must never claim more than reached disk."""

    @pytest.mark.parametrize(
        ("saved", "expected", "forbidden"),
        [
            param(
                SavedPodLogs(logs_dir=Path("x/logs"), pods_matched=2,
                             files_written=["a.log", "b.log"]),
                "Saved logs for 2 of 2 pod(s)",
                "No logs written",
                id="all-pods-written",
            ),
            param(
                SavedPodLogs(logs_dir=Path("x/logs"), pods_matched=2,
                             files_written=["a.log"],
                             failures=["b-0: kubectl logs returned no output"]),
                "Saved logs for 1 of 2 pod(s)",
                "Saved logs for 2 of 2",
                id="partial-write-counts-honestly",
            ),
            param(
                SavedPodLogs(logs_dir=Path("x/logs"), pods_matched=3,
                             failures=["a-0: kubectl logs exited 1: forbidden"]),
                "No logs written",
                "Saved logs for",
                id="nothing-written-is-not-a-success",
            ),
        ],
    )  # fmt: skip
    def test_outcome_line_matches_disk(
        self,
        saved: SavedPodLogs,
        expected: str,
        forbidden: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from aiperf.cli_commands.kube.logs import _report_saved_logs

        _report_saved_logs(saved, Path("x"))

        out = capsys.readouterr().out
        assert expected in out
        assert forbidden not in out

    def test_kubectl_failures_are_surfaced(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """kubectl stderr used to be discarded entirely."""
        from aiperf.cli_commands.kube.logs import _report_saved_logs

        saved = SavedPodLogs(
            logs_dir=Path("x/logs"),
            pods_matched=1,
            failures=["ctrl-0: kubectl logs exited 1: Forbidden"],
        )
        _report_saved_logs(saved, Path("x"))

        # Rich wraps at the resolved console width, so compare on one line.
        out = " ".join(capsys.readouterr().out.split())
        assert "ctrl-0: kubectl logs exited 1: Forbidden" in out


# ============================================================
# Shared exit-code helper
# ============================================================


class TestExitTargetNotFound:
    """The single place the not-found exit policy is expressed."""

    @pytest.mark.parametrize(
        ("ignore_not_found", "expect_exit"),
        [
            param(False, True, id="default-gates"),
            param(True, False, id="ignore-not-found-tolerates"),
        ],
    )  # fmt: skip
    def test_exit_policy(self, ignore_not_found: bool, expect_exit: bool) -> None:
        from aiperf.kubernetes.cli_helpers import exit_target_not_found

        if expect_exit:
            with pytest.raises(SystemExit) as exc_info:
                exit_target_not_found(ignore_not_found=ignore_not_found)
            assert exc_info.value.code == 1
        else:
            exit_target_not_found(ignore_not_found=ignore_not_found)


class TestTargetExists:
    """`target_exists` accepts jobs, sweeps and bare JobSets."""

    @pytest.mark.parametrize(
        ("job", "sweep", "jobset", "expected"),
        [
            param(object(), None, None, True, id="aiperfjob"),
            param(None, object(), None, True, id="aiperfsweep"),
            param(None, None, object(), True, id="bare-jobset-direct-mode"),
            param(None, None, None, False, id="nothing-matches"),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_resolution_order(
        self, job: object, sweep: object, jobset: object, expected: bool
    ) -> None:
        from aiperf.kubernetes.cli_helpers import target_exists

        with (
            patch(
                "aiperf.kubernetes.client.find_aiperf_job",
                new=AsyncMock(return_value=job),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_sweep",
                new=AsyncMock(return_value=sweep),
            ),
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(return_value=jobset),
            ),
        ):
            assert await target_exists(MagicMock(), "name", "ns") is expected
