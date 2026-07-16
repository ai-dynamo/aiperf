# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for kube logs/attach/results target resolution.

Focuses on:
- `-v` / `-t` AIPerfSweep child-name resolution before job lookup.
- namespace, kubeconfig, and kube-context propagation across command boundaries.
- missing child short-circuits that avoid downstream attach/log/result work.
- JSON list-runs cleanliness when machine-readable output is requested.
- kubectl subprocess wrapper boundaries for special-character target names.

Out of scope: live Kubernetes API behavior and HTTP artifact transfer bodies; those
are covered by the Kubernetes client and results-operator tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from pytest import param

from aiperf.cli_commands.kube._kube_common import resolve_child_name
from aiperf.config.kube import KubeManageOptions
from aiperf.kubernetes.subproc import CommandResult

# ============================================================
# Helpers
# ============================================================


@dataclass(slots=True)
class _ResolvedJobStub:
    """Minimal resolved-job surface consumed by kube CLI commands."""

    job_id: str = "latency-sweep-v07-t2"
    namespace: str = "bench-prod"
    api: object = "api-client"
    phase: str = "Running"
    closed: bool = False
    job_info: MagicMock = field(init=False)

    def __post_init__(self) -> None:
        self.job_info = MagicMock()
        self.job_info.name = self.job_id
        self.job_info.phase = self.phase

    async def aclose(self) -> None:
        self.closed = True


class _PortForwardStub:
    """Async context manager that records port-forward entry."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.entered = False

    async def __aenter__(self) -> int:
        self.entered = True
        return self.port

    async def __aexit__(self, *_exc: object) -> None:
        return None


def _pod(name: str) -> MagicMock:
    pod = MagicMock()
    pod.metadata.name = name
    return pod


# ============================================================
# Pure child-name resolution contract
# ============================================================


class TestSweepChildNameResolution:
    """Sweep child selectors mirror sweep-controller naming exactly."""

    @pytest.mark.parametrize(
        ("parent", "variation", "trial", "expected"),
        [
            ("latency-sweep", 0, None, "latency-sweep-v00"),
            ("latency-sweep", 7, 2, "latency-sweep-v07-t2"),
            param(
                "llama-3.1-throughput",
                12,
                0,
                "llama-3.1-throughput-v12-t0",
                id="dot-in-parent-is-not-shell-split",
            ),
        ],
    )  # fmt: skip
    def test_resolve_child_name_variation_and_trial_return_child_job_name(
        self, parent: str, variation: int, trial: int | None, expected: str
    ) -> None:
        assert resolve_child_name(parent, variation=variation, trial=trial) == expected


# ============================================================
# Attach command target resolution
# ============================================================


class TestAttachSweepChildResolution:
    """`aiperf kube attach` resolves sweep children before opening the API client."""

    @pytest.mark.asyncio
    async def test_attach_sweep_child_propagates_namespace_and_kube_context(
        self,
    ) -> None:
        from aiperf.cli_commands.kube.attach import attach

        resolved = _ResolvedJobStub(
            job_id="latency-sweep-v07-t2",
            namespace="bench-prod",
            api="http://operator-api:38465",
            phase="Profiling",
        )
        opts = KubeManageOptions(
            namespace="tenant-a",
            kubeconfig="/secure/kubeconfigs/dgx-prod.yaml",
            kube_context="dgx-prod-admin",
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
            await attach(
                job_id="latency-sweep",
                manage_options=opts,
                port=19090,
                variation=7,
                trial=2,
            )

        assert mock_resolve.await_args.args == ("latency-sweep-v07-t2", "tenant-a")
        assert mock_resolve.await_args.kwargs == {
            "kubeconfig": "/secure/kubeconfigs/dgx-prod.yaml",
            "kube_context": "dgx-prod-admin",
        }
        assert mock_attach.await_args.args == (
            "latency-sweep-v07-t2",
            "bench-prod",
            19090,
            "http://operator-api:38465",
        )
        assert mock_attach.await_args.kwargs == {
            "phase": "Profiling",
            "kubeconfig": "/secure/kubeconfigs/dgx-prod.yaml",
            "kube_context": "dgx-prod-admin",
        }
        assert resolved.closed is True

    @pytest.mark.asyncio
    async def test_attach_missing_sweep_child_skips_attach(self) -> None:
        from aiperf.cli_commands.kube.attach import attach

        opts = KubeManageOptions(namespace="tenant-a")
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_job",
                new=AsyncMock(return_value=None),
            ) as mock_resolve,
            patch(
                "aiperf.kubernetes.attach.attach_to_benchmark",
                new=AsyncMock(),
            ) as mock_attach,
        ):
            await attach(
                job_id="missing-sweep",
                manage_options=opts,
                variation=3,
                trial=1,
            )

        assert mock_resolve.await_args.args == ("missing-sweep-v03-t1", "tenant-a")
        mock_attach.assert_not_awaited()


# ============================================================
# Logs command target resolution
# ============================================================


class TestLogsSweepChildResolution:
    """`aiperf kube logs` resolves children before pod lookup or log saving."""

    @pytest.mark.asyncio
    async def test_logs_sweep_child_saves_resolved_child_with_kube_context(
        self, tmp_path: Path
    ) -> None:
        from aiperf.cli_commands.kube.logs import logs

        opts = KubeManageOptions(
            namespace="tenant-a",
            kubeconfig="/secure/kubeconfigs/dgx-prod.yaml",
            kube_context="dgx-prod-admin",
        )
        output = tmp_path / "adversarial-logs"
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_job_id_and_namespace",
                return_value=("latency-sweep-v12", "tenant-a"),
            ) as mock_resolve,
            patch(
                "aiperf.cli_commands.kube.logs._save_logs_to_directory",
                new=AsyncMock(),
            ) as mock_save,
            patch(
                "aiperf.cli_commands.kube.logs._print_pod_logs",
                new=AsyncMock(),
            ) as mock_print,
        ):
            await logs(
                job_id="latency-sweep",
                manage_options=opts,
                output=output,
                variation=12,
            )

        assert mock_resolve.call_args.args == ("latency-sweep-v12", "tenant-a")
        assert mock_save.await_args.args == (
            "latency-sweep-v12",
            "tenant-a",
            output,
            opts,
        )
        mock_print.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_logs_missing_sweep_child_skips_pod_log_fetch(self) -> None:
        from aiperf.cli_commands.kube.logs import logs

        opts = KubeManageOptions(namespace="tenant-a")
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_job_id_and_namespace",
                return_value=None,
            ) as mock_resolve,
            patch(
                "aiperf.cli_commands.kube.logs._save_logs_to_directory",
                new=AsyncMock(),
            ) as mock_save,
            patch(
                "aiperf.cli_commands.kube.logs._print_pod_logs",
                new=AsyncMock(),
            ) as mock_print,
        ):
            await logs(
                job_id="missing-sweep", manage_options=opts, variation=2, trial=0
            )

        assert mock_resolve.call_args.args == ("missing-sweep-v02-t0", "tenant-a")
        mock_save.assert_not_awaited()
        mock_print.assert_not_awaited()


# ============================================================
# Results command target resolution and JSON output
# ============================================================


class TestResultsSweepChildResolution:
    """`aiperf kube results` resolves a selected child as a single-job target."""

    @pytest.mark.asyncio
    async def test_results_sweep_child_invokes_single_job_branch_with_kube_context(
        self, tmp_path: Path
    ) -> None:
        from aiperf.cli_commands.kube.results import results

        opts = KubeManageOptions(
            namespace="tenant-a",
            kubeconfig="/secure/kubeconfigs/dgx-prod.yaml",
            kube_context="dgx-prod-admin",
        )
        with patch(
            "aiperf.cli_commands.kube.results._run_results",
            new=AsyncMock(),
        ) as mock_run:
            await results(
                job_id="latency-sweep",
                manage_options=opts,
                output=tmp_path / "results",
                variation=4,
                trial=9,
                run="1770001234",
            )

        assert mock_run.await_args.kwargs["job_id"] == "latency-sweep-v04-t9"
        assert mock_run.await_args.kwargs["manage_options"] is opts
        assert mock_run.await_args.kwargs["run"] == "1770001234"

    @pytest.mark.asyncio
    async def test_run_results_child_operator_path_propagates_kube_context(
        self, tmp_path: Path
    ) -> None:
        from aiperf.cli_commands.kube.results import _run_results

        resolved = _ResolvedJobStub(
            job_id="latency-sweep-v04-t9",
            namespace="bench-prod",
            phase="Completed",
        )
        opts = KubeManageOptions(
            namespace="tenant-a",
            kubeconfig="/secure/kubeconfigs/dgx-prod.yaml",
            kube_context="dgx-prod-admin",
        )
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_target",
                new=AsyncMock(return_value=resolved),
            ) as mock_resolve,
            patch(
                "aiperf.cli_commands.kube.results._resolve_op_ns",
                new=AsyncMock(return_value="aiperf-ops"),
            ),
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(return_value=MagicMock()),
            ),
            patch(
                "aiperf.kubernetes.results.retrieve_results_from_operator",
                new=AsyncMock(return_value=True),
            ) as mock_retrieve,
            patch("aiperf.cli_commands.kube.results._validate_run_arg"),
            patch("aiperf.kubernetes.console.print_results_summary"),
        ):
            await _run_results(
                job_id="latency-sweep-v04-t9",
                manage_options=opts,
                output=tmp_path / "child-results",
                from_pods=False,
                all_artifacts=True,
                shutdown=False,
                port=19091,
                operator_namespace="aiperf-ops",
                run="1770001234",
            )

        assert mock_resolve.await_args.args == ("latency-sweep-v04-t9", "tenant-a")
        assert mock_resolve.await_args.kwargs == {
            "kubeconfig": "/secure/kubeconfigs/dgx-prod.yaml",
            "kube_context": "dgx-prod-admin",
        }
        assert mock_retrieve.await_args.args[:3] == (
            "latency-sweep-v04-t9",
            "bench-prod",
            tmp_path / "child-results",
        )
        assert mock_retrieve.await_args.kwargs == {
            "local_port": 19091,
            "operator_namespace": "aiperf-ops",
            "run": "1770001234",
            "kubeconfig": "/secure/kubeconfigs/dgx-prod.yaml",
            "kube_context": "dgx-prod-admin",
        }
        assert resolved.closed is True

    @pytest.mark.asyncio
    async def test_list_runs_json_output_is_parseable_and_resolver_quiet(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from aiperf.cli_commands.kube.results import _run_list_runs

        resolved = _ResolvedJobStub(
            job_id="latency-sweep-v04-t9", namespace="bench-prod"
        )
        port_forward = _PortForwardStub(port=31081)
        payload = {
            "namespace": "bench-prod",
            "job_id": "latency-sweep-v04-t9",
            "runs": [{"run": "1770001234", "created_at": "2026-05-18T12:00:00Z"}],
        }
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_job",
                new=AsyncMock(return_value=resolved),
            ) as mock_resolve,
            patch(
                "aiperf.cli_commands.kube.results._resolve_op_ns",
                new=AsyncMock(return_value="aiperf-ops"),
            ) as mock_op_ns,
            patch(
                "aiperf.kubernetes.client.find_operator_pod",
                new=AsyncMock(return_value=("aiperf-operator-7f2a", "Running")),
            ),
            patch(
                "aiperf.kubernetes.port_forward.port_forward_with_status",
                return_value=port_forward,
            ) as mock_port_forward,
            patch(
                "aiperf.cli_commands.kube.results._fetch_runs_and_retention",
                new=AsyncMock(return_value=(payload, None)),
            ),
        ):
            await _run_list_runs(
                job_id="latency-sweep-v04-t9",
                manage_options=KubeManageOptions(
                    namespace="tenant-a",
                    kubeconfig="/secure/kubeconfigs/dgx-prod.yaml",
                    kube_context="dgx-prod-admin",
                ),
                output="json",
                preview=False,
                operator_namespace=None,
            )

        out = capsys.readouterr().out
        assert orjson.loads(out) == payload
        assert "Auto-detected operator namespace" not in out
        assert mock_resolve.await_args.kwargs["quiet"] is True
        assert mock_op_ns.await_args.kwargs["quiet"] is True
        assert mock_port_forward.call_args.kwargs == {
            "remote_port": 8081,
            "verify_api": False,
            "kubeconfig": "/secure/kubeconfigs/dgx-prod.yaml",
            "kube_context": "dgx-prod-admin",
        }
        assert port_forward.entered is True
        assert resolved.closed is True


# ============================================================
# Subprocess wrapper boundaries
# ============================================================


class TestKubectlSubprocessBoundaries:
    """kubectl paths use the project subprocess wrapper and pass names as argv cells."""

    @pytest.mark.asyncio
    async def test_save_pod_logs_uses_run_command_with_special_character_pod_name(
        self, tmp_path: Path
    ) -> None:
        from aiperf.kubernetes.logs import save_pod_logs

        with (
            patch(
                "aiperf.kubernetes.logs.get_pods",
                new=AsyncMock(return_value=[_pod("latency-sweep-v04-t9.ctrl-0")]),
            ),
            patch(
                "aiperf.kubernetes.logs.run_command",
                new=AsyncMock(
                    return_value=CommandResult(0, "controller log line\n", "")
                ),
            ) as mock_run,
        ):
            await save_pod_logs(
                "latency-sweep-v04-t9",
                "bench-prod",
                tmp_path,
                MagicMock(),
                kubeconfig="/secure/kubeconfigs/dgx-prod.yaml",
                kube_context="dgx-prod-admin",
            )

        assert mock_run.await_args.args[0] == [
            "kubectl",
            "logs",
            "-n",
            "bench-prod",
            "latency-sweep-v04-t9.ctrl-0",
            "--all-containers=true",
            "--prefix",
            "--kubeconfig",
            "/secure/kubeconfigs/dgx-prod.yaml",
            "--context",
            "dgx-prod-admin",
        ]
        assert (tmp_path / "logs" / "latency-sweep-v04-t9.ctrl-0.log").read_text() == (
            "controller log line\n"
        )

    @pytest.mark.asyncio
    async def test_kubectl_copy_results_failure_uses_run_command_for_cp_and_ls(
        self, tmp_path: Path
    ) -> None:
        from aiperf.kubernetes.results import kubectl_copy_results

        run_command = AsyncMock(
            side_effect=[
                CommandResult(1, "", "tar: results not ready"),
                CommandResult(0, "total 0\n", ""),
            ]
        )
        with (
            patch("aiperf.kubernetes.results.run_command", new=run_command),
            patch("aiperf.kubernetes.results.console"),
            patch("aiperf.kubernetes.results.print_error"),
            patch("aiperf.kubernetes.results.print_info"),
        ):
            ok = await kubectl_copy_results(
                "bench-prod",
                "latency-sweep-v04-t9.ctrl-0",
                "control-plane",
                tmp_path / "child-results",
                kubeconfig="/secure/kubeconfigs/dgx-prod.yaml",
                kube_context="dgx-prod-admin",
            )

        assert ok is False
        assert run_command.await_args_list[0].args[0] == [
            "kubectl",
            "cp",
            "-n",
            "bench-prod",
            "-c",
            "control-plane",
            "latency-sweep-v04-t9.ctrl-0:/results/.",
            str(tmp_path / "child-results"),
            "--kubeconfig",
            "/secure/kubeconfigs/dgx-prod.yaml",
            "--context",
            "dgx-prod-admin",
        ]
        assert run_command.await_args_list[1].args[0] == [
            "kubectl",
            "exec",
            "-n",
            "bench-prod",
            "-c",
            "control-plane",
            "latency-sweep-v04-t9.ctrl-0",
            "--kubeconfig",
            "/secure/kubeconfigs/dgx-prod.yaml",
            "--context",
            "dgx-prod-admin",
            "--",
            "ls",
            "-la",
            "/results",
        ]
