# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for kube debug trust boundaries.

Focuses on:
- missing-target paths that must return without touching pod diagnostics;
- kubeconfig, kube-context, namespace, and sweep child selector propagation;
- best-effort aggregation when one diagnostic source is malformed;
- per-container stderr/log-fetch placeholders rather than all-or-nothing failure;
- no direct ``print`` or ``rich.print`` bypass around the kube console facade.

Out of scope: JSON/result dataclass schema and subprocess stderr handling because
``aiperf kube debug`` currently exposes text-only diagnostics and performs no
subprocess shellouts. See ``test_preflight_adversarial.py`` for those contracts.
"""

from __future__ import annotations

import ast
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.cli_commands.kube import debug as debug_cmd
from aiperf.cli_commands.kube.debug import (
    _debug_namespace,
    _get_namespace_events,
    _get_node_resources,
    _get_problem_pod_logs,
    debug,
)
from aiperf.config.kube import KubeManageOptions
from aiperf.kubernetes.models import AIPerfJobInfo, AIPerfSweepInfo, JobSetInfo

# ============================================================
# Helpers
# ============================================================


@dataclass(slots=True)
class _K8sClientCall:
    """Captured arguments passed to ``k8s_client`` by the CLI wrapper."""

    kubeconfig: str | None
    context: str | None


@dataclass(slots=True)
class _RecordingK8sClientFactory:
    """Async-context-manager factory recording cluster selection kwargs."""

    api: MagicMock
    calls: list[_K8sClientCall] = field(default_factory=list)

    def __call__(
        self, *, kubeconfig: str | None = None, context: str | None = None
    ) -> _RecordingK8sClientContext:
        self.calls.append(_K8sClientCall(kubeconfig=kubeconfig, context=context))
        return _RecordingK8sClientContext(self.api)


@dataclass(slots=True)
class _RecordingK8sClientContext:
    """Async context manager returned by ``_RecordingK8sClientFactory``."""

    api: MagicMock

    async def __aenter__(self) -> MagicMock:
        return self.api

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


@dataclass(slots=True)
class _CallExpression:
    """Static call expression found in a source file."""

    path: Path
    line: int
    expression: str


def _pod_info_with_two_problem_containers() -> list[dict[str, object]]:
    """Return one problem pod with two containers for partial log aggregation."""
    return [
        {
            "name": "llama-crash-v07-t0-worker-0",
            "namespace": "llama-benchmarks",
            "phase": "Running",
            "restarts": 4,
            "problems": [
                {
                    "container": "worker",
                    "state": "CrashLoopBackOff",
                    "severity": "CRITICAL",
                    "suggestion": "Check worker logs",
                    "message": "back-off restarting failed worker",
                }
            ],
            "container_statuses": [
                {"name": "worker", "restartCount": 4, "state": {}},
                {"name": "record-processor", "restartCount": 0, "state": {}},
            ],
            "node": "gpu-node-7",
        }
    ]


def _listing_response(items: list[object]) -> SimpleNamespace:
    """Build the ``.items`` shape returned by kubernetes_asyncio list calls."""
    return SimpleNamespace(items=items)


@asynccontextmanager
async def _fake_k8s_client(api: MagicMock) -> AsyncIterator[MagicMock]:
    """Yield a fake Kubernetes API client from an async context manager."""
    yield api


async def _collect_events(api: MagicMock) -> list[dict[str, object]]:
    """Collect namespace events through the same API boundary as debug."""
    return await _get_namespace_events(api, "llama-benchmarks")


async def _collect_nodes(api: MagicMock) -> list[dict[str, object]]:
    """Collect node resources through the same API boundary as debug."""
    return await _get_node_resources(api)


def _source_files_under_debug_command() -> list[Path]:
    """Return source files that implement ``aiperf kube debug`` output paths."""
    root = Path(__file__).resolve().parents[4]
    return [
        root / "src/aiperf/cli_commands/kube/debug.py",
        root / "src/aiperf/cli_commands/kube/_debug_extract.py",
        root / "src/aiperf/cli_commands/kube/_debug_report.py",
    ]


def _print_bypass_calls(path: Path) -> list[_CallExpression]:
    """Return bare ``print``/``rich.print`` calls that bypass kube_console."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls: list[_CallExpression] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "print":
            calls.append(_CallExpression(path, node.lineno, "print"))
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "print"
            and isinstance(func.value, ast.Name)
            and func.value.id == "rich"
        ):
            calls.append(_CallExpression(path, node.lineno, "rich.print"))
    return calls


# ============================================================
# Missing targets
# ============================================================


class TestDebugMissingTargets:
    """Missing or empty target selections must return before pod diagnostics."""

    @pytest.mark.asyncio
    async def test_debug_without_namespace_or_last_job_returns_without_report(
        self,
    ) -> None:
        api = MagicMock()

        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                return_value=_fake_k8s_client(api),
            ),
            patch(
                "aiperf.kubernetes.cli_helpers.resolve_job_id_and_namespace",
                return_value=None,
            ) as mock_resolve_last,
            patch(
                "aiperf.cli_commands.kube.debug._get_node_resources",
                new=AsyncMock(return_value=[]),
            ) as mock_nodes,
            patch("aiperf.kubernetes.client.get_pods", new=AsyncMock()) as mock_pods,
            patch("aiperf.cli_commands.kube.debug._print_report") as mock_report,
        ):
            await debug(manage_options=KubeManageOptions())

        mock_resolve_last.assert_called_once_with(None, None)
        mock_nodes.assert_not_called()
        mock_pods.assert_not_called()
        mock_report.assert_not_called()

    @pytest.mark.asyncio
    async def test_debug_all_namespaces_with_no_jobsets_warns_and_skips_report(
        self,
    ) -> None:
        api = MagicMock()

        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                return_value=_fake_k8s_client(api),
            ),
            patch(
                "aiperf.kubernetes.client.list_jobsets",
                new=AsyncMock(return_value=[]),
            ) as mock_list_jobsets,
            patch("aiperf.kubernetes.console.print_warning") as mock_warning,
            patch("aiperf.kubernetes.client.get_pods", new=AsyncMock()) as mock_pods,
            patch("aiperf.cli_commands.kube.debug._print_report") as mock_report,
        ):
            await debug(all_namespaces=True)

        mock_list_jobsets.assert_called_once_with(api, all_namespaces=True)
        mock_warning.assert_called_once_with(
            "No AIPerf deployments found in any namespace"
        )
        mock_pods.assert_not_called()
        mock_report.assert_not_called()

    @pytest.mark.asyncio
    async def test_debug_job_id_not_found_prints_job_id_and_skips_report(self) -> None:
        api = MagicMock()

        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                return_value=_fake_k8s_client(api),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_job",
                new=AsyncMock(return_value=None),
            ) as mock_find_job,
            patch(
                "aiperf.kubernetes.client.find_aiperf_sweep",
                new=AsyncMock(return_value=None),
            ) as mock_find_sweep,
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(return_value=None),
            ) as mock_find_jobset,
            patch("aiperf.kubernetes.console.print_error") as mock_error,
            patch("aiperf.kubernetes.client.get_pods", new=AsyncMock()) as mock_pods,
            patch("aiperf.cli_commands.kube.debug._print_report") as mock_report,
        ):
            await debug(
                manage_options=KubeManageOptions(namespace="llama-benchmarks"),
                job_id="llama-crash-v07",
            )

        mock_find_job.assert_called_once_with(
            api, "llama-crash-v07", "llama-benchmarks"
        )
        mock_find_sweep.assert_called_once_with(
            api, "llama-crash-v07", "llama-benchmarks"
        )
        mock_find_jobset.assert_called_once_with(
            api, "llama-crash-v07", "llama-benchmarks"
        )
        mock_error.assert_called_once_with(
            "No AIPerf job found with ID: llama-crash-v07"
        )
        mock_pods.assert_not_called()
        mock_report.assert_not_called()

    @pytest.mark.asyncio
    async def test_debug_job_id_finds_completed_aiperfjob_cr_after_jobset_cleanup(
        self,
    ) -> None:
        api = MagicMock()
        job = AIPerfJobInfo(
            name="job-a",
            namespace="archived-benchmarks",
            phase="Completed",
            job_id="job-a",
        )

        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                return_value=_fake_k8s_client(api),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_job",
                new=AsyncMock(return_value=job),
            ) as mock_find_job,
            patch(
                "aiperf.kubernetes.client.find_aiperf_sweep",
                new=AsyncMock(return_value=None),
            ) as mock_find_sweep,
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(return_value=None),
            ) as mock_find_jobset,
            patch("aiperf.kubernetes.console.print_error") as mock_error,
            patch(
                "aiperf.kubernetes.client.get_pods",
                new=AsyncMock(return_value=[]),
            ) as mock_get_pods,
            patch(
                "aiperf.cli_commands.kube.debug._get_node_resources",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aiperf.cli_commands.kube.debug._get_namespace_events",
                new=AsyncMock(return_value=[]),
            ),
            patch("aiperf.cli_commands.kube.debug._print_report") as mock_report,
        ):
            await debug(job_id="job-a")

        mock_find_job.assert_called_once_with(api, "job-a", None)
        mock_find_sweep.assert_not_called()
        mock_find_jobset.assert_not_called()
        mock_error.assert_not_called()
        assert mock_get_pods.call_args.args[1] == "archived-benchmarks"
        assert "job-a" in mock_get_pods.call_args.args[2]
        mock_report.assert_called_once()

    @pytest.mark.asyncio
    async def test_debug_sweep_child_finds_aiperfjob_cr_after_jobset_cleanup(
        self,
    ) -> None:
        api = MagicMock()
        child = AIPerfJobInfo(
            name="sweep-a-v00",
            namespace="sweep-benchmarks",
            phase="Completed",
            job_id="sweep-a-v00",
            sweep_name="sweep-a",
            variation_index=0,
        )

        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                return_value=_fake_k8s_client(api),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_job",
                new=AsyncMock(return_value=child),
            ) as mock_find_job,
            patch(
                "aiperf.kubernetes.client.find_aiperf_sweep",
                new=AsyncMock(return_value=None),
            ) as mock_find_sweep,
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(return_value=None),
            ) as mock_find_jobset,
            patch("aiperf.kubernetes.console.print_error") as mock_error,
            patch(
                "aiperf.kubernetes.client.get_pods",
                new=AsyncMock(return_value=[]),
            ) as mock_get_pods,
            patch(
                "aiperf.cli_commands.kube.debug._get_node_resources",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aiperf.cli_commands.kube.debug._get_namespace_events",
                new=AsyncMock(return_value=[]),
            ),
            patch("aiperf.cli_commands.kube.debug._print_report"),
        ):
            await debug(job_id="sweep-a", variation=0)

        mock_find_job.assert_called_once_with(api, "sweep-a-v00", None)
        mock_find_sweep.assert_not_called()
        mock_find_jobset.assert_not_called()
        mock_error.assert_not_called()
        assert mock_get_pods.call_args.args[1] == "sweep-benchmarks"
        assert "sweep-a-v00" in mock_get_pods.call_args.args[2]

    @pytest.mark.asyncio
    async def test_debug_parent_sweep_cr_resolves_namespace_after_jobset_cleanup(
        self,
    ) -> None:
        api = MagicMock()
        sweep = AIPerfSweepInfo(
            name="sweep-a",
            namespace="sweep-benchmarks",
            phase="Completed",
        )

        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                return_value=_fake_k8s_client(api),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_job",
                new=AsyncMock(return_value=None),
            ) as mock_find_job,
            patch(
                "aiperf.kubernetes.client.find_aiperf_sweep",
                new=AsyncMock(return_value=sweep),
            ) as mock_find_sweep,
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(return_value=None),
            ) as mock_find_jobset,
            patch("aiperf.kubernetes.console.print_error") as mock_error,
            patch(
                "aiperf.kubernetes.client.get_pods",
                new=AsyncMock(return_value=[]),
            ) as mock_get_pods,
            patch(
                "aiperf.cli_commands.kube.debug._get_node_resources",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aiperf.cli_commands.kube.debug._get_namespace_events",
                new=AsyncMock(return_value=[]),
            ),
            patch("aiperf.cli_commands.kube.debug._print_report") as mock_report,
        ):
            await debug(job_id="sweep-a")

        mock_find_job.assert_called_once_with(api, "sweep-a", None)
        mock_find_sweep.assert_called_once_with(api, "sweep-a", None)
        mock_find_jobset.assert_not_called()
        mock_error.assert_not_called()
        assert mock_get_pods.call_args.args[1] == "sweep-benchmarks"
        mock_report.assert_called_once()


# ============================================================
# Namespace and kube-context propagation
# ============================================================


class TestDebugClusterSelectionPropagation:
    """Cluster-selection flags must reach the only Kubernetes client boundary."""

    @pytest.mark.asyncio
    async def test_debug_explicit_namespace_forwards_kubeconfig_and_context(
        self,
    ) -> None:
        api = MagicMock()
        factory = _RecordingK8sClientFactory(api=api)

        with (
            patch("aiperf.kubernetes.client.k8s_client", new=factory),
            patch(
                "aiperf.kubernetes.client.get_pods",
                new=AsyncMock(return_value=[]),
            ) as mock_get_pods,
            patch(
                "aiperf.cli_commands.kube.debug._get_node_resources",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aiperf.cli_commands.kube.debug._get_namespace_events",
                new=AsyncMock(return_value=[]),
            ),
            patch("aiperf.cli_commands.kube.debug._print_report") as mock_report,
        ):
            await debug(
                manage_options=KubeManageOptions(
                    namespace="llama-benchmarks",
                    kubeconfig="/opt/ci/kubeconfigs/aiperf-ci.yaml",
                    kube_context="kind-aiperf-ci",
                )
            )

        assert factory.calls == [
            _K8sClientCall(
                kubeconfig="/opt/ci/kubeconfigs/aiperf-ci.yaml",
                context="kind-aiperf-ci",
            )
        ]
        assert mock_get_pods.call_args.args[0] is api
        assert mock_get_pods.call_args.args[1] == "llama-benchmarks"
        mock_report.assert_called_once()

    @pytest.mark.asyncio
    async def test_debug_sweep_child_selector_is_resolved_before_namespace_lookup(
        self,
    ) -> None:
        api = MagicMock()
        jobset = JobSetInfo(
            name="llama-sweep-v07-t0",
            namespace="llama-benchmarks",
            jobset={},
            status="Running",
        )

        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                return_value=_fake_k8s_client(api),
            ),
            patch(
                "aiperf.kubernetes.client.find_aiperf_job",
                new=AsyncMock(return_value=None),
            ) as mock_find_job,
            patch(
                "aiperf.kubernetes.client.find_aiperf_sweep",
                new=AsyncMock(return_value=None),
            ) as mock_find_sweep,
            patch(
                "aiperf.kubernetes.client.find_jobset",
                new=AsyncMock(return_value=jobset),
            ) as mock_find_jobset,
            patch(
                "aiperf.kubernetes.client.get_pods",
                new=AsyncMock(return_value=[]),
            ) as mock_get_pods,
            patch(
                "aiperf.cli_commands.kube.debug._get_node_resources",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aiperf.cli_commands.kube.debug._get_namespace_events",
                new=AsyncMock(return_value=[]),
            ),
            patch("aiperf.cli_commands.kube.debug._print_report"),
        ):
            await debug(
                manage_options=KubeManageOptions(namespace="llama-benchmarks"),
                job_id="llama-sweep",
                variation=7,
                trial=0,
            )

        mock_find_job.assert_called_once_with(
            api, "llama-sweep-v07-t0", "llama-benchmarks"
        )
        mock_find_sweep.assert_called_once_with(
            api, "llama-sweep-v07-t0", "llama-benchmarks"
        )
        mock_find_jobset.assert_called_once_with(
            api, "llama-sweep-v07-t0", "llama-benchmarks"
        )
        assert "llama-sweep-v07-t0" in mock_get_pods.call_args.args[2]

    @pytest.mark.asyncio
    async def test_debug_all_namespaces_sorts_report_order_for_stable_output(
        self,
    ) -> None:
        api = MagicMock()
        jobsets = [
            JobSetInfo(
                name="run-b", namespace="zeta-bench", jobset={}, status="Running"
            ),
            JobSetInfo(
                name="run-a", namespace="alpha-bench", jobset={}, status="Running"
            ),
        ]

        with (
            patch(
                "aiperf.kubernetes.client.k8s_client",
                return_value=_fake_k8s_client(api),
            ),
            patch(
                "aiperf.kubernetes.client.list_jobsets",
                new=AsyncMock(return_value=jobsets),
            ),
            patch("aiperf.kubernetes.client.get_pods", new=AsyncMock(return_value=[])),
            patch(
                "aiperf.cli_commands.kube.debug._get_node_resources",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "aiperf.cli_commands.kube.debug._get_namespace_events",
                new=AsyncMock(return_value=[]),
            ),
            patch("aiperf.cli_commands.kube.debug._print_report") as mock_report,
        ):
            await debug(all_namespaces=True)

        assert [call.args[0] for call in mock_report.call_args_list] == [
            "alpha-bench",
            "zeta-bench",
        ]


# ============================================================
# Partial diagnostic aggregation
# ============================================================


class TestDebugPartialAggregation:
    """Malformed optional diagnostics must not erase healthy sections."""

    @pytest.mark.asyncio
    async def test_debug_namespace_event_api_failure_still_prints_pod_report(
        self,
    ) -> None:
        pod = MagicMock()
        pod.name = "llama-crash-v07-t0-worker-0"
        pod.raw = {
            "metadata": {"name": pod.name, "namespace": "llama-benchmarks"},
            "spec": {"nodeName": "gpu-node-7"},
            "status": {"phase": "Running", "containerStatuses": []},
        }
        api = MagicMock()

        with (
            patch(
                "aiperf.kubernetes.client.get_pods", new=AsyncMock(return_value=[pod])
            ),
            patch(
                "aiperf.cli_commands.kube.debug._get_namespace_events",
                new=AsyncMock(return_value=[]),
            ),
            patch("aiperf.cli_commands.kube.debug._print_report") as mock_report,
        ):
            await _debug_namespace(
                api,
                ns="llama-benchmarks",
                job_id=None,
                verbose=False,
                node_resources=[{"name": "gpu-node-7", "pressure": []}],
            )

        assert mock_report.call_args.args[0] == "llama-benchmarks"
        assert mock_report.call_args.kwargs["pod_infos"][0]["name"] == pod.name
        assert mock_report.call_args.kwargs["events"] == []

    @pytest.mark.parametrize(
        "collector_name,collector",
        [
            param("events", _collect_events, id="event-serializer-error"),
            param("nodes", _collect_nodes, id="node-serializer-error"),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_best_effort_collectors_swallow_serializer_errors(
        self,
        collector_name: str,
        collector: Callable[[MagicMock], Awaitable[list[dict[str, object]]]],
    ) -> None:
        api = MagicMock()
        api.sanitize_for_serialization.side_effect = RuntimeError(
            f"malformed {collector_name} payload from apiserver"
        )
        core = MagicMock()
        core.list_namespaced_event = AsyncMock(
            return_value=_listing_response([SimpleNamespace(metadata={})])
        )
        core.list_node = AsyncMock(
            return_value=_listing_response([SimpleNamespace(status={})])
        )

        with patch("kubernetes_asyncio.client.CoreV1Api", return_value=core):
            result = await collector(api)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_problem_pod_logs_preserves_each_container_error_placeholder(
        self,
    ) -> None:
        api = MagicMock()
        core = MagicMock()
        core.read_namespaced_pod_log = AsyncMock(
            side_effect=[
                ApiException(status=500, reason="log backend unavailable"),
                RuntimeError("stderr stream truncated by kubelet"),
            ]
        )

        with patch("kubernetes_asyncio.client.CoreV1Api", return_value=core):
            result = await _get_problem_pod_logs(
                api, _pod_info_with_two_problem_containers(), tail_lines=40
            )

        assert result == {
            "llama-crash-v07-t0-worker-0": {
                "worker": "<logs unavailable>",
                "record-processor": "<error fetching logs>",
            }
        }
        assert core.read_namespaced_pod_log.call_args_list[0].kwargs == {
            "name": "llama-crash-v07-t0-worker-0",
            "namespace": "llama-benchmarks",
            "container": "worker",
            "tail_lines": 40,
        }


# ============================================================
# Output facade boundary
# ============================================================


class TestDebugOutputFacadeBoundary:
    """The text-only debug command must route user output through kube_console."""

    def test_debug_command_modules_do_not_call_print_or_rich_print(self) -> None:
        bypasses = [
            bypass
            for path in _source_files_under_debug_command()
            for bypass in _print_bypass_calls(path)
        ]

        assert bypasses == []

    def test_debug_module_public_surface_does_not_claim_json_output(self) -> None:
        """Debug is text-only today; JSON/dataclass schema tests are not applicable."""
        parameters = debug_cmd.debug.__annotations__
        assert "output" not in parameters
