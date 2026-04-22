# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.cli_commands.kube module."""

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.cli_commands.kube.attach import attach
from aiperf.cli_commands.kube.list_ import list_jobs
from aiperf.cli_commands.kube.logs import logs
from aiperf.cli_commands.kube.results import results
from aiperf.config import AIPerfConfig
from aiperf.config.kube import KubeManageOptions
from aiperf.kubernetes.cli_helpers import resolve_job_id_and_namespace
from aiperf.kubernetes.console import LastBenchmarkInfo
from aiperf.kubernetes.models import AIPerfJobInfo, JobSetInfo
from aiperf.kubernetes.ui_dispatch import print_progress_message, print_realtime_metrics


@asynccontextmanager
async def _fake_k8s_client(api: Any):
    """Drop-in replacement for ``k8s_client`` yielding the provided api object."""
    yield api


_BUILD_CONFIG = "aiperf.config.cli_converter.build_aiperf_config"


def _test_config() -> AIPerfConfig:
    """Create a minimal AIPerfConfig for testing."""
    return AIPerfConfig(
        models=["test-model"],
        endpoint={
            "urls": ["http://localhost:8000/v1/chat/completions"],
        },
        datasets={
            "main": {
                "type": "synthetic",
                "entries": 10,
                "prompts": {"isl": 32, "osl": 16},
            },
        },
        phases={
            "default": {
                "type": "concurrency",
                "requests": 10,
                "concurrency": 1,
            },
        },
    )


def _sample_job_info(
    name: str = "aiperf-abc123",
    namespace: str = "default",
    phase: str = "Running",
    job_id: str = "abc123",
) -> AIPerfJobInfo:
    """Create a sample AIPerfJobInfo for testing."""
    return AIPerfJobInfo(
        name=name,
        namespace=namespace,
        phase=phase,
        job_id=job_id,
        created="2026-01-15T10:00:00Z",
        model="test-model",
    )


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def manage_options():
    """Create a KubeManageOptions instance for testing."""
    return KubeManageOptions(kubeconfig=None, namespace=None)


@pytest.fixture
def mock_kube_client():
    """Mock an ApiClient + patched free-function call sites for CLI tests.

    Exposes an ``AsyncMock`` whose attributes (``list_jobs``, ``find_job``,
    ``find_jobset``, ``list_jobsets``, ``get_pods``, …) back the patched
    free functions so tests can set return values via
    ``mock_kube_client.list_jobs.return_value = ...`` and match call
    assertions with `mock_kube_client.list_jobs.assert_called_once_with(...)`.
    Each wrapper drops the leading ``api`` arg the real free function expects.
    """
    mock_client = AsyncMock()
    api = MagicMock()
    api.close = AsyncMock()

    async def _strip_api_list_jobs(_api, **kwargs):
        return await mock_client.list_jobs(**kwargs)

    async def _strip_api_find_job(_api, *args, **kwargs):
        return await mock_client.find_job(*args, **kwargs)

    async def _strip_api_find_jobset(_api, *args, **kwargs):
        return await mock_client.find_jobset(*args, **kwargs)

    async def _strip_api_list_jobsets(_api, **kwargs):
        return await mock_client.list_jobsets(**kwargs)

    async def _strip_api_get_pods(_api, *args, **kwargs):
        return await mock_client.get_pods(*args, **kwargs)

    async def _strip_api_find_controller_pod(_api, *args, **kwargs):
        return await mock_client.find_controller_pod(*args, **kwargs)

    async def _strip_api_find_operator_pod(_api, *args, **kwargs):
        return await mock_client.find_operator_pod(*args, **kwargs)

    with (
        patch(
            "aiperf.kubernetes.client.k8s_client",
            return_value=_fake_k8s_client(api),
        ),
        patch(
            "aiperf.kubernetes.cli_helpers._open_api_client",
            new=AsyncMock(return_value=api),
        ),
        patch(
            "aiperf.kubernetes.client.list_aiperf_jobs",
            new=_strip_api_list_jobs,
        ),
        patch(
            "aiperf.kubernetes.client.find_aiperf_job",
            new=_strip_api_find_job,
        ),
        patch(
            "aiperf.kubernetes.client.find_jobset",
            new=_strip_api_find_jobset,
        ),
        patch(
            "aiperf.kubernetes.client.list_jobsets",
            new=_strip_api_list_jobsets,
        ),
        patch(
            "aiperf.kubernetes.client.get_pods",
            new=_strip_api_get_pods,
        ),
        patch(
            "aiperf.kubernetes.client.find_controller_pod",
            new=_strip_api_find_controller_pod,
        ),
        patch(
            "aiperf.kubernetes.client.find_operator_pod",
            new=_strip_api_find_operator_pod,
        ),
    ):
        yield mock_client


@pytest.fixture
def sample_jobset_item() -> JobSetInfo:
    """Create a sample JobSetInfo from Kubernetes API."""
    raw = {
        "metadata": {
            "name": "aiperf-abc123",
            "namespace": "default",
            "creationTimestamp": "2026-01-15T10:00:00Z",
            "labels": {
                "app": "aiperf",
                "aiperf.nvidia.com/job-id": "abc123",
            },
        },
        "status": {
            "conditions": [{"type": "Completed", "status": "True"}],
        },
    }
    return JobSetInfo.from_raw(raw)


@pytest.fixture
def running_jobset_info():
    """Create a JobSetInfo with Running status."""
    return JobSetInfo(
        name="aiperf-abc123",
        namespace="default",
        jobset={"metadata": {"creationTimestamp": "2026-01-15T10:00:00Z"}},
        status="Running",
    )


@pytest.fixture
def completed_jobset_info():
    """Create a JobSetInfo with Completed status."""
    return JobSetInfo(
        name="aiperf-abc123",
        namespace="default",
        jobset={"metadata": {"creationTimestamp": "2026-01-15T10:00:00Z"}},
        status="Completed",
    )


@pytest.fixture
def failed_jobset_info():
    """Create a JobSetInfo with Failed status."""
    return JobSetInfo(
        name="aiperf-abc123",
        namespace="default",
        jobset={"metadata": {"creationTimestamp": "2026-01-15T10:00:00Z"}},
        status="Failed",
    )


# =============================================================================
# List Command Tests
# =============================================================================


class TestListJobsCommand:
    """Tests for the kube list_jobs command (uses AIPerfJob CRs)."""

    async def test_list_no_jobs_found(
        self, mock_kube_client, manage_options, capsys
    ) -> None:
        """Test list_jobs when no jobs are found."""
        mock_kube_client.list_jobs.return_value = []
        await list_jobs(manage_options=manage_options)

        captured = capsys.readouterr()
        assert "No AIPerf jobs found" in captured.out

    async def test_list_shows_jobs(
        self, mock_kube_client, manage_options, capsys
    ) -> None:
        """Test list shows found jobs."""
        from aiperf.kubernetes.console import console as _console

        _console.width = 200
        try:
            mock_kube_client.list_jobs.return_value = [_sample_job_info()]
            await list_jobs(manage_options=manage_options)
        finally:
            _console.width = None

        captured = capsys.readouterr()
        assert "aiperf-abc123" in captured.out
        assert "default" in captured.out

    async def test_list_with_namespace(self, mock_kube_client, manage_options) -> None:
        """Test list with specific namespace."""
        opts = KubeManageOptions(namespace="my-namespace")
        mock_kube_client.list_jobs.return_value = []
        await list_jobs(manage_options=opts)

        mock_kube_client.list_jobs.assert_called_once_with(
            namespace="my-namespace",
            all_namespaces=False,
            status_filter=None,
        )

    async def test_list_all_namespaces(self, mock_kube_client, manage_options) -> None:
        """Test list with all namespaces flag."""
        mock_kube_client.list_jobs.return_value = []
        await list_jobs(manage_options=manage_options, all_namespaces=True)

        mock_kube_client.list_jobs.assert_called_once_with(
            namespace=None,
            all_namespaces=True,
            status_filter=None,
        )

    async def test_list_with_job_id(self, mock_kube_client, manage_options) -> None:
        """Test list with specific job ID uses find_job."""
        mock_kube_client.find_job.return_value = None
        await list_jobs(manage_options=manage_options, job_id="abc123")

        mock_kube_client.find_job.assert_called_once_with("abc123", None)

    async def test_list_running_filter(self, mock_kube_client, manage_options) -> None:
        """Test list with --running filter."""
        mock_kube_client.list_jobs.return_value = []
        await list_jobs(manage_options=manage_options, running=True)

        mock_kube_client.list_jobs.assert_called_once_with(
            namespace=None,
            all_namespaces=True,
            status_filter="Running",
        )

    async def test_list_completed_filter(
        self, mock_kube_client, manage_options
    ) -> None:
        """Test list with --completed filter."""
        mock_kube_client.list_jobs.return_value = []
        await list_jobs(manage_options=manage_options, completed=True)

        mock_kube_client.list_jobs.assert_called_once_with(
            namespace=None,
            all_namespaces=True,
            status_filter="Completed",
        )

    async def test_list_failed_filter(self, mock_kube_client, manage_options) -> None:
        """Test list with --failed filter."""
        mock_kube_client.list_jobs.return_value = []
        await list_jobs(manage_options=manage_options, failed=True)

        mock_kube_client.list_jobs.assert_called_once_with(
            namespace=None,
            all_namespaces=True,
            status_filter="Failed",
        )

    async def test_list_filter_message_includes_phase(
        self, mock_kube_client, manage_options, capsys
    ) -> None:
        """Test list with filter shows phase in message."""
        mock_kube_client.list_jobs.return_value = []
        await list_jobs(manage_options=manage_options, running=True)

        captured = capsys.readouterr()
        assert "No AIPerf jobs found with phase 'Running'" in captured.out


# =============================================================================
# Logs Command Tests
# =============================================================================


class TestLogsCommand:
    """Tests for the kube logs command."""

    async def test_logs_no_pods_found(
        self, mock_kube_client, manage_options, capsys
    ) -> None:
        """Test logs when no pods are found."""
        mock_kube_client.get_pods.return_value = []
        await logs(job_id="nonexistent", manage_options=manage_options)

        captured = capsys.readouterr()
        assert "No pods found" in captured.out

    async def test_logs_with_namespace(
        self, mock_kube_client, manage_options, capsys
    ) -> None:
        """Test logs with specific namespace."""
        opts = KubeManageOptions(namespace="default")

        mock_pod = MagicMock()
        mock_pod.name = "aiperf-abc123-controller-0-0"
        mock_pod.raw = {
            "spec": {"containers": [{"name": "control-plane"}]},
        }

        async def _mock_logs(*args, **kwargs):
            yield "log output"

        mock_pod.logs = _mock_logs
        mock_kube_client.get_pods.return_value = [mock_pod]

        await logs(job_id="abc123", manage_options=opts)


# =============================================================================
# Results Command Tests
# =============================================================================


class TestResultsCommand:
    """Tests for the kube results command."""

    async def test_results_from_api_via_from_pods(
        self, mock_kube_client, manage_options, completed_jobset_info, capsys, tmp_path
    ) -> None:
        """Test results retrieval from API service with --from-pods --summary-only."""
        output_dir = tmp_path / "results"

        mock_kube_client.find_job.return_value = _sample_job_info(phase="Completed")
        mock_kube_client.find_jobset.return_value = completed_jobset_info
        with patch(
            "aiperf.kubernetes.results.retrieve_results_from_api",
            new=AsyncMock(return_value=True),
        ):
            await results(
                job_id="abc123",
                manage_options=manage_options,
                output=output_dir,
                from_pods=True,
                all_artifacts=False,
            )

        captured = capsys.readouterr()
        assert "Results" in captured.out
        assert "Saved to" in captured.out

    async def test_results_from_pods_api_fails_tries_kubectl_cp(
        self, mock_kube_client, manage_options, running_jobset_info, capsys, tmp_path
    ) -> None:
        """Test --from-pods falls back to kubectl cp when API fails."""
        output_dir = tmp_path / "results"

        mock_kube_client.find_job.return_value = _sample_job_info()
        mock_kube_client.find_jobset.return_value = running_jobset_info
        with (
            patch(
                "aiperf.kubernetes.results.retrieve_results_from_api",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "aiperf.kubernetes.results.find_controller_pod",
                new=AsyncMock(return_value=None),
            ),
        ):
            await results(
                job_id="abc123",
                manage_options=manage_options,
                output=output_dir,
                from_pods=True,
                all_artifacts=False,
            )

        captured = capsys.readouterr()
        assert "No controller pod found" in captured.out


# =============================================================================
# Attach Command Tests
# =============================================================================


class TestAttachCommand:
    """Tests for the kube attach command."""

    async def test_attach_job_not_found(
        self, mock_kube_client, manage_options, capsys
    ) -> None:
        """Test attach when job is not found via resolve_job."""
        mock_kube_client.find_job.return_value = None
        mock_kube_client.find_jobset.return_value = None
        await attach(job_id="nonexistent", manage_options=manage_options)

        captured = capsys.readouterr()
        assert "No AIPerf job found" in captured.out

    async def test_attach_completed_job(
        self, mock_kube_client, manage_options, completed_jobset_info, capsys
    ) -> None:
        """Test attach on completed job (resolved via CR, attach checks JobSet)."""
        mock_kube_client.find_job.return_value = _sample_job_info(phase="Completed")
        mock_kube_client.find_jobset.return_value = completed_jobset_info
        await attach(job_id="abc123", manage_options=manage_options)

        captured = capsys.readouterr()
        assert "already completed" in captured.out

    async def test_attach_failed_job(
        self, mock_kube_client, manage_options, failed_jobset_info, capsys
    ) -> None:
        """Test attach on failed job."""
        mock_kube_client.find_job.return_value = _sample_job_info(phase="Failed")
        mock_kube_client.find_jobset.return_value = failed_jobset_info
        await attach(job_id="abc123", manage_options=manage_options)

        captured = capsys.readouterr()
        assert "has failed" in captured.out


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestPrintProgressMessage:
    """Tests for print_progress_message helper."""

    def test_credit_phase_start(self, capsys) -> None:
        """Test printing CREDIT_PHASE_START message."""
        print_progress_message(
            {
                "message_type": "CREDIT_PHASE_START",
                "stats": {"phase": "profiling"},
            }
        )

        captured = capsys.readouterr()
        assert "[PHASE]" in captured.out
        assert "profiling" in captured.out

    def test_credit_phase_progress(self, capsys) -> None:
        """Test printing CREDIT_PHASE_PROGRESS message."""
        print_progress_message(
            {
                "message_type": "CREDIT_PHASE_PROGRESS",
                "stats": {
                    "phase": "profiling",
                    "requests_completed": 500,
                    "total_expected_requests": 1000,
                },
            }
        )

        captured = capsys.readouterr()
        assert "500/1000" in captured.out
        assert "50.0%" in captured.out

    def test_credit_phase_complete(self, capsys) -> None:
        """Test printing CREDIT_PHASE_COMPLETE message."""
        print_progress_message(
            {
                "message_type": "CREDIT_PHASE_COMPLETE",
                "stats": {"phase": "warmup"},
            }
        )

        captured = capsys.readouterr()
        assert "[PHASE]" in captured.out
        assert "Completed" in captured.out
        assert "warmup" in captured.out

    def test_all_records_received(self, capsys) -> None:
        """Test printing ALL_RECORDS_RECEIVED message."""
        print_progress_message({"message_type": "ALL_RECORDS_RECEIVED"})

        captured = capsys.readouterr()
        assert "[COMPLETE]" in captured.out

    def test_subscribed_message_ignored(self, capsys) -> None:
        """Test subscribed message is ignored."""
        print_progress_message({"message_type": "subscribed"})

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_worker_status_summary(self, capsys) -> None:
        """Test printing WORKER_STATUS_SUMMARY message."""
        print_progress_message(
            {
                "message_type": "WORKER_STATUS_SUMMARY",
                "workers": {
                    "worker-1": {"status": "HEALTHY"},
                    "worker-2": {"status": "HEALTHY"},
                    "worker-3": {"status": "UNHEALTHY"},
                },
            }
        )

        captured = capsys.readouterr()
        assert "[WORKERS]" in captured.out
        assert "2/3" in captured.out


class TestPrintRealtimeMetrics:
    """Tests for print_realtime_metrics helper."""

    def test_prints_key_metrics(self, capsys) -> None:
        """Test printing key metrics from realtime metrics message."""
        print_realtime_metrics(
            {
                "metrics": [
                    {"tag": "throughput", "value": 100.5, "display_unit": "req/s"},
                    {"tag": "latency_p50", "value": 50.2, "unit": "ms"},
                    {"tag": "latency_p99", "value": 120.3, "display_unit": "ms"},
                    {"tag": "ttft_p50", "value": 25.0, "unit": "ms"},
                ]
            }
        )

        captured = capsys.readouterr()
        assert "[METRIC]" in captured.out
        assert "throughput: 100.50" in captured.out
        assert "latency_p50: 50.20" in captured.out

    def test_no_output_when_no_key_metrics(self, capsys) -> None:
        """Test no output when no key metrics are present."""
        print_realtime_metrics(
            {
                "metrics": [
                    {"tag": "other_metric", "value": 42.0, "unit": "things"},
                ]
            }
        )

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_empty_metrics_list(self, capsys) -> None:
        """Test handling empty metrics list."""
        print_realtime_metrics({"metrics": []})

        captured = capsys.readouterr()
        assert captured.out == ""


# =============================================================================
# Resolve Job ID and Namespace Tests
# =============================================================================


class TestResolveJobIdAndNamespace:
    """Tests for resolve_job_id_and_namespace helper."""

    def test_returns_provided_job_id(self) -> None:
        """Test returns provided job_id and namespace."""
        result = resolve_job_id_and_namespace("abc123", "my-namespace")
        assert result == ("abc123", "my-namespace")

    def test_returns_job_id_with_default_namespace(self) -> None:
        """Test returns job_id with default namespace when only job_id provided."""
        result = resolve_job_id_and_namespace("abc123", None)
        assert result == ("abc123", "aiperf-benchmarks")

    def test_uses_last_benchmark_when_no_job_id(self, capsys) -> None:
        """Test uses last benchmark info when no job_id specified."""
        with patch(
            "aiperf.kubernetes.cli_helpers.get_last_benchmark",
            return_value=LastBenchmarkInfo(job_id="last-job", namespace="last-ns"),
        ):
            result = resolve_job_id_and_namespace(None, None)

        assert result == ("last-job", "last-ns")
        captured = capsys.readouterr()
        assert "Using last benchmark" in captured.out

    def test_uses_provided_namespace_over_last_benchmark(self, capsys) -> None:
        """Test provided namespace overrides last benchmark namespace."""
        with patch(
            "aiperf.kubernetes.cli_helpers.get_last_benchmark",
            return_value=LastBenchmarkInfo(job_id="last-job", namespace="last-ns"),
        ):
            result = resolve_job_id_and_namespace(None, "override-ns")

        assert result == ("last-job", "override-ns")

    def test_returns_none_when_no_job_id_and_no_last_benchmark(self, capsys) -> None:
        """Test returns None when no job_id and no last benchmark."""
        with patch(
            "aiperf.kubernetes.cli_helpers.get_last_benchmark", return_value=None
        ):
            result = resolve_job_id_and_namespace(None, None)

        assert result is None
        captured = capsys.readouterr()
        assert "No job_id specified" in captured.out
        assert "no previous benchmark found" in captured.out


# =============================================================================
# Logs Command Additional Tests
# =============================================================================


class TestLogsCommandAdditional:
    """Additional tests for logs command."""

    async def test_logs_uses_last_benchmark(self, mock_kube_client, capsys) -> None:
        """Test logs uses last benchmark when no job_id specified."""
        manage_options = KubeManageOptions()

        mock_kube_client.get_pods.return_value = []

        with patch(
            "aiperf.kubernetes.cli_helpers.get_last_benchmark",
            return_value=LastBenchmarkInfo(job_id="last123", namespace="last-ns"),
        ):
            await logs(manage_options=manage_options)

        captured = capsys.readouterr()
        assert "Using last benchmark" in captured.out

    async def test_logs_returns_early_when_no_job_found(
        self, mock_kube_client, capsys
    ) -> None:
        """Test logs returns early when no job_id and no last benchmark."""
        manage_options = KubeManageOptions()

        with patch(
            "aiperf.kubernetes.cli_helpers.get_last_benchmark", return_value=None
        ):
            await logs(manage_options=manage_options)

        captured = capsys.readouterr()
        assert "No job_id specified" in captured.out


# =============================================================================
# Results Command Additional Tests
# =============================================================================


class TestResultsCommandAdditional:
    """Additional tests for results command."""

    async def test_results_uses_last_benchmark(
        self, mock_kube_client, capsys, tmp_path
    ) -> None:
        """Test results uses last benchmark when no job_id specified."""
        manage_options = KubeManageOptions()
        output_dir = tmp_path / "results"

        mock_kube_client.find_job.return_value = _sample_job_info()
        mock_kube_client.find_jobset.return_value = None
        with (
            patch(
                "aiperf.kubernetes.cli_helpers.get_last_benchmark",
                return_value=LastBenchmarkInfo(job_id="last123", namespace="last-ns"),
            ),
            patch(
                "aiperf.kubernetes.results.retrieve_results_from_operator",
                new=AsyncMock(return_value=False),
            ),
        ):
            await results(
                manage_options=manage_options,
                output=output_dir,
            )

        captured = capsys.readouterr()
        assert "Using last benchmark" in captured.out

    async def test_results_from_pods_flag(
        self, mock_kube_client, running_jobset_info, tmp_path
    ) -> None:
        """Test results with --from-pods flag uses API then kubectl cp."""
        manage_options = KubeManageOptions()
        output_dir = tmp_path / "results"

        mock_kube_client.find_job.return_value = _sample_job_info()
        mock_kube_client.find_jobset.return_value = running_jobset_info
        with patch(
            "aiperf.kubernetes.results.retrieve_results_from_api",
            new_callable=AsyncMock,
        ) as mock_api:
            mock_api.return_value = True
            await results(
                job_id="abc123",
                manage_options=manage_options,
                output=output_dir,
                from_pods=True,
                all_artifacts=False,
            )

            mock_api.assert_called_once()

    async def test_results_all_artifacts_from_pods(
        self, mock_kube_client, running_jobset_info, tmp_path
    ) -> None:
        """Test results with --from-pods --all downloads all artifacts."""
        manage_options = KubeManageOptions()
        output_dir = tmp_path / "results"

        mock_kube_client.find_job.return_value = _sample_job_info()
        mock_kube_client.find_jobset.return_value = running_jobset_info
        with patch(
            "aiperf.kubernetes.results.retrieve_all_artifacts",
            new_callable=AsyncMock,
        ) as mock_archive:
            await results(
                job_id="abc123",
                manage_options=manage_options,
                output=output_dir,
                from_pods=True,
                all_artifacts=True,
            )

            mock_archive.assert_called_once()

    async def test_results_default_uses_operator(
        self, mock_kube_client, running_jobset_info, tmp_path
    ) -> None:
        """Test results defaults to operator storage."""
        manage_options = KubeManageOptions()
        output_dir = tmp_path / "results"

        mock_kube_client.find_job.return_value = _sample_job_info()
        mock_kube_client.find_jobset.return_value = running_jobset_info
        with patch(
            "aiperf.kubernetes.results.retrieve_results_from_operator",
            new_callable=AsyncMock,
        ) as mock_operator:
            mock_operator.return_value = True
            await results(
                job_id="abc123",
                manage_options=manage_options,
                output=output_dir,
            )

            mock_operator.assert_called_once()


# =============================================================================
# Attach Command Additional Tests
# =============================================================================


class TestAttachCommandAdditional:
    """Additional tests for attach command."""

    async def test_attach_uses_last_benchmark(
        self, mock_kube_client, completed_jobset_info, capsys
    ) -> None:
        """Test attach uses last benchmark when no job_id specified."""
        manage_options = KubeManageOptions()

        mock_kube_client.find_job.return_value = _sample_job_info(phase="Completed")
        mock_kube_client.find_jobset.return_value = completed_jobset_info
        with patch(
            "aiperf.kubernetes.cli_helpers.get_last_benchmark",
            return_value=LastBenchmarkInfo(job_id="abc123", namespace="default"),
        ):
            await attach(manage_options=manage_options)

        captured = capsys.readouterr()
        assert "Using last benchmark" in captured.out

    async def test_attach_no_controller_pod(
        self, mock_kube_client, running_jobset_info, capsys
    ) -> None:
        """Test attach when no controller pod found."""
        manage_options = KubeManageOptions()

        mock_kube_client.find_job.return_value = _sample_job_info()
        mock_kube_client.find_jobset.return_value = running_jobset_info
        with patch(
            "aiperf.kubernetes.attach.find_controller_pod",
            new=AsyncMock(return_value=None),
        ):
            await attach(job_id="abc123", manage_options=manage_options)

        captured = capsys.readouterr()
        assert "No controller pod found" in captured.out


# =============================================================================
# List Command Wide Output Tests
# =============================================================================


class TestListCommandWide:
    """Tests for list command with wide output."""

    async def test_list_wide_output(
        self, mock_kube_client, manage_options, capsys
    ) -> None:
        """Test list with wide output shows model column."""
        from aiperf.kubernetes.console import console as _console

        _console.width = 200
        try:
            mock_kube_client.list_jobs.return_value = [
                _sample_job_info(phase="Completed")
            ]
            await list_jobs(manage_options=manage_options, wide=True)
        finally:
            _console.width = None

        captured = capsys.readouterr()
        assert "aiperf-abc123" in captured.out
        assert "MODEL" in captured.out

    async def test_list_no_filter_shows_all(
        self, mock_kube_client, manage_options
    ) -> None:
        """Test list with no filter passes None as status_filter."""
        mock_kube_client.list_jobs.return_value = []
        await list_jobs(manage_options=manage_options)

        mock_kube_client.list_jobs.assert_called_once_with(
            namespace=None,
            all_namespaces=True,
            status_filter=None,
        )
