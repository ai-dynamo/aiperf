# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.attach module.

Focuses on:
- attach_to_benchmark: early returns for missing/completed/failed jobs and non-running pods
- auto_attach_workflow: wait vs no-wait paths, ws vs log streaming, result retrieval
- retrieve_and_display_results: artifact retrieval, custom name handling, success/failure display
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client import ApiClient

from aiperf.kubernetes.attach import (
    attach_to_benchmark,
    auto_attach_workflow,
    retrieve_and_display_results,
)
from aiperf.kubernetes.enums import PodPhase

# =============================================================================
# Helpers
# =============================================================================

_MODULE = "aiperf.kubernetes.attach"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_api() -> MagicMock:
    """Mock kubernetes_asyncio ApiClient."""
    api = MagicMock(spec=ApiClient)
    api.close = AsyncMock()
    return api


@pytest.fixture
def patch_k8s_client(mock_api: MagicMock):
    """Patch k8s_client to yield mock_api."""

    @asynccontextmanager
    async def _cm(**_kwargs):
        yield mock_api

    with patch(f"{_MODULE}.k8s_client", side_effect=_cm):
        yield mock_api


@pytest.fixture
def patch_console():
    """Patch all console output functions to prevent terminal side-effects."""
    names = [
        "print_error",
        "print_warning",
        "print_action",
        "print_info",
        "print_success",
        "print_benchmark_complete",
        "print_results_summary",
        "logger",
    ]
    mocks = {}
    patches = []
    for name in names:
        p = patch(f"{_MODULE}.{name}")
        mock = p.start()
        mocks[name] = mock
        patches.append(p)
    yield mocks
    for p in patches:
        p.stop()


# =============================================================================
# attach_to_benchmark Tests
# =============================================================================


class TestAttachToBenchmark:
    """Verify attach_to_benchmark early exits and happy path."""

    @pytest.mark.asyncio
    async def test_completed_phase_prints_warning(
        self,
        mock_api: MagicMock,
        patch_console: dict,
    ) -> None:
        with patch(
            f"{_MODULE}.find_controller_pod", new_callable=AsyncMock
        ) as mock_find:
            await attach_to_benchmark(
                "job-1", "default", 8080, mock_api, phase="Completed"
            )

        patch_console["print_warning"].assert_called_once()
        patch_console["print_action"].assert_called_once()
        mock_find.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_phase_prints_error(
        self,
        mock_api: MagicMock,
        patch_console: dict,
    ) -> None:
        with patch(f"{_MODULE}._fetch_and_print_pod_logs", new_callable=AsyncMock):
            await attach_to_benchmark(
                "job-1", "default", 8080, mock_api, phase="Failed"
            )

        patch_console["print_error"].assert_called_once()
        patch_console["print_action"].assert_called_once()

    @pytest.mark.asyncio
    async def test_no_controller_pod_prints_warning(
        self,
        mock_api: MagicMock,
        patch_console: dict,
    ) -> None:
        with patch(
            f"{_MODULE}.find_controller_pod",
            new_callable=AsyncMock,
            return_value=None,
        ):
            await attach_to_benchmark("job-1", "default", 8080, mock_api)

        patch_console["print_warning"].assert_called_once()
        assert "controller pod" in str(patch_console["print_warning"].call_args).lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "pod_phase",
        [
            PodPhase.PENDING,
            PodPhase.SUCCEEDED,
            PodPhase.FAILED,
            PodPhase.UNKNOWN,
        ],
    )  # fmt: skip
    async def test_non_running_pod_prints_warning(
        self,
        mock_api: MagicMock,
        patch_console: dict,
        pod_phase: PodPhase,
    ) -> None:
        with patch(
            f"{_MODULE}.find_controller_pod",
            new_callable=AsyncMock,
            return_value=("ctrl-0", pod_phase),
        ):
            await attach_to_benchmark("job-1", "default", 8080, mock_api)

        patch_console["print_warning"].assert_called_once()

    @pytest.mark.asyncio
    async def test_running_pod_starts_port_forward_and_streams(
        self,
        mock_api: MagicMock,
        patch_console: dict,
    ) -> None:
        with (
            patch(
                f"{_MODULE}.find_controller_pod",
                new_callable=AsyncMock,
                return_value=("ctrl-0", PodPhase.RUNNING),
            ),
            patch(f"{_MODULE}.port_forward_with_status") as mock_pf,
            patch(f"{_MODULE}.stream_progress", new_callable=AsyncMock) as mock_stream,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = 9999
            mock_ctx.__aexit__.return_value = False
            mock_pf.return_value = mock_ctx

            await attach_to_benchmark("job-1", "bench-ns", 8080, mock_api)

            mock_pf.assert_called_once_with(
                "bench-ns",
                "ctrl-0",
                8080,
                kubeconfig=None,
                kube_context=None,
            )
            mock_stream.assert_awaited_once_with("ws://localhost:9999/ws")

    @pytest.mark.asyncio
    async def test_passes_kube_creds_to_port_forward(
        self,
        mock_api: MagicMock,
        patch_console: dict,
    ) -> None:
        with (
            patch(
                f"{_MODULE}.find_controller_pod",
                new_callable=AsyncMock,
                return_value=("ctrl-0", PodPhase.RUNNING),
            ),
            patch(f"{_MODULE}.port_forward_with_status") as mock_pf,
            patch(f"{_MODULE}.stream_progress", new_callable=AsyncMock),
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = 9999
            mock_ctx.__aexit__.return_value = False
            mock_pf.return_value = mock_ctx

            await attach_to_benchmark(
                "job-1",
                "default",
                8080,
                mock_api,
                kubeconfig="/my/config",
                kube_context="my-ctx",
            )

            mock_pf.assert_called_once_with(
                "default",
                "ctrl-0",
                8080,
                kubeconfig="/my/config",
                kube_context="my-ctx",
            )

    @pytest.mark.asyncio
    async def test_none_phase_proceeds_to_find_pod(
        self,
        mock_api: MagicMock,
        patch_console: dict,
    ) -> None:
        with patch(
            f"{_MODULE}.find_controller_pod",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_find:
            await attach_to_benchmark("job-1", "default", 8080, mock_api, phase=None)

        mock_find.assert_awaited_once()


# =============================================================================
# auto_attach_workflow Tests
# =============================================================================


class TestAutoAttachWorkflow:
    """Verify auto_attach_workflow wait/no-wait and streaming paths."""

    @pytest.mark.asyncio
    async def test_wait_for_ready_calls_wait_for_controller(
        self,
        patch_k8s_client: MagicMock,
        patch_console: dict,
    ) -> None:
        with (
            patch(
                f"{_MODULE}.wait_for_controller_pod_ready",
                new_callable=AsyncMock,
                return_value="ctrl-pod-0",
            ) as mock_wait,
            patch(f"{_MODULE}.stream_controller_logs", new_callable=AsyncMock),
            patch(f"{_MODULE}.retrieve_and_display_results", new_callable=AsyncMock),
        ):
            await auto_attach_workflow("job-1", "ns1", 8080, wait_for_ready=True)

        mock_wait.assert_awaited_once_with(
            patch_k8s_client, "ns1", "job-1", timeout=300
        )

    @pytest.mark.asyncio
    async def test_no_wait_uses_find_controller_pod(
        self,
        patch_k8s_client: MagicMock,
        patch_console: dict,
    ) -> None:
        with (
            patch(
                f"{_MODULE}.find_controller_pod",
                new_callable=AsyncMock,
                return_value=("ctrl-0", PodPhase.RUNNING),
            ) as mock_find,
            patch(
                f"{_MODULE}.wait_for_controller_pod_ready", new_callable=AsyncMock
            ) as mock_wait,
            patch(f"{_MODULE}.stream_controller_logs", new_callable=AsyncMock),
            patch(f"{_MODULE}.retrieve_and_display_results", new_callable=AsyncMock),
        ):
            await auto_attach_workflow("job-1", "ns1", 8080, wait_for_ready=False)

        mock_find.assert_awaited_once_with(patch_k8s_client, "ns1", "job-1")
        mock_wait.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_wait_no_pod_raises_runtime_error(
        self,
        patch_k8s_client: MagicMock,
        patch_console: dict,
    ) -> None:
        with (
            patch(
                f"{_MODULE}.find_controller_pod",
                new_callable=AsyncMock,
                return_value=None,
            ),
            pytest.raises(RuntimeError, match="No controller pod found"),
        ):
            await auto_attach_workflow("job-1", "ns1", 8080, wait_for_ready=False)

    @pytest.mark.asyncio
    async def test_stream_ws_uses_port_forward(
        self,
        patch_k8s_client: MagicMock,
        patch_console: dict,
    ) -> None:
        with (
            patch(
                f"{_MODULE}.wait_for_controller_pod_ready",
                new_callable=AsyncMock,
                return_value="ctrl-pod-0",
            ),
            patch(f"{_MODULE}.port_forward_with_status") as mock_pf,
            patch(f"{_MODULE}.stream_progress", new_callable=AsyncMock) as mock_stream,
            patch(f"{_MODULE}.retrieve_and_display_results", new_callable=AsyncMock),
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = 9999
            mock_ctx.__aexit__.return_value = False
            mock_pf.return_value = mock_ctx

            await auto_attach_workflow("job-1", "ns1", 8080, stream_ws=True)

            mock_pf.assert_called_once()
            mock_stream.assert_awaited_once_with("ws://localhost:9999/ws")

    @pytest.mark.asyncio
    async def test_no_stream_ws_uses_controller_logs(
        self,
        patch_k8s_client: MagicMock,
        patch_console: dict,
    ) -> None:
        with (
            patch(
                f"{_MODULE}.wait_for_controller_pod_ready",
                new_callable=AsyncMock,
                return_value="ctrl-pod-0",
            ),
            patch(
                f"{_MODULE}.stream_controller_logs", new_callable=AsyncMock
            ) as mock_logs,
            patch(f"{_MODULE}.retrieve_and_display_results", new_callable=AsyncMock),
        ):
            await auto_attach_workflow("job-1", "ns1", 8080, stream_ws=False)

            mock_logs.assert_awaited_once_with(
                "ns1",
                "ctrl-pod-0",
                container="control-plane",
                kubeconfig=None,
                kube_context=None,
            )

    @pytest.mark.asyncio
    async def test_calls_retrieve_and_display_results(
        self,
        patch_k8s_client: MagicMock,
        patch_console: dict,
    ) -> None:
        with (
            patch(
                f"{_MODULE}.wait_for_controller_pod_ready",
                new_callable=AsyncMock,
                return_value="ctrl-pod-0",
            ),
            patch(f"{_MODULE}.stream_controller_logs", new_callable=AsyncMock),
            patch(
                f"{_MODULE}.retrieve_and_display_results", new_callable=AsyncMock
            ) as mock_retrieve,
        ):
            await auto_attach_workflow("job-1", "ns1", 8080)

            mock_retrieve.assert_awaited_once_with(
                "job-1", "ns1", patch_k8s_client, kubeconfig=None, kube_context=None
            )

    @pytest.mark.asyncio
    async def test_prints_benchmark_complete(
        self,
        patch_k8s_client: MagicMock,
        patch_console: dict,
    ) -> None:
        with (
            patch(
                f"{_MODULE}.wait_for_controller_pod_ready",
                new_callable=AsyncMock,
                return_value="ctrl-pod-0",
            ),
            patch(f"{_MODULE}.stream_controller_logs", new_callable=AsyncMock),
            patch(f"{_MODULE}.retrieve_and_display_results", new_callable=AsyncMock),
        ):
            await auto_attach_workflow("job-1", "ns1", 8080)

            patch_console["print_benchmark_complete"].assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_kube_creds_through(
        self,
        patch_k8s_client: MagicMock,
        patch_console: dict,
    ) -> None:
        with (
            patch(
                f"{_MODULE}.wait_for_controller_pod_ready",
                new_callable=AsyncMock,
                return_value="ctrl-pod-0",
            ),
            patch(f"{_MODULE}.stream_controller_logs", new_callable=AsyncMock),
            patch(
                f"{_MODULE}.retrieve_and_display_results", new_callable=AsyncMock
            ) as mock_retrieve,
        ):
            await auto_attach_workflow(
                "job-1",
                "ns1",
                8080,
                kubeconfig="/kc",
                kube_context="ctx",
            )

            mock_retrieve.assert_awaited_once_with(
                "job-1", "ns1", patch_k8s_client, kubeconfig="/kc", kube_context="ctx"
            )


# =============================================================================
# retrieve_and_display_results Tests
# =============================================================================


class TestRetrieveAndDisplayResults:
    """Verify artifact retrieval and result display logic."""

    @pytest.fixture
    def mock_deps(self):
        """Patch retrieve_all_artifacts and save_pod_logs."""
        with (
            patch(
                f"{_MODULE}.retrieve_all_artifacts", new_callable=AsyncMock
            ) as mock_retrieve,
            patch(f"{_MODULE}.save_pod_logs", new_callable=AsyncMock) as mock_logs,
            patch(f"{_MODULE}.find_jobset", new_callable=AsyncMock) as mock_find,
            patch(f"{_MODULE}.print_results_summary") as mock_summary,
            patch(f"{_MODULE}.print_warning") as mock_warn,
            patch(f"{_MODULE}.print_action") as mock_action,
            patch("pathlib.Path.mkdir") as mock_mkdir,
        ):
            mock_retrieve.return_value = True
            mock_find.return_value = None
            yield {
                "retrieve": mock_retrieve,
                "logs": mock_logs,
                "find_jobset": mock_find,
                "summary": mock_summary,
                "warning": mock_warn,
                "action": mock_action,
                "mkdir": mock_mkdir,
            }

    @pytest.mark.asyncio
    async def test_success_prints_results_summary(
        self,
        mock_api: MagicMock,
        mock_deps: dict,
    ) -> None:
        mock_deps["retrieve"].return_value = True

        await retrieve_and_display_results("job-1", "ns1", mock_api)

        mock_deps["summary"].assert_called_once()

    @pytest.mark.asyncio
    async def test_failure_prints_warning(
        self,
        mock_api: MagicMock,
        mock_deps: dict,
    ) -> None:
        mock_deps["retrieve"].return_value = False

        await retrieve_and_display_results("job-1", "ns1", mock_api)

        mock_deps["warning"].assert_called_once()
        mock_deps["action"].assert_called_once()
        mock_deps["summary"].assert_not_called()

    @pytest.mark.asyncio
    async def test_job_id_used_for_output_dir(
        self,
        mock_api: MagicMock,
        mock_deps: dict,
    ) -> None:
        await retrieve_and_display_results("job-1", "ns1", mock_api)

        call_args = mock_deps["retrieve"].call_args
        output_dir = call_args[0][2]  # third positional arg
        assert "job-1" in str(output_dir)

    @pytest.mark.asyncio
    async def test_save_pod_logs_always_called(
        self,
        mock_api: MagicMock,
        mock_deps: dict,
    ) -> None:
        mock_deps["retrieve"].return_value = False

        await retrieve_and_display_results("job-1", "ns1", mock_api)

        mock_deps["logs"].assert_awaited_once()

    @pytest.mark.asyncio
    async def test_passes_kube_creds_to_dependencies(
        self,
        mock_api: MagicMock,
        mock_deps: dict,
    ) -> None:
        await retrieve_and_display_results(
            "job-1", "ns1", mock_api, kubeconfig="/kc", kube_context="ctx"
        )

        retrieve_kwargs = mock_deps["retrieve"].call_args[1]
        assert retrieve_kwargs["kubeconfig"] == "/kc"
        assert retrieve_kwargs["kube_context"] == "ctx"

        logs_kwargs = mock_deps["logs"].call_args[1]
        assert logs_kwargs["kubeconfig"] == "/kc"
        assert logs_kwargs["kube_context"] == "ctx"

    @pytest.mark.asyncio
    async def test_creates_output_directory(
        self,
        mock_api: MagicMock,
        mock_deps: dict,
    ) -> None:
        await retrieve_and_display_results("job-1", "ns1", mock_api)

        mock_deps["mkdir"].assert_called_once_with(parents=True, exist_ok=True)
