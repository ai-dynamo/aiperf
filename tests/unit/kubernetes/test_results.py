# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.results module.

Focuses on:
- _kubectl_kube_args argument building
- retrieve_results_from_api download behavior and error paths
- retrieve_results_from_pod with various pod phases
- kubectl_copy_results success/failure handling
- display_copied_results file display and summary parsing
- retrieve_all_artifacts listing and download flow
- shutdown_api_service response handling
- stream_controller_logs process lifecycle
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import orjson
import pytest
from kubernetes_asyncio.client import ApiClient
from pytest import param

from aiperf.kubernetes.constants import Containers
from aiperf.kubernetes.enums import PodPhase
from aiperf.kubernetes.models import JobSetInfo
from aiperf.kubernetes.results import (
    API_RESULTS_FILES_PATH,
    API_RESULTS_LIST_PATH,
    KEY_RESULT_FILES,
    _download_and_decompress,
    _kubectl_kube_args,
    display_copied_results,
    kubectl_copy_results,
    retrieve_all_artifacts,
    retrieve_results_from_api,
    retrieve_results_from_operator,
    retrieve_results_from_pod,
    shutdown_api_service,
    stream_controller_logs,
)
from aiperf.kubernetes.subproc import CommandResult

# ============================================================
# Helpers
# ============================================================


def _make_api() -> MagicMock:
    """Create a mock kubernetes_asyncio ApiClient."""
    api = MagicMock(spec=ApiClient)
    api.close = AsyncMock()
    return api


def _make_jobset_info(status: str = "Running") -> JobSetInfo:
    """Create a minimal JobSetInfo for testing."""
    return JobSetInfo(
        name="test-jobset",
        namespace="default",
        jobset={
            "metadata": {
                "name": "test-jobset",
                "namespace": "default",
                "labels": {"aiperf.nvidia.com/job-id": "job-123"},
            },
            "status": {},
        },
        status=status,
    )


def _ok_result(stdout: str = "") -> CommandResult:
    """Create a successful CommandResult."""
    return CommandResult(returncode=0, stdout=stdout, stderr="")


def _fail_result(stderr: str = "error") -> CommandResult:
    """Create a failed CommandResult."""
    return CommandResult(returncode=1, stdout="", stderr=stderr)


@asynccontextmanager
async def _mock_port_forward(port: int = 9999):
    """Fake port_forward_with_status that yields a fixed port."""
    yield port


class FakeResponse:
    """Minimal fake aiohttp response for testing."""

    def __init__(
        self,
        status: int = 200,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
        json_data: dict | None = None,
    ) -> None:
        self.status = status
        self._body = body
        self.headers = headers or {}
        self._json_data = json_data

    @property
    def content_length(self) -> int | None:
        return None

    async def read(self) -> bytes:
        return self._body

    async def json(self) -> dict:
        if self._json_data is not None:
            return self._json_data
        return orjson.loads(self._body)

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=self.status,
                message="error",
            )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeSession:
    """Fake aiohttp.ClientSession for controlling responses per URL."""

    def __init__(
        self,
        responses: dict[str, FakeResponse] | None = None,
        **_kwargs,
    ) -> None:
        self._responses = responses or {}
        self._default = FakeResponse(status=404)

    def get(self, url: str) -> FakeResponse:
        return self._responses.get(url, self._default)

    def post(self, url: str) -> FakeResponse:
        return self._responses.get(url, self._default)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


_MODULE = "aiperf.kubernetes.results"
_OPERATOR_MODULE = "aiperf.kubernetes.results_operator"
_ARTIFACTS_MODULE = "aiperf.kubernetes.results_artifacts"


def _patch_find_retrievable_pod(result, module: str = _MODULE):
    """Patch find_retrievable_pod import inside the given module."""
    return patch(
        f"{module}.find_retrievable_pod", new_callable=AsyncMock, return_value=result
    )


def _patch_find_controller_pod(result, module: str = _MODULE):
    """Patch find_controller_pod import inside the given module."""
    return patch(
        f"{module}.find_controller_pod", new_callable=AsyncMock, return_value=result
    )


def _patch_find_operator_pod(result):
    """Patch find_operator_pod import inside the operator module."""
    return patch(
        f"{_OPERATOR_MODULE}.find_operator_pod",
        new_callable=AsyncMock,
        return_value=result,
    )


# ============================================================
# _kubectl_kube_args
# ============================================================


class TestKubectlKubeArgs:
    """Verify kubectl argument building for kubeconfig and context."""

    @pytest.mark.parametrize(
        "kubeconfig,kube_context,expected",
        [
            (None, None, []),
            ("/path/to/config", None, ["--kubeconfig", "/path/to/config"]),
            (None, "my-ctx", ["--context", "my-ctx"]),
            (
                "/path/to/config",
                "my-ctx",
                ["--kubeconfig", "/path/to/config", "--context", "my-ctx"],
            ),
        ],
    )  # fmt: skip
    def test_builds_correct_args(
        self,
        kubeconfig: str | None,
        kube_context: str | None,
        expected: list[str],
    ) -> None:
        assert _kubectl_kube_args(kubeconfig, kube_context) == expected

    def test_empty_string_kubeconfig_treated_as_falsy(self) -> None:
        result = _kubectl_kube_args("", None)
        assert result == []


# ============================================================
# retrieve_results_from_api
# ============================================================


class TestRetrieveResultsFromApi:
    """Verify API-based result retrieval via port-forward."""

    @pytest.mark.asyncio
    async def test_no_jobset_info_returns_false(self, tmp_path: Path) -> None:
        api = _make_api()
        result = await retrieve_results_from_api("job-1", "ns", tmp_path, None, api)
        assert result is False

    @pytest.mark.asyncio
    async def test_no_retrievable_pod_returns_false(self, tmp_path: Path) -> None:
        api = _make_api()
        with _patch_find_retrievable_pod(None):
            result = await retrieve_results_from_api(
                "job-1", "ns", tmp_path, _make_jobset_info(), api
            )
        assert result is False

    @pytest.mark.asyncio
    async def test_downloads_key_files_successfully(self, tmp_path: Path) -> None:
        api = _make_api()

        metrics_content = orjson.dumps({"throughput": 100})
        profile_content = b"profile data"
        console_content = b"console output"
        ansi_content = b"\x1b[1mconsole output\x1b[0m"

        responses = {
            f"http://localhost:9999{API_RESULTS_FILES_PATH}/metrics.json": FakeResponse(
                body=metrics_content
            ),
            f"http://localhost:9999{API_RESULTS_FILES_PATH}/profile_export_aiperf.json": FakeResponse(
                body=profile_content
            ),
            f"http://localhost:9999{API_RESULTS_FILES_PATH}/profile_export_console.txt": FakeResponse(
                body=console_content
            ),
            f"http://localhost:9999{API_RESULTS_FILES_PATH}/profile_export_console.ansi": FakeResponse(
                body=ansi_content
            ),
        }
        session = FakeSession(responses)

        with (
            _patch_find_retrievable_pod(("pod-0", PodPhase.RUNNING)),
            patch(
                "aiperf.kubernetes.results.port_forward_with_status",
                side_effect=lambda *a, **kw: _mock_port_forward(9999),
            ),
            patch("aiohttp.ClientSession", return_value=session),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=None,
            ),
        ):
            result = await retrieve_results_from_api(
                "job-1", "ns", tmp_path, _make_jobset_info(), api
            )

        assert result is True
        assert (tmp_path / "metrics.json").read_bytes() == metrics_content
        assert (tmp_path / "profile_export_aiperf.json").read_bytes() == profile_content
        assert (tmp_path / "profile_export_console.txt").read_bytes() == console_content
        assert (tmp_path / "profile_export_console.ansi").read_bytes() == ansi_content

    @pytest.mark.asyncio
    async def test_404_files_skipped_silently(self, tmp_path: Path) -> None:
        api = _make_api()

        responses = {
            f"http://localhost:9999{API_RESULTS_FILES_PATH}/metrics.json": FakeResponse(
                body=b'{"ok": true}'
            ),
        }
        session = FakeSession(responses)

        with (
            _patch_find_retrievable_pod(("pod-0", PodPhase.RUNNING)),
            patch(
                "aiperf.kubernetes.results.port_forward_with_status",
                side_effect=lambda *a, **kw: _mock_port_forward(9999),
            ),
            patch("aiohttp.ClientSession", return_value=session),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=None,
            ),
        ):
            result = await retrieve_results_from_api(
                "job-1", "ns", tmp_path, _make_jobset_info(), api
            )

        assert result is True
        assert not (tmp_path / "profile_export_aiperf.json").exists()

    @pytest.mark.asyncio
    async def test_malformed_metrics_json_still_downloads(self, tmp_path: Path) -> None:
        api = _make_api()

        bad_json = b"not valid json"
        responses = {
            f"http://localhost:9999{API_RESULTS_FILES_PATH}/metrics.json": FakeResponse(
                body=bad_json
            ),
        }
        session = FakeSession(responses)

        with (
            _patch_find_retrievable_pod(("pod-0", PodPhase.RUNNING)),
            patch(
                "aiperf.kubernetes.results.port_forward_with_status",
                side_effect=lambda *a, **kw: _mock_port_forward(9999),
            ),
            patch("aiohttp.ClientSession", return_value=session),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=None,
            ),
        ):
            result = await retrieve_results_from_api(
                "job-1", "ns", tmp_path, _make_jobset_info(), api
            )

        assert result is True
        assert (tmp_path / "metrics.json").read_bytes() == bad_json

    @pytest.mark.asyncio
    async def test_port_forward_exception_returns_false(self, tmp_path: Path) -> None:
        api = _make_api()

        @asynccontextmanager
        async def _failing_pf(*a, **kw):
            raise ConnectionError("port forward failed")
            yield  # noqa: F841, RET503

        with (
            _patch_find_retrievable_pod(("pod-0", PodPhase.RUNNING)),
            patch(
                "aiperf.kubernetes.results.port_forward_with_status",
                side_effect=lambda *a, **kw: _failing_pf(),
            ),
        ):
            result = await retrieve_results_from_api(
                "job-1", "ns", tmp_path, _make_jobset_info(), api
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_connector_error_breaks_loop_returns_false(
        self, tmp_path: Path
    ) -> None:
        api = _make_api()

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(
            side_effect=aiohttp.ClientConnectorError(
                connection_key=MagicMock(), os_error=OSError("refused")
            )
        )

        with (
            _patch_find_retrievable_pod(("pod-0", PodPhase.RUNNING)),
            patch(
                "aiperf.kubernetes.results.port_forward_with_status",
                side_effect=lambda *a, **kw: _mock_port_forward(9999),
            ),
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=None,
            ),
        ):
            result = await retrieve_results_from_api(
                "job-1", "ns", tmp_path, _make_jobset_info(), api
            )

        assert result is False


# ============================================================
# retrieve_results_from_pod
# ============================================================


class TestRetrieveResultsFromPod:
    """Verify pod-based result retrieval with kubectl cp."""

    @pytest.mark.asyncio
    async def test_no_jobset_info_returns_false(self, tmp_path: Path) -> None:
        api = _make_api()
        result = await retrieve_results_from_pod("job-1", "ns", tmp_path, None, api)
        assert result is False

    @pytest.mark.asyncio
    async def test_no_controller_pod_returns_false(self, tmp_path: Path) -> None:
        api = _make_api()
        with _patch_find_controller_pod(None):
            result = await retrieve_results_from_pod(
                "job-1", "ns", tmp_path, _make_jobset_info(), api
            )
        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "phase",
        [
            param(PodPhase.PENDING, id="pending"),
            param(PodPhase.FAILED, id="failed"),
            param(PodPhase.UNKNOWN, id="unknown"),
        ],
    )  # fmt: skip
    async def test_non_retrievable_phase_returns_false(
        self, tmp_path: Path, phase: PodPhase
    ) -> None:
        api = _make_api()
        with _patch_find_controller_pod(("pod-0", phase)):
            result = await retrieve_results_from_pod(
                "job-1", "ns", tmp_path, _make_jobset_info(), api
            )
        assert result is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "phase",
        [PodPhase.RUNNING, PodPhase.SUCCEEDED],
    )  # fmt: skip
    async def test_retrievable_phase_calls_kubectl_copy(
        self, tmp_path: Path, phase: PodPhase
    ) -> None:
        api = _make_api()
        with (
            _patch_find_controller_pod(("pod-0", phase)),
            patch(
                "aiperf.kubernetes.results.kubectl_copy_results",
                new_callable=AsyncMock,
                return_value=False,
            ) as mock_cp,
        ):
            result = await retrieve_results_from_pod(
                "job-1", "ns", tmp_path, _make_jobset_info(), api
            )

        mock_cp.assert_awaited_once_with(
            "ns",
            "pod-0",
            Containers.CONTROL_PLANE,
            tmp_path,
            kubeconfig=None,
            kube_context=None,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_successful_copy_displays_results(self, tmp_path: Path) -> None:
        api = _make_api()
        jobset_info = _make_jobset_info()

        with (
            _patch_find_controller_pod(("pod-0", PodPhase.RUNNING)),
            patch(
                "aiperf.kubernetes.results.kubectl_copy_results",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "aiperf.kubernetes.results.display_copied_results",
                return_value=True,
            ) as mock_display,
        ):
            result = await retrieve_results_from_pod(
                "job-1", "ns", tmp_path, jobset_info, api
            )

        mock_display.assert_called_once_with(tmp_path, jobset_info)
        assert result is True


# ============================================================
# kubectl_copy_results
# ============================================================


class TestKubectlCopyResults:
    """Verify kubectl cp invocation and fallback listing."""

    @pytest.mark.asyncio
    async def test_successful_copy_returns_true(self, tmp_path: Path) -> None:
        with patch(
            "aiperf.kubernetes.results.run_command",
            new_callable=AsyncMock,
            return_value=_ok_result(),
        ):
            result = await kubectl_copy_results("ns", "pod-0", "container", tmp_path)
        assert result is True

    @pytest.mark.asyncio
    async def test_successful_copy_with_stdout(self, tmp_path: Path) -> None:
        with patch(
            "aiperf.kubernetes.results.run_command",
            new_callable=AsyncMock,
            return_value=_ok_result(stdout="copied files"),
        ):
            result = await kubectl_copy_results("ns", "pod-0", "container", tmp_path)
        assert result is True

    @pytest.mark.asyncio
    async def test_failed_copy_attempts_ls_and_returns_false(
        self, tmp_path: Path
    ) -> None:
        with patch(
            "aiperf.kubernetes.results.run_command",
            new_callable=AsyncMock,
            side_effect=[
                _fail_result("copy error"),
                _ok_result("file1\nfile2"),
            ],
        ) as mock_run:
            result = await kubectl_copy_results("ns", "pod-0", "container", tmp_path)

        assert result is False
        assert mock_run.await_count == 2
        ls_cmd = mock_run.call_args_list[1][0][0]
        assert "exec" in ls_cmd
        assert "ls" in ls_cmd

    @pytest.mark.asyncio
    async def test_failed_copy_and_failed_ls(self, tmp_path: Path) -> None:
        with patch(
            "aiperf.kubernetes.results.run_command",
            new_callable=AsyncMock,
            side_effect=[
                _fail_result("copy error"),
                _fail_result("exec error"),
            ],
        ):
            result = await kubectl_copy_results("ns", "pod-0", "container", tmp_path)
        assert result is False

    @pytest.mark.asyncio
    async def test_kubeconfig_args_passed_to_both_commands(
        self, tmp_path: Path
    ) -> None:
        with patch(
            "aiperf.kubernetes.results.run_command",
            new_callable=AsyncMock,
            side_effect=[
                _fail_result("copy error"),
                _ok_result(),
            ],
        ) as mock_run:
            await kubectl_copy_results(
                "ns",
                "pod-0",
                "container",
                tmp_path,
                kubeconfig="/my/config",
                kube_context="ctx",
            )

        for call in mock_run.call_args_list:
            cmd = call[0][0]
            assert "--kubeconfig" in cmd
            assert "/my/config" in cmd
            assert "--context" in cmd
            assert "ctx" in cmd


# ============================================================
# display_copied_results
# ============================================================


class TestDisplayCopiedResults:
    """Verify result display and summary parsing."""

    def test_empty_directory_returns_false(self, tmp_path: Path) -> None:
        result = display_copied_results(tmp_path, _make_jobset_info())
        assert result is False

    def test_files_present_returns_true(self, tmp_path: Path) -> None:
        (tmp_path / "metrics.json").write_text("{}")
        result = display_copied_results(tmp_path, _make_jobset_info())
        assert result is True

    def test_summary_file_parsed_when_present(self, tmp_path: Path) -> None:
        summary = {"summary": {"requests": 100, "latency_p99": "50ms"}}
        (tmp_path / "profile_export_aiperf.json").write_bytes(orjson.dumps(summary))
        result = display_copied_results(tmp_path, _make_jobset_info())
        assert result is True

    def test_malformed_summary_file_does_not_crash(self, tmp_path: Path) -> None:
        (tmp_path / "profile_export_aiperf.json").write_bytes(b"not json")
        result = display_copied_results(tmp_path, _make_jobset_info())
        assert result is True

    def test_summary_without_summary_key_no_error(self, tmp_path: Path) -> None:
        (tmp_path / "profile_export_aiperf.json").write_bytes(
            orjson.dumps({"other": "data"})
        )
        result = display_copied_results(tmp_path, _make_jobset_info())
        assert result is True


# ============================================================
# retrieve_all_artifacts
# ============================================================


class TestRetrieveAllArtifacts:
    """Verify full artifact retrieval via API listing + download."""

    @pytest.mark.asyncio
    async def test_no_jobset_info_returns_false(self, tmp_path: Path) -> None:
        api = _make_api()
        result = await retrieve_all_artifacts("job-1", "ns", tmp_path, None, api, 0)
        assert result is False

    @pytest.mark.asyncio
    async def test_no_retrievable_pod_returns_false(self, tmp_path: Path) -> None:
        api = _make_api()
        with _patch_find_retrievable_pod(None, module=_ARTIFACTS_MODULE):
            result = await retrieve_all_artifacts(
                "job-1", "ns", tmp_path, _make_jobset_info(), api, 0
            )
        assert result is False

    @pytest.mark.asyncio
    async def test_downloads_listed_files(self, tmp_path: Path) -> None:
        api = _make_api()

        list_resp = FakeResponse(
            json_data={"files": [{"name": "a.json"}, {"name": "b.txt"}]}
        )
        file_a = FakeResponse(body=b'{"data": 1}')
        file_b = FakeResponse(body=b"text content")

        base = "http://localhost:9999"
        responses = {
            f"{base}{API_RESULTS_LIST_PATH}": list_resp,
            f"{base}{API_RESULTS_FILES_PATH}/a.json": file_a,
            f"{base}{API_RESULTS_FILES_PATH}/b.txt": file_b,
        }
        session = FakeSession(responses)

        with (
            _patch_find_retrievable_pod(
                ("pod-0", PodPhase.RUNNING), module=_ARTIFACTS_MODULE
            ),
            patch(
                "aiperf.kubernetes.results_artifacts.port_forward_with_status",
                side_effect=lambda *a, **kw: _mock_port_forward(9999),
            ),
            patch("aiohttp.ClientSession", return_value=session),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=None,
            ),
        ):
            result = await retrieve_all_artifacts(
                "job-1", "ns", tmp_path, _make_jobset_info(), api, 0
            )

        assert result is True
        assert (tmp_path / "a.json").exists()
        assert (tmp_path / "b.txt").read_bytes() == b"text content"

    @pytest.mark.asyncio
    async def test_x_filename_header_overrides_dest(self, tmp_path: Path) -> None:
        api = _make_api()

        list_resp = FakeResponse(json_data={"files": [{"name": "file1"}]})
        file_resp = FakeResponse(body=b"data", headers={"x-filename": "renamed.json"})

        base = "http://localhost:9999"
        responses = {
            f"{base}{API_RESULTS_LIST_PATH}": list_resp,
            f"{base}{API_RESULTS_FILES_PATH}/file1": file_resp,
        }
        session = FakeSession(responses)

        with (
            _patch_find_retrievable_pod(
                ("pod-0", PodPhase.RUNNING), module=_ARTIFACTS_MODULE
            ),
            patch(
                "aiperf.kubernetes.results_artifacts.port_forward_with_status",
                side_effect=lambda *a, **kw: _mock_port_forward(9999),
            ),
            patch("aiohttp.ClientSession", return_value=session),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=None,
            ),
        ):
            result = await retrieve_all_artifacts(
                "job-1", "ns", tmp_path, _make_jobset_info(), api, 0
            )

        assert result is True
        assert (tmp_path / "renamed.json").exists()
        assert not (tmp_path / "file1").exists()

    @pytest.mark.asyncio
    async def test_empty_file_list_returns_false(self, tmp_path: Path) -> None:
        api = _make_api()

        list_resp = FakeResponse(json_data={"files": []})
        base = "http://localhost:9999"
        responses = {f"{base}{API_RESULTS_LIST_PATH}": list_resp}
        session = FakeSession(responses)

        with (
            _patch_find_retrievable_pod(
                ("pod-0", PodPhase.RUNNING), module=_ARTIFACTS_MODULE
            ),
            patch(
                "aiperf.kubernetes.results_artifacts.port_forward_with_status",
                side_effect=lambda *a, **kw: _mock_port_forward(9999),
            ),
            patch("aiohttp.ClientSession", return_value=session),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=None,
            ),
        ):
            result = await retrieve_all_artifacts(
                "job-1", "ns", tmp_path, _make_jobset_info(), api, 0
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_list_request_failure_returns_false(self, tmp_path: Path) -> None:
        api = _make_api()

        list_resp = FakeResponse(status=500)
        base = "http://localhost:9999"
        responses = {f"{base}{API_RESULTS_LIST_PATH}": list_resp}
        session = FakeSession(responses)

        with (
            _patch_find_retrievable_pod(
                ("pod-0", PodPhase.RUNNING), module=_ARTIFACTS_MODULE
            ),
            patch(
                "aiperf.kubernetes.results_artifacts.port_forward_with_status",
                side_effect=lambda *a, **kw: _mock_port_forward(9999),
            ),
            patch("aiohttp.ClientSession", return_value=session),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=None,
            ),
        ):
            result = await retrieve_all_artifacts(
                "job-1", "ns", tmp_path, _make_jobset_info(), api, 0
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_port_forward_exception_returns_false(self, tmp_path: Path) -> None:
        api = _make_api()

        @asynccontextmanager
        async def _failing_pf(*a, **kw):
            raise RuntimeError("port forward died")
            yield  # noqa: F841, RET503

        with (
            _patch_find_retrievable_pod(
                ("pod-0", PodPhase.RUNNING), module=_ARTIFACTS_MODULE
            ),
            patch(
                "aiperf.kubernetes.results_artifacts.port_forward_with_status",
                side_effect=lambda *a, **kw: _failing_pf(),
            ),
        ):
            result = await retrieve_all_artifacts(
                "job-1", "ns", tmp_path, _make_jobset_info(), api, 0
            )

        assert result is False


# ============================================================
# shutdown_api_service
# ============================================================


class TestShutdownApiService:
    """Verify shutdown request handling."""

    @pytest.mark.asyncio
    async def test_no_retrievable_pod_no_controller_returns_false(self) -> None:
        api = _make_api()
        with (
            _patch_find_retrievable_pod(None),
            _patch_find_controller_pod(None),
        ):
            result = await shutdown_api_service("job-1", "ns", api)
        assert result is False

    @pytest.mark.asyncio
    async def test_no_retrievable_pod_but_controller_exists_returns_true(self) -> None:
        api = _make_api()
        with (
            _patch_find_retrievable_pod(None),
            _patch_find_controller_pod(("pod-0", PodPhase.SUCCEEDED)),
        ):
            result = await shutdown_api_service("job-1", "ns", api)
        assert result is True

    @pytest.mark.asyncio
    async def test_shutdown_200_returns_true(self) -> None:
        api = _make_api()

        resp = FakeResponse(status=200)
        base = "http://localhost:9999"
        session = FakeSession({f"{base}/api/shutdown": resp})

        with (
            _patch_find_retrievable_pod(("pod-0", PodPhase.RUNNING)),
            patch(
                "aiperf.kubernetes.results.port_forward_with_status",
                side_effect=lambda *a, **kw: _mock_port_forward(9999),
            ),
            patch("aiohttp.ClientSession", return_value=session),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=None,
            ),
        ):
            result = await shutdown_api_service("job-1", "ns", api)

        assert result is True

    @pytest.mark.asyncio
    async def test_shutdown_409_returns_false(self) -> None:
        api = _make_api()

        resp = FakeResponse(status=409)
        base = "http://localhost:9999"
        session = FakeSession({f"{base}/api/shutdown": resp})

        with (
            _patch_find_retrievable_pod(("pod-0", PodPhase.RUNNING)),
            patch(
                "aiperf.kubernetes.results.port_forward_with_status",
                side_effect=lambda *a, **kw: _mock_port_forward(9999),
            ),
            patch("aiohttp.ClientSession", return_value=session),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=None,
            ),
        ):
            result = await shutdown_api_service("job-1", "ns", api)

        assert result is False

    @pytest.mark.asyncio
    async def test_shutdown_unexpected_status_returns_false(self) -> None:
        api = _make_api()

        resp = FakeResponse(status=503)
        base = "http://localhost:9999"
        session = FakeSession({f"{base}/api/shutdown": resp})

        with (
            _patch_find_retrievable_pod(("pod-0", PodPhase.RUNNING)),
            patch(
                "aiperf.kubernetes.results.port_forward_with_status",
                side_effect=lambda *a, **kw: _mock_port_forward(9999),
            ),
            patch("aiohttp.ClientSession", return_value=session),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=None,
            ),
        ):
            result = await shutdown_api_service("job-1", "ns", api)

        assert result is False

    @pytest.mark.asyncio
    async def test_shutdown_exception_returns_false(self) -> None:
        api = _make_api()

        @asynccontextmanager
        async def _failing_pf(*a, **kw):
            raise OSError("connection refused")
            yield  # noqa: F841, RET503

        with (
            _patch_find_retrievable_pod(("pod-0", PodPhase.RUNNING)),
            patch(
                "aiperf.kubernetes.results.port_forward_with_status",
                side_effect=lambda *a, **kw: _failing_pf(),
            ),
        ):
            result = await shutdown_api_service("job-1", "ns", api)

        assert result is False


# ============================================================
# stream_controller_logs
# ============================================================


class TestStreamControllerLogs:
    """Verify log streaming process management."""

    @pytest.mark.asyncio
    async def test_streams_lines_until_eof(self) -> None:
        mock_proc = MagicMock()
        lines = [b"line 1\n", b"line 2\n", b""]
        line_iter = iter(lines)
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline = AsyncMock(side_effect=lambda: next(line_iter))
        mock_proc.wait = AsyncMock()

        with (
            patch(
                "aiperf.kubernetes.results.start_streaming_process",
                new_callable=AsyncMock,
                return_value=mock_proc,
            ),
            patch(
                "aiperf.kubernetes.results.terminate_process",
                new_callable=AsyncMock,
            ) as mock_term,
        ):
            await stream_controller_logs("ns", "pod-0")

        mock_proc.wait.assert_awaited_once()
        mock_term.assert_awaited_once_with(mock_proc)

    @pytest.mark.asyncio
    async def test_cancellation_terminates_process(self) -> None:
        mock_proc = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline = AsyncMock(side_effect=asyncio.CancelledError)
        mock_proc.terminate = MagicMock()

        with (
            patch(
                "aiperf.kubernetes.results.start_streaming_process",
                new_callable=AsyncMock,
                return_value=mock_proc,
            ),
            patch(
                "aiperf.kubernetes.results.terminate_process",
                new_callable=AsyncMock,
            ) as mock_term,
            pytest.raises(asyncio.CancelledError),
        ):
            await stream_controller_logs("ns", "pod-0")

        mock_proc.terminate.assert_called_once()
        mock_term.assert_awaited_once_with(mock_proc)

    @pytest.mark.asyncio
    async def test_stdout_none_exits_immediately(self) -> None:
        mock_proc = MagicMock()
        mock_proc.stdout = None

        with (
            patch(
                "aiperf.kubernetes.results.start_streaming_process",
                new_callable=AsyncMock,
                return_value=mock_proc,
            ),
            patch(
                "aiperf.kubernetes.results.terminate_process",
                new_callable=AsyncMock,
            ) as mock_term,
        ):
            await stream_controller_logs("ns", "pod-0")

        mock_term.assert_awaited_once_with(mock_proc)

    @pytest.mark.asyncio
    async def test_kubeconfig_args_in_command(self) -> None:
        mock_proc = MagicMock()
        mock_proc.stdout = None

        with (
            patch(
                "aiperf.kubernetes.results.start_streaming_process",
                new_callable=AsyncMock,
                return_value=mock_proc,
            ) as mock_start,
            patch(
                "aiperf.kubernetes.results.terminate_process",
                new_callable=AsyncMock,
            ),
        ):
            await stream_controller_logs(
                "ns",
                "pod-0",
                kubeconfig="/kc",
                kube_context="ctx",
            )

        cmd = mock_start.call_args[0][0]
        assert "--kubeconfig" in cmd
        assert "/kc" in cmd
        assert "--context" in cmd
        assert "ctx" in cmd

    @pytest.mark.asyncio
    async def test_default_container_is_control_plane(self) -> None:
        mock_proc = MagicMock()
        mock_proc.stdout = None

        with (
            patch(
                "aiperf.kubernetes.results.start_streaming_process",
                new_callable=AsyncMock,
                return_value=mock_proc,
            ) as mock_start,
            patch(
                "aiperf.kubernetes.results.terminate_process",
                new_callable=AsyncMock,
            ),
        ):
            await stream_controller_logs("ns", "pod-0")

        cmd = mock_start.call_args[0][0]
        c_idx = cmd.index("-c")
        assert cmd[c_idx + 1] == Containers.CONTROL_PLANE


# ============================================================
# Module-level constants
# ============================================================


class TestModuleConstants:
    """Verify module-level constants are correctly defined."""

    def test_key_result_files_contains_expected(self) -> None:
        assert "metrics.json" in KEY_RESULT_FILES
        assert "profile_export_aiperf.json" in KEY_RESULT_FILES
        assert "profile_export_console.txt" in KEY_RESULT_FILES
        assert "profile_export_console.ansi" in KEY_RESULT_FILES

    def test_api_paths(self) -> None:
        assert API_RESULTS_FILES_PATH.startswith("/api/")
        assert API_RESULTS_LIST_PATH.startswith("/api/")


# ============================================================
# _download_and_decompress
# ============================================================


class _FakeStreamContent:
    """Fake aiohttp response content with iter_chunked support."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_chunked(self, chunk_size: int):
        for chunk in self._chunks:
            yield chunk


class _FakeDownloadResponse:
    """Fake aiohttp.ClientResponse for _download_and_decompress testing."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.content = _FakeStreamContent(chunks)


class TestDownloadAndDecompress:
    """Verify _download_and_decompress handles all encoding variants."""

    @pytest.mark.asyncio
    async def test_identity_encoding_saves_raw(self, tmp_path: Path) -> None:
        content = b'{"metric": "value"}'
        resp = _FakeDownloadResponse([content])
        dest = tmp_path / "file.json"

        await _download_and_decompress(resp, dest, "identity")

        assert dest.read_bytes() == content

    @pytest.mark.asyncio
    async def test_gzip_encoding_decompresses(self, tmp_path: Path) -> None:
        import gzip

        original = b'{"metric": "gzipped"}'
        compressed = gzip.compress(original)
        resp = _FakeDownloadResponse([compressed])
        dest = tmp_path / "file.json"

        await _download_and_decompress(resp, dest, "gzip")

        assert dest.read_bytes() == original

    @pytest.mark.asyncio
    async def test_zstd_encoding_decompresses(self, tmp_path: Path) -> None:
        import zstandard

        original = b'{"metric": "zstd_value"}'
        compressed = zstandard.ZstdCompressor().compress(original)
        resp = _FakeDownloadResponse([compressed])
        dest = tmp_path / "file.json"

        await _download_and_decompress(resp, dest, "zstd")

        assert dest.read_bytes() == original

    @pytest.mark.asyncio
    async def test_multi_chunk_identity(self, tmp_path: Path) -> None:
        resp = _FakeDownloadResponse([b"chunk1", b"chunk2", b"chunk3"])
        dest = tmp_path / "file.json"

        await _download_and_decompress(resp, dest, "identity")

        assert dest.read_bytes() == b"chunk1chunk2chunk3"

    @pytest.mark.asyncio
    async def test_empty_response(self, tmp_path: Path) -> None:
        resp = _FakeDownloadResponse([])
        dest = tmp_path / "file.json"

        await _download_and_decompress(resp, dest, "identity")

        assert dest.exists()
        assert dest.read_bytes() == b""

    @pytest.mark.asyncio
    async def test_unknown_encoding_treated_as_identity(self, tmp_path: Path) -> None:
        content = b"raw bytes"
        resp = _FakeDownloadResponse([content])
        dest = tmp_path / "file.json"

        await _download_and_decompress(resp, dest, "br")

        assert dest.read_bytes() == content


# ============================================================
# retrieve_results_from_operator
# ============================================================


class TestRetrieveResultsFromOperator:
    """Verify operator-based result retrieval via port-forward."""

    @pytest.mark.asyncio
    async def test_operator_pod_not_found_returns_false(self, tmp_path: Path) -> None:
        api = _make_api()
        with _patch_find_operator_pod(None):
            result = await retrieve_results_from_operator("job-1", "ns", tmp_path, api)
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_failure_returns_false(self, tmp_path: Path) -> None:
        api = _make_api()

        health_resp = FakeResponse(status=503)
        session = FakeSession(
            {
                "http://localhost:9999/healthz": health_resp,
            }
        )

        with (
            _patch_find_operator_pod(("operator-0", PodPhase.RUNNING)),
            patch(
                "aiperf.kubernetes.results_operator.port_forward_with_status",
                side_effect=lambda *a, **kw: _mock_port_forward(9999),
            ),
            patch("aiohttp.ClientSession", return_value=session),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=None,
            ),
        ):
            result = await retrieve_results_from_operator("job-1", "ns", tmp_path, api)

        assert result is False

    @pytest.mark.asyncio
    async def test_job_not_found_returns_false(self, tmp_path: Path) -> None:
        api = _make_api()

        base = "http://localhost:9999"
        health_session = FakeSession({f"{base}/healthz": FakeResponse()})
        list_session = FakeSession(
            {
                f"{base}/api/v1/results/ns/job-1": FakeResponse(status=404),
            }
        )

        call_count = 0

        def session_factory(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return health_session
            return list_session

        with (
            _patch_find_operator_pod(("operator-0", PodPhase.RUNNING)),
            patch(
                "aiperf.kubernetes.results_operator.port_forward_with_status",
                side_effect=lambda *a, **kw: _mock_port_forward(9999),
            ),
            patch("aiohttp.ClientSession", side_effect=session_factory),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=None,
            ),
        ):
            result = await retrieve_results_from_operator("job-1", "ns", tmp_path, api)

        assert result is False

    @pytest.mark.asyncio
    async def test_port_forward_exception_returns_false(self, tmp_path: Path) -> None:
        api = _make_api()

        @asynccontextmanager
        async def _failing_pf(*a, **kw):
            raise ConnectionError("port forward failed")
            yield  # noqa: F841, RET503

        with (
            _patch_find_operator_pod(("operator-0", PodPhase.RUNNING)),
            patch(
                "aiperf.kubernetes.results_operator.port_forward_with_status",
                side_effect=lambda *a, **kw: _failing_pf(),
            ),
        ):
            result = await retrieve_results_from_operator("job-1", "ns", tmp_path, api)

        assert result is False

    @pytest.mark.asyncio
    async def test_empty_file_list_returns_false(self, tmp_path: Path) -> None:
        api = _make_api()

        base = "http://localhost:9999"
        health_session = FakeSession({f"{base}/healthz": FakeResponse()})
        list_session = FakeSession(
            {
                f"{base}/api/v1/results/ns/job-1": FakeResponse(
                    json_data={"namespace": "ns", "job_id": "job-1", "files": []}
                ),
            }
        )

        call_count = 0

        def session_factory(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return health_session
            return list_session

        with (
            _patch_find_operator_pod(("operator-0", PodPhase.RUNNING)),
            patch(
                "aiperf.kubernetes.results_operator.port_forward_with_status",
                side_effect=lambda *a, **kw: _mock_port_forward(9999),
            ),
            patch("aiohttp.ClientSession", side_effect=session_factory),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=None,
            ),
        ):
            result = await retrieve_results_from_operator("job-1", "ns", tmp_path, api)

        assert result is False
