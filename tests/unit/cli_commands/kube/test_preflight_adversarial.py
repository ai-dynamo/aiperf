# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for kube preflight trust boundaries.

Focuses on:
- CLI option propagation into the Kubernetes client context.
- JSON output remaining machine-parseable when human logs would otherwise render.
- Namespace, RBAC, timeout, and dispatcher error classification.
- Partial aggregation when one non-connectivity check fails.
- Subprocess trust boundaries for missing tools and stderr preservation.

Out of scope: full kube deployment, pod log streaming, and result downloads; see
``tests/unit/cli_commands/kube/test_profile_deploy.py``,
``test_logs.py``, and ``test_kube_results_sweep.py`` for those paths.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from kubernetes_asyncio.client import ApiClient, CoreV1Api
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.cli_commands.kube import preflight as preflight_cmd
from aiperf.config.kube import KubeManageOptions
from aiperf.kubernetes import preflight_checks, subproc
from aiperf.kubernetes.preflight import (
    CheckResult,
    CheckStatus,
    CLIPreflightChecker,
    PreflightResults,
)


@pytest.fixture(autouse=True)
def _hermetic_k8s_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test in this module to a stub ``k8s_client``.

    ``CLIPreflightChecker.run_all_checks`` / ``run_quick_checks`` open
    ``k8s_client()`` before dispatching the individual checks. Tests that
    override the ``_check_*`` methods but do not patch ``k8s_client`` would
    otherwise reach a live client-open, which falls through to
    ``load_kube_config()`` and depends on the developer's ``~/.kube/config``.
    Tests that assert on the kubeconfig/context forwarded to the client install
    their own factory via ``patch("aiperf.kubernetes.preflight.k8s_client", ...)``
    — that re-patches over this default within the ``with`` block. Patched on the
    ``preflight`` module because it binds ``k8s_client`` at import time.
    """
    from contextlib import asynccontextmanager

    import aiperf.kubernetes.preflight as preflight_mod

    @asynccontextmanager
    async def _stub(*, kubeconfig: str | None = None, context: str | None = None):
        yield MagicMock(spec=ApiClient)

    monkeypatch.setattr(preflight_mod, "k8s_client", _stub)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _K8sClientCall:
    """Captured arguments passed through ``k8s_client``."""

    kubeconfig: str | None
    context: str | None


@dataclass(slots=True)
class _RecordingK8sClientFactory:
    """Context-manager factory that records kubeconfig and context."""

    api: ApiClient
    calls: list[_K8sClientCall] = field(default_factory=list)

    def __call__(
        self, *, kubeconfig: str | None = None, context: str | None = None
    ) -> _RecordingK8sClientContext:
        self.calls.append(_K8sClientCall(kubeconfig=kubeconfig, context=context))
        return _RecordingK8sClientContext(self.api)


@dataclass(slots=True)
class _RecordingK8sClientContext:
    """Async context manager returned by the fake k8s client factory."""

    api: ApiClient

    async def __aenter__(self) -> ApiClient:
        return self.api

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


@dataclass(slots=True)
class _CapturedCheckerInit:
    """Constructor surface captured from the CLI wrapper."""

    namespace: str
    kubeconfig: str | None
    kube_context: str | None
    image: str | None
    image_pull_secrets: list[str] | None
    secrets: list[str] | None
    endpoint_url: str | None
    workers: int


@dataclass(slots=True)
class _ProcessStub:
    """Minimal process stub for subprocess boundary tests."""

    returncode: int | None
    stdout: bytes = b""
    stderr: bytes = b""
    pid: int = 4242
    terminated: bool = False
    killed: bool = False

    async def communicate(self) -> tuple[bytes, bytes]:
        return self.stdout, self.stderr

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> None:
        self.returncode = -15


def _result(name: str, status: CheckStatus = CheckStatus.PASS) -> CheckResult:
    """Return a named check result with a realistic message."""
    return CheckResult(name=name, status=status, message=f"{name} completed")


def _async_result(
    name: str, status: CheckStatus = CheckStatus.PASS
) -> Callable[[], Awaitable[CheckResult]]:
    """Return an async check function for ``CLIPreflightChecker`` monkeypatching."""

    async def _check() -> CheckResult:
        return _result(name, status)

    return _check


async def _raising_check() -> CheckResult:
    raise RuntimeError("apiserver returned malformed discovery payload")


async def _passing_check() -> CheckResult:
    return _result("Cluster Connectivity")


# ---------------------------------------------------------------------------
# CLI trust boundary
# ---------------------------------------------------------------------------


class TestPreflightCliTrustBoundary:
    """The CLI wrapper forwards cluster-selection options and keeps JSON clean."""

    @pytest.mark.asyncio
    async def test_check_cluster_connectivity_with_kube_context_passes_context_to_client(
        self,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        factory = _RecordingK8sClientFactory(api=api)
        checker = CLIPreflightChecker(
            namespace="aiperf-ci",
            kubeconfig="/opt/ci/kubeconfigs/aiperf-ci.yaml",
            kube_context="kind-aiperf-ci",
        )
        # The orchestrator opens the shared client and stamps self._api; the
        # per-check methods consume it. Stub the quick checks so this test
        # asserts only that the kubeconfig/context are forwarded to k8s_client.
        checker._check_cluster_connectivity = _async_result("Cluster Connectivity")
        checker._check_jobset_crd = _async_result("JobSet CRD")
        checker._check_rbac_permissions = _async_result("RBAC Permissions")

        with patch("aiperf.kubernetes.preflight.k8s_client", new=factory):
            results = await checker.run_quick_checks()

        assert results.passed is True
        assert factory.calls == [
            _K8sClientCall(
                kubeconfig="/opt/ci/kubeconfigs/aiperf-ci.yaml",
                context="kind-aiperf-ci",
            )
        ]

    @pytest.mark.asyncio
    async def test_run_preflight_json_output_is_parseable_and_restores_logger(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        captured: list[_CapturedCheckerInit] = []
        kube_logger = logging.getLogger("aiperf.kube")
        kube_logger.setLevel(logging.INFO)
        original_level = kube_logger.level

        class FakeChecker:
            def __init__(
                self,
                namespace: str,
                *,
                kubeconfig: str | None = None,
                kube_context: str | None = None,
                image: str | None = None,
                image_pull_secrets: list[str] | None = None,
                secrets: list[str] | None = None,
                endpoint_url: str | None = None,
                workers: int = 1,
            ) -> None:
                captured.append(
                    _CapturedCheckerInit(
                        namespace=namespace,
                        kubeconfig=kubeconfig,
                        kube_context=kube_context,
                        image=image,
                        image_pull_secrets=image_pull_secrets,
                        secrets=secrets,
                        endpoint_url=endpoint_url,
                        workers=workers,
                    )
                )

            async def run_all_checks(self) -> PreflightResults:
                logging.getLogger("aiperf.kube").info("human preflight line")
                return PreflightResults(
                    checks=[
                        CheckResult(
                            name="Cluster Connectivity",
                            status=CheckStatus.PASS,
                            message="connected to kind-aiperf-ci",
                        )
                    ]
                )

        with patch("aiperf.kubernetes.preflight.CLIPreflightChecker", new=FakeChecker):
            await preflight_cmd._run_preflight(
                manage_options=KubeManageOptions(
                    namespace="aiperf-ci",
                    kubeconfig="/opt/ci/kubeconfigs/aiperf-ci.yaml",
                    kube_context="kind-aiperf-ci",
                ),
                image="nvcr.io/nvidia/aiperf:ci",
                image_pull_secrets=["nvcr-creds", "fallback-creds"],
                secrets=["endpoint-api-key", "dataset-token"],
                endpoint_url="http://llama.default.svc.cluster.local:8000/v1",
                workers=7,
                output="json",
            )

        payload = orjson.loads(capsys.readouterr().out)
        assert payload["passed"] is True
        assert payload["checks"][0]["name"] == "Cluster Connectivity"
        assert captured == [
            _CapturedCheckerInit(
                namespace="aiperf-ci",
                kubeconfig="/opt/ci/kubeconfigs/aiperf-ci.yaml",
                kube_context="kind-aiperf-ci",
                image="nvcr.io/nvidia/aiperf:ci",
                image_pull_secrets=["nvcr-creds", "fallback-creds"],
                secrets=["endpoint-api-key", "dataset-token"],
                endpoint_url="http://llama.default.svc.cluster.local:8000/v1",
                workers=7,
            )
        ]
        assert kube_logger.level == original_level

    @pytest.mark.asyncio
    async def test_run_preflight_default_namespace_is_aiperf_benchmarks(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        captured_namespaces: list[str] = []

        class FakeChecker:
            def __init__(
                self,
                namespace: str,
                *,
                kubeconfig: str | None = None,
                kube_context: str | None = None,
                image: str | None = None,
                image_pull_secrets: list[str] | None = None,
                secrets: list[str] | None = None,
                endpoint_url: str | None = None,
                workers: int = 1,
            ) -> None:
                captured_namespaces.append(namespace)

            async def run_all_checks(self) -> PreflightResults:
                return PreflightResults(checks=[_result("Cluster Connectivity")])

        with patch("aiperf.kubernetes.preflight.CLIPreflightChecker", new=FakeChecker):
            await preflight_cmd._run_preflight(
                manage_options=KubeManageOptions(),
                image=None,
                image_pull_secrets=None,
                secrets=None,
                endpoint_url=None,
                workers=1,
                output="json",
            )

        assert orjson.loads(capsys.readouterr().out)["passed"] is True
        assert captured_namespaces == ["aiperf-benchmarks"]

    def test_preflight_cli_parses_repeated_secret_references(self) -> None:
        _, bound, _ = preflight_cmd.app.parse_args(
            [
                "--image-pull-secret",
                "nvcr-creds",
                "--image-pull-secret",
                "fallback-creds",
                "--secret",
                "endpoint-api-key",
                "--secret",
                "dataset-token",
            ],
            exit_on_error=False,
            print_error=False,
        )

        assert bound.arguments["image_pull_secrets"] == [
            "nvcr-creds",
            "fallback-creds",
        ]
        assert bound.arguments["secrets"] == ["endpoint-api-key", "dataset-token"]


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


class TestPreflightErrorClassification:
    """Boundary failures classify as fail, warn, skip, or aggregate-only failures."""

    @pytest.mark.parametrize(
        "api_error,expected_status,expected_message",
        [
            param(
                ApiException(status=404),
                CheckStatus.FAIL,
                "does not exist",
                id="missing-namespace-denied-create-fails",
            ),
            param(
                ApiException(status=403),
                CheckStatus.SKIP,
                "permission denied",
                id="namespace-get-denied-skips",
            ),
            param(
                ApiException(status=500),
                CheckStatus.FAIL,
                "HTTP 500",
                id="namespace-server-error-fails",
            ),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_check_namespace_api_errors_preserve_namespace_and_classification(
        self,
        api_error: ApiException,
        expected_status: CheckStatus,
        expected_message: str,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        core = MagicMock(spec=CoreV1Api)
        core.read_namespace = AsyncMock(side_effect=api_error)

        with (
            patch(
                "aiperf.kubernetes.preflight_checks.client.CoreV1Api", return_value=core
            ),
            patch(
                "aiperf.kubernetes.preflight_checks._shared_check_rbac_access",
                new=AsyncMock(return_value=False),
            ),
        ):
            result = await preflight_checks.check_namespace(
                api, namespace="llama-benchmarks"
            )

        assert result.status == expected_status
        assert "llama-benchmarks" in result.message or "namespace" in result.message
        assert expected_message in result.message

    @pytest.mark.parametrize(
        "check_name,check_fn,expected_status,expected_message",
        [
            param(
                "Cluster Connectivity",
                preflight_checks.check_cluster_connectivity,
                CheckStatus.FAIL,
                "Failed to connect",
                id="connectivity-timeout-fails",
            ),
            param(
                "Kubernetes Version",
                preflight_checks.check_kubernetes_version,
                CheckStatus.WARN,
                "Could not determine version",
                id="version-timeout-warns",
            ),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_api_timeout_classification_distinguishes_critical_connectivity(
        self,
        check_name: str,
        check_fn: Callable[[ApiClient], Awaitable[CheckResult]],
        expected_status: CheckStatus,
        expected_message: str,
    ) -> None:
        api = MagicMock(spec=ApiClient)
        version_api = MagicMock()
        version_api.get_code = AsyncMock(side_effect=TimeoutError("slow apiserver"))

        with patch(
            "aiperf.kubernetes.preflight_checks.client.VersionApi",
            return_value=version_api,
        ):
            result = await check_fn(api)

        assert result.name == check_name
        assert result.status == expected_status
        assert expected_message in result.message
        assert "slow apiserver" in result.message

    @pytest.mark.asyncio
    async def test_check_rbac_permissions_transient_errors_warn_without_missing_permissions(
        self,
    ) -> None:
        api = MagicMock(spec=ApiClient)

        with patch(
            "aiperf.kubernetes.preflight_checks._shared_check_rbac_access",
            new=AsyncMock(
                side_effect=RuntimeError("authorization endpoint unavailable")
            ),
        ):
            result = await preflight_checks.check_rbac_permissions(
                api, namespace="llama-benchmarks"
            )

        assert result.status == CheckStatus.WARN
        assert "transient apiserver errors" in result.message
        assert any(
            "authorization endpoint unavailable" in detail for detail in result.details
        )
        assert any("Re-run preflight" in hint for hint in result.hints)


# ---------------------------------------------------------------------------
# Aggregation and short-circuiting
# ---------------------------------------------------------------------------


class TestPreflightAggregation:
    """Dispatcher errors are contained and connectivity failure remains special."""

    @pytest.mark.asyncio
    async def test_run_all_checks_non_connectivity_exception_aggregates_remaining_checks(
        self,
    ) -> None:
        checker = CLIPreflightChecker(namespace="llama-benchmarks")
        checker._check_cluster_connectivity = _passing_check
        checker._check_kubernetes_version = _raising_check
        checker._check_namespace = _async_result("Namespace")
        checker._check_rbac_permissions = _async_result("RBAC Permissions")
        checker._check_jobset_crd = _async_result("JobSet CRD")
        checker._check_jobset_controller = _async_result("JobSet Controller")
        checker._check_resource_quotas = _async_result("Resource Quotas")
        checker._check_node_resources = _async_result("Node Resources")
        checker._check_secrets = _async_result("Secrets", CheckStatus.SKIP)
        checker._check_image = _async_result("Image Pull", CheckStatus.SKIP)
        checker._check_network_policies = _async_result("Network Policies")
        checker._check_dns = _async_result("DNS Resolution")
        checker._check_endpoint_connectivity = _async_result(
            "Endpoint Connectivity", CheckStatus.SKIP
        )

        with (
            patch("aiperf.kubernetes.preflight.logger.info"),
            patch("aiperf.kubernetes.preflight.logger.error"),
        ):
            results = await checker.run_all_checks()

        assert results.passed is False
        assert len(results.checks) == 13
        version = results.checks[1]
        assert version.name == "Kubernetes Version"
        assert version.status == CheckStatus.FAIL
        assert "malformed discovery payload" in version.message
        assert results.checks[-1].name == "Endpoint Connectivity"

    @pytest.mark.asyncio
    async def test_run_all_checks_connectivity_failure_short_circuits_later_checks(
        self,
    ) -> None:
        checker = CLIPreflightChecker(namespace="llama-benchmarks")
        checker._check_cluster_connectivity = _async_result(
            "Cluster Connectivity", CheckStatus.FAIL
        )
        checker._check_kubernetes_version = _raising_check

        with (
            patch("aiperf.kubernetes.preflight.logger.info"),
            patch("aiperf.kubernetes.preflight.logger.error"),
        ):
            results = await checker.run_all_checks()

        assert results.passed is False
        assert [check.name for check in results.checks] == ["Cluster Connectivity"]


# ---------------------------------------------------------------------------
# Subprocess trust boundary
# ---------------------------------------------------------------------------


class TestPreflightSubprocessTrustBoundary:
    """Subprocess helpers preserve diagnostic stderr and classify missing tools."""

    @pytest.mark.asyncio
    async def test_run_command_nonzero_exit_preserves_stderr_for_user_diagnostics(
        self,
    ) -> None:
        proc = _ProcessStub(
            returncode=2,
            stdout=b"client version v1.31.0\n",
            stderr=b"error: context kind-aiperf-ci does not exist\n",
        )

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
            result = await subproc.run_command(
                ["kubectl", "--context", "kind-aiperf-ci", "version"], timeout=5
            )

        assert result.ok is False
        assert result.returncode == 2
        assert "client version" in result.stdout
        assert "kind-aiperf-ci does not exist" in result.stderr

    @pytest.mark.parametrize(
        "tool,cmd",
        [
            param("kubectl", ["kubectl", "version", "--client"], id="missing-kubectl"),
            param("helm", ["helm", "version", "--short"], id="missing-helm"),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_check_command_missing_kubectl_or_helm_returns_false_not_traceback(
        self, tool: str, cmd: list[str]
    ) -> None:
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(side_effect=FileNotFoundError(tool)),
        ):
            ok = await subproc.check_command(cmd, timeout=5)

        assert ok is False
