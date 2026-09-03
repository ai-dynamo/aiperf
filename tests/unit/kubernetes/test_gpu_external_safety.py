# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Safety contracts for live external GPU test runs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.kubernetes.gpu.conftest import (
    _OPTIONS,
    GPUTestSettings,
    _release_gpu,
    _resolve_settings,
    jobset_controller,
)
from tests.kubernetes.gpu.dynamo import conftest as dynamo_conftest
from tests.kubernetes.helpers.benchmark import BenchmarkDeployer


class _FakeConfig:
    """Pytest-config-shaped stub exposing both ``option`` and ``getoption``.

    ``_resolve_settings`` reads ``--gpu-*`` flags off ``config.option`` but
    reads xdist's ``numprocesses`` through the ``config.getoption`` API, so the
    stub must honour both access paths.
    """

    def __init__(self, **options: object) -> None:
        self.option = SimpleNamespace(**options)

    def getoption(self, name: str, default: object = None) -> object:
        return getattr(self.option, name, default)


def _config(**options: object) -> _FakeConfig:
    """Build a pytest-config-shaped object with no explicit CLI options."""
    return _FakeConfig(**options)


def _set_safe_external_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure every mutable GPU test namespace for the external-cluster guard."""
    monkeypatch.setenv("GPU_TEST_CONTEXT", "external")
    monkeypatch.setenv("GPU_TEST_EXTERNAL_EXISTING_OPERATOR", "1")
    monkeypatch.setenv("GPU_TEST_NAMESPACE_PREFIX", "user-scope-")
    for suffix in (
        "BENCHMARK_NAMESPACE",
        "VLLM_NAMESPACE",
        "TRTLLM_NAMESPACE",
        "SGLANG_NAMESPACE",
        "DYNAMO_NAMESPACE",
    ):
        monkeypatch.setenv(f"GPU_TEST_{suffix}", "user-scope-gpu-e2e")


def test_resolve_settings_external_cluster_requires_namespace_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External execution must reject a run with no configured namespace prefix."""
    monkeypatch.setenv("GPU_TEST_CONTEXT", "external")
    monkeypatch.setenv("GPU_TEST_EXTERNAL_EXISTING_OPERATOR", "1")

    with pytest.raises(pytest.UsageError, match="--gpu-namespace-prefix"):
        _resolve_settings(_config())


def test_resolve_settings_external_cluster_requires_user_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External execution must reject namespaces outside the configured prefix."""
    monkeypatch.setenv("GPU_TEST_CONTEXT", "external")
    monkeypatch.setenv("GPU_TEST_EXTERNAL_EXISTING_OPERATOR", "1")
    monkeypatch.setenv("GPU_TEST_NAMESPACE_PREFIX", "user-scope-")

    with pytest.raises(pytest.UsageError, match="user-scope-"):
        _resolve_settings(_config())


def test_resolve_settings_external_cluster_accepts_explicit_user_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External execution accepts only the fully explicit user-owned scope."""
    _set_safe_external_environment(monkeypatch)

    settings = _resolve_settings(_config())

    assert settings.benchmark_namespace == "user-scope-gpu-e2e"
    assert settings.vllm_namespace == "user-scope-gpu-e2e"
    assert settings.dynamo_namespace == "user-scope-gpu-e2e"
    assert settings.external_existing_operator is True


def test_resolve_settings_local_cluster_uses_dedicated_benchmark_namespace() -> None:
    """Local GPU benchmarks must generate manifests with an explicit namespace."""
    settings = _resolve_settings(_config())

    assert settings.benchmark_namespace == "aiperf-gpu-benchmark"


def test_resolve_settings_external_cluster_rejects_parallel_xdist_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parallel workers contend for the same GPUs, so -n > 1 must be refused."""
    _set_safe_external_environment(monkeypatch)

    with pytest.raises(pytest.UsageError, match="must be serial"):
        _resolve_settings(_config(numprocesses=2))


@pytest.mark.asyncio
async def test_release_gpu_external_cluster_never_lists_or_deletes_namespaces() -> None:
    """The historical GPU release loop must be inert against a shared cluster."""

    class ExternalKubectl:
        context = "external"

        async def run(self, *args: str, **kwargs: object) -> None:
            raise AssertionError(f"unexpected namespace lookup: {args}, {kwargs}")

        async def delete_namespace(self, *args: str, **kwargs: object) -> None:
            raise AssertionError(f"unexpected namespace deletion: {args}, {kwargs}")

    await _release_gpu(ExternalKubectl(), "user-scope-gpu-e2e")


def test_gpu_option_surface_contains_all_external_namespace_controls() -> None:
    """The safe external invocation does not rely on hidden fixture defaults."""
    flags = {flag for flag, *_rest in _OPTIONS}

    assert {
        "--gpu-benchmark-namespace",
        "--gpu-vllm-namespace",
        "--gpu-trtllm-namespace",
        "--gpu-sglang-namespace",
        "--gpu-dynamo-namespace",
        "--gpu-external-existing-operator",
    } <= flags


@pytest.mark.asyncio
async def test_jobset_controller_installs_for_local_kind_context_when_missing() -> None:
    """A local Kind context must not be treated as an external cluster."""

    class LocalKindKubectl:
        context = "kind-aiperf-gpu"

        def __init__(self) -> None:
            self.applied_url = ""
            self.waited_for = False

        async def run(self, *args: str, **kwargs: object) -> SimpleNamespace:
            assert args == ("get", "crd", "jobsets.jobset.x-k8s.io")
            return SimpleNamespace(returncode=1)

        async def apply_server_side(self, url: str) -> None:
            self.applied_url = url

        async def wait_for_condition(self, *args: str, **kwargs: object) -> None:
            self.waited_for = True

    kubectl = LocalKindKubectl()

    await jobset_controller.__wrapped__(kubectl, GPUTestSettings())

    assert kubectl.applied_url
    assert kubectl.waited_for


@pytest.mark.asyncio
async def test_dynamo_operator_installs_for_local_kind_context_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local Kind context must not disable Dynamo operator installation."""

    class LocalKindKubectl:
        context = "kind-aiperf-gpu"

        async def run(self, *args: str, **kwargs: object) -> SimpleNamespace:
            assert args == ("get", "crd", "dynamographdeployments.nvidia.com")
            return SimpleNamespace(returncode=1)

    monkeypatch.setattr(
        dynamo_conftest, "_dynamo_operator_is_running", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        dynamo_conftest,
        "_remove_stale_dynamo_crds_release",
        AsyncMock(side_effect=RuntimeError("installer reached")),
    )

    with pytest.raises(RuntimeError, match="installer reached"):
        await dynamo_conftest.dynamo_operator.__wrapped__(
            LocalKindKubectl(), GPUTestSettings()
        )


def test_dynamo_helm_sets_omit_removed_chart_settings() -> None:
    """Dynamo 1.x chart values must not use retired chart settings."""
    helm_sets = dynamo_conftest._dynamo_helm_sets()

    assert "dynamo-operator.webhook.enabled=false" not in helm_sets
    assert "dynamo-operator.dynamo.mpiRun.sshKeygen.enabled=false" not in helm_sets
    assert "grove.enabled=false" not in helm_sets
    assert "global.grove.install=false" in helm_sets
    assert "global.grove.enabled=false" in helm_sets
    assert "kai-scheduler.enabled=false" not in helm_sets
    assert "global.kai-scheduler.install=false" in helm_sets
    assert "global.kai-scheduler.enabled=false" in helm_sets


@pytest.mark.asyncio
async def test_pull_secret_copy_uses_only_explicit_source_namespace() -> None:
    """A safe external run must not enumerate or read other users' namespaces."""

    calls: list[tuple[str, ...]] = []

    class Kubectl:
        async def run(self, *args: str, **kwargs: object) -> SimpleNamespace:
            calls.append(args)
            if args[:3] == ("get", "secret", "nvcr-pull") and args[-1] == "target":
                return SimpleNamespace(returncode=1, stdout="")
            assert args == (
                "get",
                "secret",
                "nvcr-pull",
                "-n",
                "user-scope-aiperf-bench",
                "-o",
                "yaml",
            )
            return SimpleNamespace(
                returncode=0, stdout="apiVersion: v1\nmetadata:\n  name: nvcr-pull\n"
            )

        async def apply(self, manifest: str, namespace: str) -> None:
            assert namespace == "target"
            assert "namespace:" not in manifest

    deployer = BenchmarkDeployer(
        kubectl=Kubectl(),  # type: ignore[arg-type]
        project_root=Path.cwd(),
        default_image_pull_secret_source_namespace="user-scope-aiperf-bench",
    )

    await deployer._ensure_pull_secrets_in_namespace("target", ["nvcr-pull"])

    assert (
        "get",
        "namespaces",
        "-o",
        "jsonpath={.items[*].metadata.name}",
    ) not in calls
